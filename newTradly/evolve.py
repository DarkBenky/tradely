import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger
from createTraningSet import create_training_set
from params import WINDOW_SIZE, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE
import random
import json
import os
import gc

MAX_PARAMS = 100_000_000
BASE_MUTATION_RATE = 0.35
BASE_ADD_LAYER_RATE = 0.25
BASE_REMOVE_LAYER_RATE = 0.15
ELITE_RATIO = 0.5
POPULATION_SIZE = 40
GENERATIONS = 10_000
EPOCHS_PER_GEN = 3
EXPLORATION_PHASE_RATIO = 0.7
TRAIN_SAMPLES_BASE = 7_500
TRAIN_SAMPLES_MAX = 125_000

LAYER_TYPES = ['lstm', 'gru', 'dense', 'transformer', 'cnn']

def count_params(genome):
    total = 0
    input_size = WINDOW_SIZE * 5
    current_size = input_size
    
    for layer in genome:
        layer_type = layer['type']
        units = layer['units']
        
        if layer_type == 'lstm':
            total += 4 * units * (current_size + units + 1)
            current_size = units
        elif layer_type == 'gru':
            total += 3 * units * (current_size + units + 1)
            current_size = units
        elif layer_type == 'dense':
            total += (current_size + 1) * units
            current_size = units
        elif layer_type == 'transformer':
            num_heads = layer.get('num_heads', 4)
            key_dim = units // num_heads
            total += units * current_size * 3 + units * key_dim * num_heads
            total += units * 4 * units + units
            current_size = units
        elif layer_type == 'cnn':
            filters = units
            kernel_size = layer.get('kernel_size', 3)
            total += kernel_size * current_size * filters + filters
            current_size = filters
    
    total += current_size * 3
    return total

def create_random_genome():
    genome = []
    remaining_params = MAX_PARAMS
    input_size = WINDOW_SIZE * 5
    
    while remaining_params > 1000:
        layer_type = random.choice(LAYER_TYPES)
        max_units = min(2048, remaining_params // 10)
        
        if max_units < 16:
            break
            
        units = random.randint(16, max_units)
        
        layer = {'type': layer_type, 'units': units}
        
        if layer_type == 'transformer':
            layer['num_heads'] = random.choice([2, 4, 8])
        elif layer_type == 'cnn':
            layer['kernel_size'] = random.choice([3, 5, 7])
        
        genome.append(layer)
        test_params = count_params(genome)
        
        if test_params > MAX_PARAMS:
            genome.pop()
            break
        
        remaining_params = MAX_PARAMS - test_params
        
        if len(genome) > 10 or random.random() < 0.3:
            break
    
    if len(genome) == 0:
        genome.append({'type': 'dense', 'units': 1024})
    
    return genome

def build_model_from_genome(genome, input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for i, layer in enumerate(genome):
        layer_type = layer['type']
        units = layer['units']
        
        try:
            if layer_type == 'lstm':
                if len(x.shape) == 2:
                    x = layers.Reshape((1, -1))(x)
                return_sequences = i < len(genome) - 1 and genome[i+1]['type'] in ['lstm', 'gru', 'transformer', 'cnn']
                x = layers.LSTM(units, return_sequences=return_sequences)(x)
            elif layer_type == 'gru':
                if len(x.shape) == 2:
                    x = layers.Reshape((1, -1))(x)
                return_sequences = i < len(genome) - 1 and genome[i+1]['type'] in ['lstm', 'gru', 'transformer', 'cnn']
                x = layers.GRU(units, return_sequences=return_sequences)(x)
            elif layer_type == 'dense':
                if len(x.shape) == 3:
                    x = layers.Flatten()(x)
                x = layers.Dense(units, activation='relu')(x)
            elif layer_type == 'transformer':
                num_heads = layer.get('num_heads', 4)
                if num_heads > units:
                    num_heads = max(1, units // 4)
                key_dim = max(1, units // num_heads)
                
                if len(x.shape) == 2:
                    x = layers.Reshape((1, -1))(x)
                
                current_dim = int(x.shape[-1])
                if current_dim != units:
                    x = layers.Dense(units)(x)
                
                attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
                x = layers.LayerNormalization()(x + attn)
                ffn = layers.Dense(units * 4, activation='relu')(x)
                ffn = layers.Dense(units)(ffn)
                x = layers.LayerNormalization()(x + ffn)
            elif layer_type == 'cnn':
                kernel_size = layer.get('kernel_size', 3)
                if len(x.shape) == 2:
                    x = layers.Reshape((1, -1))(x)
                
                seq_len = int(x.shape[1])
                if kernel_size > seq_len:
                    kernel_size = max(1, seq_len)
                
                current_dim = int(x.shape[-1])
                if current_dim != units:
                    x = layers.Conv1D(units, 1, padding='same')(x)
                
                x = layers.Conv1D(units, kernel_size, padding='same', activation='relu')(x)
                if seq_len >= 4:
                    x = layers.MaxPooling1D(2, padding='same')(x)
            
            x = layers.Dropout(DROPOUT_RATE)(x)
        except Exception as e:
            print(f"Warning: Skipping {layer_type} layer due to: {e}")
            continue
    
    if len(x.shape) == 3:
        x = layers.Flatten()(x)
    
    output_next = layers.Dense(1, name='next')(x)
    output_half = layers.Dense(1, name='half')(x)
    output_full = layers.Dense(1, name='full')(x)
    
    model = keras.Model(inputs=inputs, outputs=[output_next, output_half, output_full])
    return model

def mutate_genome(genome, mutation_rate, add_layer_rate, remove_layer_rate):
    genome = [layer.copy() for layer in genome]
    
    if random.random() < add_layer_rate and len(genome) < 10:
        new_layer_type = random.choice(LAYER_TYPES)
        remaining = MAX_PARAMS - count_params(genome)
        max_units = min(1024, remaining // 10)
        
        if max_units >= 16:
            units = random.randint(16, max_units)
            layer = {'type': new_layer_type, 'units': units}
            
            if new_layer_type == 'transformer':
                layer['num_heads'] = random.choice([2, 4, 8])
            elif new_layer_type == 'cnn':
                layer['kernel_size'] = random.choice([3, 5, 7])
            
            insert_pos = random.randint(0, len(genome))
            genome.insert(insert_pos, layer)
            
            if count_params(genome) > MAX_PARAMS:
                genome.pop(insert_pos)
    
    if random.random() < remove_layer_rate and len(genome) > 1:
        genome.pop(random.randint(0, len(genome) - 1))
    
    if random.random() < mutation_rate and len(genome) > 0:
        idx = random.randint(0, len(genome) - 1)
        remaining = MAX_PARAMS - count_params(genome) + count_params([genome[idx]])
        max_units = min(2048, remaining // 10)
        
        if max_units >= 16:
            genome[idx]['units'] = random.randint(16, max_units)
            
            if count_params(genome) > MAX_PARAMS:
                genome[idx]['units'] = genome[idx]['units'] // 2
    
    return genome

def evaluate_individual(genome, X_train, y_train, X_val, y_val, generation, individual_id, epochs=1, use_callbacks=False):
    try:
        model = build_model_from_genome(genome, (WINDOW_SIZE, 5))
        
        initial_lr = LEARNING_RATE if epochs == 1 else LEARNING_RATE * 0.5
        optimizer = keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=0.01)
        model.compile(
            optimizer=optimizer,
            loss={'next': 'mse', 'half': 'mse', 'full': 'mse'},
            metrics={'next': ['mae'], 'half': ['mae'], 'full': ['mae']}
        )
        
        callbacks = []
        if use_callbacks and epochs > 1:
            callbacks.extend([
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                    verbose=0
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=0
                )
            ])
        
        history = model.fit(
            X_train,
            {'next': y_train[0], 'half': y_train[1], 'full': y_train[2]},
            validation_data=(X_val, {'next': y_val[0], 'half': y_val[1], 'full': y_val[2]}),
            batch_size=BATCH_SIZE,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )
        
        val_loss = history.history['val_loss'][-1]
        fitness = -val_loss
        
        layer_counts = {}
        total_units = 0
        for layer in genome:
            layer_type = layer['type']
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            total_units += layer['units']
        
        individual_metrics = {
            'fitness': fitness,
            'val_loss': val_loss,
            'num_layers': len(genome),
            'total_params': count_params(genome),
            'total_units': total_units,
            'lstm_count': layer_counts.get('lstm', 0),
            'gru_count': layer_counts.get('gru', 0),
            'dense_count': layer_counts.get('dense', 0),
            'transformer_count': layer_counts.get('transformer', 0),
            'cnn_count': layer_counts.get('cnn', 0),
        }
        
        model.save(f'models/gen_{generation}_ind_{individual_id}.keras')
        
        del model
        keras.backend.clear_session()
        tf.keras.backend.clear_session()
        gc.collect()
        
        return fitness, individual_metrics
    except Exception as e:
        print(f"Error evaluating genome: {e}")
        return -1e9, None

def evolve():
    os.makedirs('models', exist_ok=True)
    os.makedirs('genomes', exist_ok=True)
    
    checkpoint_file = 'evolution_checkpoint.json'
    start_generation = 0
    population = []
    best_fitness_ever = -1e9
    best_genome_ever = None
    
    if os.path.exists(checkpoint_file):
        print("Loading checkpoint...")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        start_generation = checkpoint['generation'] + 1
        population = checkpoint['population']
        best_fitness_ever = checkpoint['best_fitness_ever']
        best_genome_ever = checkpoint['best_genome_ever']
        print(f"Resuming from generation {start_generation}")
        print(f"Best fitness so far: {best_fitness_ever:.6f}")
    
    wandb.init(project="tradely-evolution", config={
        "population_size": POPULATION_SIZE,
        "generations": GENERATIONS,
        "max_params": MAX_PARAMS,
        "base_mutation_rate": BASE_MUTATION_RATE,
        "base_add_layer_rate": BASE_ADD_LAYER_RATE,
        "base_remove_layer_rate": BASE_REMOVE_LAYER_RATE,
        "elite_ratio": ELITE_RATIO
    })
    
    print("Generating initial training data...")
    X_train, (y_next, y_half, y_full) = create_training_set(TRAIN_SAMPLES_BASE, 5_000)
    print(f"Training data: {X_train.shape}")
    
    print("Generating validation data...")
    X_val, (y_val_next, y_val_half, y_val_full) = create_training_set(2500, 1000)
    
    y_train = (y_next, y_half, y_full)
    y_val = (y_val_next, y_val_half, y_val_full)
    
    exploration_phase_end = int(GENERATIONS * EXPLORATION_PHASE_RATIO)
    
    if len(population) == 0:
        for i in range(POPULATION_SIZE):
            genome = create_random_genome()
            population.append({'genome': genome, 'fitness': None})
            print(f"Individual {i+1}: {len(genome)} layers, {count_params(genome)} params")
    
    for generation in range(start_generation, GENERATIONS):
        print(f"\n=== Generation {generation + 1}/{GENERATIONS} ===")
        
        progress = generation / GENERATIONS
        exploration_phase = GENERATIONS * EXPLORATION_PHASE_RATIO
        
        if generation < exploration_phase:
            decay_factor = 1.0
            epochs_this_gen = EPOCHS_PER_GEN
            use_callbacks = False
            
            if generation > 0 and generation % 10 == 0:
                print("\nRefreshing training data...")
                X_train, (y_next, y_half, y_full) = create_training_set(TRAIN_SAMPLES_BASE, 5_000)
                y_train = (y_next, y_half, y_full)
                print(f"New training data: {X_train.shape}")
                
                print("Refreshing validation data...")
                X_val, (y_val_next, y_val_half, y_val_full) = create_training_set(2500, 1000)
                y_val = (y_val_next, y_val_half, y_val_full)
        else:
            consolidation_progress = (generation - exploration_phase) / (GENERATIONS - exploration_phase)
            decay_factor = 1.0 - consolidation_progress
            epochs_this_gen = EPOCHS_PER_GEN + int(consolidation_progress * 9)
            use_callbacks = True
            
            if generation == exploration_phase_end:
                print("\n*** ENTERING CONSOLIDATION PHASE ***")
                print("Regenerating larger training dataset...")
                train_samples = int(TRAIN_SAMPLES_BASE + (TRAIN_SAMPLES_MAX - TRAIN_SAMPLES_BASE) * 0.5)
                X_train, (y_next, y_half, y_full) = create_training_set(train_samples, 2000)
                y_train = (y_next, y_half, y_full)
                print(f"New training data: {X_train.shape}")
            elif generation > exploration_phase_end and (generation - exploration_phase_end) % 500 == 0:
                print("\nScaling up training data...")
                train_samples = int(TRAIN_SAMPLES_BASE + (TRAIN_SAMPLES_MAX - TRAIN_SAMPLES_BASE) * consolidation_progress)
                X_train, (y_next, y_half, y_full) = create_training_set(train_samples, 2000)
                y_train = (y_next, y_half, y_full)
                print(f"New training data: {X_train.shape}")
        
        mutation_rate = BASE_MUTATION_RATE * decay_factor
        add_layer_rate = BASE_ADD_LAYER_RATE * decay_factor
        remove_layer_rate = BASE_REMOVE_LAYER_RATE * decay_factor
        
        print(f"Phase: {'Exploration' if generation < exploration_phase else 'Consolidation'}")
        print(f"Mutation rates: {mutation_rate:.3f} / Add: {add_layer_rate:.3f} / Remove: {remove_layer_rate:.3f}")
        print(f"Epochs: {epochs_this_gen} | Training samples: {len(X_train)}")
        
        for i, individual in enumerate(population):
            if individual['fitness'] is None:
                print(f"Evaluating individual {i+1}/{len(population)}...")
                fitness, metrics = evaluate_individual(
                    individual['genome'],
                    X_train, y_train,
                    X_val, y_val,
                    generation, i, epochs_this_gen, use_callbacks
                )
                individual['fitness'] = fitness
                individual['metrics'] = metrics
                print(f"Fitness: {fitness:.6f}")
        
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        best_fitness = population[0]['fitness']
        avg_fitness = np.mean([ind['fitness'] for ind in population])
        
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_genome_ever = population[0]['genome']
        
        print(f"\nBest fitness: {best_fitness:.6f}")
        print(f"Average fitness: {avg_fitness:.6f}")
        print(f"Best genome: {len(population[0]['genome'])} layers, {count_params(population[0]['genome'])} params")
        
        best_genome = population[0]['genome']
        layer_counts = {}
        layer_units = []
        for layer in best_genome:
            layer_type = layer['type']
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            layer_units.append(layer['units'])
        
        wandb.log({
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "best_params": count_params(best_genome),
            "best_layers": len(best_genome),
            "best_lstm_count": layer_counts.get('lstm', 0),
            "best_gru_count": layer_counts.get('gru', 0),
            "best_dense_count": layer_counts.get('dense', 0),
            "best_transformer_count": layer_counts.get('transformer', 0),
            "best_cnn_count": layer_counts.get('cnn', 0),
            "best_avg_units": np.mean(layer_units) if layer_units else 0,
            "best_max_units": max(layer_units) if layer_units else 0,
            "best_min_units": min(layer_units) if layer_units else 0,
            "mutation_rate": mutation_rate,
            "add_layer_rate": add_layer_rate,
            "remove_layer_rate": remove_layer_rate,
            "epochs_per_gen": epochs_this_gen,
            "population_avg_params": np.mean([count_params(ind['genome']) for ind in population]),
            "population_avg_layers": np.mean([len(ind['genome']) for ind in population]),
            "population_fitness_std": np.std([ind['fitness'] for ind in population]),
        })
        
        num_elites = int(POPULATION_SIZE * ELITE_RATIO)
        elites = population[:num_elites]
        
        new_population = [{'genome': elite['genome'], 'fitness': elite['fitness'], 'metrics': elite.get('metrics')} for elite in elites]
        
        while len(new_population) < POPULATION_SIZE:
            parent = random.choice(elites)
            child_genome = mutate_genome(parent['genome'], mutation_rate, add_layer_rate, remove_layer_rate)
            new_population.append({'genome': child_genome, 'fitness': None, 'metrics': None})
        
        population = new_population
        
        with open(f'genomes/best_genome_gen_{generation}.json', 'w') as f:
            json.dump(population[0]['genome'], f, indent=2)
        
        checkpoint = {
            'generation': generation,
            'population': population,
            'best_fitness_ever': best_fitness_ever,
            'best_genome_ever': best_genome_ever
        }
        with open('evolution_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"Checkpoint saved at generation {generation}")
        
        for model_file in os.listdir('models'):
            if model_file.startswith('gen_') and model_file.endswith('.keras'):
                file_gen = int(model_file.split('_')[1])
                if file_gen < generation:
                    os.remove(f'models/{model_file}')
    
    print(f"\n=== Evolution Complete ===")
    print(f"Best fitness ever: {best_fitness_ever:.6f}")
    print(f"Best genome: {len(best_genome_ever)} layers, {count_params(best_genome_ever)} params")
    
    with open('genomes/best_genome_final.json', 'w') as f:
        json.dump(best_genome_ever, f, indent=2)
    
    print("\nBuilding and saving final model...")
    final_model = build_model_from_genome(best_genome_ever, (WINDOW_SIZE, 5))
    optimizer = keras.optimizers.AdamW(learning_rate=LEARNING_RATE)
    final_model.compile(
        optimizer=optimizer,
        loss={'next': 'mse', 'half': 'mse', 'full': 'mse'},
        metrics={'next': ['mae'], 'half': ['mae'], 'full': ['mae']}
    )
    
    final_model.fit(
        X_train,
        {'next': y_train[0], 'half': y_train[1], 'full': y_train[2]},
        validation_data=(X_val, {'next': y_val[0], 'half': y_val[1], 'full': y_val[2]}),
        batch_size=BATCH_SIZE,
        epochs=5,
        verbose=1
    )
    
    final_model.save('models/evolved_model.keras')
    print("Evolved model saved to models/evolved_model.keras")
    
    wandb.finish()

if __name__ == "__main__":
    evolve()
