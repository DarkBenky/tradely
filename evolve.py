import tensorflow as tf
import numpy as np
import os
import shutil
from dataclasses import dataclass, field

@dataclass
class PopulationMember:
    id: int
    generation: int
    parent_id: int
    fitness: float
    episodes_trained: int
    weight_path: str
    fitness_history: list = field(default_factory=list)


class Population:
    def __init__(self, population_size, elite_count, population_dir, 
                 mutation_rate=0.02, mutation_decay=0.995, crossover_rate=0.3):
        self.population_size = population_size
        self.elite_count = elite_count
        self.population_dir = population_dir
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.mutation_decay = mutation_decay
        self.crossover_rate = crossover_rate
        self.members = []
        self.member_id_counter = 0
        self.generation = 0
        self.best_ever_fitness = -float('inf')
        self.best_model_path = None
        
        os.makedirs(population_dir, exist_ok=True)
    
    def initialize(self, model, base_weights_path=None):
        if base_weights_path and os.path.exists(base_weights_path):
            model.load_weights(base_weights_path)
        
        base_weights = [w.numpy() for w in model.trainable_weights]
        
        for i in range(self.population_size):
            member_path = os.path.join(self.population_dir, f"member_{i}.weights.h5")
            
            for w, base_w in zip(model.trainable_weights, base_weights):
                w.assign(base_w)
            
            if i > 0:
                self._mutate_model(model, self.mutation_rate * (1 + i * 0.1))
            
            model.save_weights(member_path)
            
            member = PopulationMember(
                id=self.member_id_counter,
                generation=0,
                parent_id=-1 if i == 0 else 0,
                fitness=0.0,
                episodes_trained=0,
                weight_path=member_path
            )
            self.members.append(member)
            self.member_id_counter += 1
        
        for w, base_w in zip(model.trainable_weights, base_weights):
            w.assign(base_w)
        
        return self.members
    
    def _mutate_model(self, model, rate):
        for w in model.trainable_weights:
            noise = tf.random.normal(w.shape, mean=0.0, stddev=rate)
            w.assign(w + noise * tf.abs(w))
    
    def _crossover(self, model, weights1, weights2):
        for w, p1, p2 in zip(model.trainable_weights, weights1, weights2):
            mask = tf.random.uniform(w.shape) < 0.5
            new_weights = tf.where(mask, p1, p2)
            w.assign(new_weights)
    
    def update_fitness(self, member_idx, fitness_value):
        member = self.members[member_idx]
        member.fitness_history.append(fitness_value)
        if len(member.fitness_history) > 10:
            member.fitness_history.pop(0)
        member.fitness = np.mean(member.fitness_history)
    
    def evolve(self, model, save_best_path=None):
        self.members.sort(key=lambda m: m.fitness, reverse=True)
        
        best_fitness = self.members[0].fitness
        
        if best_fitness > self.best_ever_fitness:
            self.best_ever_fitness = best_fitness
            if save_best_path:
                shutil.copy(self.members[0].weight_path, save_best_path)
                self.best_model_path = save_best_path
        
        elite_weights = []
        for i in range(self.elite_count):
            model.load_weights(self.members[i].weight_path)
            elite_weights.append([w.numpy() for w in model.trainable_weights])
        
        for i in range(self.elite_count, self.population_size):
            parent_idx = i % self.elite_count
            parent = self.members[parent_idx]
            
            model.load_weights(parent.weight_path)
            
            if self.elite_count > 1 and np.random.random() < self.crossover_rate:
                other_idx = (parent_idx + 1) % self.elite_count
                self._crossover(model, elite_weights[parent_idx], elite_weights[other_idx])
            
            self._mutate_model(model, self.mutation_rate)
            
            member_path = self.members[i].weight_path
            model.save_weights(member_path)
            
            self.members[i] = PopulationMember(
                id=self.member_id_counter,
                generation=self.generation + 1,
                parent_id=parent.id,
                fitness=0.0,
                episodes_trained=0,
                weight_path=member_path
            )
            self.member_id_counter += 1
        
        for i in range(self.elite_count):
            self.members[i].generation = self.generation + 1
        
        self.mutation_rate *= self.mutation_decay
        self.generation += 1
        
        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': np.mean([m.fitness for m in self.members]),
            'worst_fitness': self.members[-1].fitness,
            'best_ever_fitness': self.best_ever_fitness,
            'mutation_rate': self.mutation_rate
        }
    
    def get_member_weight_path(self, idx):
        return self.members[idx].weight_path
    
    def increment_episodes(self, idx):
        self.members[idx].episodes_trained += 1
    
    def get_stats(self):
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'elite_count': self.elite_count,
            'mutation_rate': self.mutation_rate,
            'best_ever_fitness': self.best_ever_fitness,
            'fitnesses': [m.fitness for m in self.members]
        }
