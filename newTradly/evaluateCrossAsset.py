import tensorflow as tf
import numpy as np
from crossAssetData import create_cross_asset_dataset, ASSETS, WINDOW_SIZE, HORIZON
from crossAssetModel import create_cross_asset_model
import json

def evaluate_model(model_path: str, n_batches: int = 50):
    model = tf.keras.models.load_model(model_path, compile=False)
    
    all_predictions = []
    all_targets = []
    
    print(f"Evaluating model on {n_batches} batches...")
    
    for batch in range(n_batches):
        try:
            X, y = create_cross_asset_dataset(32)
            y_pred = model(X, training=False).numpy()
            
            all_predictions.append(y_pred)
            all_targets.append(y)
        except Exception as e:
            print(f"Error in batch {batch}: {e}")
            continue
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    rank_corr = []
    for i in range(len(predictions)):
        pred_rank = np.argsort(predictions[i])
        true_rank = np.argsort(targets[i])
        corr = np.corrcoef(pred_rank, true_rank)[0, 1]
        rank_corr.append(corr)
    
    avg_rank_corr = np.mean(rank_corr)
    
    print("\n=== Evaluation Results ===")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Rank Correlation: {avg_rank_corr:.4f}")
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rank_correlation": float(avg_rank_corr)
    }

def backtest_strategy(model_path: str, n_periods: int = 100, top_k: int = 3):
    model = tf.keras.models.load_model(model_path, compile=False)
    
    portfolio_values = [1.0]
    positions = []
    
    print(f"\nBacktesting strategy over {n_periods} periods...")
    print(f"Strategy: Long top {top_k}, Short bottom {top_k}")
    
    for period in range(n_periods):
        try:
            X, y = create_cross_asset_dataset(1)
            
            scores = model(X, training=False).numpy()[0]
            
            scores = scores - scores.mean()
            
            ranked_indices = np.argsort(scores)
            
            long_positions = ranked_indices[-top_k:]
            short_positions = ranked_indices[:top_k]
            
            actual_returns = y[0]
            
            long_return = np.mean(actual_returns[long_positions])
            short_return = np.mean(actual_returns[short_positions])
            
            period_return = (long_return - short_return) / 2
            
            portfolio_values.append(portfolio_values[-1] * (1 + period_return))
            
            positions.append({
                "period": period,
                "long": long_positions.tolist(),
                "short": short_positions.tolist(),
                "return": float(period_return),
                "portfolio_value": portfolio_values[-1]
            })
            
            if period % 10 == 0:
                print(f"Period {period}: Return={period_return:.4f}, Portfolio={portfolio_values[-1]:.4f}")
        
        except Exception as e:
            print(f"Error in period {period}: {e}")
            portfolio_values.append(portfolio_values[-1])
            continue
    
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] - 1.0) * 100
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_dd = np.min(portfolio_values / np.maximum.accumulate(portfolio_values) - 1) * 100
    
    print("\n=== Backtest Results ===")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Final Portfolio Value: {portfolio_values[-1]:.4f}")
    
    results = {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "final_value": float(portfolio_values[-1]),
        "positions": positions
    }
    
    return results

def analyze_predictions(model_path: str, n_samples: int = 10):
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f"\nAnalyzing {n_samples} sample predictions...")
    
    for sample in range(n_samples):
        try:
            X, y = create_cross_asset_dataset(1)
            scores = model(X, training=False).numpy()[0]
            
            print(f"\nSample {sample + 1}:")
            print("Asset | Predicted | Actual  | Diff")
            print("-" * 40)
            
            for i in range(ASSETS):
                print(f"  {i:2d}  | {scores[i]:8.4f} | {y[0, i]:7.4f} | {scores[i] - y[0, i]:7.4f}")
            
            pred_rank = np.argsort(scores)
            true_rank = np.argsort(y[0])
            
            print(f"\nTop 3 Predicted: {pred_rank[-3:][::-1].tolist()}")
            print(f"Top 3 Actual:    {true_rank[-3:][::-1].tolist()}")
            
        except Exception as e:
            print(f"Error in sample {sample}: {e}")
            continue

if __name__ == "__main__":
    import sys
    
    model_path = "models/cross_asset_best.keras"
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "eval":
            results = evaluate_model(model_path)
            with open("evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)
        
        elif command == "backtest":
            results = backtest_strategy(model_path)
            with open("backtest_results.json", "w") as f:
                json.dump(results, f, indent=2)
        
        elif command == "analyze":
            analyze_predictions(model_path)
        
        else:
            print("Usage: python evaluateCrossAsset.py [eval|backtest|analyze]")
    
    else:
        print("Running all evaluations...")
        eval_results = evaluate_model(model_path)
        backtest_results = backtest_strategy(model_path)
        analyze_predictions(model_path, n_samples=5)
        
        all_results = {
            "evaluation": eval_results,
            "backtest": backtest_results
        }
        
        with open("full_evaluation.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("\nResults saved to full_evaluation.json")
