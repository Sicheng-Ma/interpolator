"""
Performance and Profiling Script for the 5D Regressor.

This script benchmarks training time, memory usage, and model accuracy
across different dataset sizes as required by Task 8.

Usage:
    python benchmark.py
    
Output:
    - Console output with detailed metrics
    - benchmark_results.json with raw data
    - benchmark_report.md with formatted report
"""

import json
import time
import tracemalloc
import gc
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

from fivedreg import create_sample_dataset, FiveDRegressor


def measure_memory(func):
    """Decorator to measure peak memory usage of a function."""
    def wrapper(*args, **kwargs):
        gc.collect()
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return result, peak / 1024 / 1024  # Convert to MB
    return wrapper


def benchmark_training(n_samples: int, config: Dict[str, Any], n_runs: int = 3) -> Dict[str, Any]:
    """
    Benchmark training performance for a given dataset size.
    
    Args:
        n_samples: Number of samples in the dataset
        config: Model configuration
        n_runs: Number of runs to average
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking with {n_samples:,} samples")
    print(f"{'='*60}")
    
    training_times = []
    memory_usages = []
    metrics_list = []
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end=" ")
        
        # Create fresh dataset for each run
        dataset = create_sample_dataset(n_samples=n_samples, random_state=42 + run)
        dataset.handle_missing_values(strategy="mean")
        train, val, test = dataset.split(test_size=0.2, val_size=0.1, random_state=42)
        
        # Create model
        model = FiveDRegressor(
            hidden_layers=config["hidden_layers"],
            learning_rate=config["learning_rate"],
            max_epochs=config["max_epochs"],
            batch_size=config["batch_size"],
            patience=config["patience"],
            verbose=False,
            random_state=42
        )
        
        # Measure training time and memory
        gc.collect()
        tracemalloc.start()
        
        start_time = time.perf_counter()
        model.fit(train.X, train.y)
        end_time = time.perf_counter()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        training_time = end_time - start_time
        memory_mb = peak / 1024 / 1024
        
        # Get metrics
        metrics = model.score(test.X, test.y)
        
        training_times.append(training_time)
        memory_usages.append(memory_mb)
        metrics_list.append(metrics)
        
        print(f"Time: {training_time:.2f}s, Memory: {memory_mb:.1f}MB, R²: {metrics['r2']:.4f}")
    
    # Calculate averages
    results = {
        "n_samples": n_samples,
        "n_runs": n_runs,
        "training_time": {
            "mean": np.mean(training_times),
            "std": np.std(training_times),
            "min": np.min(training_times),
            "max": np.max(training_times),
            "all_runs": training_times
        },
        "memory_mb": {
            "mean": np.mean(memory_usages),
            "std": np.std(memory_usages),
            "peak": np.max(memory_usages),
            "all_runs": memory_usages
        },
        "metrics": {
            "mse": {
                "mean": np.mean([m["mse"] for m in metrics_list]),
                "std": np.std([m["mse"] for m in metrics_list])
            },
            "rmse": {
                "mean": np.mean([m["rmse"] for m in metrics_list]),
                "std": np.std([m["rmse"] for m in metrics_list])
            },
            "mae": {
                "mean": np.mean([m["mae"] for m in metrics_list]),
                "std": np.std([m["mae"] for m in metrics_list])
            },
            "r2": {
                "mean": np.mean([m["r2"] for m in metrics_list]),
                "std": np.std([m["r2"] for m in metrics_list])
            }
        }
    }
    
    print(f"\n  Summary for {n_samples:,} samples:")
    print(f"    Avg Training Time: {results['training_time']['mean']:.2f}s ± {results['training_time']['std']:.2f}s")
    print(f"    Peak Memory: {results['memory_mb']['peak']:.1f} MB")
    print(f"    Avg R² Score: {results['metrics']['r2']['mean']:.4f} ± {results['metrics']['r2']['std']:.4f}")
    print(f"    Avg RMSE: {results['metrics']['rmse']['mean']:.4f} ± {results['metrics']['rmse']['std']:.4f}")
    
    return results


def benchmark_prediction(model: FiveDRegressor, n_samples_list: List[int]) -> Dict[str, Any]:
    """
    Benchmark prediction performance for different batch sizes.
    
    Args:
        model: Trained model
        n_samples_list: List of sample sizes to test
        
    Returns:
        Dictionary with prediction benchmark results
    """
    print(f"\n{'='*60}")
    print("Benchmarking Prediction Performance")
    print(f"{'='*60}")
    
    results = {}
    
    for n_samples in n_samples_list:
        X = np.random.randn(n_samples, 5).astype(np.float32)
        
        # Warm-up
        _ = model.predict(X[:10])
        
        # Measure prediction time
        gc.collect()
        tracemalloc.start()
        
        start_time = time.perf_counter()
        _ = model.predict(X)
        end_time = time.perf_counter()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        pred_time = end_time - start_time
        memory_mb = peak / 1024 / 1024
        time_per_sample = pred_time / n_samples * 1000  # ms
        
        results[n_samples] = {
            "total_time_ms": pred_time * 1000,
            "time_per_sample_ms": time_per_sample,
            "memory_mb": memory_mb,
            "samples_per_second": n_samples / pred_time
        }
        
        print(f"  {n_samples:,} samples: {pred_time*1000:.2f}ms total, {time_per_sample:.4f}ms/sample, {memory_mb:.1f}MB")
    
    return results


def generate_report(all_results: Dict[str, Any], output_path: str = "benchmark_report.md"):
    """Generate a markdown report from benchmark results."""
    
    report = f"""# Performance and Profiling Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Overview

This report presents the performance analysis of the 5D Neural Network Regressor,
measuring training time, memory usage, and model accuracy across different dataset sizes.

### Model Configuration

- **Architecture:** {all_results['config']['hidden_layers']}
- **Learning Rate:** {all_results['config']['learning_rate']}
- **Max Epochs:** {all_results['config']['max_epochs']}
- **Batch Size:** {all_results['config']['batch_size']}
- **Early Stopping Patience:** {all_results['config']['patience']}

## 2. Training Performance

### 2.1 Training Time vs Dataset Size

| Dataset Size | Training Time (s) | Std Dev (s) |
|--------------|-------------------|-------------|
"""
    
    for result in all_results['training_results']:
        n = result['n_samples']
        t_mean = result['training_time']['mean']
        t_std = result['training_time']['std']
        report += f"| {n:,} | {t_mean:.2f} | {t_std:.2f} |\n"
    
    report += """
### 2.2 Memory Usage vs Dataset Size

| Dataset Size | Peak Memory (MB) | Avg Memory (MB) |
|--------------|------------------|-----------------|
"""
    
    for result in all_results['training_results']:
        n = result['n_samples']
        m_peak = result['memory_mb']['peak']
        m_mean = result['memory_mb']['mean']
        report += f"| {n:,} | {m_peak:.1f} | {m_mean:.1f} |\n"
    
    report += """
### 2.3 Scaling Analysis

"""
    
    # Calculate scaling factors
    results = all_results['training_results']
    if len(results) >= 2:
        t1, t2 = results[0]['training_time']['mean'], results[-1]['training_time']['mean']
        n1, n2 = results[0]['n_samples'], results[-1]['n_samples']
        scaling_factor = (t2 / t1) / (n2 / n1)
        report += f"- Training time scaling: When dataset size increases {n2/n1:.0f}x (from {n1:,} to {n2:,}), "
        report += f"training time increases {t2/t1:.1f}x\n"
        report += f"- This indicates approximately **{'linear' if 0.8 < scaling_factor < 1.2 else 'sub-linear' if scaling_factor < 0.8 else 'super-linear'}** scaling\n"
    
    report += """
## 3. Model Accuracy

### 3.1 Accuracy Metrics vs Dataset Size

| Dataset Size | R² Score | RMSE | MSE | MAE |
|--------------|----------|------|-----|-----|
"""
    
    for result in all_results['training_results']:
        n = result['n_samples']
        r2 = result['metrics']['r2']['mean']
        rmse = result['metrics']['rmse']['mean']
        mse = result['metrics']['mse']['mean']
        mae = result['metrics']['mae']['mean']
        report += f"| {n:,} | {r2:.4f} | {rmse:.4f} | {mse:.6f} | {mae:.4f} |\n"
    
    report += """
### 3.2 Accuracy Observations

"""
    r2_values = [r['metrics']['r2']['mean'] for r in results]
    report += f"- Best R² Score: {max(r2_values):.4f} (at {results[r2_values.index(max(r2_values))]['n_samples']:,} samples)\n"
    report += f"- R² Score range: {min(r2_values):.4f} to {max(r2_values):.4f}\n"
    
    if max(r2_values) - min(r2_values) < 0.01:
        report += "- Model accuracy is **stable** across different dataset sizes\n"
    else:
        report += "- Larger datasets generally lead to better model accuracy\n"
    
    report += """
## 4. Prediction Performance

### 4.1 Inference Time

| Batch Size | Total Time (ms) | Time per Sample (ms) | Throughput (samples/s) |
|------------|-----------------|----------------------|------------------------|
"""
    
    for n, data in all_results['prediction_results'].items():
        report += f"| {n:,} | {data['total_time_ms']:.2f} | {data['time_per_sample_ms']:.4f} | {data['samples_per_second']:,.0f} |\n"
    
    report += """
## 5. Conclusions

### 5.1 Performance Summary

"""
    
    # Training time for 10K
    result_10k = next((r for r in results if r['n_samples'] == 10000), results[-1])
    report += f"- **Training Time (10K samples):** {result_10k['training_time']['mean']:.2f}s "
    report += f"({'✅ Under 1 minute requirement' if result_10k['training_time']['mean'] < 60 else '❌ Exceeds 1 minute'})\n"
    report += f"- **Peak Memory:** {max(r['memory_mb']['peak'] for r in results):.1f} MB\n"
    report += f"- **Best R² Score:** {max(r2_values):.4f}\n"
    
    report += """
### 5.2 Recommendations

1. **For Production:** The model trains efficiently on CPU, meeting the <1 minute requirement for 10K samples
2. **Memory Efficiency:** Memory usage scales reasonably with dataset size
3. **Batch Prediction:** Larger batch sizes are more efficient for bulk predictions

---
*Report generated by benchmark.py*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")
    return report


def main():
    """Run the complete benchmark suite."""
    print("=" * 60)
    print("5D Regressor Performance Benchmark")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model configuration
    config = {
        "hidden_layers": [64, 32, 16],
        "learning_rate": 0.005,
        "max_epochs": 100,
        "batch_size": 256,
        "patience": 20
    }
    
    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Dataset sizes to benchmark
    dataset_sizes = [1000, 5000, 10000]
    
    # Run training benchmarks
    training_results = []
    for n_samples in dataset_sizes:
        result = benchmark_training(n_samples, config, n_runs=3)
        training_results.append(result)
    
    # Train a model for prediction benchmarks
    print("\nTraining model for prediction benchmarks...")
    dataset = create_sample_dataset(n_samples=5000, random_state=42)
    model = FiveDRegressor(**config, verbose=False, random_state=42)
    model.fit(dataset.X, dataset.y)
    
    # Run prediction benchmarks
    prediction_results = benchmark_prediction(model, [100, 1000, 5000, 10000])
    
    # Compile all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "training_results": training_results,
        "prediction_results": prediction_results
    }
    
    # Save raw results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nRaw results saved to: benchmark_results.json")
    
    # Generate markdown report
    generate_report(all_results)
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()