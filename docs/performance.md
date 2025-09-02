# Performance Benchmarking and Optimization

OpenEval Lab includes comprehensive performance monitoring and benchmarking tools to help optimize evaluation workflows and adapter performance.

## Features

### Performance Monitoring
- Real-time memory usage tracking
- CPU utilization monitoring
- Disk I/O metrics
- Latency statistics (mean, median, P95, P99)
- Throughput calculations

### Benchmarking Suite
- Function-level benchmarking with warmup runs
- Adapter performance profiling with different batch sizes
- Comparison benchmarking between multiple implementations
- Detailed performance reports with optimization recommendations

### System Metrics
- Peak memory usage detection
- Memory delta tracking
- Background memory monitoring
- Cross-platform system information

## Usage

### Basic Performance Monitoring
Enable performance monitoring during evaluation:
```bash
openeval run spec.json --benchmark
```

### Memory Profiling
Enable detailed memory profiling:
```bash
openeval run spec.json --profile-memory
```

### Save Performance Report
Save detailed performance metrics to file:
```bash
openeval run spec.json --benchmark --save-performance performance_report.json
```

### Adapter Benchmarking
Benchmark adapter performance across different batch sizes:
```bash
openeval benchmark spec.json --batch-sizes "1,5,10,20,50" --iterations 20 --output adapter_benchmark.json
```

### Advanced Benchmarking
Benchmark with custom configuration:
```bash
openeval benchmark spec.json \
  --batch-sizes "1,2,4,8,16,32" \
  --iterations 50 \
  --output detailed_benchmark.json
```

## Performance Report Structure

### Basic Performance Metrics
```json
{
  "performance": {
    "wall_time": 45.23,
    "cpu_time": 42.18,
    "peak_memory_mb": 512.7,
    "memory_delta_mb": 245.3,
    "cpu_percent": 85.2,
    "throughput": 2.21,
    "latency_stats": {
      "mean": 0.452,
      "median": 0.431,
      "p95": 0.789,
      "p99": 1.234,
      "min": 0.123,
      "max": 2.456,
      "std": 0.234
    }
  }
}
```

### Benchmark Report
```json
{
  "adapter": "OpenAIAdapter",
  "benchmark_config": {
    "batch_sizes": [1, 5, 10, 20],
    "iterations": 10,
    "examples_count": 100
  },
  "results": {
    "OpenAIAdapter_batch_1": [...],
    "OpenAIAdapter_batch_5": [...]
  },
  "recommendations": [
    {
      "type": "batch_size_optimization",
      "adapter": "OpenAIAdapter",
      "recommendation": "Use batch size 10 for optimal throughput",
      "throughput_data": {
        "1": 1.2,
        "5": 4.8,
        "10": 8.9,
        "20": 8.1
      }
    }
  ]
}
```

## Integration with Code

### Performance Context Manager
```python
from openeval.performance import performance_context, PerformanceMonitor

monitor = PerformanceMonitor()
with performance_context(monitor, "my_operation", item_count=100):
    # Your evaluation code here
    pass
```

### Function Benchmarking
```python
from openeval.performance import benchmark_function

@benchmark_function(iterations=100, warmup=10)
def my_evaluation_function():
    # Your function to benchmark
    pass

results = my_evaluation_function()
print(results['benchmark_summary'])
```

### Adapter Profiling
```python
from openeval.performance import AdapterPerformanceProfiler

profiler = AdapterPerformanceProfiler()
profiler.profile_adapter(adapter, examples, batch_sizes=[1, 5, 10, 20])
recommendations = profiler.get_optimization_recommendations()
```

### Custom Performance Suite
```python
from openeval.performance import create_performance_suite

suite = create_performance_suite()
monitor = suite['monitor']
benchmark = suite['benchmark']
```

## Optimization Recommendations

The system automatically provides optimization recommendations:

### Batch Size Optimization
- Analyzes throughput across different batch sizes
- Identifies optimal batch size for maximum efficiency
- Considers memory constraints and latency requirements

### Memory Optimization
- Identifies high-memory operations (>1GB)
- Suggests memory optimization strategies
- Recommends smaller batch sizes for memory-constrained environments

### Performance Patterns
- Detects performance degradation patterns
- Identifies bottlenecks in evaluation pipelines
- Suggests architectural improvements

## System Requirements

### Optional Dependencies
- `psutil`: For detailed system metrics (CPU, memory, disk I/O)
- Without psutil, basic timing metrics are still available

### Installation
```bash
pip install psutil  # For full system monitoring
```

### Platform Support
- Cross-platform monitoring (Windows, macOS, Linux)
- Adaptive fallbacks when system metrics unavailable
- Consistent API across all platforms

## Best Practices

### For Development
1. **Use benchmarking** during adapter development to optimize performance
2. **Profile memory usage** for large-scale evaluations
3. **Test different batch sizes** to find optimal configuration
4. **Monitor latency statistics** to identify outliers

### For Production
1. **Enable basic monitoring** to track evaluation performance over time
2. **Save performance reports** for historical analysis
3. **Set up alerts** based on performance thresholds
4. **Use recommendations** to optimize deployment configurations

### For Research
1. **Compare adapter performance** using standardized benchmarks
2. **Track performance regressions** across model versions
3. **Document performance characteristics** in research papers
4. **Share benchmark results** for reproducibility

## Troubleshooting

### High Memory Usage
- Use smaller batch sizes
- Enable memory profiling to identify bottlenecks
- Consider streaming evaluation for large datasets

### Slow Performance
- Check CPU utilization and optimize concurrency
- Profile individual operation latencies
- Consider caching strategies for repeated operations

### Inconsistent Results
- Ensure sufficient warmup iterations
- Account for system load variations
- Use multiple benchmark runs for statistical significance

## Performance Metrics Glossary

- **Wall Time**: Total elapsed time including waiting
- **CPU Time**: Actual CPU computation time
- **Peak Memory**: Maximum memory usage during operation
- **Memory Delta**: Change in memory from start to end
- **Throughput**: Items processed per second
- **Latency**: Time per individual operation
- **P95/P99**: 95th/99th percentile latency values
