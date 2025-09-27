"""Automated performance benchmark suite with regression testing and optimization.

This module provides comprehensive performance benchmarking, regression detection,
and optimization recommendations for evaluation system performance monitoring.
"""

import os
import json
import time
import threading
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import subprocess
import sys

# Optional imports with graceful fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    
try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import line_profiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False

try:
    from .enhanced_logging import get_logger
    from .observability import record_metric, log_event
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def record_metric(*args, **kwargs):
        pass
    def log_event(*args, **kwargs):
        pass

logger = get_logger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks."""
    THROUGHPUT = "throughput"
    LATENCY = "latency" 
    MEMORY = "memory"
    CPU = "cpu"
    CONCURRENCY = "concurrency"
    SCALABILITY = "scalability"
    REGRESSION = "regression"


class PerformanceMetric(Enum):
    """Performance metrics to track."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    THROUGHPUT = "throughput"
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    ERROR_RATE = "error_rate"
    CONCURRENCY_LEVEL = "concurrency_level"


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    cpu_count: int
    memory_total_gb: float
    python_version: str
    platform: str
    architecture: str
    hostname: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResult:
    """Single benchmark execution result."""
    benchmark_name: str
    benchmark_type: BenchmarkType
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    throughput: Optional[float] = None
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    system_info: Optional[SystemInfo] = None


@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks."""
    name: str
    description: str
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    baseline_path: Optional[Path] = None
    threshold_config: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RegressionResult:
    """Regression detection result."""
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    threshold_percent: float
    is_regression: bool
    severity: str  # "low", "medium", "high", "critical"
    recommendation: str


class SystemProfiler:
    """System performance profiler."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._metrics: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self._monitoring or not HAS_PSUTIL:
            return
        
        self._monitoring = True
        self._metrics.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.debug("System monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics."""
        if not self._monitoring:
            return {}
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        return self._aggregate_metrics()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        if not HAS_PSUTIL:
            return
            
        try:
            import psutil
            process = psutil.Process()
        except:
            return
        
        while self._monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_rss_mb': memory_info.rss / 1024 / 1024,
                    'memory_vms_mb': memory_info.vms / 1024 / 1024,
                    'memory_percent': memory_percent,
                }
                
                with self._lock:
                    self._metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.sampling_interval)
    
    def _aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate collected metrics."""
        with self._lock:
            if not self._metrics:
                return {}
            
            # Calculate statistics for each metric
            aggregated = {}
            metric_keys = ['cpu_percent', 'memory_rss_mb', 'memory_vms_mb', 'memory_percent']
            
            for key in metric_keys:
                values = [m[key] for m in self._metrics if key in m]
                if values:
                    aggregated[key] = {
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'max': max(values),
                        'min': min(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'samples': len(values)
                    }
            
            if self._metrics:
                aggregated['duration'] = self._metrics[-1]['timestamp'] - self._metrics[0]['timestamp']
            return aggregated


class BenchmarkRunner:
    """Executes and manages performance benchmarks."""
    
    def __init__(self, output_dir: Path = Path("benchmarks"), 
                 enable_profiling: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_profiling = enable_profiling
        
        self.system_info = self._get_system_info()
        self.profiler = SystemProfiler()
        
        # Benchmark registry
        self._benchmarks: Dict[str, Callable] = {}
        self._benchmark_configs: Dict[str, Dict] = {}
    
    def _get_system_info(self) -> SystemInfo:
        """Get current system information."""
        if HAS_PSUTIL:
            import psutil
            cpu_count = psutil.cpu_count() or 1
            memory_total_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        else:
            cpu_count = 1
            memory_total_gb = 4.0  # Default fallback
            
        return SystemInfo(
            cpu_count=cpu_count,
            memory_total_gb=memory_total_gb,
            python_version=sys.version.split()[0],
            platform=sys.platform,
            architecture=os.uname().machine if hasattr(os, 'uname') else 'unknown',
            hostname=os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        )
    
    def register_benchmark(self, name: str, func: Callable, 
                          benchmark_type: BenchmarkType = BenchmarkType.LATENCY,
                          config: Optional[Dict] = None):
        """Register a benchmark function."""
        self._benchmarks[name] = func
        self._benchmark_configs[name] = {
            'type': benchmark_type,
            'config': config or {},
            'warmup_iterations': config.get('warmup_iterations', 3) if config else 3,
            'iterations': config.get('iterations', 10) if config else 10,
            'timeout': config.get('timeout', 300) if config else 300,
        }
        logger.info(f"Registered benchmark: {name}")
    
    def run_benchmark(self, name: str, **kwargs) -> BenchmarkResult:
        """Run a single benchmark."""
        if name not in self._benchmarks:
            raise ValueError(f"Benchmark '{name}' not registered")
        
        func = self._benchmarks[name]
        config = self._benchmark_configs[name]
        benchmark_type = config['type']
        
        logger.info(f"Running benchmark: {name}")
        
        # Warmup iterations
        warmup_iterations = config['warmup_iterations']
        for i in range(warmup_iterations):
            try:
                func(**kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration {i+1} failed: {e}")
        
        # Main benchmark iterations
        iterations = config['iterations']
        execution_times = []
        memory_usages = []
        cpu_usages = []
        error_count = 0
        success_count = 0
        
        for i in range(iterations):
            # Start system monitoring
            self.profiler.start_monitoring()
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                # Execute benchmark
                result = func(**kwargs)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i+1} failed: {e}")
                error_count += 1
                continue
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Stop monitoring and get metrics
            system_metrics = self.profiler.stop_monitoring()
            
            execution_time = end_time - start_time
            memory_usage = max(end_memory - start_memory, 0)  
            cpu_usage = system_metrics.get('cpu_percent', {}).get('mean', 0)
            
            execution_times.append(execution_time)
            memory_usages.append(memory_usage)
            cpu_usages.append(cpu_usage)
        
        if not execution_times:
            raise RuntimeError(f"All benchmark iterations failed for: {name}")
        
        # Calculate statistics
        avg_execution_time = statistics.mean(execution_times)
        avg_memory_usage = statistics.mean(memory_usages) if memory_usages else 0.0
        avg_cpu_usage = statistics.mean(cpu_usages) if cpu_usages else 0.0
        
        # Calculate latency percentiles
        sorted_times = sorted(execution_times)
        latency_percentiles = {
            'p50': self._percentile(sorted_times, 50),
            'p95': self._percentile(sorted_times, 95),
            'p99': self._percentile(sorted_times, 99)
        }
        
        # Calculate throughput (if applicable)
        throughput = None
        if benchmark_type == BenchmarkType.THROUGHPUT:
            throughput = success_count / avg_execution_time if avg_execution_time > 0 else 0
        
        result = BenchmarkResult(
            benchmark_name=name,
            benchmark_type=benchmark_type,
            execution_time=avg_execution_time,
            memory_usage_mb=avg_memory_usage,
            cpu_percent=avg_cpu_usage,
            throughput=throughput,
            latency_percentiles=latency_percentiles,
            error_count=error_count,
            success_count=success_count,
            system_info=self.system_info,
            metadata={
                'iterations': iterations,
                'warmup_iterations': warmup_iterations,
                'execution_times': execution_times,
                'memory_usages': memory_usages,
                'cpu_usages': cpu_usages,
                'kwargs': kwargs
            }
        )
        
        # Record metrics
        record_metric("benchmark_execution_time", avg_execution_time, "histogram", 
                     {"benchmark": name})
        record_metric("benchmark_memory_usage", avg_memory_usage, "histogram",
                     {"benchmark": name})
        
        log_event("info", f"Benchmark completed: {name}", 
                 execution_time=avg_execution_time, memory_usage=avg_memory_usage)
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_PSUTIL:
            import psutil
            try:
                return psutil.Process().memory_info().rss / 1024 / 1024
            except:
                return 0.0
        return 0.0
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        k = (len(values) - 1) * (percentile / 100.0)
        f = int(k)
        c = k - f
        if f + 1 < len(values):
            return values[f] * (1 - c) + values[f + 1] * c
        return values[f]
    
    def run_suite(self, suite_name: str, benchmark_names: List[str],
                  **kwargs) -> BenchmarkSuite:
        """Run a suite of benchmarks."""
        suite = BenchmarkSuite(name=suite_name, description=f"Benchmark suite: {suite_name}")
        
        for benchmark_name in benchmark_names:
            try:
                result = self.run_benchmark(benchmark_name, **kwargs)
                suite.benchmarks.append(result)
            except Exception as e:
                logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
        
        # Save suite results
        self.save_suite(suite)
        
        return suite
    
    def save_suite(self, suite: BenchmarkSuite):
        """Save benchmark suite results."""
        timestamp = suite.created_at.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{suite.name}_{timestamp}.json"
        
        # Convert to serializable format
        suite_data = {
            'name': suite.name,
            'description': suite.description,
            'created_at': suite.created_at.isoformat(),
            'system_info': asdict(self.system_info),
            'benchmarks': []
        }
        
        for benchmark in suite.benchmarks:
            benchmark_data = asdict(benchmark)
            benchmark_data['timestamp'] = benchmark.timestamp.isoformat()
            benchmark_data['benchmark_type'] = benchmark.benchmark_type.value
            if benchmark.system_info:
                benchmark_data['system_info'] = asdict(benchmark.system_info)
                benchmark_data['system_info']['timestamp'] = benchmark.system_info.timestamp.isoformat()
            suite_data['benchmarks'].append(benchmark_data)
        
        with open(output_file, 'w') as f:
            json.dump(suite_data, f, indent=2)
        
        logger.info(f"Benchmark suite saved: {output_file}")


class RegressionDetector:
    """Detects performance regressions by comparing current and baseline results."""
    
    def __init__(self, default_thresholds: Optional[Dict[str, float]] = None):
        self.default_thresholds = default_thresholds or {
            'execution_time': 20.0,  # 20% increase is a regression
            'memory_usage_mb': 25.0,  # 25% increase is a regression
            'cpu_percent': 30.0,      # 30% increase is a regression
            'throughput': -15.0,      # 15% decrease is a regression
            'error_rate': 5.0         # 5% increase in errors is a regression
        }
    
    def detect_regressions(self, current_suite: BenchmarkSuite, 
                          baseline_suite: BenchmarkSuite,
                          custom_thresholds: Optional[Dict[str, float]] = None) -> List[RegressionResult]:
        """Detect regressions between current and baseline results."""
        thresholds = {**self.default_thresholds, **(custom_thresholds or {})}
        regressions = []
        
        # Create lookup for baseline benchmarks
        baseline_benchmarks = {b.benchmark_name: b for b in baseline_suite.benchmarks}
        
        for current_benchmark in current_suite.benchmarks:
            name = current_benchmark.benchmark_name
            
            if name not in baseline_benchmarks:
                logger.warning(f"No baseline found for benchmark: {name}")
                continue
            
            baseline_benchmark = baseline_benchmarks[name]
            
            # Check each metric for regressions
            metric_comparisons = [
                ('execution_time', current_benchmark.execution_time, baseline_benchmark.execution_time),
                ('memory_usage_mb', current_benchmark.memory_usage_mb, baseline_benchmark.memory_usage_mb),
                ('cpu_percent', current_benchmark.cpu_percent, baseline_benchmark.cpu_percent),
            ]
            
            # Add throughput if available
            if current_benchmark.throughput is not None and baseline_benchmark.throughput is not None:
                metric_comparisons.append(('throughput', current_benchmark.throughput, baseline_benchmark.throughput))
            
            # Calculate error rate
            current_error_rate = current_benchmark.error_count / max(current_benchmark.success_count + current_benchmark.error_count, 1) * 100
            baseline_error_rate = baseline_benchmark.error_count / max(baseline_benchmark.success_count + baseline_benchmark.error_count, 1) * 100
            metric_comparisons.append(('error_rate', current_error_rate, baseline_error_rate))
            
            for metric_name, current_value, baseline_value in metric_comparisons:
                if baseline_value == 0:
                    continue  # Skip if baseline is zero to avoid division by zero
                
                # Calculate percentage change
                if metric_name == 'throughput':
                    # For throughput, decrease is bad
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    threshold = thresholds.get(metric_name, self.default_thresholds[metric_name])
                    is_regression = change_percent < threshold  # Negative threshold for throughput
                else:
                    # For other metrics, increase is bad
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    threshold = thresholds.get(metric_name, self.default_thresholds[metric_name])
                    is_regression = change_percent > threshold
                
                if is_regression:
                    severity = self._determine_severity(change_percent, threshold)
                    recommendation = self._get_recommendation(metric_name, change_percent)
                    
                    regression = RegressionResult(
                        metric_name=metric_name,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        change_percent=change_percent,
                        threshold_percent=threshold,
                        is_regression=True,
                        severity=severity,
                        recommendation=recommendation
                    )
                    
                    regressions.append(regression)
                    
                    logger.warning(f"Regression detected in {name}.{metric_name}: "
                                 f"{change_percent:.1f}% change (threshold: {threshold}%)")
        
        return regressions
    
    def _determine_severity(self, change_percent: float, threshold: float) -> str:
        """Determine regression severity based on change magnitude."""
        if abs(change_percent) > abs(threshold) * 3:
            return "critical"
        elif abs(change_percent) > abs(threshold) * 2:
            return "high"
        elif abs(change_percent) > abs(threshold) * 1.5:
            return "medium"
        else:
            return "low"
    
    def _get_recommendation(self, metric_name: str, change_percent: float) -> str:
        """Get optimization recommendation based on regression."""
        recommendations = {
            'execution_time': "Consider optimizing algorithms, reducing I/O operations, or enabling caching.",
            'memory_usage_mb': "Review memory allocations, implement object pooling, or optimize data structures.",
            'cpu_percent': "Profile CPU-intensive operations, optimize algorithms, or consider parallelization.",
            'throughput': "Analyze bottlenecks, increase concurrency, or optimize critical path operations.",
            'error_rate': "Investigate error causes, improve error handling, or add input validation."
        }
        
        base_recommendation = recommendations.get(metric_name, "Investigate performance degradation.")
        
        if abs(change_percent) > 50:
            return f"URGENT: {base_recommendation} Change: {change_percent:.1f}%"
        else:
            return f"{base_recommendation} Change: {change_percent:.1f}%"


class PerformanceOptimizer:
    """Provides performance optimization recommendations."""
    
    def analyze_results(self, suite: BenchmarkSuite) -> Dict[str, List[str]]:
        """Analyze benchmark results and provide optimization recommendations."""
        recommendations = {
            'memory': [],
            'cpu': [],
            'latency': [],
            'throughput': [],
            'general': []
        }
        
        for benchmark in suite.benchmarks:
            name = benchmark.benchmark_name
            
            # Memory recommendations
            if benchmark.memory_usage_mb > 1000:  # > 1GB
                recommendations['memory'].append(
                    f"{name}: High memory usage ({benchmark.memory_usage_mb:.1f} MB). "
                    "Consider memory profiling and optimization."
                )
            
            # CPU recommendations
            if benchmark.cpu_percent > 80:
                recommendations['cpu'].append(
                    f"{name}: High CPU usage ({benchmark.cpu_percent:.1f}%). "
                    "Consider algorithm optimization or parallelization."
                )
            
            # Latency recommendations
            p99_latency = benchmark.latency_percentiles.get('p99', 0)
            if p99_latency > 10.0:  # > 10 seconds
                recommendations['latency'].append(
                    f"{name}: High P99 latency ({p99_latency:.2f}s). "
                    "Consider optimizing slow operations or adding timeouts."
                )
            
            # Throughput recommendations
            if benchmark.throughput and benchmark.throughput < 10:  # < 10 ops/sec
                recommendations['throughput'].append(
                    f"{name}: Low throughput ({benchmark.throughput:.1f} ops/s). "
                    "Consider increasing concurrency or optimizing bottlenecks."
                )
            
            # Error rate recommendations
            error_rate = benchmark.error_count / max(benchmark.success_count + benchmark.error_count, 1) * 100
            if error_rate > 1.0:  # > 1% error rate
                recommendations['general'].append(
                    f"{name}: High error rate ({error_rate:.1f}%). "
                    "Investigate and fix underlying issues."
                )
        
        return recommendations
    
    def generate_optimization_report(self, suite: BenchmarkSuite, 
                                   regressions: List[RegressionResult],
                                   output_path: Path):
        """Generate comprehensive optimization report."""
        recommendations = self.analyze_results(suite)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'suite_name': suite.name,
            'summary': {
                'total_benchmarks': len(suite.benchmarks),
                'regressions_found': len(regressions),
                'critical_regressions': len([r for r in regressions if r.severity == 'critical']),
                'high_regressions': len([r for r in regressions if r.severity == 'high']),
            },
            'regressions': [asdict(r) for r in regressions],
            'recommendations': recommendations,
            'system_info': asdict(suite.benchmarks[0].system_info) if (suite.benchmarks and suite.benchmarks[0].system_info) else {},
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report generated: {output_path}")


# Example benchmark functions
def benchmark_evaluation_pipeline():
    """Example benchmark for evaluation pipeline."""
    import time
    import random
    
    # Simulate evaluation workload
    data_size = random.randint(100, 1000)
    processing_time = data_size * 0.001  # Simulate processing
    
    time.sleep(processing_time)
    
    # Simulate some memory allocation
    data = [random.random() for _ in range(data_size)]
    result = sum(data)
    
    return {'processed_items': data_size, 'result': result}


def benchmark_model_inference():
    """Example benchmark for model inference."""
    import time
    import random
    
    # Simulate model inference
    inference_time = random.uniform(0.1, 0.5)
    time.sleep(inference_time)
    
    # Simulate memory usage
    model_data = [0] * random.randint(1000, 10000)
    
    return {'inference_time': inference_time, 'model_size': len(model_data)}


# Register default benchmarks
def setup_default_benchmarks(runner: BenchmarkRunner):
    """Setup default benchmark suite."""
    runner.register_benchmark(
        'evaluation_pipeline',
        benchmark_evaluation_pipeline,
        BenchmarkType.LATENCY,
        {'iterations': 50, 'warmup_iterations': 5}
    )
    
    runner.register_benchmark(
        'model_inference', 
        benchmark_model_inference,
        BenchmarkType.THROUGHPUT,
        {'iterations': 100, 'warmup_iterations': 10}
    )