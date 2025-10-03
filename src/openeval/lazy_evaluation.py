"""
Lazy Evaluation and Deferred Computation for OpenEval Lab

This module provides lazy evaluation patterns, deferred computation, and on-demand
processing to improve memory efficiency and computational performance.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Generic, TypeVar
from dataclasses import dataclass, field

try:
    import dask
    import dask.delayed
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    dask = None

from .enhanced_logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class ComputationNode:
    """A node in the computation graph."""
    func: Callable[..., Any]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[ComputationNode] = field(default_factory=list)
    result: Optional[Any] = None
    computed: bool = False
    computation_time: float = 0.0
    cache_key: Optional[str] = None

    def __hash__(self) -> int:
        """Hash for caching."""
        if self.cache_key:
            return hash(self.cache_key)
        return hash((id(self.func), tuple(self.args), tuple(sorted(self.kwargs.items()))))


class LazyEvaluator:
    """
    Lazy evaluator that defers computation until results are actually needed.
    """

    def __init__(self, cache_results: bool = True, max_cache_size: int = 1000):
        self.cache_results = cache_results
        self.max_cache_size = max_cache_size
        self._cache: Dict[int, Any] = {}
        self._cache_order: List[int] = []
        self._lock = threading.RLock()

    def lazy(self, func: Callable[..., T]) -> Callable[..., LazyValue[T]]:
        """
        Decorator to make a function lazy.

        Args:
            func: Function to make lazy

        Returns:
            Lazy version of the function
        """
        def lazy_func(*args, **kwargs) -> LazyValue[T]:
            return LazyValue(
                func=func,
                args=args,
                kwargs=kwargs,
                evaluator=self
            )

        return lazy_func

    def compute(self, lazy_value: LazyValue[T]) -> T:
        """
        Compute a lazy value.

        Args:
            lazy_value: The lazy value to compute

        Returns:
            Computed result
        """
        return lazy_value.compute()

    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """
        Invalidate cached results.

        Args:
            pattern: Optional pattern to match cache keys
        """
        with self._lock:
            if pattern:
                keys_to_remove = [k for k in self._cache.keys() if pattern in str(k)]
                for key in keys_to_remove:
                    del self._cache[key]
                    if key in self._cache_order:
                        self._cache_order.remove(key)
            else:
                self._cache.clear()
                self._cache_order.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "max_cache_size": self.max_cache_size,
                "cache_hit_rate": 0.0,  # Would need hit/miss counters
                "memory_usage_estimate": len(self._cache) * 1000  # Rough estimate
            }


@dataclass
class LazyValue(Generic[T]):
    """A lazily computed value."""

    func: Callable[..., T]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    evaluator: Optional[LazyEvaluator] = None
    _result: Optional[T] = None
    _computed: bool = False
    _computing: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def compute(self) -> T:
        """Compute the lazy value."""
        with self._lock:
            if self._computed:
                return self._result

            if self._computing:
                # Avoid recursive computation
                raise RuntimeError("Circular dependency detected in lazy computation")

            self._computing = True

        try:
            # Check cache first
            if self.evaluator and self.evaluator.cache_results:
                cache_key = hash((id(self.func), tuple(self.args), tuple(sorted(self.kwargs.items()))))
                with self.evaluator._lock:
                    if cache_key in self.evaluator._cache:
                        self._result = self.evaluator._cache[cache_key]
                        self._computed = True
                        return self._result

            # Compute the result
            start_time = time.time()
            self._result = self.func(*self.args, **self.kwargs)
            computation_time = time.time() - start_time

            self._computed = True

            # Cache the result
            if self.evaluator and self.evaluator.cache_results:
                cache_key = hash((id(self.func), tuple(self.args), tuple(sorted(self.kwargs.items()))))
                with self.evaluator._lock:
                    if len(self.evaluator._cache) >= self.evaluator.max_cache_size:
                        # Remove oldest entry
                        oldest_key = self.evaluator._cache_order.pop(0)
                        del self.evaluator._cache[oldest_key]

                    self.evaluator._cache[cache_key] = self._result
                    self.evaluator._cache_order.append(cache_key)

            logger.debug(f"Computed lazy value in {computation_time:.4f}s")
            return self._result

        finally:
            with self._lock:
                self._computing = False

    def is_computed(self) -> bool:
        """Check if the value has been computed."""
        return self._computed

    def __str__(self) -> str:
        if self._computed:
            return str(self._result)
        return f"<LazyValue: {self.func.__name__} (not computed)>"

    def __repr__(self) -> str:
        return self.__str__()


class LazyDataset(Iterator[T]):
    """
    A dataset that loads and processes data lazily.
    """

    def __init__(
        self,
        data_source: Union[List[T], Iterator[T], Callable[[], Iterator[T]]],
        transform: Optional[Callable[[T], T]] = None,
        batch_size: int = 1,
        prefetch: int = 0
    ):
        self.data_source = data_source
        self.transform = transform
        self.batch_size = batch_size
        self.prefetch = prefetch
        self._iterator: Optional[Iterator[T]] = None
        self._buffer: List[T] = []
        self._exhausted = False

    def __iter__(self) -> Iterator[T]:
        if isinstance(self.data_source, list):
            self._iterator = iter(self.data_source)
        elif callable(self.data_source):
            self._iterator = self.data_source()
        else:
            self._iterator = self.data_source

        self._buffer = []
        self._exhausted = False
        return self

    def __next__(self) -> T:
        if self._exhausted:
            raise StopIteration

        # Fill buffer if needed
        while len(self._buffer) < self.batch_size and not self._exhausted:
            try:
                item = next(self._iterator)
                if self.transform:
                    item = self.transform(item)
                self._buffer.append(item)
            except StopIteration:
                self._exhausted = True
                break

        if not self._buffer:
            raise StopIteration

        # Return batch or single item
        if self.batch_size == 1:
            return self._buffer.pop(0)
        else:
            batch = self._buffer[:self.batch_size]
            self._buffer = self._buffer[self.batch_size:]
            return batch

    def peek(self) -> Optional[T]:
        """Peek at the next item without consuming it."""
        try:
            if not self._buffer and not self._exhausted:
                # Fill buffer minimally
                item = next(self._iterator)
                if self.transform:
                    item = self.transform(item)
                self._buffer.append(item)
            return self._buffer[0] if self._buffer else None
        except StopIteration:
            return None


class DeferredComputationGraph:
    """
    A computation graph that supports deferred execution and optimization.
    """

    def __init__(self):
        self.nodes: Dict[str, ComputationNode] = {}
        self._lock = threading.RLock()

    def add_node(
        self,
        name: str,
        func: Callable[..., Any],
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Add a computation node to the graph.

        Args:
            name: Unique name for the node
            func: Function to compute
            args: Positional arguments
            kwargs: Keyword arguments
            dependencies: Names of dependent nodes
        """
        with self._lock:
            node = ComputationNode(
                func=func,
                args=args or [],
                kwargs=kwargs or {},
                dependencies=[self.nodes[dep] for dep in (dependencies or []) if dep in self.nodes]
            )
            self.nodes[name] = node

    def compute_node(self, name: str) -> Any:
        """
        Compute a node and its dependencies.

        Args:
            name: Name of the node to compute

        Returns:
            Computed result
        """
        with self._lock:
            if name not in self.nodes:
                raise KeyError(f"Node {name} not found")

            node = self.nodes[name]

            # Compute dependencies first
            for dep in node.dependencies:
                if not dep.computed:
                    self._compute_node_recursive(dep)

            # Compute this node
            return self._compute_node_recursive(node)

    def _compute_node_recursive(self, node: ComputationNode) -> Any:
        """Recursively compute a node."""
        if node.computed:
            return node.result

        # Compute dependencies
        computed_args = []
        for arg in node.args:
            if isinstance(arg, ComputationNode):
                computed_args.append(self._compute_node_recursive(arg))
            else:
                computed_args.append(arg)

        computed_kwargs = {}
        for key, value in node.kwargs.items():
            if isinstance(value, ComputationNode):
                computed_kwargs[key] = self._compute_node_recursive(value)
            else:
                computed_kwargs[key] = value

        # Execute function
        start_time = time.time()
        node.result = node.func(*computed_args, **computed_kwargs)
        node.computation_time = time.time() - start_time
        node.computed = True

        return node.result

    def optimize_graph(self) -> None:
        """Optimize the computation graph (placeholder for future optimizations)."""
        # Future: Implement graph optimization techniques like:
        # - Common subexpression elimination
        # - Dead code elimination
        # - Reordering for better cache locality
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get computation graph statistics."""
        with self._lock:
            total_time = sum(node.computation_time for node in self.nodes.values())
            computed_nodes = sum(1 for node in self.nodes.values() if node.computed)

            return {
                "total_nodes": len(self.nodes),
                "computed_nodes": computed_nodes,
                "total_computation_time": total_time,
                "average_computation_time": total_time / computed_nodes if computed_nodes > 0 else 0
            }


class LazyMetric:
    """
    A metric that computes results lazily and caches intermediate results.
    """

    def __init__(self, metric_func: Callable[..., Any], cache_intermediates: bool = True):
        self.metric_func = metric_func
        self.cache_intermediates = cache_intermediates
        self._cache: Dict[str, Any] = {}
        self._computation_count = 0

    def compute(
        self,
        predictions: LazyValue[List[Any]],
        references: LazyValue[List[Any]],
        **kwargs
    ) -> LazyValue[Any]:
        """
        Compute metric lazily.

        Args:
            predictions: Lazy predictions
            references: Lazy references
            **kwargs: Additional arguments

        Returns:
            Lazy metric result
        """
        def _lazy_compute():
            # Compute inputs if needed
            preds = predictions.compute() if isinstance(predictions, LazyValue) else predictions
            refs = references.compute() if isinstance(references, LazyValue) else references

            # Create cache key
            cache_key = f"{hash(tuple(preds))}_{hash(tuple(refs))}_{hash(tuple(sorted(kwargs.items())))}"

            if self.cache_intermediates and cache_key in self._cache:
                return self._cache[cache_key]

            result = self.metric_func(preds, refs, **kwargs)
            self._computation_count += 1

            if self.cache_intermediates:
                self._cache[cache_key] = result

            return result

        return LazyValue(func=_lazy_compute, evaluator=None)


# Utility functions for easy integration
def lazy_property(func: Callable[[Any], T]) -> property:
    """
    Decorator to create lazy properties.

    Args:
        func: Property getter function

    Returns:
        Lazy property
    """
    private_name = f"_{func.__name__}"

    def getter(self):
        if not hasattr(self, private_name):
            setattr(self, private_name, func(self))
        return getattr(self, private_name)

    return property(getter)


def create_lazy_dataset(
    data_factory: Callable[[], Iterator[T]],
    transform: Optional[Callable[[T], T]] = None,
    batch_size: int = 1
) -> LazyDataset[T]:
    """
    Create a lazy dataset from a data factory function.

    Args:
        data_factory: Function that returns an iterator
        transform: Optional transformation function
        batch_size: Batch size for iteration

    Returns:
        Lazy dataset
    """
    return LazyDataset(data_factory, transform, batch_size)


def benchmark_lazy_evaluation(
    func: Callable[[], Any],
    iterations: int = 100,
    use_lazy: bool = True
) -> Dict[str, Any]:
    """
    Benchmark lazy vs eager evaluation.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        use_lazy: Whether to use lazy evaluation

    Returns:
        Benchmark results
    """
    import time

    evaluator = LazyEvaluator() if use_lazy else None

    times = []
    for _ in range(iterations):
        start_time = time.time()

        if use_lazy:
            lazy_result = evaluator.lazy(func)()
            result = lazy_result.compute()
        else:
            result = func()

        times.append(time.time() - start_time)

    return {
        "evaluation_type": "lazy" if use_lazy else "eager",
        "iterations": iterations,
        "total_time": sum(times),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "cache_stats": evaluator.get_cache_stats() if evaluator else None
    }