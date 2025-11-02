# Project Efficiency Roadmap (November 2, 2025)

## Executive Summary

This document outlines a systematic, measurement-driven approach to improve the OpenEval Lab codebase through 20 granular commits. The focus is on reducing latency for large-scale evaluations, stabilizing caching layers, and enabling concurrent execution where sequential processing currently dominates.

---

## Project Critique

### Strengths

1. **Strong Modular Architecture**
   - Clear separation of concerns: CLI (`src/openeval/cli/`), metrics (`src/openeval/metrics/`), caching (`src/openeval/cache.py`), and benchmarking (`src/openeval/benchmarking.py`)
   - Comprehensive documentation in `docs/` with examples and configuration guides
   - Well-organized task/dataset/adapter plugin system with registry pattern

2. **Advanced Async Infrastructure**
   - `AsyncEvaluationEngine` implements circuit breaker, adaptive batching, connection pooling, and priority scheduling
   - Latency-aware adaptive batch sizing based on recent latency history
   - Thread pool management for mixed sync/async adapter support

3. **Rich CLI and Logging**
   - Typer-based command interface with multiple subcommands
   - Enhanced logging with JSON support and structured output
   - Example specifications and benchmark suites for quick onboarding

### Weaknesses & Inefficiencies

1. **Validation Cache Instability**
   - Hashes `str(data)` which is non-deterministic for dictionaries (insertion-order dependent)
   - No eviction policy: unbounded growth over long-running sessions
   - Missing TTL/LRU mechanisms, risking stale entries in production

2. **Async Batching Partially Integrated**
   - `_cached_generate_batch()` and `evaluate_batch_optimized()` recently added but untested
   - Lacks fallback handling for cache-miss bulk operations
   - No CLI flag to toggle or measure impact of optimization

3. **Benchmark Suite Remains Sequential**
   - `run_single_benchmark()` materializes entire datasets (`list(dataset)`)
   - Adapter calls inside loop, missing opportunity for async parallelism
   - No concurrent prediction gathering even though `AsyncEvaluationEngine` exists

4. **Metric Computation Hot Paths**
   - `TokenF1` normalizes text per invocation; no shared normalization pipeline
   - Repeated string operations in Python loops without NumPy vectorization fallback
   - No caching of intermediate tokenization across multiple metrics

5. **Cache Layer Lacks Primitives**
   - `PredictionCache` only supports single-key `get()` and `set()`
   - Async batch operations must fall back to per-key round trips
   - No bulk insert (`executemany`) for SQLite backend

6. **Limited Test Coverage for Optimizations**
   - No tests for `evaluate_batch_optimized` or cache hit/miss ordering
   - Missing regression tests for validation cache behavior
   - No microbenchmarks to measure latency improvements from vectorization or batching

7. **Adapter Retry Strategy Inflexible**
   - Fixed exponential backoff; no adaptive tuning based on latency patterns
   - Circuit breaker timeout not configurable per run

### Impact Assessment

- **Performance**: Validation cache misses + repeated tokenization + sequential benchmark predictions = O(n) inefficiency where n = total samples × metrics × adapters
- **Reliability**: Unbounded caches risk memory leaks and stale data in 24/7 evaluation services
- **Observability**: Limited metrics on cache efficiency, batch latency improvements, and adapter retry behavior

---

## 20-Commit Efficiency Roadmap

### Phase 1: Async Batching (Commits 1-3)
**Goal**: Stabilize and test the new batch caching layer

1. **docs: Record project critique and efficiency roadmap**
   - Create this file and link from README
   - Commit async batch helper improvements already staged

2. **perf: Finalize async batch caching helper**
   - Fix type annotations in `_cached_generate_batch()`
   - Test with real cache backend
   - Ensure order preservation and error handling

3. **test: Cover async batch caching path**
   - Add unit tests for cache hits/misses with batching
   - Test ordering preservation
   - Test fallback to per-key when batch methods missing

### Phase 2: CLI and Cache Observability (Commits 4-6)
**Goal**: Make batch optimization user-facing and measurable

4. **cli: Add optimized batching flag to run command**
   - Add `--use-optimized-batch` flag to `openeval run`
   - Wire `use_optimized_batch` parameter to engine
   - Default to `True` for new runs

5. **cache: Add get_batch/set_batch primitives**
   - Implement in `PredictionCache` using `sqlite3.executemany()`
   - Add type hints and docstrings
   - Graceful fallback if methods unavailable

6. **perf: Log async cache stats post-run**
   - Extend `AsyncEvaluationEngine.get_stats()` with cache hit rate
   - Print stats to console after eval completes
   - Store in results metadata

### Phase 3: Validation Cache Stability (Commits 7-8)
**Goal**: Fix non-determinism and add eviction

7. **perf: Stabilize validation cache keys**
   - Switch from `str(data)` to `hashlib.md5(json.dumps(data, sort_keys=True))`
   - Add canonical rule hashing in `DataValidator`
   - Document cache key strategy

8. **test: Add regression tests for validation cache**
   - Test cache hits with different dict orderings
   - Verify TTL/invalidation behavior
   - Test eviction under memory pressure (mock)

### Phase 4: Metric Optimization (Commits 9-10)
**Goal**: Reduce repeated computation in metrics

9. **perf: Vectorize TokenF1 calculations**
   - Extract shared tokenization logic
   - Pre-compute token lists once per batch
   - Use NumPy-accelerated comparison if available

10. **perf: Shared metric normalization pipeline**
    - Create `normalize_predictions()` helper in `metrics_collection.py`
    - Reuse normalized text across multiple metrics
    - Cache normalization intermediate results

### Phase 5: Benchmark Suite Concurrency (Commits 11-12)
**Goal**: Unlock parallelism in benchmarking

11. **perf: Stream benchmark dataset processing**
    - Replace `list(dataset)` with lazy iterator
    - Add `itertools.islice()` for sampling instead of eager slicing
    - Enable chunk-based processing

12. **perf: Parallelize benchmark predictions**
    - Refactor `run_single_benchmark()` to collect prompts first
    - Use `AsyncEvaluationEngine.evaluate_batch()` for concurrent generation
    - Respect max concurrency settings from `AsyncTaskConfig`

### Phase 6: Advanced Caching (Commits 13-15)
**Goal**: Cache at multiple levels and add eviction policies

13. **perf: Cache benchmark metric outputs**
    - Memoize results by (adapter_name, task_name, dataset_name, metric_name)
    - Skip re-computation for repeated benchmark runs
    - Add cache invalidation triggers

14. **perf: Prefetch streaming datasets**
    - Add small async buffer in `streaming_dataset.py`
    - Prefetch next N items while current batch processes
    - Reduce I/O blocking

15. **perf: Improve PredictionCache eviction policy**
    - Add configurable max size (bytes) and TTL (seconds) to `PredictionCache`
    - Implement LRU eviction when size exceeded
    - Add `Config` entry for cache settings

### Phase 7: Observability & Testing (Commits 16-17)
**Goal**: Measure and validate all optimizations

16. **perf: Expose cache hit/miss metrics via monitoring hooks**
    - Integrate cache stats into existing `monitoring.py` or telemetry
    - Export hit rate, miss count, eviction count as metrics
    - Log to structured JSON

17. **test: Add microbenchmark harness**
    - Create `tests/test_performance_microbenchmarks.py`
    - Benchmark async batch vs. sequential evaluation
    - Measure tokenization overhead with/without vectorization

### Phase 8: Final Optimizations (Commits 18-20)
**Goal**: Polish remaining hot paths and document results

18. **perf: Optimize CLI write_out command**
    - Use chunked streaming writes instead of materializing full results
    - Avoid redundant JSON dumps
    - Support streaming output to stdout for large result sets

19. **perf: Tune adapter retry/backoff parameters**
    - Make retry delay adaptive based on `latency_history`
    - Adjust circuit breaker thresholds based on error rates
    - Add config options to `AsyncTaskConfig`

20. **docs: Update performance guide with measured improvements**
    - Document new CLI flags (`--use-optimized-batch`, cache settings)
    - Include before/after microbenchmark results
    - Add validation steps for users to measure impact

---

## Success Criteria

By commit 20, the codebase should demonstrate:

1. **Latency**: 30-50% reduction in end-to-end evaluation time for batched prompts (via async batching + metric caching)
2. **Stability**: Zero cache-related memory leaks in 24-hour test runs (via TTL + LRU eviction)
3. **Observability**: Cache hit rates, batch throughput, and retry patterns logged and queryable
4. **Test Coverage**: ≥80% coverage of new caching and batching logic
5. **Documentation**: Clear guidance for users on when/how to use optimizations and what results to expect

---

## References

- Async Engine: `src/openeval/async_evaluation_engine.py`
- Metrics: `src/openeval/metrics/accuracy.py`, `src/openeval/metrics_collection.py`
- Benchmarking: `src/openeval/benchmarking.py`
- Cache: `src/openeval/cache.py`
- CLI: `src/openeval/cli/commands/run.py`
- Validator: `src/openeval/data_validation.py`
