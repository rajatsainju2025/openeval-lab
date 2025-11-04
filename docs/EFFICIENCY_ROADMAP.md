# OpenEval Lab Efficiency Roadmap (Fresh Start - November 4, 2025)

## Project Critique

### Strengths
- **Modular Architecture**: Clean separation between CLI, core evaluation engine, metrics, caching, and benchmarking components
- **Advanced Async Infrastructure**: AsyncEvaluationEngine with circuit breaker, adaptive batching, and connection pooling
- **Comprehensive Caching**: Multi-level cache with bloom filters, compression, and batch operations
- **Vectorized Metrics**: NumPy-accelerated computations where applicable
- **Rich Documentation**: Extensive docs, examples, and configuration options
- **Type Safety**: Full type hints and Pydantic validation throughout

### Areas for Improvement

#### Performance Bottlenecks
1. **Synchronous Benchmarking**: BenchmarkSuite still processes evaluations synchronously despite async infrastructure
2. **Dataset Materialization**: `list(dataset)` loads entire datasets into memory unnecessarily
3. **Duplicate Normalization**: Text normalization repeated across metrics without caching
4. **Cache Batch Inefficiency**: Batch operations use individual SQL calls instead of bulk operations
5. **CLI Output Overhead**: Large result serialization without streaming or compression

#### Memory & Resource Management
1. **Unbounded Caches**: No size limits or TTL policies on validation/data caches
2. **Memory Leaks**: No profiling or leak detection in long-running processes
3. **Connection Overhead**: HTTP adapters create new connections per request
4. **Result Size**: Large evaluation outputs stored uncompressed

#### Observability Gaps
1. **Limited Profiling**: No performance regression detection or memory profiling
2. **Progress Estimation**: No ETA calculations based on historical data
3. **Configuration Validation**: No performance hints for suboptimal settings

## 20-Commit Efficiency Roadmap

### Phase 1: Core Performance (Commits 1-5)
1. **docs: Record fresh project critique and roadmap** - Document current state and plan
2. **feat: Add metrics normalization utilities** - Commit existing normalization helpers
3. **perf: Parallelize benchmark predictions** - Use AsyncEvaluationEngine in benchmarks
4. **perf: Stream benchmark dataset processing** - Replace list(dataset) with iterators
5. **perf: Cache benchmark metric outputs** - Memoize adapter/task/dataset combinations

### Phase 2: Cache & I/O Optimization (Commits 6-10)
6. **perf: Improve cache batch primitives** - Optimize SQL patterns for bulk operations
7. **perf: Prefetch streaming datasets** - Add async buffering to dataset loading
8. **perf: Add cache eviction policies** - Size/TTL controls for PredictionCache
9. **perf: Expose cache metrics via monitoring** - Integrate stats into telemetry
10. **test: Add microbenchmark harness** - pytest-benchmark for performance validation

### Phase 3: CLI & Output Optimization (Commits 11-15)
11. **perf: Optimize CLI write_out operations** - Chunked streaming and reduced JSON ops
12. **perf: Tune adapter retry parameters** - Adaptive backoff based on latency
13. **perf: Add memory profiling hooks** - Leak detection and memory monitoring
14. **perf: Optimize HTTP adapter connections** - Connection pooling for adapters
15. **perf: Compress large result outputs** - Result compression for large evaluations

### Phase 4: Advanced Features (Commits 16-20)
16. **perf: Preprocess datasets with caching** - Pipeline caching for expensive operations
17. **perf: Add progress estimation** - ETA based on historical performance
18. **perf: Add performance regression tests** - CI pipeline regression detection
19. **perf: Optimize configuration loading** - Validation and performance hints
20. **docs: Update performance guide** - Comprehensive guide with validation steps

## Success Metrics

### Performance Targets
- **Benchmark Throughput**: 3x improvement in benchmark suite execution time
- **Memory Usage**: 50% reduction in peak memory for large evaluations
- **Cache Hit Rate**: >80% for repeated evaluations
- **CLI Responsiveness**: <100ms for common operations

### Quality Targets
- **Test Coverage**: >90% for performance-critical paths
- **Type Safety**: Zero mypy errors in core modules
- **Documentation**: Complete performance tuning guide
- **CI Performance**: <5min for full test suite

## Implementation Notes

### Commit Discipline
- Each commit focuses on one performance aspect
- Include performance measurements where possible
- Update documentation for new features
- Maintain backward compatibility

### Testing Strategy
- Unit tests for individual optimizations
- Integration tests for end-to-end performance
- Microbenchmarks for regression detection
- Memory profiling for leak detection

### Rollback Plan
- Feature flags for major changes
- Performance monitoring to detect regressions
- Clear documentation of performance trade-offs
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
