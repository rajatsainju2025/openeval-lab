# Optimization Checklist for OpenEval Lab Users

## Pre-Evaluation Setup (Do This First)

- [ ] Clear old cache: `rm -rf ~/.openeval/cache`
- [ ] Verify disk space: `df -h` (at least 5GB recommended)
- [ ] Check memory: `free -h` (at least 4GB recommended)
- [ ] Update OpenEval: `pip install --upgrade openeval-lab`

## Enable Performance Features

- [ ] Enable caching: `--cache` flag
- [ ] Enable compression: `--compress` flag
- [ ] Set batch size: `--batch-size 32` or higher
- [ ] Enable resource monitoring: `--monitor-resources`

## Configuration Optimization

### For Small Datasets (< 1K items)
```bash
openeval run spec.json --batch-size 64 --cache
```

### For Medium Datasets (1K-100K items)
```bash
openeval run spec.json --batch-size 32 --cache --compress --stream
```

### For Large Datasets (> 100K items)
```bash
openeval run spec.json --batch-size 16 --cache --compress --stream --max-workers 4
```

## Monitoring & Debugging

- [ ] Enable verbose logging: `--verbose`
- [ ] Enable profiling: `--profile`
- [ ] Monitor resources: `--monitor-resources`
- [ ] Check cache stats: `openeval cache stats`

## Advanced Optimization (Code Level)

```python
from openeval.streaming import stream_dataset
from openeval.batch_operations import BatchCacheOps
from openeval.validation_cache import get_cached_validation

# 1. Stream large datasets
for item in stream_dataset(dataset):
    process(item)

# 2. Cache validation results
cached = get_cached_validation(spec)
if not cached:
    result = validate(spec)

# 3. Use batch operations
BatchCacheOps.batch_set(cache, items)

# 4. Profile functions
from openeval.profiling_decorators import profile_time

@profile_time
def expensive_operation():
    pass
```

## Memory Optimization

- [ ] Use `--stream` for datasets > 10K items
- [ ] Enable compression: `--compress`
- [ ] Reduce batch size if memory usage high
- [ ] Monitor with: `watch free -h`

## Speed Optimization

- [ ] Increase batch size: `--batch-size 128`
- [ ] Enable caching: `--cache`
- [ ] Use validation cache for repeated specs
- [ ] Profile with: `--profile`

## Troubleshooting

### If Slow
1. Check batch size: Should be 16-128
2. Enable caching: `--cache`
3. Profile: `--profile`
4. Check network: `openeval doctor`

### If Out of Memory
1. Enable streaming: `--stream`
2. Reduce batch size: `--batch-size 8`
3. Enable compression: `--compress`
4. Clear cache: `rm -rf ~/.openeval/cache`

### If Cache Not Working
1. Check cache dir: `ls ~/.openeval/cache`
2. Verify permissions: `chmod 755 ~/.openeval/cache`
3. Clear and rebuild: `rm -rf ~/.openeval/cache && openeval run ...`

## Validation Benchmarks

After optimization, you should see:

| Metric | Expected | How to Measure |
|--------|----------|----------------|
| Startup time | < 2s | `time openeval --help` |
| Memory (10K items) | < 300MB | Monitor during run |
| Cache hit rate | > 90% | `openeval cache stats` |
| Throughput | > 100 items/s | Check time for known dataset |

## Continuous Improvement

- [ ] Run benchmarks regularly: `python -m openeval.performance_benchmarks`
- [ ] Monitor cache stats: `openeval cache stats`
- [ ] Review logs: `tail -f ~/.openeval/logs/eval.log`
- [ ] Profile hotspots: `--profile > profile.txt`

## Documentation References

- [Full Performance Tuning Guide](PERFORMANCE_TUNING.md)
- [Efficiency Improvements](EFFICIENCY_IMPROVEMENTS.md)
- [API Boundaries](API_BOUNDARIES.md)
- [Configuration Guide](configuration.md)

## Getting Help

Performance issues? Check:
1. [Performance Tuning Guide](PERFORMANCE_TUNING.md)
2. [Troubleshooting](error_handling.md)
3. Run `openeval doctor` for environment check
4. Open issue on GitHub with `--verbose` output
