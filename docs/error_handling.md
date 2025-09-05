# Error Handling and Robustness

OpenEval Lab provides comprehensive error handling and retry mechanisms to make evaluations more robust in production environments.

## Features

### Error Tracking
- Categorized error logging with severity levels
- Error context and stack trace preservation
- Error count and type statistics
- Recoverable vs non-recoverable error classification

### Retry Mechanisms
- Configurable retry attempts and delays
- Exponential backoff with jitter
- Type-specific retry logic
- Network and temporary failure handling

### Circuit Breaker Pattern
- Prevent cascading failures
- Automatic service degradation
- Configurable failure thresholds
- Timeout-based recovery

### Error Recovery
- Automatic recovery for common errors
- Missing package installation attempts
- Directory creation for missing paths
- Graceful degradation strategies

## Usage

### Basic Robust Mode
```bash
openeval run spec.json --robust
```

### Advanced Configuration
```bash
openeval run spec.json --robust --max-retry-attempts 5 --error-summary
```

### Error Testing
```bash
# Test retry mechanisms
openeval error-test retry --max-attempts 3

# Test circuit breaker
openeval error-test circuit-breaker

# Test error recovery
openeval error-test recovery
```

## Error Types and Handling

### Network Errors
- `ConnectionError`, `TimeoutError`, `OSError`
- Automatic retry with exponential backoff
- Circuit breaker protection for external services

### Temporary Failures
- `RuntimeError`, `ValueError` (configurable)
- Retry with shorter delays
- Context-aware error logging

### Non-Recoverable Errors
- `ImportError` (with auto-recovery attempt)
- `FileNotFoundError` (with directory creation)
- `TypeError`, `AttributeError` (immediate failure)

## Configuration Examples

### Custom Retry Configuration
```python
from openeval.error_handling import RetryConfig, retry_with_config

config = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    exponential_backoff=True,
    jitter=True,
    retryable_errors=[ConnectionError, TimeoutError]
)

@retry_with_config(config)
def your_function():
    # Your evaluation code here
    pass
```

### Circuit Breaker Setup
```python
from openeval.error_handling import CircuitBreaker

@CircuitBreaker(failure_threshold=3, timeout=30.0)
def external_service_call():
    # Call to external service
    pass
```

### Error Context Creation
```python
from openeval.error_handling import create_robust_evaluation_context

context = create_robust_evaluation_context()
error_tracker = context["error_tracker"]
network_retry = context["network_retry"]
```

## Best Practices

1. **Enable robust mode** for production evaluations
2. **Set appropriate retry limits** based on your service constraints
3. **Monitor error summaries** to identify patterns
4. **Use circuit breakers** for external service calls
5. **Test error handling** in development environments

## Error Summary Output

When `--error-summary` is enabled, you'll see:

```
Error Summary:
Total errors: 3
Critical errors: 0
Recoverable errors: 3

Error types:
  ConnectionError: 2
  TimeoutError: 1

Recent errors:
  [medium] ConnectionError: Connection refused
  [medium] TimeoutError: Request timeout
  [medium] ConnectionError: Network unreachable
```

## Integration with Existing Code

The error handling system is designed to be minimally invasive:

- Enable with CLI flags for immediate benefits
- Decorate existing functions for targeted protection
- Use context managers for scoped error handling
- Integrate with logging systems automatically

## Performance Impact

- Minimal overhead when no errors occur
- Retry delays only when errors happen
- Circuit breakers prevent wasted resources
- Error tracking uses efficient data structures
