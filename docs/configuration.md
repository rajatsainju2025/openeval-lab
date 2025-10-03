# OpenEval Configuration Management

OpenEval Lab provides a comprehensive configuration management system that supports multiple file formats, environment-specific settings, and secure secret handling.

## Features

### Multi-Format Support
- YAML (recommended)
- JSON
- TOML (with optional dependency)
- Environment variables

### Environment Management
- Environment-specific configurations
- Development, testing, and production presets
- Automatic environment detection

### Secret Management
- Secure loading from environment variables
- Support for .env files
- Separate secrets files per environment

### Configuration Hierarchy
1. Default values
2. Configuration files
3. Environment variables (with OPENEVAL_ prefix)
4. Command-line arguments (highest priority)

## Usage

### Initialize Configuration
Create a new configuration file:
```bash
openeval config init
openeval config init --output myconfig.yaml --format yaml
```

### View Configuration
Show current configuration:
```bash
openeval config show
openeval config show --config myconfig.yaml --env production
```

### Validate Configuration
Check configuration for errors:
```bash
openeval config validate
openeval config validate --config myconfig.yaml
```

### Set Configuration Values
Update configuration values:
```bash
openeval config set evaluation.default_concurrency 8
openeval config set adapters.rate_limit_rpm 100 --config myconfig.yaml
```

### Use Configuration in Evaluation
```bash
openeval run spec.json --config myconfig.yaml --env production
```

## Configuration Sections

### Project Settings
```yaml
project_name: "my-evaluation-project"
version: "1.0.0"
description: "Project description"
environment: "development"
debug_mode: true
```

### Directory Settings
```yaml
data_dir: "data"
output_dir: "outputs"
cache_dir: ".cache"
log_dir: "logs"
```

### Evaluation Settings
```yaml
evaluation:
  # Performance
  default_concurrency: 4
  default_max_retries: 3
  default_timeout: 60.0

  # Resources
  max_memory_mb: 8192
  max_cpu_percent: 80.0

  # Output
  default_output_format: "json"
  include_records_by_default: false

  # Caching
  default_cache_mode: "rw"
  cache_compression: true

  # Error handling
  enable_robust_mode: true
  max_retry_attempts: 5
```

### Adapter Settings
```yaml
adapters:
  # API configuration
  api_base_urls:
    openai: "https://api.openai.com/v1"

  api_keys:
    openai: "${OPENAI_API_KEY}"

  # Request defaults
  default_temperature: 0.0
  default_max_tokens: 2048
  request_timeout: 30.0

  # Rate limiting
  rate_limit_rpm: 60
  rate_limit_tpm: 150000
```

### Logging Configuration
```yaml
logging:
  level: "INFO"
  log_to_file: true
  log_file_path: "logs/openeval.log"
  structured_logging: false
  redact_sensitive: true
```

## Environment Variables

All configuration can be overridden with environment variables using the `OPENEVAL_` prefix:

```bash
# Set evaluation concurrency
export OPENEVAL_EVALUATION_DEFAULT_CONCURRENCY=8

# Set API key
export OPENEVAL_ADAPTERS_API_KEYS_OPENAI="sk-..."

# Set logging level
export OPENEVAL_LOGGING_LEVEL="DEBUG"
```

## Environment-Specific Configurations

### Development Environment
```yaml
environment: "development"
debug_mode: true
logging:
  level: "DEBUG"
evaluation:
  enable_benchmarking: true
```

### Testing Environment
```yaml
environment: "testing"
debug_mode: false
evaluation:
  default_concurrency: 1
  enable_robust_mode: true
logging:
  level: "INFO"
```

### Production Environment
```yaml
environment: "production"
debug_mode: false
logging:
  level: "WARNING"
evaluation:
  enable_robust_mode: true
  save_performance_by_default: true
```

## Secret Management

### Using Environment Variables
```yaml
adapters:
  api_keys:
    openai: "${OPENAI_API_KEY}"
    anthropic: "${ANTHROPIC_API_KEY}"
```

### Using .env Files
Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

### Environment-Specific Secrets
Create separate files for each environment:
- `.secrets.development.yaml`
- `.secrets.testing.yaml`
- `.secrets.production.yaml`

```yaml
# .secrets.production.yaml
openai_api_key: "sk-prod-key"
database_url: "postgresql://..."
monitoring_token: "token-here"
```

## Configuration File Locations

OpenEval searches for configuration files in this order:

1. Path specified with `--config` flag
2. `./openeval.yaml`
3. `./openeval.yml`
4. `./openeval.json`
5. `~/.openeval.yaml`
6. `~/.config/openeval/config.yaml`

## Best Practices

### Development
- Use YAML format for readability
- Enable debug mode and verbose logging
- Use local cache and testing APIs
- Include performance monitoring

### Testing
- Disable concurrency for reproducible results
- Enable robust mode with retries
- Use deterministic settings
- Separate test data directories

### Production
- Use environment variables for secrets
- Enable robust mode and performance monitoring
- Set appropriate resource limits
- Use structured logging
- Configure rate limiting

### Security
- Never commit API keys to version control
- Use environment variables or separate secrets files
- Enable sensitive data redaction in logs
- Restrict file permissions on secrets files

## Integration with CI/CD

### GitHub Actions
```yaml
- name: Run evaluation
  env:
    OPENEVAL_ADAPTERS_API_KEYS_OPENAI: ${{ secrets.OPENAI_API_KEY }}
    OPENEVAL_ENVIRONMENT: "testing"
  run: |
    openeval run spec.json --config .github/openeval.yaml
```

### Docker
```dockerfile
# Copy configuration
COPY openeval.yaml /app/

# Set environment
ENV OPENEVAL_ENVIRONMENT=production
ENV OPENEVAL_CONFIG=/app/openeval.yaml

# Run evaluation
CMD ["openeval", "run", "spec.json"]
```

## Troubleshooting

### Configuration Not Found
- Check file paths and permissions
- Verify configuration file syntax
- Use `openeval config show` to debug

### Environment Variables Not Working
- Ensure correct `OPENEVAL_` prefix
- Use underscore for nested keys: `OPENEVAL_EVALUATION_DEFAULT_CONCURRENCY`
- Check environment variable values with `env | grep OPENEVAL`

### Secret Loading Issues
- Verify .env file location and syntax
- Check environment variable expansion: `${VAR_NAME}`
- Ensure secrets files have correct permissions

### Performance Issues
- Review resource limits in configuration
- Check concurrency and timeout settings
- Monitor memory usage with profiling enabled
