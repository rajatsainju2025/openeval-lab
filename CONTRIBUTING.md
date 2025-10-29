# Contributing to OpenEval Lab

Thanks for your interest in contributing to OpenEval Lab! We're excited to have you here.

## Code of Conduct

Please be respectful and constructive in all interactions. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/openeval-lab.git
   cd openeval-lab
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e '.[dev]'
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Verify setup**
   ```bash
   pytest tests/test_smoke.py
   openeval --version
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style (black + ruff handle this automatically)
- Add or update tests
- Update documentation if needed

### 3. Run Tests and Checks

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_smoke.py

# Run with coverage
pytest --cov=openeval

# Format code (automatic with pre-commit)
black src/ tests/
ruff check --fix src/ tests/
```

### 4. Commit Your Changes

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
git commit -m "feat: add new metric for token-level accuracy"
git commit -m "fix: resolve caching issue with large datasets"
git commit -m "docs: improve profiling examples"
git commit -m "refactor: optimize string operations in metrics"
git commit -m "test: add tests for new profiling utilities"
git commit -m "chore: update dependencies"
```

**Commit types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring (no functional changes)
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub with:
- Clear title describing the change
- Description of what was changed and why
- Link to related issues if applicable
- Screenshots or examples if relevant

## Coding Guidelines

### Python Style

- **Black** for formatting (line length: 100)
- **Ruff** for linting
- **Type hints** for all public APIs
- **Docstrings** for all public functions/classes (Google style)

Example:

```python
from __future__ import annotations

from typing import List, Optional


def process_samples(
    samples: List[str],
    batch_size: int = 32,
    verbose: bool = False
) -> List[str]:
    """Process a list of samples in batches.

    Args:
        samples: Input samples to process
        batch_size: Number of samples per batch
        verbose: Whether to print progress

    Returns:
        Processed samples

    Raises:
        ValueError: If batch_size is less than 1
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    # Implementation...
```

### Testing Guidelines

- Write tests for new features
- Maintain or improve code coverage
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

```python
def test_metric_computation_with_empty_input():
    """Test that metric handles empty input correctly."""
    # Arrange
    metric = ExactMatch()
    predictions = []
    references = []

    # Act
    result = metric.compute(predictions, references)

    # Assert
    assert result == {"accuracy": 0.0}
```

### Performance Considerations

- Use streaming for large datasets
- Implement lazy loading where possible
- Pre-process data once, not repeatedly
- Use profiling tools to identify bottlenecks
- Add performance tests for critical paths

## Areas for Contribution

### High Priority

- **New metrics**: Implement additional evaluation metrics
- **Dataset loaders**: Support for more data formats
- **Adapters**: Integrations with new model APIs
- **Documentation**: Tutorials, examples, API docs
- **Bug fixes**: Check GitHub Issues

### Medium Priority

- **Performance optimizations**: Profile and optimize hot paths
- **Test coverage**: Increase test coverage
- **Type annotations**: Add missing type hints
- **Examples**: More real-world examples

### Good First Issues

Look for issues tagged with `good-first-issue` on GitHub:
- Documentation improvements
- Simple bug fixes
- Adding examples
- Improving error messages

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally (`pytest`)
- [ ] Code is formatted (`black`, `ruff`)
- [ ] Type checking passes (`mypy src/` if available)
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Commit messages follow Conventional Commits

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested these changes.

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Questions or Issues?

- **Bugs**: Open an issue on GitHub
- **Features**: Open an issue for discussion first
- **Questions**: Check docs or open a discussion

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to OpenEval Lab! 🚀
