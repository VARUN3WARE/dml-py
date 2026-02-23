# Contributing to PyDML

Thank you for your interest in contributing to PyDML! This document provides guidelines and information for contributors.

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/dml-py.git
cd dml-py
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

Or install tools separately:

```bash
pip install -e .
pip install pytest pytest-cov black flake8 mypy
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

## Development Workflow

### Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/issue-123
```

### Make Changes

1. Write code following the style guide
2. Add tests for new functionality
3. Update documentation
4. Run tests locally

### Commit Messages

Follow the conventional commits format:

```
feat: Add new feature X
fix: Fix issue #123
docs: Update documentation for Y
test: Add tests for Z
refactor: Improve code structure
```

### Submit Pull Request

1. Push your branch to GitHub
2. Create a pull request
3. Describe your changes
4. Link related issues

## Code Style

### Python Style

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use double quotes for strings
- Use type hints where possible

### Format Code

```bash
# Auto-format with black
black pydml/ tests/ examples/

# Check style
flake8 pydml/ tests/

# Type checking
mypy pydml/
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: int, arg2: str) -> bool:
    """Short one-line description.

    Longer description if needed, explaining what the function does,
    its behavior, and any important notes.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When validation fails
        TypeError: When type is incorrect

    Example:
        >>> result = my_function(42, "test")
        >>> print(result)
        True
    """
    # Implementation
    pass
```

## Testing

### Run Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_dml.py

# With coverage
pytest --cov=pydml --cov-report=html

# Verbose output
pytest -v
```

### Write Tests

```python
import pytest
from pydml import DMLTrainer

class TestDMLTrainer:
    def test_initialization(self):
        """Test trainer initialization."""
        models = [...]
        trainer = DMLTrainer(models)
        assert len(trainer.models) == len(models)

    def test_invalid_input(self):
        """Test input validation."""
        with pytest.raises(ValueError):
            trainer = DMLTrainer([])  # Empty list
```

### Test Coverage

Aim for >80% code coverage for new features.

## Documentation

### Build Documentation

```bash
cd docs
make html
# Open docs/_build/html/index.html
```

### Documentation Style

- Use Markdown for narrative docs
- Use reStructuredText for API docs
- Include code examples
- Add cross-references

### Update Documentation

When adding features:

1. Add API documentation (docstrings)
2. Update relevant user guide pages
3. Add examples if applicable
4. Update changelog

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally
- [ ] Code is formatted (black)
- [ ] Documentation is updated
- [ ] Changelog is updated
- [ ] No merge conflicts

### PR Description

Include:

- What changes were made
- Why the changes were needed
- How to test the changes
- Related issues

### Review Process

1. Maintainers will review your PR
2. Address any feedback
3. Once approved, PR will be merged

## Issue Guidelines

### Reporting Bugs

Include:

- PyDML version
- Python version
- Operating system
- Minimal reproducible example
- Expected vs. actual behavior
- Error messages/stack traces

### Feature Requests

Include:

- Use case / motivation
- Proposed API / interface
- Alternative solutions considered
- Willingness to implement

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on what is best for the community

### Communication

- Use GitHub Issues for bugs and features
- Use Discussions for questions and ideas
- Be patient and courteous

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:

- Open an issue for clarification
- Start a discussion
- Reach out to maintainers

Thank you for contributing to PyDML! 
