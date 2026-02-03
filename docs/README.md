# PyDML Documentation

This directory contains the Sphinx documentation for PyDML.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Clean Build

```bash
make clean
make html
```

## Documentation Structure

```
docs/
├── index.md                    # Main documentation page
├── installation.md             # Installation guide
├── quickstart.md               # Quickstart guide
├── examples.md                 # Examples overview
├── conf.py                     # Sphinx configuration
├── Makefile                    # Build commands
├── api/                        # API reference
│   ├── core.md                 # Core components
│   ├── trainers.md             # Trainer classes
│   ├── models.md               # Model architectures
│   ├── losses.md               # Loss functions
│   ├── strategies.md           # Training strategies
│   ├── analysis.md             # Analysis tools
│   └── utils.md                # Utilities
├── user_guide/                 # User guides
│   ├── core_concepts.md        # Core concepts
│   ├── trainers.md             # Trainers guide
│   ├── models.md               # Models guide
│   ├── losses.md               # Losses guide
│   ├── callbacks.md            # Callbacks guide
│   └── utilities.md            # Utilities guide
├── tutorials/                  # Step-by-step tutorials
│   ├── basic_dml.md            # Basic DML tutorial
│   ├── knowledge_distillation.md  # KD tutorial
│   ├── advanced_features.md    # Advanced features
│   ├── custom_models.md        # Custom models
│   └── production_deployment.md  # Production guide
├── changelog.md                # Changelog
├── contributing.md             # Contributing guide
└── license.md                  # License information
```

## Viewing Documentation Locally

After building:

```bash
# Option 1: Open directly in browser
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows

# Option 2: Serve with Python
cd _build/html
python -m http.server 8000
# Then visit http://localhost:8000
```

## Documentation Formats

Build different formats:

```bash
# HTML (default)
make html

# PDF (requires LaTeX)
make latexpdf

# ePub
make epub

# Plain text
make text
```

## Auto-rebuild During Development

Install sphinx-autobuild for live reloading:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

Then visit http://127.0.0.1:8000 - changes will auto-rebuild.

## Writing Documentation

### Markdown Support

We use MyST parser for Markdown support. Both `.md` and `.rst` files are supported.

### Code Examples

Use fenced code blocks:

\```python
from pydml import DMLTrainer
trainer = DMLTrainer(models)
\```

### API Documentation

API docs are auto-generated from docstrings using Sphinx autodoc:

\```{eval-rst}
.. autoclass:: pydml.trainers.DMLTrainer
:members:
:show-inheritance:
\```

### Cross-References

Link to other pages:

```markdown
See [Installation](installation.md) for details.
```

Link to API docs:

```markdown
See {class}`pydml.trainers.DMLTrainer` for the API.
```

## Publishing Documentation

Documentation can be hosted on:

- **Read the Docs**: https://readthedocs.org/
- **GitHub Pages**: Via GitHub Actions
- **Custom hosting**: Upload `_build/html/`

## Troubleshooting

### Module Import Errors

If Sphinx can't import pydml:

```bash
# Install pydml in development mode
pip install -e ..
```

### Missing Dependencies

Install all documentation dependencies:

```bash
pip install -r ../requirements.txt
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

### Build Warnings

Common warnings and fixes:

- **Duplicate object description**: Add `:noindex:` to one occurrence
- **Missing cross-reference**: Check the link target exists
- **Failed to import**: Ensure the module/function actually exists

## Contributing

When adding new features to PyDML:

1. Write clear docstrings (Google style)
2. Update relevant API documentation
3. Add examples if applicable
4. Update changelog.md

See [Contributing Guide](contributing.md) for more details.
