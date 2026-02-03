# Installation

## Requirements

PyDML requires:

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- tqdm >= 4.65.0

## Install from PyPI

The easiest way to install PyDML is via pip:

```bash
pip install pytorch-dml
```

This will install PyDML and all required dependencies.

## Install from Source

For the latest development version or to contribute to PyDML:

### Clone the Repository

```bash
git clone https://github.com/VARUN3WARE/dml-py.git
cd dml-py
```

### Using pip

```bash
pip install -e .
```

### Using uv (Fast Alternative)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyDML
uv pip install -e .
```

## Verify Installation

Test your installation:

```bash
python examples/test_installation.py
```

Or run a quick Python check:

```python
import pydml
print(pydml.__version__)
print("PyDML installed successfully!")
```

## GPU Support

PyDML will automatically use CUDA if available. To verify GPU support:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

### Installing PyTorch with CUDA

If you need CUDA support, install PyTorch first:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more options.

## Development Installation

For development with testing and documentation tools:

```bash
# Clone repository
git clone https://github.com/VARUN3WARE/dml-py.git
cd dml-py

# Install with dev dependencies
pip install -e ".[dev]"

# Or install additional tools manually
pip install pytest pytest-cov black flake8 mypy
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

## Docker Installation

Use PyDML in a Docker container:

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Install PyDML
RUN pip install pytorch-dml

# Copy your training scripts
COPY . /workspace

CMD ["python", "train.py"]
```

## Common Issues

### Import Error: No module named 'pydml'

Make sure you've installed the package:

```bash
pip install pytorch-dml
```

If installed from source with `-e`, ensure you're in the correct virtual environment.

### CUDA Out of Memory

Reduce batch size or use gradient checkpointing:

```python
from pydml import DMLTrainer

trainer = DMLTrainer(
    models,
    device='cuda',
    # Use smaller batch size
)
```

### Version Conflicts

Create a fresh virtual environment:

```bash
python -m venv pydml_env
source pydml_env/bin/activate  # On Windows: pydml_env\Scripts\activate
pip install pytorch-dml
```

## Next Steps

- Read the [Quickstart Guide](quickstart.md)
- Explore [Examples](examples.md)
- Check out the [User Guide](user_guide/core_concepts.md)
