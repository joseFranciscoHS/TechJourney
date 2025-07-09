# DWMRI Processing Package

A comprehensive Python package for Diffusion Weighted Magnetic Resonance Imaging (DWMRI) processing, featuring multiple algorithms and tools.

## Features

- **MDS2S**: Multi-Dimensional Signal to Signal processing
- **P2S**: Point to Signal processing module
- **DRCNet**: Deep Residual Convolutional Network
- **Utils**: Common utilities for DWMRI processing

## Installation

### From Source

```bash
git clone <repository-url>
cd dwmri-processing
pip install -e .
```

### Development Installation

```bash
pip install -e .[dev]
```

## Usage

### Running MDS2S

```bash
# From command line
mds2s

# Or directly
python mds2s/run.py
```

### Using as a Python Package

```python
from mds2s.run import main
from utils.utils import load_config

# Load configuration
config = load_config("mds2s/config.yaml")

# Run main processing
main()
```

## Project Structure

```
src/
├── mds2s/           # Multi-Dimensional Signal to Signal processing
│   ├── __init__.py
│   ├── run.py       # Main entry point
│   ├── model.py     # Model definitions
│   ├── data.py      # Data loading and processing
│   ├── fit.py       # Training and fitting functions
│   └── config.yaml  # Configuration file
├── p2s/             # Point to Signal processing
│   └── __init__.py
├── drcnet/          # Deep Residual Convolutional Network
│   └── __init__.py
├── utils/           # Common utilities
│   ├── __init__.py
│   ├── utils.py     # Configuration loading
│   ├── metrics.py   # Evaluation metrics
│   ├── checkpoint.py # Model checkpointing
│   └── img.py       # Image processing utilities
├── setup.py         # Package setup
└── README.md        # This file
```

## Configuration

The package uses YAML configuration files. Example configuration:

```yaml
train:
  num_epochs: 100
  num_volumes: 6
  device: cpu
  mask_p: 0.3
  dropout_p: 0.3
  checkpoint_dir: "checkpoints"
  learning_rate: 0.00001
  batch_size: 4

model:
  in_channel: 10
  out_channel: 10

data:
  take_volumes: 6
  noise_sigma: 0.01
  bvalue: 2500
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this package in your research, please cite:

```bibtex
@software{dwmri_processing,
  title={DWMRI Processing Package},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dwmri-processing}
}
``` 