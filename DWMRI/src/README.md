# DWMRI Processing Package

A comprehensive Python package for Diffusion Weighted Magnetic Resonance Imaging (DWMRI) processing, featuring multiple algorithms and tools.

## Features

- **MDS2S**: Multi-Dimensional Signal to Signal processing
- **P2S**: Point to Signal processing module
- **DRCNet**: Deep Residual Convolutional Network
- **Utils**: Common utilities for DWMRI processing

## Installation

### From Source (uv, recommended)

From the `DWMRI` directory (parent of this `src/` folder):

```bash
git clone <repository-url>
cd TechJourney/DWMRI   # or your clone path ending in DWMRI
uv sync --extra dev    # creates .venv, installs deps + dev tools, project in editable mode
```

Dependencies come from `uv.lock`; the package is installed editable by default.

### Pip (legacy)

```bash
cd /path/to/DWMRI
pip install -e ".[dev]"
```

### Dependencies

Runtime dependencies are defined in [pyproject.toml](../pyproject.toml) and pinned in [uv.lock](../uv.lock). **DIPY** is required for data loading, Patch2Self, MP-PCA, and DTI metrics. A pip-only list remains in [requirements.txt](requirements.txt).

### Reproducibility (training)

- **`train.seed`** in `drcnet_hybrid_rgs/config.yaml` / `restormer_hybrid_rgs/config.yaml` sets Python, NumPy, and PyTorch RNGs and seeds per-sample RGS / Bernoulli masks in the lazy dataset.
- **`train.reproducible: true`** disables cuDNN autotune and enables deterministic algorithms where supported (slower; **GPU runs are still not guaranteed bit-identical**).
- **Default** (`reproducible: false`): cuDNN benchmark on for throughput (`utils.repro_seed.configure_cudnn`).

After a validated pilot, refresh the lockfile with `uv lock` at the `DWMRI/` root (see comments in `requirements.lock` here).

## Usage

### Paper final protocol runbook

Use this runbook to launch the frozen paper protocol (D-Brain + Stanford) with the final manifest.

1) **Install/sync dependencies**

```bash
cd /path/to/DWMRI
uv sync --extra dev
```

2) **Quick dependency checks**

```bash
PYTHONPATH=src uv run python -c "import numpy, torch; print('numpy/torch ok')"
PYTHONPATH=src uv run python -c "import dipy; print('dipy ok')"
```

3) **Validate protocol/manfiest coherence**

```bash
python experiments/validate_protocol_final.py \
  --protocol experiments/paper_protocol_final.yaml \
  --manifest experiments/paper_manifest_final.yaml \
  --out /tmp/paper_final_out/protocol_validation.json
```

4) **Launch full final campaign**

```bash
export EXP_ID=paper_final_v1
export OUT=/tmp/paper_final_out

python experiments/driver.py \
  --manifest experiments/paper_manifest_final.yaml \
  --exp-id "$EXP_ID" \
  --output-root "$OUT" \
  --registry-path "$OUT/registry.jsonl" \
  --fail-fast
```

5) **Resume after interruption**

```bash
python experiments/driver.py \
  --manifest experiments/paper_manifest_final.yaml \
  --exp-id "$EXP_ID" \
  --output-root "$OUT" \
  --registry-path "$OUT/registry.jsonl" \
  --resume --retry-failed --fail-fast
```

6) **Generate closure artifact**

```bash
python experiments/protocol_closure_report.py \
  --validation-json "$OUT/protocol_validation.json" \
  --dependency-json "$OUT/dependency_probe.json" \
  --out "$OUT/protocol_closure_report.json"
```

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
└── README.md        # This file
```

Project metadata and `uv.lock` live in the parent `DWMRI/` directory (`pyproject.toml`, `setup.py` shim).

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

### Format and lint (Ruff)

From the `DWMRI` directory:

```bash
ruff check src experiments
ruff format src experiments
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