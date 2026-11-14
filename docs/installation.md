# Installation

## Requirements

- Python 3.10 or later.
- PyTorch 2.1 or later.
- CPU works for small examples and tests; CUDA is optional.

The base package also installs NumPy, Pillow, PyYAML, and Safetensors. It does
not install Transformers, LeRobot, diffusion libraries, ROS, or simulation
packages.

## Virtual environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

To install test and development tools too:

```bash
python -m pip install -e ".[dev]"
```

## PyTorch device builds

Choose the PyTorch wheel that matches your operating system, Python version,
and CPU or CUDA environment. Install that PyTorch build in the activated
environment before installing VLAForge. This lets pip keep the selected build
when it resolves VLAForge's `torch>=2.1` requirement. VLAForge itself does not
select a CUDA version or device index.

If you expect CUDA, check `torch.cuda.is_available()` in Python. The library
selects CUDA automatically when `device="auto"` is used and PyTorch reports
CUDA availability.

## Kaggle and offline use

The library does not assume a Kaggle directory. Pass dataset and checkpoint
paths explicitly. For offline use, provide already available encoder modules
and tokenizer; VLAForge does not download pretrained weights automatically.

## Editable install and tests

Editable installation is useful while changing the source tree:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

If a system-level pytest plugin interferes with collection, see the environment
workaround in [Contributing](contributing.md#local-checks).
