# Contributing to VLAForge

Contributions should keep the public interfaces small, preserve model and
dataset independence, and document user-visible behavior changes.

## Development environment

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Install a
PyTorch build that matches your CPU or CUDA environment first when you need a
specific build.

## Local checks

Run the checks used in CI:

```bash
ruff check src tests examples --select F
python -m pytest
```

If a system ROS pytest plugin fails before test collection, disable external
plugin autoloading:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

PowerShell:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest
```

## Code standards

- Add type annotations to public APIs.
- Keep model-specific loss logic in the model's `compute_loss` method.
- Keep dataset storage and preprocessing independent of model architectures.
- Keep optional integrations optional and avoid unnecessary base dependencies.
- Raise descriptive errors for invalid inputs.
- Add focused tests that do not need network access, GPU hardware, private
  datasets, or pretrained downloads.
- Update relevant documentation when public behavior changes.

## Proposing changes

1. Open an issue for substantial behavior changes or new features.
2. Create a focused branch from the default branch.
3. Implement the change with relevant tests and documentation.
4. Run the local checks above.
5. Open a pull request explaining the motivation, implementation, and checks.

Bug reports should include a minimal reproduction, expected and actual behavior,
Python/PyTorch versions, and relevant operating-system or device details. Do not
attach private robot data or credentials.

## Adding a model

1. Implement the `BaseVLA` contract and register a unique model name.
2. Keep architecture-specific behavior in the model implementation.
3. Add synthetic tests for output shapes, finite loss, backward propagation,
   checkpoint save/load, and inference.
4. Document encoder interfaces, tensor shapes, and optional packages.

The trainer should work through `compute_loss` without model-specific branches.

## Adding a dataset adapter

1. Implement `BaseVLADataset` and return the documented sample fields.
2. Provide an `ActionSpec`; fit normalization statistics on training data only.
3. Reuse shared transforms and collation where appropriate.
4. Add tests using small generated fixtures rather than downloaded datasets.
5. Document storage assumptions and optional dependencies.

## Pull request checklist

- [ ] The change has a clear purpose and focused scope.
- [ ] Relevant tests were added or updated.
- [ ] Ruff and pytest checks pass.
- [ ] Public behavior and examples are documented.
- [ ] No dataset, checkpoint, credential, or generated binary was committed.
