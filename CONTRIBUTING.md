# Contributing to VLAForge

Thanks for helping improve VLAForge. Contributions should keep the public
interfaces small, preserve model and dataset independence, and document any
user-visible behavior changes.

## Development environment

Use Python 3.10 or later. Create a virtual environment from the repository
root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

In Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Install a PyTorch build that matches your CPU or CUDA environment before
installing VLAForge if you need a specific build. The library and test suite do
not download pretrained encoders or datasets.

## Local checks

Run the same checks used in CI:

```bash
ruff check src tests examples --select F
python -m pytest
```

Some ROS installations expose unrelated pytest plugins through the active
environment. If pytest fails while importing a ROS plugin before collecting
VLAForge tests, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

On PowerShell, use:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest
```

## Code standards

- Add type annotations to public APIs.
- Keep model-specific losses inside the model's `compute_loss` method.
- Keep dataset storage and preprocessing independent of model architectures.
- Keep optional integrations optional; do not add a new base dependency without
  a clear need.
- Raise descriptive errors for invalid inputs instead of silently substituting
  values.
- Add or update focused tests for behavior changes. Tests must not require
  network access, GPU hardware, private datasets, or pretrained downloads.
- Update the relevant documentation when changing a public API.

## Proposing a change

1. Open an issue for substantial behavior changes or new features.
2. Create a focused branch from the current default branch.
3. Make a small change with tests and documentation.
4. Run the local checks above.
5. Open a pull request describing the motivation, implementation, and checks.

Bug reports should include a minimal reproduction, expected behavior, actual
behavior, Python/PyTorch versions, and relevant operating-system or device
details. Do not attach private robot data or credentials.

## Adding a model

1. Implement the `BaseVLA` contract.
2. Register the model with a unique name.
3. Keep architecture-specific behavior in the model implementation.
4. Add synthetic tests for output shape, finite loss, backward propagation,
   checkpoint save/load, and inference.
5. Document required encoder interfaces, tensor shapes, and optional packages.

The trainer should work with the model through `compute_loss`; it should not
need a model-specific branch.

## Adding a dataset adapter

1. Implement `BaseVLADataset` and return the documented sample fields.
2. Provide an `ActionSpec` and compute statistics from training data only.
3. Reuse the shared transforms and collator where appropriate.
4. Add tests using small generated fixtures instead of downloading datasets.
5. Document storage assumptions and optional dependencies.

## Pull request checklist

- [ ] The change has a clear purpose and focused scope.
- [ ] Relevant tests were added or updated.
- [ ] Ruff and pytest checks pass.
- [ ] Public behavior and examples are documented.
- [ ] No dataset, checkpoint, credential, or generated binary was committed.
