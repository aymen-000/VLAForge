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

## Test organization and optional integrations

Keep fast, isolated tests in `tests/unit/`. Put tests that connect VLAForge to
real optional packages or external data formats in `tests/integration/`.
Unit tests should use small fixtures and must not need network access or
downloaded datasets. The normal `pytest` command discovers both directories;
opt-in integration tests should skip when their dependencies or configuration
are not present.

The LeRobot integration test is opt-in. Install the optional dependency and
set a dataset repository ID to run it:

```bash
python -m pip install -e ".[dev,lerobot]"
export VLAFORGE_LEROBOT_REPO_ID="lerobot/xarm_lift_medium_image"
python -m pytest tests/integration/test_lerobot_integration.py
```

The suggested dataset is a small image dataset (78.8 MB total on the Hub).
The test opens its first two episodes. LeRobot may download complete underlying
data or video chunks for those episodes, rather than only the selected frames,
if they are not already cached. Set `VLAFORGE_LEROBOT_ROOT` to use a local
dataset directory.
The action, task, and proprioception field names can be overridden with
`VLAFORGE_LEROBOT_ACTION_FIELD`, `VLAFORGE_LEROBOT_TASK_FIELD`, and
`VLAFORGE_LEROBOT_PROPRIOCEPTION_FIELD`.

In Windows PowerShell, set the repository ID with:

```powershell
$env:VLAFORGE_LEROBOT_REPO_ID = "lerobot/xarm_lift_medium_image"
python -m pytest tests/integration/test_lerobot_integration.py
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
