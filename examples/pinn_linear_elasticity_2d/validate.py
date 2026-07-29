"""Standalone correctness validation for the 2D linear-elasticity PINN.

This driver does not require pytest. It checks the manufactured solution,
failure guards, a one-update checkpoint smoke test, and a configurable
training baseline. It prints a concise acceptance summary and does not write
validation artifacts into the repository.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import platform
import subprocess
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from fealpy.backend import bm
from fealpy.ml.modules import Solution
from model import LinearElasticityPINNModel
from soptx.model.linear_elasticity_2d import TriSolHomoDirHuZhang2d


EXAMPLE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_DIR.parents[1]
class ExactSOPTDisplacement(torch.nn.Module):
    """The SOPT-X manufactured displacement represented as a network."""

    def forward(self, p):
        x, y = p[:, 0], p[:, 1]
        u1 = torch.exp(x - y) * x * (1 - x) * y * (1 - y)
        u2 = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        return torch.stack([u1, u2], dim=-1)


class ZeroDisplacement(torch.nn.Module):
    """A zero mapping used to integrate the exact-solution L2 norm."""

    def forward(self, p):
        return torch.zeros_like(p)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate the 2D plane-strain linear-elasticity PINN baseline.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2000,
        help='Parameter updates in the training baseline; use zero to skip training.',
    )
    parser.add_argument(
        '--max-validation-loss',
        type=float,
        default=5.0e-2,
        help='Maximum best fixed-validation loss for the baseline gate.',
    )
    parser.add_argument(
        '--max-relative-l2',
        type=float,
        default=1.0e-1,
        help='Maximum relative displacement L2 error at the best checkpoint.',
    )
    return parser.parse_args()


def make_options(**overrides: Any) -> dict[str, Any]:
    """Return a small deterministic configuration for structural checks."""
    options = {
        'epochs': 1,
        'hidden_size': (4,),
        'npde': 4,
        'nbc': 2,
        'nval_pde': 4,
        'nval_bc': 2,
        'log_interval': 1,
        'pbar_log': False,
        'seed': 7,
        'device': 'cpu',
        'dtype': 'float64',
    }
    options.update(overrides)
    return options


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ['git', *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def environment_record() -> dict[str, Any]:
    status = git_value('status', '--porcelain')
    return {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'fealpy': package_version('fealpy'),
        'soptx': package_version('soptx'),
        'cuda_available': torch.cuda.is_available(),
        'git_revision': git_value('rev-parse', 'HEAD'),
        'git_dirty': None if status is None else bool(status),
    }


def plain_options(options: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in options.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        elif isinstance(value, (tuple, list)):
            result[key] = list(value)
        else:
            result[key] = repr(value)
    return result


def add_check(
    result: dict[str, Any],
    name: str,
    passed: bool,
    *,
    metrics: dict[str, Any] | None = None,
    detail: str = '',
) -> None:
    result['checks'][name] = {
        'passed': bool(passed),
        'metrics': {} if metrics is None else metrics,
        'detail': detail,
    }
    if not passed:
        result['failures'].append(name if not detail else f'{name}: {detail}')


def expect_exception(
    action: Callable[[], Any],
    exception_type: type[BaseException],
    message_fragment: str,
) -> tuple[bool, str]:
    try:
        action()
    except exception_type as error:
        message = str(error)
        return message_fragment in message, message
    except BaseException as error:  # noqa: BLE001 - report wrong exception types.
        return False, f'{type(error).__name__}: {error}'
    return False, 'No exception was raised.'


def validate_constructor(result: dict[str, Any]) -> None:
    model = LinearElasticityPINNModel(make_options())
    point = torch.zeros((3, 2), dtype=torch.float64)
    prediction = model.predict(point)
    passed = (
        model.gd == 2
        and model.pde.__class__.__name__ == 'TriSolHomoDirHuZhang2d'
        and tuple(prediction.shape) == (3, 2)
    )
    add_check(
        result,
        'constructor_and_shape',
        passed,
        metrics={
            'geometric_dimension': model.gd,
            'pde_class': model.pde.__class__.__name__,
            'prediction_shape': list(prediction.shape),
        },
    )


def validate_exact_solution(result: dict[str, Any]) -> None:
    model = LinearElasticityPINNModel(make_options())
    model.net = Solution(ExactSOPTDisplacement()).to(dtype=torch.float64)

    interior = torch.tensor(
        [[0.2, 0.3], [0.4, 0.6], [0.7, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    boundary = torch.tensor(
        [[0.0, 0.2], [1.0, 0.4], [0.5, 0.0], [0.8, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    strain = model.strain(interior)
    equilibrium = model.equilibrium_residual(interior, create_graph=False)
    boundary_residual = model.dirichlet_residual(boundary)

    symmetry_max = float(
        torch.max(torch.abs(strain - strain.transpose(-1, -2))).item()
    )
    equilibrium_component_max = torch.max(
        torch.abs(equilibrium),
        dim=0,
    ).values
    equilibrium_max = float(torch.max(torch.abs(equilibrium)).item())
    boundary_max = float(torch.max(torch.abs(boundary_residual)).item())
    thresholds = result['thresholds']
    passed = (
        symmetry_max <= thresholds['exact_strain_symmetry_max_abs']
        and equilibrium_max <= thresholds['exact_equilibrium_max_abs']
        and boundary_max <= thresholds['exact_boundary_max_abs']
    )
    add_check(
        result,
        'manufactured_solution_consistency',
        passed,
        metrics={
            'strain_symmetry_max_abs': symmetry_max,
            'equilibrium_x_max_abs': float(equilibrium_component_max[0].item()),
            'equilibrium_y_max_abs': float(equilibrium_component_max[1].item()),
            'equilibrium_residual_max_abs': equilibrium_max,
            'dirichlet_residual_max_abs': boundary_max,
        },
    )


def validate_failure_guards(result: dict[str, Any]) -> None:
    guard_results: dict[str, dict[str, Any]] = {}

    bm.set_backend('numpy')
    try:
        passed, detail = expect_exception(
            lambda: LinearElasticityPINNModel(make_options()),
            RuntimeError,
            'PyTorch backend',
        )
    finally:
        bm.set_backend('pytorch')
    guard_results['non_pytorch_backend'] = {'passed': passed, 'detail': detail}

    class ThreeDimensionalProblem(TriSolHomoDirHuZhang2d):
        def geo_dimension(self):
            return 3

    passed, detail = expect_exception(
        lambda: LinearElasticityPINNModel(
            make_options(pde=ThreeDimensionalProblem())
        ),
        ValueError,
        'Only 2D plane strain',
    )
    guard_results['three_dimensional_problem'] = {
        'passed': passed,
        'detail': detail,
    }

    plane_stress_data = TriSolHomoDirHuZhang2d()
    plane_stress_data._plane_type = 'plane_stress'
    passed, detail = expect_exception(
        lambda: LinearElasticityPINNModel(make_options(pde=plane_stress_data)),
        ValueError,
        "plane_type='plane_strain'",
    )
    guard_results['plane_stress_problem'] = {'passed': passed, 'detail': detail}

    mixed_data = TriSolHomoDirHuZhang2d()
    mixed_data.is_displacement_boundary = lambda p: torch.zeros(
        p.shape[:-1], dtype=torch.bool, device=p.device
    )
    mixed_model = LinearElasticityPINNModel(make_options(pde=mixed_data))
    boundary = torch.tensor(
        [[0.0, 0.2], [1.0, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    passed, detail = expect_exception(
        lambda: mixed_model.dirichlet_residual(boundary),
        ValueError,
        'requires every sampled boundary point',
    )
    guard_results['mixed_boundary_problem'] = {'passed': passed, 'detail': detail}

    add_check(
        result,
        'unsupported_problem_guards',
        all(record['passed'] for record in guard_results.values()),
        metrics=guard_results,
    )


def validate_one_update(result: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix='soptx-pinn-smoke-') as directory:
        model = LinearElasticityPINNModel(
            make_options(checkpoint_dir=directory)
        )
        before = [
            parameter.detach().clone()
            for parameter in model.net.parameters()
        ]
        history = model.run()
        changed = any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, model.net.parameters())
        )
        best_exists = (Path(directory) / 'best.pt').is_file()
        last_exists = (Path(directory) / 'last.pt').is_file()
        finite = all(
            math.isfinite(value)
            for key in ('train_loss', 'validation_loss')
            for value in history[key]
        )
        passed = (
            history['epoch'] == [1]
            and changed
            and best_exists
            and last_exists
            and finite
        )
        add_check(
            result,
            'one_update_and_checkpoints',
            passed,
            metrics={
                'recorded_epochs': history['epoch'],
                'parameters_changed': changed,
                'best_checkpoint_created': best_exists,
                'last_checkpoint_created': last_exists,
                'finite_losses': finite,
            },
        )


def fixed_diagnostic_points(
    domain: tuple[float, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    xmin, xmax, ymin, ymax = domain
    x = torch.linspace(xmin, xmax, 34, dtype=dtype, device=device)[1:-1]
    y = torch.linspace(ymin, ymax, 34, dtype=dtype, device=device)[1:-1]
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    interior = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    interior.requires_grad_(True)

    parameter = torch.linspace(0.0, 1.0, 101, dtype=dtype, device=device)
    left = torch.stack([
        torch.full_like(parameter, xmin),
        ymin + (ymax - ymin) * parameter,
    ], dim=-1)
    right = torch.stack([
        torch.full_like(parameter, xmax),
        ymin + (ymax - ymin) * parameter,
    ], dim=-1)
    bottom = torch.stack([
        xmin + (xmax - xmin) * parameter,
        torch.full_like(parameter, ymin),
    ], dim=-1)
    top = torch.stack([
        xmin + (xmax - xmin) * parameter,
        torch.full_like(parameter, ymax),
    ], dim=-1)
    return interior, torch.cat([left, right, bottom, top], dim=0)


def relative_l2_metrics(
    model: LinearElasticityPINNModel,
) -> dict[str, Any]:
    error_components, absolute_combined = model.displacement_l2_error()
    if error_components is None or absolute_combined is None:
        raise RuntimeError('Unable to evaluate displacement L2 error.')

    zero = Solution(ZeroDisplacement()).to(
        device=model.device,
        dtype=model.dtype,
    )
    exact_norm_components = zero.estimate_error(
        model.pde.disp_solution,
        model.mesh,
        coordtype='c',
    ).detach()
    exact_norm_combined = float(
        torch.linalg.vector_norm(exact_norm_components).item()
    )
    if exact_norm_combined == 0.0:
        raise RuntimeError('The exact displacement has zero L2 norm.')

    relative_components = error_components / exact_norm_components
    return {
        'absolute_components': [
            float(value) for value in error_components.flatten().tolist()
        ],
        'absolute_combined': absolute_combined,
        'exact_norm_components': [
            float(value) for value in exact_norm_components.flatten().tolist()
        ],
        'exact_norm_combined': exact_norm_combined,
        'relative_components': [
            float(value) for value in relative_components.flatten().tolist()
        ],
        'relative_combined': absolute_combined / exact_norm_combined,
    }


def validate_training_baseline(
    result: dict[str, Any],
    *,
    epochs: int,
) -> None:
    if epochs == 0:
        result['training_baseline_performed'] = False
        add_check(
            result,
            'training_baseline',
            True,
            metrics={'performed': False},
            detail='Training baseline explicitly skipped.',
        )
        return
    if epochs < 0:
        raise ValueError("'epochs' must be non-negative.")

    result['training_baseline_performed'] = True
    with tempfile.TemporaryDirectory(prefix='soptx-pinn-baseline-') as directory:
        options = LinearElasticityPINNModel.get_options(args=[])
        options.update({
            'epochs': epochs,
            'checkpoint_dir': directory,
            'device': 'cpu',
            'dtype': 'float64',
            'pbar_log': False,
        })
        model = LinearElasticityPINNModel(options)
        result['training_config'] = plain_options(model.options)

        start = time.perf_counter()
        history = model.run()
        elapsed = time.perf_counter() - start

        best_path = Path(directory) / 'best.pt'
        last_path = Path(directory) / 'last.pt'
        if not best_path.is_file() or not last_path.is_file():
            raise RuntimeError('Training did not create both temporary checkpoints.')

        best_payload = torch.load(
            best_path,
            map_location=model.device,
            weights_only=False,
        )
        model.net.load_state_dict(best_payload['model_state_dict'])

        l2 = relative_l2_metrics(model)
        interior, boundary = fixed_diagnostic_points(
            model.domain,
            dtype=model.dtype,
            device=model.device,
        )
        equilibrium = model.equilibrium_residual(interior, create_graph=False)
        with torch.no_grad():
            boundary_error = model.predict(boundary) - model.pde.dirichlet_bc(boundary)

        equilibrium_rms = float(torch.sqrt(torch.mean(equilibrium**2)).item())
        boundary_max = float(torch.max(torch.abs(boundary_error)).item())
        best_validation = float(min(history['validation_loss']))
        first_validation = float(history['validation_loss'][0])
        final_validation = float(history['validation_loss'][-1])
        all_finite = all(
            value is None or math.isfinite(value)
            for key, values in history.items()
            if key != 'epoch'
            for item in values
            for value in (item if isinstance(item, list) else [item])
        )

        thresholds = result['thresholds']
        validation_passed = (
            best_validation <= thresholds['best_validation_loss_max']
            and best_validation < first_validation
        )
        accuracy_passed = (
            l2['relative_combined']
            <= thresholds['relative_displacement_l2_max']
        )
        passed = all_finite and validation_passed and accuracy_passed

        add_check(
            result,
            'training_baseline',
            passed,
            metrics={
                'performed': True,
                'elapsed_seconds': elapsed,
                'recorded_updates': history['epoch'],
                'first_validation_loss': first_validation,
                'best_validation_loss': best_validation,
                'final_validation_loss': final_validation,
                'validation_gate_passed': validation_passed,
                'relative_l2_gate_passed': accuracy_passed,
                'best_checkpoint_epoch': int(best_payload['epoch']),
                'best_checkpoint_created': True,
                'last_checkpoint_created': True,
                'all_history_values_finite': all_finite,
                'best_checkpoint_displacement_l2': l2,
                'fixed_equilibrium_residual_rms': equilibrium_rms,
                'fixed_boundary_error_max_abs': boundary_max,
                'history': history,
            },
        )


def print_result(result: dict[str, Any]) -> None:
    def print_metrics(metrics: dict[str, Any], indent: str = '    ') -> None:
        for key, value in metrics.items():
            if key == 'history' or isinstance(value, (list, tuple)):
                continue
            if isinstance(value, dict):
                print(f'{indent}{key}:')
                print_metrics(value, indent + '  ')
            else:
                print(f'{indent}{key}: {value}')

    environment = result.get('environment', {})
    print(
        'environment: '
        f"python={environment.get('python')}, "
        f"torch={environment.get('torch')}, "
        f"cuda_available={environment.get('cuda_available')}, "
        f"git={environment.get('git_revision')}, "
        f"dirty={environment.get('git_dirty')}"
    )
    print('validation gates:')
    for name, record in result.get('checks', {}).items():
        marker = 'PASS' if record.get('passed') else 'FAIL'
        print(f'  [{marker}] {name}')
        print_metrics(record.get('metrics', {}))
    print(f"validation status: {result['status']}")
    for failure in result.get('failures', []):
        print(f'failure: {failure}')


def run_validation(arguments: argparse.Namespace) -> dict[str, Any]:
    bm.set_backend('pytorch')
    result: dict[str, Any] = {
        'schema_version': 1,
        'scope': '2d-plane-strain-all-dirichlet-pinn-baseline',
        'status': 'running',
        'environment': environment_record(),
        'thresholds': {
            'exact_strain_symmetry_max_abs': 1.0e-12,
            'exact_equilibrium_max_abs': 1.0e-10,
            'exact_boundary_max_abs': 1.0e-12,
            'best_validation_loss_max': arguments.max_validation_loss,
            'relative_displacement_l2_max': arguments.max_relative_l2,
        },
        'checks': {},
        'failures': [],
    }
    validate_constructor(result)
    validate_exact_solution(result)
    validate_failure_guards(result)
    validate_one_update(result)
    validate_training_baseline(result, epochs=arguments.epochs)
    if result['failures']:
        result['status'] = 'failed'
    elif result['training_baseline_performed']:
        result['status'] = 'passed'
    else:
        result['status'] = 'partial'
    return result


def main() -> int:
    arguments = get_arguments()
    try:
        result = run_validation(arguments)
    except BaseException as error:  # noqa: BLE001 - always report the failure.
        result = {
            'schema_version': 1,
            'scope': '2d-plane-strain-all-dirichlet-pinn-baseline',
            'status': 'error',
            'environment': environment_record(),
            'checks': {},
            'failures': [f'{type(error).__name__}: {error}'],
            'traceback': traceback.format_exc(),
        }

    print_result(result)
    return 0 if result['status'] in {'passed', 'partial'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
