"""PINN model for a two-dimensional plane-strain linear-elasticity problem.

The first implementation intentionally supports only the all-Dirichlet
manufactured solution ``TriSolHomoDirHuZhang2d`` owned by SOPT-X.
Traction and mixed boundary conditions need boundary normals and are therefore
kept out of this self-contained, verifiable baseline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from fealpy.backend import bm
from fealpy.typing import TensorLike

from fealpy.ml.grad import gradient
from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler
from fealpy.ml.torch_mapping import activations, optimizers
from problem import make_default_problem


class PlaneStrainDirichletProblem(Protocol):
    """SOPT-X-owned contract for the first linear-elasticity PINN problem."""

    @property
    def domain(self) -> Sequence[float]: ...

    @property
    def plane_type(self) -> str: ...

    @property
    def lam(self) -> float: ...

    @property
    def mu(self) -> float: ...

    def init_mesh(self, **kwargs): ...
    def body_force(self, points: TensorLike) -> TensorLike: ...
    def disp_solution(self, points: TensorLike) -> TensorLike: ...
    def dirichlet_bc(self, points: TensorLike) -> TensorLike: ...
    def is_dirichlet_boundary(self): ...


class LinearElasticityPINNModel:
    """A reproducible PINN baseline for 2D plane-strain linear elasticity.

    The neural network maps coordinates ``(x, y)`` to displacement
    ``(u_x, u_y)``.  The model enforces

    ``-div(sigma(u)) = body_force`` in the domain and the prescribed
    displacement on every boundary face.  It deliberately rejects plane
    stress, 3D, traction, and mixed-boundary data so that its first use has a
    clear mechanics contract.
    """

    def __init__(self, options: Optional[dict[str, Any]] = None):
        self.options = self.get_options(args=[])
        if options is not None:
            self.options.update(options)

        self._require_pytorch_backend()
        self._configure_logger()

        self.seed = int(self.options['seed'])
        self.device = torch.device(self.options['device'])
        if self.options['dtype'] != 'float64':
            raise ValueError("Only dtype='float64' is supported in this baseline.")
        self.dtype = torch.float64
        self._set_seed()

        self.lr = float(self.options['lr'])
        self.epochs = int(self.options['epochs'])
        self.hidden_size = tuple(self.options['hidden_size'])
        self.activation = activations[self.options['activation']]
        self.npde = int(self.options['npde'])
        self.nbc = int(self.options['nbc'])
        self.nval_pde = int(self.options['nval_pde'])
        self.nval_bc = int(self.options['nval_bc'])
        self.weights = tuple(float(value) for value in self.options['weights'])
        self.log_interval = int(self.options['log_interval'])
        self.checkpoint_dir = self.options['checkpoint_dir']

        if self.epochs < 1:
            raise ValueError("'epochs' must be at least one parameter update.")
        if min(self.npde, self.nbc, self.nval_pde, self.nval_bc) < 1:
            raise ValueError("All collocation-point counts must be positive.")
        if len(self.weights) != 2:
            raise ValueError("'weights' must contain (equilibrium, Dirichlet).")
        if self.log_interval < 1:
            raise ValueError("'log_interval' must be positive.")

        self.set_pde(self.options['pde'])
        self.set_mesh(self.options['mesh_size'])
        self.set_network()

        self.history: dict[str, list[Any]] = {}
        self.best_validation_loss = float('inf')
        self.last_metrics: dict[str, Any] = {}

    @classmethod
    def get_options(cls, args: Optional[Sequence[str]] = None) -> dict[str, Any]:
        """Return options; ``args=None`` parses the command line for examples."""
        parser = argparse.ArgumentParser(
            description="Solve a 2D plane-strain elasticity problem using PINN."
        )
        parser.add_argument('--pde', default='soptx-default',
                            choices=('soptx-default',),
                            help='SOPT-X built-in plane-strain manufactured problem.')
        parser.add_argument('--mesh_size', default=30, type=int,
                            help='Nodes per coordinate direction for L2 diagnostics.')
        parser.add_argument('--sampling_mode', default='random',
                            choices=('random', 'linspace'),
                            help="'random' samples per epoch; 'linspace' reuses one grid.")
        parser.add_argument('--npde', default=400, type=int,
                            help='Interior collocation points for random sampling.')
        parser.add_argument('--nbc', default=100, type=int,
                            help='Boundary collocation points per boundary face.')
        parser.add_argument('--nval_pde', default=400, type=int,
                            help='Fixed interior diagnostic points.')
        parser.add_argument('--nval_bc', default=100, type=int,
                            help='Fixed boundary diagnostic points per boundary face.')
        parser.add_argument('--weights', default=(1.0, 30.0), nargs=2, type=float,
                            metavar=('W_EQ', 'W_D'),
                            help='Weights for equilibrium and Dirichlet loss.')
        parser.add_argument('--hidden_size', default=(32, 32, 16), nargs='+', type=int,
                            help='Hidden-layer widths.')
        parser.add_argument('--optimizer', default='Adam', choices=tuple(optimizers),
                            help='Optimizer name.')
        parser.add_argument('--activation', default='Tanh', choices=tuple(activations),
                            help='Hidden activation.')
        parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
        parser.add_argument('--step_size', default=0, type=int,
                            help='Learning-rate decay period; zero disables StepLR.')
        parser.add_argument('--gamma', default=0.99, type=float,
                            help='StepLR multiplicative decay factor.')
        parser.add_argument('--epochs', default=2000, type=int,
                            help='Exact number of parameter updates.')
        parser.add_argument('--seed', default=0, type=int, help='PyTorch random seed.')
        parser.add_argument('--device', default='cpu', type=str,
                            help="PyTorch device, for example 'cpu' or 'cuda'.")
        parser.add_argument('--dtype', default='float64', choices=('float64',),
                            help='PINN floating-point dtype; float64 is required in v1.')
        parser.add_argument('--log_interval', default=100, type=int,
                            help='Updates between diagnostic records.')
        parser.add_argument('--checkpoint_dir', default=None, type=str,
                            help='Optional directory for best.pt and last.pt.')
        parser.add_argument('--pbar_log', action='store_true',
                            help='Send logs through the progress-bar handler.')
        parser.add_argument('--log_level', default='INFO', type=str,
                            help='Python logging level.')
        return vars(parser.parse_args(args))

    def _configure_logger(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.propagate = False
        self.logger.setLevel(self.options['log_level'])
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

    def _require_pytorch_backend(self) -> None:
        if bm.backend_name != 'pytorch':
            raise RuntimeError(
                "LinearElasticityPINNModel requires the PyTorch backend because "
                "its residual uses torch.autograd."
            )

    def _set_seed(self) -> None:
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError(f"Requested device '{self.device}' is unavailable.")
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(self.seed)

    def set_pde(
        self, pde: Union[PlaneStrainDirichletProblem, str, None] = None
    ) -> None:
        """Set a SOPT-X-owned two-dimensional plane-strain problem."""
        if pde is None or pde == 'soptx-default':
            pde = make_default_problem()
        if isinstance(pde, str):
            raise ValueError(f"Unknown SOPT-X PINN problem {pde!r}.")

        if hasattr(pde, 'geo_dimension') and pde.geo_dimension() != 2:
            raise ValueError("Only 2D plane strain is supported in this baseline.")
        if getattr(pde, 'plane_type', None) != 'plane_strain':
            raise ValueError(
                "Only PDE data with plane_type='plane_strain' are supported in this baseline."
            )

        self.pde = pde
        self.gd = 2
        self.domain = tuple(pde.domain)

    def set_mesh(self, mesh_size: int = 30, mesh=None) -> None:
        """Set the mesh used only for quadrature-based L2 diagnostics."""
        if mesh is None:
            if mesh_size < 2:
                raise ValueError("'mesh_size' must be at least two.")
            self.mesh_size = (mesh_size, mesh_size)
            self.mesh = self.pde.init_mesh(nx=mesh_size - 1, ny=mesh_size - 1)
        else:
            self.mesh = mesh

    def set_network(self, net: Optional[nn.Module] = None) -> None:
        """Set a coordinate-to-displacement network and its optimizer."""
        if net is None:
            layers: list[nn.Module] = []
            sizes = (self.gd,) + self.hidden_size + (self.gd,)
            for index in range(len(sizes) - 1):
                layers.append(nn.Linear(
                    sizes[index], sizes[index + 1], dtype=self.dtype, device=self.device
                ))
                if index < len(sizes) - 2:
                    layers.append(self.activation())
            net = nn.Sequential(*layers)

        self.net = Solution(net).to(device=self.device, dtype=self.dtype)
        optimizer_type = optimizers[self.options['optimizer']]
        self.optimizer = optimizer_type(self.net.parameters(), lr=self.lr)

        step_size = int(self.options['step_size'])
        self.steplr = (
            None if step_size == 0
            else StepLR(self.optimizer, step_size=step_size, gamma=float(self.options['gamma']))
        )

    def displacement_gradient(self, p: TensorLike) -> TensorLike:
        """Return ``grad(u)`` with shape ``(n_points, 2, 2)``.

        Each displacement component is differentiated separately.  Differentiating
        a vector output in one call would instead differentiate its component sum.
        """
        u = self.net(p)
        if u.ndim != 2 or u.shape[-1] != self.gd:
            raise ValueError(
                f"Network must return shape (n_points, {self.gd}), got {tuple(u.shape)}."
            )
        return torch.stack([
            gradient(u[:, component:component + 1], p, create_graph=True)
            for component in range(self.gd)
        ], dim=1)

    def strain(self, p: TensorLike) -> TensorLike:
        """Return the infinitesimal strain tensor with shape ``(N, 2, 2)``."""
        grad_u = self.displacement_gradient(p)
        return 0.5 * (grad_u + grad_u.transpose(-1, -2))

    def stress(self, p: TensorLike) -> TensorLike:
        """Return isotropic plane-strain stress ``lambda tr(eps) I + 2 mu eps``."""
        eps = self.strain(p)
        lam = self._field_value(self.pde.lam, p)
        mu = self._field_value(self.pde.mu, p)
        identity = torch.eye(self.gd, dtype=p.dtype, device=p.device).unsqueeze(0)
        trace = torch.diagonal(eps, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        return lam * trace.unsqueeze(-1) * identity + 2.0 * mu * eps

    def equilibrium_residual(self, p: TensorLike, *, create_graph: bool = True) -> TensorLike:
        """Return ``-div(sigma) - body_force`` with shape ``(N, 2)``."""
        sigma = self.stress(p)
        divergence_components = []
        total_derivatives = self.gd * self.gd
        derivative_index = 0
        for equation_component in range(self.gd):
            divergence_component = torch.zeros_like(p[:, 0])
            for coordinate_component in range(self.gd):
                sigma_component = sigma[:, equation_component, coordinate_component]
                is_last_derivative = derivative_index == total_derivatives - 1
                grad_sigma_component = torch.autograd.grad(
                    outputs=sigma_component,
                    inputs=p,
                    grad_outputs=torch.ones_like(sigma_component),
                    create_graph=create_graph,
                    retain_graph=create_graph or not is_last_derivative,
                )[0]
                divergence_component = (
                    divergence_component
                    + grad_sigma_component[:, coordinate_component]
                )
                derivative_index += 1
            divergence_components.append(divergence_component)
        divergence = torch.stack(divergence_components, dim=-1)
        body_force = self.pde.body_force(p)
        if body_force.shape != divergence.shape:
            raise ValueError(
                f"body_force shape {tuple(body_force.shape)} does not match "
                f"equilibrium residual shape {tuple(divergence.shape)}."
            )
        return -divergence - body_force

    def dirichlet_residual(self, p: TensorLike) -> TensorLike:
        """Return the displacement residual on an all-Dirichlet boundary."""
        mask = self._dirichlet_mask(p)
        if mask.shape != p.shape[:-1] or not bool(torch.all(mask)):
            raise ValueError(
                "This baseline requires every sampled boundary point to be Dirichlet; "
                "traction and mixed boundary conditions are not implemented."
            )
        value = self.net(p) - self.pde.dirichlet_bc(p)
        if value.shape != p.shape:
            raise ValueError(
                f"Dirichlet residual shape {tuple(value.shape)} does not match "
                f"point shape {tuple(p.shape)}."
            )
        return value

    @staticmethod
    def _field_value(field, p: TensorLike):
        return field(p) if callable(field) else field

    def _dirichlet_mask(self, p: TensorLike) -> TensorLike:
        direct_predicate = getattr(self.pde, 'is_displacement_boundary', None)
        if direct_predicate is not None:
            return direct_predicate(p)

        predicates = self.pde.is_dirichlet_boundary()
        if not isinstance(predicates, (tuple, list)):
            return predicates(p)
        mask = torch.ones(p.shape[:-1], dtype=torch.bool, device=p.device)
        for predicate in predicates:
            mask &= predicate(p)
        return mask

    def _exact_displacement(self, p: TensorLike) -> TensorLike:
        return self.pde.disp_solution(p)

    def _loss_components(
        self, interior: TensorLike, boundary: TensorLike, *, create_graph: bool
    ) -> tuple[TensorLike, TensorLike, TensorLike]:
        equilibrium = self.equilibrium_residual(interior, create_graph=create_graph)
        dirichlet = self.dirichlet_residual(boundary)
        mse = nn.MSELoss(reduction='mean')
        equilibrium_loss = mse(equilibrium, torch.zeros_like(equilibrium))
        dirichlet_loss = mse(dirichlet, torch.zeros_like(dirichlet))
        total_loss = self.weights[0] * equilibrium_loss + self.weights[1] * dirichlet_loss
        return equilibrium_loss, dirichlet_loss, total_loss

    def _make_samplers(self) -> tuple[ISampler, BoxBoundarySampler]:
        interior = ISampler(
            self.domain, mode=self.options['sampling_mode'], dtype=bm.float64,
            device=self.device, requires_grad=True,
        )
        boundary = BoxBoundarySampler(
            self.domain, mode=self.options['sampling_mode'], dtype=bm.float64,
            device=self.device, requires_grad=True,
        )
        return interior, boundary

    def _sample_pair(
        self, interior_sampler: ISampler, boundary_sampler: BoxBoundarySampler,
        n_interior: int, n_boundary: int,
    ) -> tuple[TensorLike, TensorLike]:
        return interior_sampler.run(n_interior), boundary_sampler.run(n_boundary)

    def _reset_history(self) -> None:
        self.history = {
            'epoch': [],
            'train_equilibrium_loss': [],
            'train_dirichlet_loss': [],
            'train_loss': [],
            'validation_equilibrium_loss': [],
            'validation_dirichlet_loss': [],
            'validation_loss': [],
            'learning_rate': [],
            'l2_error_components': [],
            'l2_error': [],
        }

    def displacement_l2_error(self) -> tuple[Optional[TensorLike], Optional[float]]:
        """Return component and combined displacement L2 error when available."""
        try:
            component_error = self.net.estimate_error(
                self._exact_displacement, self.mesh, coordtype='c'
            ).detach()
            combined_error = float(torch.linalg.vector_norm(component_error).item())
            return component_error, combined_error
        except (AttributeError, NotImplementedError, RuntimeError, ValueError) as error:
            self.logger.warning("Unable to evaluate displacement L2 error: %s", error)
            return None, None

    def _checkpoint_payload(self, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {}
        for key, value in self.options.items():
            if isinstance(value, (str, int, float, bool, type(None), tuple, list)):
                options[key] = value
            else:
                options[key] = repr(value)
        return {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': None if self.steplr is None else self.steplr.state_dict(),
            'options': options,
            'history': self.history,
            'metrics': metrics,
        }

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, Any]) -> None:
        if self.checkpoint_dir is None:
            return
        directory = Path(self.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_payload(epoch, metrics), directory / name)

    def _record_diagnostics(
        self, epoch: int, train_losses: tuple[TensorLike, TensorLike, TensorLike],
        validation_points: tuple[TensorLike, TensorLike],
    ) -> dict[str, Any]:
        validation_losses = self._loss_components(
            *validation_points, create_graph=False
        )
        component_error, combined_error = self.displacement_l2_error()
        component_values = (
            None if component_error is None
            else [float(value) for value in component_error.flatten().tolist()]
        )
        metrics = {
            'train_equilibrium_loss': float(train_losses[0].detach().item()),
            'train_dirichlet_loss': float(train_losses[1].detach().item()),
            'train_loss': float(train_losses[2].detach().item()),
            'validation_equilibrium_loss': float(validation_losses[0].detach().item()),
            'validation_dirichlet_loss': float(validation_losses[1].detach().item()),
            'validation_loss': float(validation_losses[2].detach().item()),
            'learning_rate': float(self.optimizer.param_groups[0]['lr']),
            'l2_error_components': component_values,
            'l2_error': combined_error,
        }
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            self.history[key].append(value)
        self.last_metrics = metrics
        self.logger.info(
            "epoch: %d, train loss: %.6e, validation loss: %.6e",
            epoch, metrics['train_loss'], metrics['validation_loss'],
        )
        return metrics

    def run(self) -> dict[str, list[Any]]:
        """Train for exactly ``epochs`` updates and return the recorded history."""
        self._set_seed()
        self._reset_history()
        self.best_validation_loss = float('inf')

        train_interior_sampler, train_boundary_sampler = self._make_samplers()
        validation_interior_sampler, validation_boundary_sampler = self._make_samplers()
        validation_points = self._sample_pair(
            validation_interior_sampler, validation_boundary_sampler,
            self.nval_pde, self.nval_bc,
        )

        train_points: Optional[tuple[TensorLike, TensorLike]] = None
        if self.options['sampling_mode'] == 'linspace':
            train_points = self._sample_pair(
                train_interior_sampler, train_boundary_sampler, self.npde, self.nbc
            )

        for epoch in range(1, self.epochs + 1):
            if train_points is None:
                train_points = self._sample_pair(
                    train_interior_sampler, train_boundary_sampler, self.npde, self.nbc
                )
            interior, boundary = train_points
            interior.grad = None
            boundary.grad = None

            self.optimizer.zero_grad()
            train_losses = self._loss_components(interior, boundary, create_graph=True)
            train_losses[2].backward()
            self.optimizer.step()
            if self.steplr is not None:
                self.steplr.step()

            if epoch == 1 or epoch % self.log_interval == 0 or epoch == self.epochs:
                metrics = self._record_diagnostics(epoch, train_losses, validation_points)
                if metrics['validation_loss'] < self.best_validation_loss:
                    self.best_validation_loss = metrics['validation_loss']
                    self._save_checkpoint('best.pt', epoch, metrics)

            if self.options['sampling_mode'] == 'random':
                train_points = None

        self._save_checkpoint('last.pt', self.epochs, self.last_metrics)
        return self.history

    def predict(self, p: TensorLike) -> TensorLike:
        """Predict the two displacement components at the supplied coordinates."""
        return self.net(p)

    def show(self) -> None:
        """Plot loss diagnostics and, when available, displacement L2 error."""
        if not self.history.get('epoch'):
            raise RuntimeError("Run training before calling show().")

        import matplotlib.pyplot as plt

        epochs = self.history['epoch']
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].semilogy(epochs, self.history['train_loss'], label='train loss')
        axes[0].semilogy(epochs, self.history['validation_loss'], label='fixed validation loss')
        axes[0].set_xlabel('parameter updates')
        axes[0].set_ylabel('loss')
        axes[0].set_title('PINN training diagnostics')
        axes[0].grid(True)
        axes[0].legend()

        l2_values = self.history['l2_error']
        if any(value is not None for value in l2_values):
            filtered = [(epoch, value) for epoch, value in zip(epochs, l2_values) if value is not None]
            axes[1].semilogy([item[0] for item in filtered], [item[1] for item in filtered])
            axes[1].set_ylabel('combined displacement L2 error')
            axes[1].set_title('Exact-solution diagnostic')
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, 'No exact displacement diagnostic', ha='center', va='center')
            axes[1].set_axis_off()
        axes[1].set_xlabel('parameter updates')
        figure.tight_layout()
        plt.show()
