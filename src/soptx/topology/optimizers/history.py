"""Optimization history data without visualization dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import Function
from fealpy.typing import TensorLike


@dataclass
class OptimizationHistory:
    """State recorded by topology optimizers across iterations."""

    iter_indices: List[int] = field(default_factory=list)
    changes: List[float] = field(default_factory=list)
    iteration_times: List[float] = field(default_factory=list)
    physical_densities: List[TensorLike] = field(default_factory=list)
    start_time: float = field(default_factory=time)
    scalar_histories: Dict[str, List[float]] = field(default_factory=dict)
    field_histories: Dict[str, List[TensorLike]] = field(default_factory=dict)

    def log_iteration(
        self,
        iter_idx: int,
        change: float,
        time_cost: float,
        physical_density: TensorLike,
        scalars: Optional[Dict[str, float]] = None,
        fields: Optional[Dict[str, TensorLike]] = None,
    ) -> None:
        if isinstance(physical_density, Function):
            rho_phys = physical_density.space.function(
                bm.copy(physical_density[:])
            )
        else:
            rho_phys = bm.copy(physical_density[:])
        self.iter_indices.append(iter_idx)
        self.changes.append(float(change))
        self.iteration_times.append(time_cost)
        self.physical_densities.append(rho_phys)
        if scalars is not None:
            for key, value in scalars.items():
                self.scalar_histories.setdefault(key, []).append(
                    float(value)
                )
        if fields is not None:
            for key, value in fields.items():
                self.field_histories.setdefault(key, []).append(value)

    def get_total_time(self) -> float:
        return time() - self.start_time

    def get_average_iteration_time(self) -> float:
        if len(self.iteration_times) <= 1:
            return 0.0
        return sum(self.iteration_times[1:]) / (
            len(self.iteration_times) - 1
        )

    def print_time_statistics(self) -> None:
        total_time = self.get_total_time()
        average = self.get_average_iteration_time()
        print("\nTime Statistics:")
        print(f"Total optimization time: {total_time:.3f} sec")
        if self.iteration_times:
            print(
                f"First iteration time: "
                f"{self.iteration_times[0]:.3f} sec"
            )
        if len(self.iteration_times) > 1:
            print(
                "Average iteration time (excluding first): "
                f"{average:.3f} sec"
            )
            print(f"Number of iterations: {len(self.iteration_times)}")
