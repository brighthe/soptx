"""Compatibility imports for optimization history and reporting."""

from soptx.topology.optimizers.history import OptimizationHistory
from soptx.topology.postprocess.history import (
    load_history_data,
    plot_optimization_history,
    plot_optimization_history_backup,
    plot_optimization_history_comparison,
    save_history_data,
    save_optimization_history,
)

__all__ = [
    "OptimizationHistory",
    "load_history_data",
    "plot_optimization_history",
    "plot_optimization_history_backup",
    "plot_optimization_history_comparison",
    "save_history_data",
    "save_optimization_history",
]
