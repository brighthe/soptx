"""Topology-optimization algorithms."""

from importlib import import_module

_EXPORTS = {
    "ALMMMAOptimizer": (".al_mma", "ALMMMAOptimizer"),
    "ALMMMAOptions": (".al_mma", "ALMMMAOptions"),
    "MMAOptimizer": (".mma", "MMAOptimizer"),
    "MMAOptions": (".mma", "MMAOptions"),
    "OCOptimizer": (".oc", "OCOptimizer"),
    "OCOptions": (".oc", "OCOptions"),
    "OptimizationHistory": (".history", "OptimizationHistory"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value
