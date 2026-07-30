"""Topology-optimization filters and projection strategies."""

from importlib import import_module

_EXPORTS = {
    "DensityStrategy": (".strategies", "DensityStrategy"),
    "Filter": (".filter", "Filter"),
    "FilterMatrixBuilder": (".matrix", "FilterMatrixBuilder"),
    "NoneStrategy": (".strategies", "NoneStrategy"),
    "ProjectionStrategy": (".strategies", "ProjectionStrategy"),
    "SensitivityStrategy": (".strategies", "SensitivityStrategy"),
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
