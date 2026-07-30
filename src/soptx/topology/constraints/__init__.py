"""Topology-optimization constraints."""

from importlib import import_module

_EXPORTS = {
    "ApparentStressConstraint": (
        ".apparent_stress",
        "ApparentStressConstraint",
    ),
    "VanishingStressConstraint": (
        ".vanishing_stress",
        "VanishingStressConstraint",
    ),
    "VolumeConstraint": (".volume", "VolumeConstraint"),
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
