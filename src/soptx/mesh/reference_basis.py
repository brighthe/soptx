# 移植自 brighthe/fealpy ``fealpy/mesh/reference_basis.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Shared Lagrange basis kernels on reference entities.

The functions in this module evaluate scalar Lagrange bases without owning
geometry, finite-element DoF, or physical-coordinate semantics.  A Schema
descriptor is required only to isolate reusable backend tensors in the
external cache; callers remain responsible for selecting the reference
entity convention and local basis ordering.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Callable

from ..backend import Tensor, bm
from .schema.descriptor import SchemaDescriptor

__all__ = [
    "simplex_lagrange_basis",
    "simplex_lagrange_grad_barycentric",
    "simplex_lagrange_grad_reference",
    "tensor_product_lagrange_basis",
    "tensor_product_lagrange_grad_barycentric",
    "tensor_product_lagrange_grad_reference",
]


@dataclass(frozen=True, slots=True)
class _ReferenceBasisCacheKey:
    descriptor: SchemaDescriptor
    operation: str
    order: tuple[int, ...]
    backend: str
    device: str
    dtype: str


_REFERENCE_BASIS_CACHE_MAXSIZE = 256
_REFERENCE_BASIS_CACHE: OrderedDict[
    _ReferenceBasisCacheKey,
    Tensor,
] = OrderedDict()
_REFERENCE_BASIS_CACHE_LOCK = RLock()


def _normalize_simplex_order(p: object) -> int:
    if type(p) is not int:
        raise TypeError(
            f"simplex basis order must be an integer, got {type(p).__name__}"
        )
    if p < 0:
        raise ValueError(f"simplex basis order must be non-negative, got {p}")
    return p


def _normalize_tensor_product_order(
    p: object,
    factor_count: int,
) -> tuple[int, ...]:
    if type(p) is not tuple:
        raise TypeError(
            "tensor-product basis order must be a tuple of integers, "
            f"got {type(p).__name__}"
        )
    if len(p) != factor_count:
        raise ValueError(
            "tensor-product basis order must contain one value per factor, "
            f"got {len(p)} orders for {factor_count} factors"
        )
    if any(type(value) is not int for value in p):
        raise TypeError(
            f"tensor-product basis order must contain integers, got {p!r}"
        )
    if any(value < 0 for value in p):
        raise ValueError(
            f"tensor-product basis order must be non-negative, got {p!r}"
        )
    return p


def _require_simplex_bcs(bcs: Tensor, name: str) -> None:
    if len(bcs.shape) != 2:
        raise ValueError(
            f"{name} expects a tensor with shape (num_points, num_vertices)"
        )
    if int(bcs.shape[-1]) < 2:
        raise ValueError(f"{name} expects at least two barycentric coordinates")
    if bcs.dtype not in (bm.float32, bm.float64):
        raise TypeError(f"{name} expects a float32 or float64 tensor")


def _require_tensor_product_bcs(bcs: object, name: str) -> tuple[Tensor, ...]:
    if type(bcs) is not tuple:
        raise TypeError(f"{name} expects a tuple of barycentric tensors")
    if not bcs:
        raise ValueError(f"{name} expects at least one reference factor")

    first_dtype: str | None = None
    first_device: str | None = None
    for factor, bc in enumerate(bcs):
        if len(bc.shape) != 2:
            raise ValueError(
                f"{name} factor {factor} expects shape (num_points, num_vertices)"
            )
        if int(bc.shape[-1]) < 2:
            raise ValueError(
                f"{name} factor {factor} expects at least two barycentric coordinates"
            )
        if bc.dtype not in (bm.float32, bm.float64):
            raise TypeError(
                f"{name} factor {factor} expects a float32 or float64 tensor"
            )
        dtype = str(bc.dtype)
        device = str(bm.get_device(bc))
        if first_dtype is None:
            first_dtype = dtype
            first_device = device
        elif dtype != first_dtype or device != first_device:
            raise ValueError(
                f"{name} expects all factors on the same device and with the same dtype"
            )
    return bcs


def _cache_key(
    descriptor: SchemaDescriptor,
    operation: str,
    order: tuple[int, ...],
    like: Tensor,
) -> _ReferenceBasisCacheKey:
    if type(descriptor) is not SchemaDescriptor:
        raise TypeError(
            "descriptor must be a SchemaDescriptor, "
            f"got {type(descriptor).__name__}"
        )
    return _ReferenceBasisCacheKey(
        descriptor=descriptor,
        operation=operation,
        order=order,
        backend=bm.backend_name,
        device=str(bm.get_device(like)),
        dtype=str(like.dtype),
    )


def _cached_tensor(
    key: _ReferenceBasisCacheKey,
    factory: Callable[[], Tensor],
) -> Tensor:
    with _REFERENCE_BASIS_CACHE_LOCK:
        tensor = _REFERENCE_BASIS_CACHE.get(key)
        if tensor is None:
            tensor = factory()
            _REFERENCE_BASIS_CACHE[key] = tensor
            if len(_REFERENCE_BASIS_CACHE) > _REFERENCE_BASIS_CACHE_MAXSIZE:
                _REFERENCE_BASIS_CACHE.popitem(last=False)
        else:
            _REFERENCE_BASIS_CACHE.move_to_end(key)
        return tensor


def _simplex_multi_index(
    descriptor: SchemaDescriptor,
    operation: str,
    p: int,
    bcs: Tensor,
    *,
    cache_order: tuple[int, ...] | None = None,
) -> Tensor:
    vertex_count = int(bcs.shape[-1])
    device = bm.get_device(bcs)
    key = _cache_key(
        descriptor,
        f"{operation}:simplex_multi_index:{vertex_count}",
        (p,) if cache_order is None else cache_order,
        bcs,
    )
    return _cached_tensor(
        key,
        lambda: bm.device_put(
            bm.multi_index_matrix(p, vertex_count - 1, dtype=bm.int32),
            device,
        ),
    )


def _normalize_permutation(
    permutation: tuple[int, ...] | None,
    basis_count: int,
) -> tuple[int, ...] | None:
    if permutation is None:
        return None
    if type(permutation) is not tuple:
        raise TypeError("permutation must be a tuple of integers or None")
    if any(type(index) is not int for index in permutation):
        raise TypeError("permutation must contain plain integers")
    if len(permutation) != basis_count or set(permutation) != set(range(basis_count)):
        raise ValueError(
            f"permutation must be a bijection of range({basis_count}), "
            f"got {permutation!r}"
        )
    return permutation


def _permutation_tensor(
    descriptor: SchemaDescriptor,
    operation: str,
    order: tuple[int, ...],
    permutation: tuple[int, ...] | None,
    basis_count: int,
    like: Tensor,
) -> Tensor | None:
    permutation = _normalize_permutation(permutation, basis_count)
    if permutation is None:
        return None

    key = _cache_key(
        descriptor,
        f"{operation}:permutation:{permutation!r}",
        order,
        like,
    )
    return _cached_tensor(
        key,
        lambda: bm.asarray(
            permutation,
            dtype=bm.int64,
            device=bm.get_device(like),
        ),
    )


def _apply_basis_permutation(
    values: Tensor,
    descriptor: SchemaDescriptor,
    operation: str,
    order: tuple[int, ...],
    permutation: tuple[int, ...] | None,
) -> Tensor:
    basis_axis = -2 if len(values.shape) >= 3 else -1
    basis_count = int(values.shape[basis_axis])
    indices = _permutation_tensor(
        descriptor,
        operation,
        order,
        permutation,
        basis_count,
        values,
    )
    if indices is None:
        return values
    if basis_axis == -1:
        return values[..., indices]
    return values[..., indices, :]


def _reference_gradient_transform(
    descriptor: SchemaDescriptor,
    operation: str,
    p: int,
    bcs: Tensor,
    *,
    cache_order: tuple[int, ...] | None = None,
) -> Tensor:
    vertex_count = int(bcs.shape[-1])
    reference_dimension = vertex_count - 1
    key = _cache_key(
        descriptor,
        f"{operation}:reference_gradient_transform:{vertex_count}",
        (p,) if cache_order is None else cache_order,
        bcs,
    )

    def factory() -> Tensor:
        device = bm.get_device(bcs)
        first_row = -bm.ones(
            (1, reference_dimension),
            dtype=bcs.dtype,
            device=device,
        )
        remaining_rows = bm.eye(
            reference_dimension,
            dtype=bcs.dtype,
            device=device,
        )
        return bm.concat((first_row, remaining_rows), axis=0)

    return _cached_tensor(key, factory)


def simplex_lagrange_basis(
    bcs: Tensor,
    p: int,
    *,
    descriptor: SchemaDescriptor,
    permutation: tuple[int, ...] | None = None,
) -> Tensor:
    """Evaluate a scalar Lagrange basis on one simplex reference entity.

    Parameters:
        bcs: Barycentric coordinates with shape ``(Q, V)``, where ``V`` is
            the number of simplex vertices.
        p: Non-negative polynomial order.  Order zero returns one constant
            basis function.
        descriptor: Concrete Schema identity used to isolate cached backend
            tensors.  It does not change the polynomial definition.
        permutation: Optional mapping from requested output positions to the
            backend multi-index order.  It must be a bijection of all basis
            columns.

    Returns:
        Basis values with shape ``(Q, L)``, where ``L`` is the number of
        simplex Lagrange nodes.  The result keeps the input dtype and device.
    """
    _require_simplex_bcs(bcs, "simplex_lagrange_basis")
    p = _normalize_simplex_order(p)
    multi_index = _simplex_multi_index(descriptor, "simplex_basis", p, bcs)
    values = bm.simplex_shape_function(bcs, p, multi_index)
    return _apply_basis_permutation(
        values,
        descriptor,
        "simplex_basis",
        (p,),
        permutation,
    )


def simplex_lagrange_grad_barycentric(
    bcs: Tensor,
    p: int,
    *,
    descriptor: SchemaDescriptor,
    permutation: tuple[int, ...] | None = None,
) -> Tensor:
    """Differentiate a simplex Lagrange basis by barycentric coordinates.

    The result has shape ``(Q, L, V)``.  Its basis axis follows
    ``permutation`` and its last axis follows the input barycentric order.
    The barycentric variables are differentiated as independent variables;
    use :func:`simplex_lagrange_grad_reference` for derivatives restricted
    by ``sum(lambda_i) = 1``.
    """
    _require_simplex_bcs(bcs, "simplex_lagrange_grad_barycentric")
    p = _normalize_simplex_order(p)
    multi_index = _simplex_multi_index(
        descriptor,
        "simplex_grad_barycentric",
        p,
        bcs,
    )
    values = bm.simplex_grad_shape_function(bcs, p, multi_index)
    return _apply_basis_permutation(
        values,
        descriptor,
        "simplex_grad_barycentric",
        (p,),
        permutation,
    )


def simplex_lagrange_grad_reference(
    bcs: Tensor,
    p: int,
    *,
    descriptor: SchemaDescriptor,
    permutation: tuple[int, ...] | None = None,
) -> Tensor:
    """Differentiate a simplex Lagrange basis by reference coordinates.

    Reference coordinates are ``(lambda_1, ..., lambda_{V-1})`` with
    ``lambda_0 = 1 - sum(reference_coordinates)``.  The result has shape
    ``(Q, L, V - 1)``.
    """
    _require_simplex_bcs(bcs, "simplex_lagrange_grad_reference")
    p = _normalize_simplex_order(p)
    grad_barycentric = simplex_lagrange_grad_barycentric(
        bcs,
        p,
        descriptor=descriptor,
    )
    transform = _reference_gradient_transform(
        descriptor,
        "simplex_grad_reference",
        p,
        bcs,
    )
    values = bm.einsum("...ij,jk->...ik", grad_barycentric, transform)
    return _apply_basis_permutation(
        values,
        descriptor,
        "simplex_grad_reference",
        (p,),
        permutation,
    )


def _tensor_product_factor_values(
    bcs: tuple[Tensor, ...],
    p: tuple[int, ...],
    descriptor: SchemaDescriptor,
    operation: str,
) -> tuple[Tensor, ...]:
    return tuple(
        bm.simplex_shape_function(
            bc,
            order,
            _simplex_multi_index(
                descriptor,
                f"{operation}:factor_{factor}",
                order,
                bc,
                cache_order=p,
            ),
        )
        for factor, (bc, order) in enumerate(zip(bcs, p, strict=True))
    )


def _tensor_product_factor_grads(
    bcs: tuple[Tensor, ...],
    p: tuple[int, ...],
    descriptor: SchemaDescriptor,
    operation: str,
    *,
    reference: bool,
) -> tuple[Tensor, ...]:
    result: list[Tensor] = []
    for factor, (bc, order) in enumerate(zip(bcs, p, strict=True)):
        multi_index = _simplex_multi_index(
            descriptor,
            f"{operation}:factor_{factor}",
            order,
            bc,
            cache_order=p,
        )
        grad = bm.simplex_grad_shape_function(bc, order, multi_index)
        if reference:
            transform = _reference_gradient_transform(
                descriptor,
                f"{operation}:factor_{factor}",
                order,
                bc,
                cache_order=p,
            )
            grad = bm.einsum("...ij,jk->...ik", grad, transform)
        result.append(grad)
    return tuple(result)


def _tensor_product_gradients(
    factor_values: tuple[Tensor, ...],
    factor_gradients: tuple[Tensor, ...],
) -> Tensor:
    blocks: list[Tensor] = []
    for differentiated_factor, gradient in enumerate(factor_gradients):
        components: list[Tensor] = []
        for component in range(int(gradient.shape[-1])):
            factors = list(factor_values)
            factors[differentiated_factor] = gradient[..., component]
            components.append(bm.tensorprod(*factors))
        blocks.append(bm.stack(components, axis=-1))
    return bm.concat(blocks, axis=-1)


def tensor_product_lagrange_basis(
    bcs: tuple[Tensor, ...],
    p: tuple[int, ...],
    *,
    descriptor: SchemaDescriptor,
    permutation: tuple[int, ...] | None = None,
) -> Tensor:
    """Evaluate a tensor product of simplex-factor Lagrange bases.

    Each factor coordinate tensor has shape ``(Q_f, V_f)``.  The output has
    shape ``(product(Q_f), product(L_f))``.  Factor order is explicit: the
    first supplied factor is the slowest-varying point and basis index.
    Interval products provide quadrilateral and hexahedron kernels, while a
    triangle-by-interval product provides the prism kernel.
    """
    bcs = _require_tensor_product_bcs(bcs, "tensor_product_lagrange_basis")
    p = _normalize_tensor_product_order(p, len(bcs))
    factor_values = _tensor_product_factor_values(
        bcs,
        p,
        descriptor,
        "tensor_product_basis",
    )
    values = bm.tensorprod(*factor_values)
    return _apply_basis_permutation(
        values,
        descriptor,
        "tensor_product_basis",
        p,
        permutation,
    )


def tensor_product_lagrange_grad_barycentric(
    bcs: tuple[Tensor, ...],
    p: tuple[int, ...],
    *,
    descriptor: SchemaDescriptor,
    permutation: tuple[int, ...] | None = None,
) -> Tensor:
    """Differentiate a tensor-product basis by all factor barycentrics.

    The result has shape ``(product(Q_f), product(L_f), sum(V_f))``.  The
    last axis concatenates independent factor barycentric derivatives in
    input order.  Reference gradients apply each factor's barycentric sum
    constraint.
    """
    bcs = _require_tensor_product_bcs(
        bcs,
        "tensor_product_lagrange_grad_barycentric",
    )
    p = _normalize_tensor_product_order(p, len(bcs))
    factor_values = _tensor_product_factor_values(
        bcs,
        p,
        descriptor,
        "tensor_product_grad_barycentric",
    )
    factor_gradients = _tensor_product_factor_grads(
        bcs,
        p,
        descriptor,
        "tensor_product_grad_barycentric",
        reference=False,
    )
    values = _tensor_product_gradients(factor_values, factor_gradients)
    return _apply_basis_permutation(
        values,
        descriptor,
        "tensor_product_grad_barycentric",
        p,
        permutation,
    )


def tensor_product_lagrange_grad_reference(
    bcs: tuple[Tensor, ...],
    p: tuple[int, ...],
    *,
    descriptor: SchemaDescriptor,
    permutation: tuple[int, ...] | None = None,
) -> Tensor:
    """Differentiate a tensor-product basis by factor reference coordinates.

    The result has shape ``(product(Q_f), product(L_f), sum(V_f - 1))``.
    Each factor uses ``(lambda_1, ..., lambda_{V_f-1})`` as its independent
    reference coordinates.
    """
    bcs = _require_tensor_product_bcs(
        bcs,
        "tensor_product_lagrange_grad_reference",
    )
    p = _normalize_tensor_product_order(p, len(bcs))
    factor_values = _tensor_product_factor_values(
        bcs,
        p,
        descriptor,
        "tensor_product_grad_reference",
    )
    factor_gradients = _tensor_product_factor_grads(
        bcs,
        p,
        descriptor,
        "tensor_product_grad_reference",
        reference=True,
    )
    values = _tensor_product_gradients(factor_values, factor_gradients)
    return _apply_basis_permutation(
        values,
        descriptor,
        "tensor_product_grad_reference",
        p,
        permutation,
    )


def _clear_reference_basis_cache() -> None:
    """Clear cached backend tensors for deterministic tests and diagnostics."""
    with _REFERENCE_BASIS_CACHE_LOCK:
        _REFERENCE_BASIS_CACHE.clear()


def _reference_basis_cache_keys() -> tuple[_ReferenceBasisCacheKey, ...]:
    """Return an immutable cache-key snapshot for tests and diagnostics."""
    with _REFERENCE_BASIS_CACHE_LOCK:
        return tuple(_REFERENCE_BASIS_CACHE)
