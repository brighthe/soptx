# 移植自 brighthe/fealpy ``fealpy/mesh/reference_basis.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参考实体上共享的 Lagrange 基函数内核.

本模块的函数只计算标量 Lagrange 基, 不涉及几何、有限元自由度或物理坐标语义.
需要 Schema 描述符只是为了在外部缓存中隔离可复用的后端张量; 参考实体的约定与局部
基函数的顺序由调用方负责选定.
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
        """构造重心坐标导数到参考坐标导数的变换矩阵 ``(V, V-1)``: 首行全为 -1, 其下为单位阵."""
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
    """计算单纯形参考实体上的标量 Lagrange 基函数.

    Parameters
    ----------
    bcs : Tensor
        重心坐标, 形状 ``(Q, V)``, ``V`` 为单纯形顶点数.
    p : int
        非负的多项式次数; 0 次返回一个常数基函数.
    descriptor : SchemaDescriptor
        具体 Schema 的身份, 用于隔离缓存的后端张量, 不影响多项式的定义.
    permutation : optional
        输出位置到后端多重指标顺序的映射, 须为全部基函数列上的双射.

    Returns
    -------
    Tensor
        基函数值, 形状 ``(Q, L)``, ``L`` 为单纯形 Lagrange 节点数; 保持输入的
        dtype 与 device.
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
    """单纯形 Lagrange 基函数对重心坐标的导数, 形状 ``(Q, L, V)``.

    基函数轴按 ``permutation`` 排列, 末轴按输入重心坐标的顺序. 各重心坐标视为独立
    变量求导; 受 ``sum(lambda_i) = 1`` 约束的导数用
    :func:`simplex_lagrange_grad_reference`.
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
    """单纯形 Lagrange 基函数对参考坐标的导数, 形状 ``(Q, L, V - 1)``.

    参考坐标为 ``(lambda_1, ..., lambda_{V-1})``, ``lambda_0 = 1 - sum(参考坐标)``.
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
    """计算各单纯形因子上 Lagrange 基函数的张量积.

    每个因子的坐标张量形状为 ``(Q_f, V_f)``, 输出形状为
    ``(product(Q_f), product(L_f))``. 因子顺序是显式的: 第一个因子对应变化最慢的
    积分点与基函数下标. 区间之积给出四边形与六面体的内核, 三角形乘区间给出三棱柱的
    内核.
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
    """张量积基函数对全部因子重心坐标的导数, 形状 ``(product(Q_f), product(L_f), sum(V_f))``.

    末轴按输入顺序拼接各因子独立的重心坐标导数; 参考坐标梯度则施加各因子重心坐标
    之和为 1 的约束.
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
    """张量积基函数对各因子参考坐标的导数, 形状 ``(product(Q_f), product(L_f), sum(V_f - 1))``.

    每个因子以 ``(lambda_1, ..., lambda_{V_f-1})`` 为独立的参考坐标.
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
    """清空缓存的后端张量, 供确定性测试与诊断使用."""
    with _REFERENCE_BASIS_CACHE_LOCK:
        _REFERENCE_BASIS_CACHE.clear()


def _reference_basis_cache_keys() -> tuple[_ReferenceBasisCacheKey, ...]:
    """返回缓存键的不可变快照, 供测试与诊断使用."""
    with _REFERENCE_BASIS_CACHE_LOCK:
        return tuple(_REFERENCE_BASIS_CACHE)
