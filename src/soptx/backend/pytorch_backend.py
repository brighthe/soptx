# 移植自 brighthe/fealpy ``fealpy/backend/pytorch_backend.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""pytorch 计算后端.

把 torch 的 ``dim``、``keepdim``、``size`` 等参数名统一为 array API 的 ``axis``、
``keepdims``、``shape``; 未手动实现的函数按 ``base`` 的名称映射从 torch 复制.
"""

from typing import Union, Optional, Tuple, Any
from itertools import combinations_with_replacement
from functools import reduce, partial
from math import factorial, prod
from scipy.spatial import KDTree

try:
    import torch
    from torch import vmap, norm, det, cross
    from torch.func import jacfwd, jacrev

except ImportError:
    raise ImportError("Name 'torch' cannot be imported. "
                      'Make sure PyTorch is installed before using '
                      'the PyTorch backend in SOPTX. '
                      'See https://pytorch.org/ for installation.')

from .base import (
    BackendProxy,
    ATTRIBUTE_MAPPING, FUNCTION_MAPPING, TRANSFORMS_MAPPING
)

Tensor = torch.Tensor
_device = torch.device

def _dim_to_axis(func):
    def wrapper(*args, axis=None, **kwargs):
        """把 ``axis`` 参数改名为 torch 的 ``dim`` 后调用."""
        if axis is None:
            return func(*args, **kwargs)
        return func(*args, dim=axis, **kwargs)
    return wrapper

def _dims_to_axes(func):
    def wrapper(*args, axes=None, **kwargs):
        """把 ``axes`` 参数改名为 torch 的 ``dims`` 后调用."""
        if axes is None:
            return func(*args, **kwargs)
        return func(*args, dims=axes, **kwargs)
    return wrapper

def _size_to_shape(func):
    def wrapper(*args, shape=None, **kwargs):
        """把 ``shape`` 参数改名为 torch 的 ``size`` 后调用."""
        if shape is None:
            return func(*args, **kwargs)
        return func(*args, size=shape, **kwargs)
    return wrapper

def _axis_keepdims_dispatch(func, **defaults):
    if len(defaults) > 0:
        def wrapper(*args, **kwargs):
            """把 ``axis``、``keepdims`` 改名为 torch 的 ``dim``、``keepdim``, 并补上默认参数后调用."""
            if 'axis' in kwargs:
                kwargs['dim'] = kwargs.pop('axis')
            if 'keepdims' in kwargs:
                kwargs['keepdim'] = kwargs.pop('keepdims')
            defaults.update(kwargs)
            kwargs = defaults
            return func(*args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            """把 ``axis``、``keepdims`` 改名为 torch 的 ``dim``、``keepdim``, 并补上默认参数后调用."""
            if 'axis' in kwargs:
                kwargs['dim'] = kwargs.pop('axis')
            if 'keepdims' in kwargs:
                kwargs['keepdim'] = kwargs.pop('keepdims')
            return func(*args, **kwargs)
    return wrapper


class PyTorchBackend(BackendProxy, backend_name='pytorch'):
    """pytorch 后端代理, 张量类型为 ``torch.Tensor``."""
    DATA_CLASS = torch.Tensor
    linalg = torch.linalg
    random = torch.random

    @staticmethod
    def context(tensor: Tensor, /):
        """张量的构造参数 ``{'dtype': ..., 'device': ...}``."""
        return {"dtype": tensor.dtype, "device": tensor.device}

    @staticmethod
    def set_default_device(device: Union[str, _device]) -> None:
        """设置 torch 的默认设备."""
        torch.set_default_device(device)

    @staticmethod
    def device_type(tensor_like: Tensor, /):
        """张量所在设备的类型, 如 ``'cpu'``、``'cuda'``."""
        return tensor_like.device.type

    @staticmethod
    def device_index(tensor_like: Tensor, /):
        """张量所在设备的编号, CPU 张量为 None."""
        return tensor_like.device.index

    @staticmethod
    def get_device(tensor_like: Tensor, /):
        """张量所在的设备."""
        return tensor_like.device

    @staticmethod
    def device_put(tensor_like: Tensor, /, device=None) -> Tensor:
        """把张量移到指定设备."""
        return tensor_like.to(device=device)

    @staticmethod
    def to_numpy(tensor_like: Tensor, /) -> Any:
        """分离计算图并复制到 CPU 后转为 numpy 数组."""
        return tensor_like.detach().cpu().numpy()

    from_numpy = staticmethod(torch.from_numpy)

    @staticmethod
    def tolist(tensor: Tensor, /):
        """转为 Python 嵌套列表."""
        return tensor.tolist()

    ### 创建函数 ###
    # Python array API 标准 v2023.12
    @staticmethod
    def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
        """参数顺序与 numpy 一致的 ``torch.arange``: 只给一个位置参数时视为终点."""
        if stop is None:
            stop = start
            start = 0
        return torch.arange(start, stop, step, dtype=dtype, device=device)

    @staticmethod
    def eye(n: int, m: Optional[int]=None, /, k: int=0, dtype=None, **kwargs) -> Tensor:
        """``torch.eye``, 只支持 ``k=0``."""
        assert k == 0, "Only k=0 is supported by `eye` in PyTorchBackend."
        if m is None:
            m = n
        return torch.eye(n, m, dtype=dtype, **kwargs)

    @staticmethod
    def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
        """``torch.linspace``; ``start``、``stop`` 为张量时逐元素生成 (经 ``vmap``). 只支持 ``endpoint=True``."""
        assert endpoint == True
        if isinstance(start, (int, float)) and isinstance(stop, (int, float)):
            return torch.linspace(start, stop, num, dtype=dtype, device=device)
        else:
            vmap_fun = partial(torch.linspace, dtype=dtype, device=device)
            for _ in range(start.ndim):
                vmap_fun = vmap(vmap_fun, in_dims=(0, 0, None), out_dims=0)
            return vmap_fun(start, stop, num)

    @staticmethod
    def tril(x, /, *, k=0):
        """取第 ``k`` 条对角线及以下的部分."""
        return torch.tril(x, k)

    @staticmethod
    def triu(x, /, *, k=0):
        """取第 ``k`` 条对角线及以上的部分."""
        return torch.triu(x, k)

    @staticmethod
    def take_along_axis(x, indices, /, *, axis):
        """``torch.take_along_dim``, 索引先转为 int64."""
        return torch.take_along_dim(x, indices.long(), dim=axis)

    @staticmethod
    def unique_counts(x, /):
        """返回 ``(唯一值, 计数)``."""
        values, counts = torch.unique(x, return_counts=True)
        return values, counts

    ### 数据类型函数 ###
    # Python array API 标准 v2023.12
    @staticmethod
    def astype(x, dtype, /, *, copy=True, device=None):
        """``Tensor.to``: 转换数据类型, 可同时换设备."""
        return x.to(dtype=dtype, device=device, copy=copy)

    ### 逐元素函数 ###
    @staticmethod # NOTE: PyTorch 内置的 equal 实为 `all(equal(x1, x2))`, 这里改为逐元素比较
    def equal(x1, x2, /):
        """逐元素相等比较 ``x1 == x2``."""
        return x1 == x2

    ### 索引函数 ###

    ### 线性代数函数 ###
    # Python array API 标准 v2023.12
    @staticmethod
    def matrix_transpose(x, /):
        """交换最后两轴."""
        return x.transpose(-1, -2)

    tensordot = staticmethod(_dims_to_axes(torch.tensordot))
    vecdot = staticmethod(_dim_to_axis(torch.linalg.vecdot))

    # 非标准
    cross = staticmethod(_dim_to_axis(torch.cross))

    @staticmethod
    def dot(x1, x2, /, *, axis=-1):
        """沿 ``axis`` 收缩两个张量 (``tensordot``)."""
        return torch.tensordot(x1, x2, dims=[[axis], [axis]])

    @staticmethod
    def trace(x, /, *, offset: int = 0, axis1=0, axis2=1):
        """取偏移 ``offset`` 的对角线 (``axis1``、``axis2`` 两轴) 并求和."""
        data = torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)
        return torch.sum(data, dim=-1)

    ### 变形函数 ###
    # Python array API 标准 v2023.12
    broadcast_to = staticmethod(_size_to_shape(torch.broadcast_to))
    concat = staticmethod(_dim_to_axis(torch.concat))
    expand_dims = staticmethod(_dim_to_axis(torch.unsqueeze))
    @staticmethod
    def flip(a, axis=None):
        """沿给定轴翻转; ``axis`` 为 None 时翻转所有轴."""
        if axis is None:
            axis = list(range(a.dim()))
        elif isinstance(axis, int):
            axis = [axis]
        elif isinstance(axis, tuple):
            axis = list(axis)
        return torch.flip(a, dims=axis)

    permute_dims = staticmethod(_dims_to_axes(torch.permute))
    repeat = staticmethod(_dim_to_axis(torch.repeat_interleave))
    @staticmethod
    def roll(x, /, shift, *, axis=None):
        """沿 ``axis`` 循环平移 (``torch.roll``)."""
        return torch.roll(x, shifts=shift, dims=axis)

    squeeze = staticmethod(_dim_to_axis(torch.squeeze))
    stack = staticmethod(_dim_to_axis(torch.stack))
    unstack = staticmethod(_dim_to_axis(torch.unbind))

    # 非标准
    @staticmethod
    def concatenate(arrays, /, axis=0, out=None, *, dtype=None):
        """``torch.cat``, 可指定结果的数据类型."""
        if dtype is not None:
            arrays = [a.to(dtype) for a in arrays]
        return torch.cat(arrays, dim=axis, out=out)

    @staticmethod
    def insert(x, obj, values, /, *, axis=None):
        """仿 ``np.insert``: 在 ``axis`` 轴的 ``obj`` 位置之前插入 ``values``; ``axis`` 为 None 时先展平."""
        kwargs = {'dtype': x.dtype, 'device': x.device}
        ndim = x.ndim
        if axis is None:
            if ndim != 1:
                x = x.ravel()
            ndim = x.ndim
            axis = ndim - 1
        else: # 规范化 axis
            if axis < -ndim or axis > ndim:
                raise IndexError(f"index {axis} is out of bounds for axis {axis} "
                                f"with size {ndim}")
            axis = axis if axis >=0 else axis + ndim

        slobj = [slice(None), ] * ndim
        N = x.shape[axis]
        newshape = list(x.shape)

        if isinstance(obj, slice):
            # 转为 range 对象
            indices = torch.arange(*obj.indices(N), dtype=torch.int64, device=x.device)
        else:
            # 须复制 obj, 因为 indices 会被原地修改
            indices = torch.tensor(obj, dtype=torch.int64, device=x.device)
            if indices.dtype == bool:
                indices = torch.as_tensor(indices, dtype=torch.int64, device=x.device)
            elif indices.ndim > 1:
                raise ValueError(
                    "index array argument obj to insert must be one dimensional "
                    "or scalar")
        if indices.numel() == 1:
            index = indices.item()
            if index < -N or index > N:
                raise IndexError(f"index {obj} is out of bounds for axis {axis} "
                                f"with size {N}")
            if (index < 0):
                index += N

            values = torch.tensor(values, **kwargs)
            if values.ndim < ndim:
                values = values.reshape((1,)*(ndim - values.ndim) + (-1, ))
            if indices.ndim == 0:
                # 这里的广播行为差别很大: a[:,0,:] = ... 与 a[:,[0],:] = ... 截然不同!
                # 下面调整 values, 使其按后一种方式工作 (即 a[:,0:1,:]).
                values = torch.moveaxis(values, 0, axis)
            numnew = values.shape[axis]
            newshape[axis] += numnew
            new = torch.empty(newshape, **kwargs)
            slobj[axis] = slice(None, index)
            new[tuple(slobj)] = x[tuple(slobj)]
            slobj[axis] = slice(index, index+numnew)
            new[tuple(slobj)] = values
            slobj[axis] = slice(index+numnew, None)
            slobj2 = [slice(None)] * ndim
            slobj2[axis] = slice(index, None)
            new[tuple(slobj)] = x[tuple(slobj2)]

            return new

        elif indices.numel() == 0 and not isinstance(obj, Tensor):
            # 空列表可以安全地转为 int64
            indices = torch.as_tensor(indices, torch.int64)

        indices[indices < 0] += N

        numnew = len(indices)
        order = indices.argsort(stable=True)   # 稳定排序
        indices[order] += torch.arange(numnew, dtype=torch.int64, device=x.device)

        newshape[axis] += numnew
        old_mask = torch.ones(newshape[axis], dtype=torch.bool, device=x.device)
        old_mask[indices] = False

        new = torch.empty(newshape, **kwargs)
        slobj2 = [slice(None)]*ndim
        slobj[axis] = indices
        slobj2[axis] = old_mask
        new[tuple(slobj)] = values
        new[tuple(slobj2)] = x

        return new

    @staticmethod
    def split(x, indices_or_sections, /, *, axis=0):
        """仿 ``np.split``: 整数表示等分的段数, 一维序列表示分割点.

        Raises
        ------
        ValueError
            整数段数不能整除, 或分割点不是一维.
        """
        # 语义对齐 np.split: int 表示等分段数, 1 维序列 (list/tuple/ndarray/Tensor)
        # 表示分割点; torch.split 吃的是各段长度, 这里做换算.
        if isinstance(indices_or_sections, int):
            if x.shape[axis] % indices_or_sections != 0:
                raise ValueError(
                    "array split does not result in an equal division"
                )
            chunk_size = x.shape[axis] // indices_or_sections
        else:
            indices = torch.as_tensor(indices_or_sections)
            if indices.ndim != 1:
                raise ValueError(
                    "indices_or_sections must be an int or a 1-D sequence of indices"
                )
            kwargs = {'dtype': indices.dtype, 'device': indices.device}
            HEAD = torch.tensor([0], **kwargs)
            TAIL = torch.tensor([x.shape[axis]], **kwargs)
            indices = torch.cat([HEAD, indices, TAIL])
            chunk_size = (indices[1:] - indices[:-1]).tolist()

        return torch.split(x, chunk_size, dim=axis)

    ### 查找函数 ###
    # Python array API 标准 v2023.12
    argmax = staticmethod(_axis_keepdims_dispatch(torch.argmax))
    argmin = staticmethod(_axis_keepdims_dispatch(torch.argmin))

    @staticmethod
    def nonzero(x, /):
        """返回各维非零索引组成的元组, 同 numpy."""
        return torch.nonzero(x, as_tuple=True)

    ### 集合函数 ###
    # Python array API 标准 v2023.12
    @staticmethod
    def unique_all(a, axis=None, **kwargs):
        """返回 ``(唯一值, 首次出现位置, 逆映射, 计数)``; ``axis`` 为 None 时先展平."""
        if axis is None:
            a = torch.flatten(a)
            axis = 0
        b, inverse, counts = torch.unique(a, return_inverse=True,
                return_counts=True,
                dim=axis, **kwargs)
        kwargs = {'dtype': inverse.dtype, 'device': inverse.device}
        indices = torch.zeros(counts.shape, **kwargs)
        idx = torch.arange(a.shape[axis]-1, -1, -1, **kwargs)
        indices.scatter_(0, inverse.flip(dims=[0]), idx)
        return b, indices, inverse, counts

    # 非标准
    @staticmethod
    def unique_all_(a, axis=None, **kwargs):
        """返回 ``(唯一值, 首次出现位置, 最后出现位置, 逆映射, 计数)``; ``axis`` 为 None 时先展平."""
        if axis is None:
            a = torch.flatten(a)
            axis = 0
        b, inverse, counts = torch.unique(a, return_inverse=True,
                return_counts=True,
                dim=axis, **kwargs)
        kwargs = {'dtype': inverse.dtype, 'device': inverse.device}

        indices0 = torch.zeros(counts.shape, **kwargs)
        indices1 = torch.zeros(counts.shape, **kwargs)

        idx = torch.arange(a.shape[axis]-1, -1, -1, **kwargs)
        indices0.scatter_(0, inverse.flip(dims=[0]), idx)

        idx = torch.arange(a.shape[0], **kwargs)
        indices1.scatter_(0, inverse, idx)
        return b, indices0, indices1, inverse, counts

    @staticmethod
    def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None, **kwargs):
        """仿 ``np.unique``: 可选返回首次出现位置、逆映射与计数; ``axis`` 为 None 时先展平."""
        if axis is None:
            b, inverse, counts = torch.unique(a.flatten(), return_inverse=True,
                                              return_counts=True,
                                              dim=0, **kwargs)
        else:
            b, inverse, counts = torch.unique(a, return_inverse=True,
                                              return_counts=True,
                                              dim=axis, **kwargs)

        any_return = return_index or return_inverse or return_counts
        if any_return:
            result = (b,)
        else:
            return b 

        if return_index:
            if axis is None:
                original_tensor = a.flatten()
                size = original_tensor.shape[0]
            else:
                original_tensor = a
                size = original_tensor.shape[axis]
            
            kwargs_device_dtype = {'dtype': inverse.dtype, 'device': inverse.device}
            indices = torch.zeros(counts.shape, **kwargs_device_dtype)
            idx_range = torch.arange(size - 1, -1, -1, **kwargs_device_dtype)
            flipped_inverse = inverse.flip(dims=[0])
            indices.scatter_(0, flipped_inverse, idx_range)
            result += (indices,)

        if return_inverse:
            if axis is None:
                inverse = inverse.reshape(a.shape)
            result += (inverse,)

        if return_counts:
            result += (counts,)

        return result

    ### 排序函数 ###
    # Python array API 标准 v2023.12
    argsort = staticmethod(_dim_to_axis(torch.argsort))
    @staticmethod
    def sort(x, /, *, axis=-1, descending=False, stable=True):
        """沿 ``axis`` 排序并只返回值, 默认稳定排序."""
        return torch.sort(x, dim=axis, descending=descending, stable=stable)[0]

    # 非标准
    @staticmethod
    def lexsort(keys: Tuple[Tensor, ...], /, *, axis: int = -1):
        """仿 ``np.lexsort``: 稳定的字典序排序, 最后一个键为主键, 返回排序下标.

        Raises
        ------
        ValueError
            键为零维, 或没有给出键.
        """
        if keys[0].ndim < 1:
            raise ValueError("keys must be at least 2 dimensional, but got "
                             f"shape {keys.shape}.")
        if len(keys) == 0:
            raise ValueError(f"Must have at least 1 key.")

        idx = keys[0].argsort(dim=axis, stable=True)

        for k in keys[1:]:
            idx = idx.gather(axis, k.gather(axis, idx).argsort(dim=axis, stable=True))

        return idx

    ### 统计函数 ###
    # Python array API 标准 v2023.12
    @staticmethod
    def max(x, /, *, axis=None, keepdims=False):
        """沿 ``axis`` 取最大值 (只返回值); ``axis`` 为 None 时对全部元素."""
        if axis is None:
            return torch.max(x)
        return torch.max(x, axis, keepdim=keepdims)[0]

    mean = staticmethod(_axis_keepdims_dispatch(torch.mean))

    @staticmethod
    def min(x, /, *, axis=None, keepdims=False):
        """沿 ``axis`` 取最小值 (只返回值); ``axis`` 为 None 时对全部元素."""
        if axis is None:
            return torch.min(x)
        return torch.min(x, axis, keepdim=keepdims)[0]

    prod = staticmethod(_axis_keepdims_dispatch(torch.prod))
    std = staticmethod(_axis_keepdims_dispatch(torch.std, correction=0.))
    sum = staticmethod(_axis_keepdims_dispatch(torch.sum))
    var = staticmethod(_axis_keepdims_dispatch(torch.var, correction=0.))

    # 非标准
    # 语义对齐 np.cumsum/np.cumprod: axis=None 表示展平后累加;
    # torch.cumsum/torch.cumprod 的 dim 为必填参数, 不能直接省略.
    @staticmethod
    def cumsum(x, axis=None, dtype=None, out=None):
        """累加, ``axis`` 为 None 时先展平, 同 numpy."""
        if axis is None:
            x, axis = x.reshape(-1), 0
        return torch.cumsum(x, dim=axis, dtype=dtype, out=out)

    @staticmethod
    def cumprod(x, axis=None, dtype=None, out=None):
        """累乘, ``axis`` 为 None 时先展平, 同 numpy."""
        if axis is None:
            x, axis = x.reshape(-1), 0
        return torch.cumprod(x, dim=axis, dtype=dtype, out=out)

    cumulative_sum = staticmethod(_dim_to_axis(torch.cumsum))

    ### 工具函数 ###
    # Python array API 标准 v2023.12
    all = staticmethod(_axis_keepdims_dispatch(torch.all))
    any = staticmethod(_axis_keepdims_dispatch(torch.any))

    # 非标准
    @staticmethod
    def size(x, /, *, axis=None):
        """元素个数; 给出 ``axis`` 时为该轴长度."""
        if axis is None:
            return x.numel()
        else:
            return x.size(axis)

    ### 其他函数 ###

    @staticmethod
    def set_at(a: Tensor, indices, src, /):
        """``a[indices] = src``, 原地修改并返回 ``a``."""
        a[indices] = src
        return a

    @staticmethod
    def add_at(a: Tensor, indices, src, /):
        """在 ``indices`` 处把 ``src`` 累加到 ``a`` 上, 重复索引全部计入.

        跨后端约定以 numpy 的 ``np.add.at`` 为准 (jax 的 ``.at[].add()`` 与之一致):
        重复索引的每次出现都要计入. ``a[indices] += src`` 不满足这一点, 高级索引赋值
        对重复索引只保留任意一次贡献, 其余被静默丢弃. ``index_put_(accumulate=True)``
        是与 ``np.add.at`` 对应的 torch 原语; 与之相同, 本方法原地修改并返回 ``a``.

        Raises
        ------
        NotImplementedError
            ``indices`` 含切片分量; 应传入显式的索引张量, 或沿单轴累加时改用
            ``index_add``.
        """
        if not isinstance(indices, tuple):
            indices = (indices,)
        if any(isinstance(idx, slice) for idx in indices):
            raise NotImplementedError(
                "slice components are not supported by add_at on the PyTorch "
                "backend; pass explicit index tensors, or use index_add to "
                "accumulate along a single axis."
            )
        indices = tuple(
            idx if isinstance(idx, Tensor) else torch.as_tensor(idx, device=a.device)
            for idx in indices
        )
        if isinstance(src, Tensor):
            src = src.to(dtype=a.dtype, device=a.device)
        else:
            src = torch.as_tensor(src, dtype=a.dtype, device=a.device)
        return a.index_put_(indices, src, accumulate=True)

    @staticmethod
    def index_add(a: Tensor, index, src, /, *, axis: int=0, alpha=1):
        """``Tensor.index_add_``: 沿 ``axis`` 在 ``index`` 处累加 ``alpha * src``, 原地修改并返回 ``a``.

        ``src`` 可以是数, 也可以是能广播到 ``index`` 展开形状的张量.
        """
        axis = a.ndim + axis if (axis < 0) else axis
        src_flat_shape = a.shape[:axis] + (index.numel(), ) + a.shape[axis+1:]

        if isinstance(src, (int, float, complex)):
            src = torch.full([1,]*len(src_flat_shape), src, dtype=a.dtype, device=a.device)
            src = torch.broadcast_to(src, src_flat_shape)
        else:
            src_shape = a.shape[:axis] + index.shape + a.shape[axis+1:]
            src = torch.broadcast_to(src, src_shape).reshape(src_flat_shape)
            if src.dtype != a.dtype:
                src = src.to(dtype=a.dtype)

        return a.index_add_(axis, index.ravel(), src, alpha=alpha)

    @staticmethod
    def scatter(x: Tensor, index, src, /, *, axis: int=0):
        """``Tensor.scatter_`` 的包装, 原地修改并返回 ``x``."""
        x.scatter_(dim=axis, index=index, src=src)
        return x

    @staticmethod
    def scatter_add(x: Tensor, index, src, /, *, axis: int=0):
        """``Tensor.scatter_add_`` 的包装, 原地修改并返回 ``x``."""
        x.scatter_add_(dim=axis, index=index, src=src)
        return x

    ### 函数式编程 ###

    @staticmethod
    def vmap(func, /, in_axes=0, out_axes=0, **kwargs):
        """``torch.vmap``, 参数名对齐 jax 的 ``in_axes``、``out_axes``."""
        return torch.vmap(func, in_dims=in_axes, out_dims=out_axes, **kwargs)

    ### 稀疏函数 ###

    # 传给这些构造函数的索引与行指针来自 SOPTX 自己的 COOTensor/CSRTensor, 它们
    # 维护着布局不变量. torch 要求显式开启或关闭不变量检查, 否则给出警告; 这里关闭
    # 检查, 免得 spmm 路径每次调用都做校验, 该路径在 Krylov 迭代中每步都要走.
    _SPARSE_KWARGS = {'check_invariants': False}

    @staticmethod
    def coo_spmm(indices, values, shape, other):
        """COO 矩阵乘以一维或二维稠密张量 (``torch.sparse.mm``).

        Raises
        ------
        NotImplementedError
            ``values`` 带批量维.
        """
        if values.ndim == 1:
            mat = torch.sparse_coo_tensor(indices, values, size=shape,
                                          **PyTorchBackend._SPARSE_KWARGS)
            return PyTorchBackend._spmm(mat, other)
        else:
            raise NotImplementedError("Batch sparse matrix multiplication has "
                                      "not been supported yet.")

    @staticmethod
    def csr_spmm(crow, col, values, shape, other):
        """CSR 矩阵乘以一维或二维稠密张量 (``torch.sparse.mm``).

        Raises
        ------
        NotImplementedError
            ``values`` 带批量维.
        """
        if values.ndim == 1:
            mat = torch.sparse_csr_tensor(crow, col, values, size=shape,
                                          **PyTorchBackend._SPARSE_KWARGS)
            return PyTorchBackend._spmm(mat, other)
        else:
            raise NotImplementedError("Batch sparse matrix multiplication has "
                                      "not been supported yet.")

    @staticmethod
    def csr_spspmm(crow1, col1, values1, shape1, crow2, col2, values2, shape2):
        """两个 CSR 矩阵相乘, 返回 ``(crow, col, values, shape)``."""
        mat1 = torch.sparse_csr_tensor(crow1, col1, values1, size=shape1,
                                       **PyTorchBackend._SPARSE_KWARGS)
        mat2 = torch.sparse_csr_tensor(crow2, col2, values2, size=shape2,
                                       **PyTorchBackend._SPARSE_KWARGS)
        mat3: Tensor = PyTorchBackend._spmm(mat1, mat2)
        return mat3.crow_indices(), mat3.col_indices(), mat3.values(), mat3.shape

    @staticmethod
    def _spmm(mat, other):
        if other.ndim == 1:
            return torch.sparse.mm(mat, other[:, None])[:, 0]
        else:
            return torch.sparse.mm(mat, other)

    @staticmethod
    def coo_tocsr(indices, values, shape):
        """COO 转 CSR (``to_sparse_csr``), 返回 ``(crow, col, values)``."""
        mat = torch.sparse_coo_tensor(indices, values, size=shape,
                                      **PyTorchBackend._SPARSE_KWARGS)
        mat = mat.to_sparse_csr()
        return mat.crow_indices(), mat.col_indices(), mat.values()

    @staticmethod
    def query_point(x, y, h, box_size, mask_self=True, periodic=[True, True, True]):
        """查找 ``y`` 中各点在 ``x`` 中半径 ``h`` 以内的近邻 (基于 scipy 的 KDTree).

        Parameters
        ----------
        x, y : TensorLike
            被查询点集与查询点集, 形状 ``(N, GD)``.
        h : float
            查询半径.
        box_size : sequence of float
            周期区域的尺寸, 只在周期边界下使用.
        mask_self : bool, optional
            为 False 时去掉点与自身配对的结果. 默认 True.
        periodic : list of bool, optional
            三个方向是否周期. 全为 True 时按二维周期区域把边界附近的点平移复制后
            查询; 全为 False 时不考虑周期; 混合取值未实现. 默认 ``[True, True, True]``.

        Returns
        -------
        tuple
            ``(node_self, neighbors)``: 每对近邻中查询点与被查询点的编号.

        Raises
        ------
        TypeError
            ``periodic`` 不是三个布尔值的列表.
        NotImplementedError
            ``periodic`` 混合取值.
        """
        if not isinstance(periodic, list) or len(periodic) != 3 or not all(isinstance(p, bool) for p in periodic):
            raise TypeError("periodic type is：[bool, bool, bool]")
        def map_points(a, b, r, positions):
            """把靠近二维周期区域边界的点平移复制到对侧的像位置.

            返回 ``(扩充后的点, 各点的原编号, 是否为原始点)``.
            """
            x, y = positions[:, 0], positions[:, 1]
            cond_1 = (0 <= x) & (x <= r) & (r < y) & (y < b - r)  # 区域[(0, r), (r, b-r)]
            cond_2 = (a - r <= x) & (x <= a) & (r < y) & (y < b - r)  # 区域[(a-r, a), (r, b-r)]
            cond_3 = (r < x) & (x < a - r) & (0 <= y) & (y <= r)  # 区域[(r, a-r), (0, r)]
            cond_4 = (r < x) & (x < a - r) & (b - r <= y) & (y <= b)  # 区域[(r, a-r), (b-r, b)]
            cond_5 = (0 <= x) & (x <= r) & (0 <= y) & (y <= r)  # 角区域[(0, r), (0, r)]
            cond_6 = (a - r <= x) & (x <= a) & (0 <= y) & (y <= r)  # 角区域[(a-r, a), (0, r)]
            cond_7 = (0 <= x) & (x <= r) & (b - r <= y) & (y <= b)  # 角区域[(0, r), (b-r, b)]
            cond_8 = (a - r <= x) & (x <= a) & (b - r <= y) & (y <= b)  # 角区域[(a-r, a), (b-r, b)]
            cond_9 = ~(cond_1 | cond_2 | cond_3 | cond_4 | cond_5 | cond_6 | cond_7 | cond_8) # 其他区域
            idx_positions = torch.arange(len(positions))
            bool_positions = torch.ones(len(positions), dtype=bool)

            cond_5_1 = torch.concatenate([(x[cond_5] + a).reshape(-1,1), (y[cond_5]).reshape(-1,1)], axis=-1) # 右侧映射
            cond_5_2 = torch.concatenate([(x[cond_5]).reshape(-1,1), (y[cond_5] + b).reshape(-1,1)], axis=-1) # 上方映射 
            cond_5_3 = torch.concatenate([(x[cond_5] + a).reshape(-1,1), (y[cond_5] + b).reshape(-1,1)], axis=-1) # 右上方映射
            _cond_5 = torch.concatenate([cond_5_1, cond_5_2, cond_5_3], axis=0)
            idx_cond_5_1 = torch.nonzero(cond_5).flatten()
            idx_5 = torch.concatenate([idx_cond_5_1, idx_cond_5_1, idx_cond_5_1], axis=0)
            bool_cond_5_1 = torch.zeros(len(cond_5_1), dtype=bool)
            bool_5 = torch.concatenate([bool_cond_5_1, bool_cond_5_1, bool_cond_5_1], axis=0)

            _cond_3 = torch.concatenate([(x[cond_3]).reshape(-1,1), (y[cond_3] + b).reshape(-1,1)], axis=-1) # 上侧映射 
            idx_3 = torch.nonzero(cond_3).flatten()
            bool_3 = torch.zeros(len(_cond_3), dtype=bool)

            cond_6_1 = torch.concatenate([(x[cond_6] - a).reshape(-1,1), (y[cond_6]).reshape(-1,1)], axis=-1) # 左侧映射
            cond_6_2 = torch.concatenate([(x[cond_6]).reshape(-1,1), (y[cond_6] + b).reshape(-1,1)], axis=-1) # 上方映射
            cond_6_3 = torch.concatenate([(x[cond_6] - a).reshape(-1,1), (y[cond_6] + b).reshape(-1,1)], axis=-1) # 左上方映射
            _cond_6 = torch.concatenate([cond_6_1, cond_6_2, cond_6_3], axis=0)
            idx_cond_6_1 = torch.nonzero(cond_6).flatten()
            idx_6 = torch.concatenate([idx_cond_6_1, idx_cond_6_1, idx_cond_6_1], axis=0)
            bool_cond_6_1 = torch.zeros(len(cond_6_1), dtype=bool)
            bool_6 = torch.concatenate([bool_cond_6_1, bool_cond_6_1, bool_cond_6_1], axis=0)

            _cond_1 = torch.concatenate([(x[cond_1] + a).reshape(-1,1), y[cond_1].reshape(-1,1)], axis=-1) # 右侧映射
            idx_1 = torch.nonzero(cond_1).flatten()
            bool_1 = torch.zeros(len(_cond_1), dtype=bool)

            _cond_2 = torch.concatenate([(x[cond_2] - a).reshape(-1,1), y[cond_2].reshape(-1,1)], axis=-1) # 左侧映射
            idx_2 = torch.nonzero(cond_2).flatten()
            bool_2 = torch.zeros(len(_cond_2), dtype=bool)

            cond_7_1 = torch.concatenate([(x[cond_7] + a).reshape(-1,1), (y[cond_7]).reshape(-1,1)], axis=-1) # 右侧映射
            cond_7_2 = torch.concatenate([(x[cond_7]).reshape(-1,1), (y[cond_7] - b).reshape(-1,1)], axis=-1) # 下方映射
            cond_7_3 = torch.concatenate([(x[cond_7] + a).reshape(-1,1), (y[cond_7] - b).reshape(-1,1)], axis=-1) # 右下方映射
            _cond_7 = torch.concatenate([cond_7_1, cond_7_2, cond_7_3], axis=0)
            idx_cond_7_1 = torch.nonzero(cond_7).flatten()
            idx_7 = torch.concatenate([idx_cond_7_1, idx_cond_7_1, idx_cond_7_1], axis=0)
            bool_cond_7_1 = torch.zeros(len(cond_7_1), dtype=bool)
            bool_7 = torch.concatenate([bool_cond_7_1, bool_cond_7_1, bool_cond_7_1], axis=0)

            _cond_4 = torch.concatenate([(x[cond_4]).reshape(-1,1), (y[cond_4] - b).reshape(-1,1)], axis=-1) #下侧映射
            idx_4 = torch.nonzero(cond_4).flatten()
            bool_4 = torch.zeros(len(_cond_4), dtype=bool)

            cond_8_1 = torch.concatenate([(x[cond_8] - a).reshape(-1,1), (y[cond_8]).reshape(-1,1)], axis=-1) # 左侧映射
            cond_8_2 = torch.concatenate([(x[cond_8]).reshape(-1,1), (y[cond_8] - b).reshape(-1,1)], axis=-1) # 下侧映射
            cond_8_3 = torch.concatenate([(x[cond_8] - a).reshape(-1,1), (y[cond_8] - b).reshape(-1,1)], axis=-1) # 左下方映射
            _cond_8 = torch.concatenate([cond_8_1, cond_8_2, cond_8_3], axis=0)
            idx_cond_8_1 = torch.nonzero(cond_8).flatten()
            idx_8 = torch.concatenate([idx_cond_8_1, idx_cond_8_1, idx_cond_8_1], axis=0)
            bool_cond_8_1 = torch.zeros(len(cond_8_1), dtype=bool)
            bool_8 = torch.concatenate([bool_cond_8_1, bool_cond_8_1, bool_cond_8_1], axis=0)

            mapped_positions = torch.concatenate([positions, _cond_5, _cond_3, _cond_6, _cond_1, _cond_2, _cond_7, _cond_4, _cond_8], axis=0)
            mapped_indices = torch.concatenate([idx_positions, idx_5, idx_3, idx_6, idx_1, idx_2, idx_7, idx_4, idx_8], axis=0)
            mapped_bool = torch.concatenate([bool_positions, bool_5, bool_3, bool_6, bool_1, bool_2, bool_7, bool_4, bool_8], axis=0)
            return mapped_positions, mapped_indices, mapped_bool

        if all(periodic):
            map_x, map_idx_x, map_bool_x = map_points(box_size[0], box_size[1], h, x)
            map_y, map_idx_y, map_bool_y= map_points(box_size[0], box_size[1], h, y)
            map_x = map_x.to('cpu')
            map_y = map_y.to('cpu')
            tree = KDTree(map_x)
            neighbors = tree.query_ball_point(map_y, h)
            lengths = torch.tensor([len(sublist) for sublist in neighbors]) 
            a = torch.arange(len(lengths))
            node_self = torch.repeat_interleave(a, lengths)
            neighbors = torch.concatenate([torch.tensor(c) for c in neighbors])
            map_bool = map_bool_x[node_self]
            neighbors = map_idx_x[neighbors]
            neighbors = neighbors[map_bool]
            node_self = node_self[node_self < x.shape[0]]
            if not mask_self:
                mask = node_self == neighbors
                node_self = node_self[~mask]
                neighbors = neighbors[~mask]
            
        elif not any(periodic):
            tree = KDTree(x)
            neighbors = tree.query_ball_point(y, h)
            lengths = torch.tensor([len(sublist) for sublist in neighbors]) 
            a = torch.arange(len(lengths))
            node_self = torch.repeat_interleave(a, lengths)
            neighbors = torch.concatenate([torch.tensor(c) for c in neighbors])
            if not mask_self:
                mask = node_self == neighbors
                node_self = node_self[~mask]
                neighbors = neighbors[~mask]

        else:
            for dim in range(3):
                if not periodic[dim]:
                    raise NotImplementedError(f"Single-side periodic boundary condition for dimension {dim} is not implemented yet.")
                    pass
        return node_self, neighbors

    ### 网格与有限元专用函数 ###

    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=None) -> Tensor:
        """``dim`` 维单纯形上 ``p`` 次 Lagrange 插值点的多重指标.

        Returns
        -------
        Tensor
            形状 ``(ldof, dim+1)``, 每行之和为 ``p``.

        Notes
        -----
        TODO: 结果固定在默认设备上, 尚未接受设备参数.
        """
        dtype = dtype or torch.int
        sep = torch.flip(torch.tensor(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=dtype
        ), dims=(0,))
        raw = torch.zeros((sep.shape[0], dim+2), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @staticmethod
    def edge_length(edge: Tensor, node: Tensor, *, out=None) -> Tensor:
        """各边的长度."""
        points = node[edge, :]
        return norm(points[..., 0, :] - points[..., 1, :], dim=-1, out=out)

    @staticmethod
    def edge_normal(edge: Tensor, node: Tensor, unit=False, *, out=None) -> Tensor:
        """二维网格各边的法向, 为切向 ``node[edge[:,1]] - node[edge[:,0]]`` 顺时针旋转 90 度;
        ``unit`` 为 True 时单位化.

        Raises
        ------
        ValueError
            几何维数不是 2.
        """
        points = node[edge, :]
        if points.shape[-1] != 2:
            raise ValueError("Only 2D meshes are supported.")
        edges = points[..., 1, :] - points[..., 0, :]
        if unit:
            edges = edges.div_(norm(edges, dim=-1, keepdim=True))
        return torch.stack([edges[..., 1], -edges[..., 0]], dim=-1, out=out)

    @staticmethod
    def edge_tangent(edge: Tensor, node: Tensor, unit=False, *, out=None) -> Tensor:
        """各边的切向 ``node[edge[:,1]] - node[edge[:,0]]``, ``unit`` 为 True 时单位化."""
        v = torch.sub(node[edge[:, 1], :], node[edge[:, 0], :], out=out)
        if unit:
            l = torch.norm(v, dim=-1, keepdim=True)
            v.div_(l)
        return v

    @staticmethod
    def tensorprod(*tensors: Tensor) -> Tensor:
        """多组一维重心坐标的张量积, 展平为 ``(NQ, NVC)``, ``NVC`` 为各组分量数之积."""
        num = len(tensors)
        NVC = reduce(lambda x, y: x * y.shape[-1], tensors, 1)
        desp1 = 'mnopq'
        desp2 = 'abcde'
        string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
        string += " -> " + desp1[:num] + desp2[:num]
        return torch.einsum(string, *tensors).reshape(-1, NVC)

    @classmethod
    def bc_to_points(cls, bcs: Union[Tensor, Tuple[Tensor, ...]], node: Tensor, entity: Tensor) -> Tensor:
        """把重心坐标映射为各实体上的直角坐标, 形状 ``(NE, NQ, GD)``; 张量积重心坐标先做 ``tensorprod``."""
        points = node[entity, :]

        if not isinstance(bcs, Tensor):
            bcs = cls.tensorprod(*bcs)
        return torch.einsum('ijk, ...j -> i...k', points, bcs)

    @staticmethod
    def barycenter(entity: Tensor, node: Tensor, loc: Optional[Tensor]=None) -> Tensor:
        """各实体顶点坐标的平均; ``loc`` 未使用."""
        return torch.mean(node[entity, :], dim=1) # TODO: polygon mesh case

    @staticmethod
    def simplex_measure(entity: Tensor, node: Tensor) -> Tensor:
        """单纯形的有向测度 ``det(edges) / TD!``, 顶点逆序时为负.

        Raises
        ------
        RuntimeError
            几何维数不等于拓扑维数.
        """
        points = node[entity, :]
        TD = points.size(-2) - 1
        if TD != points.size(-1):
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                            "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        return det(edges).div(factorial(TD))

    @classmethod
    def _simplex_shape_function_kernel(cls, bc: Tensor, p: int, mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        itype = torch.int
        device = bc.device
        shape = (1, TD+1)

        if mi is None:
            mi = cls.multi_index_matrix(p, TD, dtype=torch.int)

        c = torch.arange(1, p+1, dtype=itype, device=device)
        P = 1.0 / torch.cumprod(c, dim=0, dtype=bc.dtype)
        t = torch.arange(0, p, dtype=itype, device=device)
        Ap = p*bc.unsqueeze(-2) - t.reshape(-1, 1)
        Ap = torch.cumprod(Ap, dim=-2).clone()
        Ap = Ap.mul(P.reshape(-1, 1))
        A = torch.cat([torch.ones(shape, dtype=bc.dtype, device=device), Ap], dim=-2)
        idx = torch.arange(TD + 1, dtype=itype, device=device)
        phi = torch.prod(A[mi, idx], dim=-1)
        return phi

    @classmethod
    def simplex_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        """单纯形上 ``p`` 次 Lagrange 基函数在重心坐标处的值, 形状 ``(..., ldof)``, 沿首轴 ``vmap``."""
        fn = vmap(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        )
        return fn(bcs)

    @classmethod
    def simplex_grad_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        """``p`` 次 Lagrange 基函数对重心坐标的导数 (``jacfwd``), 形状 ``(..., ldof, TD+1)``."""
        fn = vmap(jacfwd(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        ))
        return fn(bcs)

    @classmethod
    def simplex_hess_shape_function(cls, bcs: Tensor, p: int, mi=None) -> Tensor:
        """``p`` 次 Lagrange 基函数对重心坐标的二阶导数 (两次 ``jacfwd``), 形状 ``(..., ldof, TD+1, TD+1)``."""
        fn = vmap(jacfwd(jacfwd(
            partial(cls._simplex_shape_function_kernel, p=p, mi=mi)
        )))
        return fn(bcs)

    @staticmethod
    def tensor_measure(entity: Tensor, node: Tensor) -> Tensor:
        """张量积单元的测度. 尚未实现, 调用即抛 ``NotImplementedError``."""
        # TODO: 尚未实现
        raise NotImplementedError

    @staticmethod
    def interval_grad_lambda(line: Tensor, node: Tensor) -> Tensor:
        """区间单元上两个重心坐标的梯度, 形状 ``(NC, 2, GD)``."""
        points = node[line, :]
        v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
        h2 = torch.sum(v**2, dim=-1, keepdim=True)
        v = v.div(h2)
        return torch.stack([-v, v], dim=-2)

    @staticmethod
    def triangle_area_3d(tri: Tensor, node: Tensor, out: Optional[Tensor]=None) -> Tensor:
        """三维空间中三角形的面积."""
        points = node[tri, :]
        cross_product = cross(points[..., 1, :] - points[..., 0, :],
                    points[..., 2, :] - points[..., 0, :], dim=-1, out=out) / 2.0
        result = norm(cross_product, dim=-1)
        return result

    @staticmethod
    def triangle_grad_lambda_2d(tri: Tensor, node: Tensor) -> Tensor:
        """二维三角形三个重心坐标的梯度, 形状 ``(NC, 3, 2)``."""
        shape = tri.shape[:-1] + (3, 2)
        result = torch.zeros(shape, dtype=node.dtype, device=node.device)

        result[..., 0, :] = node[tri[..., 2]] - node[tri[..., 1]]
        result[..., 1, :] = node[tri[..., 0]] - node[tri[..., 2]]
        result[..., 2, :] = node[tri[..., 1]] - node[tri[..., 0]]

        nv = result[..., 0, 0]*result[..., 1, 1] - result[..., 0, 1]*result[..., 1, 0]

        result = result.flip(-1)
        result[..., 0].mul_(-1)
        return result.div_(nv[..., None, None])

    @staticmethod
    def triangle_grad_lambda_3d(tri: Tensor, node: Tensor) -> Tensor:
        """三维空间中三角形三个重心坐标在其所在平面内的梯度, 形状 ``(NC, 3, 3)``."""
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :] # (..., 3)
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = cross(e0, e1, dim=-1) # (..., 3)
        length = norm(nv, dim=-1, keepdim=True) # (..., 1)
        n = nv.div_(length)
        return torch.stack([
            cross(n, e0, dim=-1),
            cross(n, e1, dim=-1),
            cross(n, e2, dim=-1)
        ], dim=-2).div_(length.unsqueeze(-2)) # (..., 3, 3)

    @classmethod
    def tetrahedron_grad_lambda_3d(cls, tet: Tensor, node: Tensor, localFace: Tensor) -> Tensor:
        """四面体四个重心坐标的梯度, 形状 ``(NC, 4, 3)``.

        ``localFace[i]`` 为第 ``i`` 个顶点所对面的三个局部顶点.
        """
        NC = tet.shape[0]
        kwargs = cls.context(node)
        Dlambda = torch.zeros((NC, 4, 3), **kwargs)
        volume = cls.simplex_measure(tet, node)
        for i in range(4):
            j, k, m = localFace[i]
            vjk = node[tet[:, k],:] - node[tet[:, j],:]
            vjm = node[tet[:, m],:] - node[tet[:, j],:]
            Dlambda[:, i, :] = cross(vjm, vjk, dim=-1) / (6*volume.reshape(-1, 1))
        return Dlambda


PyTorchBackend.attach_attributes(ATTRIBUTE_MAPPING, torch)
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(
    array='tensor',
    bitwise_invert='bitwise_not',
    broadcast_arrays='broadcast_tensors',
    copy='clone',
    compile='compile',
)
PyTorchBackend.attach_methods(function_mapping, torch)
PyTorchBackend.attach_methods(TRANSFORMS_MAPPING, torch.func)

PyTorchBackend.random.rand = torch.rand
PyTorchBackend.random.rand_like = torch.rand_like
PyTorchBackend.random.randint = torch.randint
PyTorchBackend.random.randint_like = torch.randint_like
PyTorchBackend.random.randn = torch.randn
PyTorchBackend.random.randn_like = torch.randn_like
PyTorchBackend.random.randperm = torch.randperm
