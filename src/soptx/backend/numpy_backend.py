# 移植自 brighthe/fealpy ``fealpy/backend/numpy_backend.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""numpy 计算后端.

只支持 CPU; 构造函数会丢弃 ``device`` 参数. 未手动实现的函数按 ``base`` 的名称映射
从 numpy 复制.
"""

from typing import Optional, Union, Tuple
import threading
from functools import reduce
from math import factorial
from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from numpy.linalg import det
from scipy.sparse._sparsetools import coo_matvec, csr_matvec, csr_matvecs, coo_tocsr

from .base import (
    ModuleProxy, BackendProxy,
    ATTRIBUTE_MAPPING, FUNCTION_MAPPING
)

def _remove_device(func):
    def wrapper(*args, **kwargs):
        """丢弃 numpy 不接受的 ``device`` 参数后调用."""
        if 'device' in kwargs:
            kwargs.pop('device')
        return func(*args, **kwargs)
    return wrapper

class NumPyBackend(BackendProxy, backend_name='numpy'):
    """numpy 后端代理, 张量类型为 ``np.ndarray``."""
    DATA_CLASS = np.ndarray

    linalg = np.linalg
    random = np.random

    @staticmethod
    def context(tensor):
        """张量的构造参数 ``{'dtype': ...}``; numpy 没有设备概念."""
        return {"dtype": tensor.dtype}

    @staticmethod
    def set_default_device(device) -> None:
        """numpy 后端不支持, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError("`set_default_device` is not supported by NumPyBackend")

    @staticmethod
    def device_type(tensor_like, /):
        """恒为 ``'cpu'``."""
        return 'cpu'

    @staticmethod
    def device_index(tensor_like, /):
        """恒为 0."""
        return 0

    @staticmethod
    def get_device(tensor_like: NDArray, /):
        """恒为 ``'cpu'``."""
        return 'cpu'

    @staticmethod
    def device_put(tensor_like, /, device=None):
        """只支持 CPU, 原样返回.

        Raises
        ------
        NotImplementedError
            ``device`` 不是 None 或 ``'cpu'``.
        """
        if device not in {None, 'cpu'}:
            raise NotImplementedError("only cpu device is supported by NumPyBackend ")
        return tensor_like

    @staticmethod
    def to_numpy(tensor_like: NDArray, /) -> NDArray:
        """原样返回."""
        return tensor_like

    @staticmethod
    def from_numpy(ndarray: NDArray, /) -> NDArray:
        """原样返回."""
        return ndarray

    @staticmethod
    def tolist(tensor: NDArray, /):
        """转为 Python 嵌套列表."""
        return tensor.tolist()

    ### 创建函数 ###
    if np.__version__ < '2.0.0':
        arange = staticmethod(_remove_device(np.arange))
        asarray = staticmethod(_remove_device(np.asarray))
        empty = staticmethod(_remove_device(np.empty))
        empty_like = staticmethod(_remove_device(np.empty_like))
        eye = staticmethod(_remove_device(np.eye))
        full = staticmethod(_remove_device(np.full))
        full_like = staticmethod(_remove_device(np.full_like))
        linspace = staticmethod(_remove_device(np.linspace))
        ones = staticmethod(_remove_device(np.ones))
        ones_like = staticmethod(_remove_device(np.ones_like))
        zeros = staticmethod(_remove_device(np.zeros))
        zeros_like = staticmethod(_remove_device(np.zeros_like))

    array = staticmethod(_remove_device(np.array))
    tensor = staticmethod(_remove_device(np.array))

    ### 数据类型函数 ###

    ### 逐元素函数 ###

    ### 索引函数 ###

    ### 线性代数函数 ###
    # 非标准
    @staticmethod
    def einsum(*args, **kwargs):
        """``np.einsum``, 固定 ``optimize=True`` 以自动选择收缩路径."""
        return np.einsum(*args, **kwargs, optimize=True)

    ### 变形函数 ###
    # Python array API 标准 v2023.12
    @staticmethod
    def unstack(x, /, *, axis: int=0):
        """沿 ``axis`` 拆成切片元组, 切片去掉该轴, 与 array API 及 pytorch 后端一致."""
        return tuple(np.moveaxis(x, axis, 0))

    ### 查找函数 ###

    ### 集合函数 ###
    # 非标准

    ### 排序函数 ###

    ### 统计函数 ###
    # Python array API 标准 v2023.12
    @staticmethod
    def sum(x, /, *, axis=None, dtype=None, keepdims=False):
        """``np.add.reduce`` 求和."""
        return np.add.reduce(x, axis=axis, dtype=dtype, keepdims=keepdims)

    ### 工具函数 ###

    ### 其他函数 ### (非标准)
    @staticmethod
    def set_at(a: NDArray, indices, src, /) -> NDArray:
        """``a[indices] = src``, 原地修改并返回 ``a``."""
        a[indices] = src
        return a

    @staticmethod
    def add_at(a: NDArray, indices, src, /) -> NDArray:
        """``np.add.at``: 在 ``indices`` 处累加 ``src``, 重复索引全部计入; 原地修改并返回 ``a``."""
        np.add.at(a, indices, src)
        return a

    @staticmethod
    def index_add(a: NDArray, index, src, /, *, axis=0, alpha=1):
        """沿 ``axis`` 在 ``index`` 处累加 ``alpha * src``, 重复索引全部计入; 原地修改并返回 ``a``."""
        indexing = [slice(None)] * a.ndim
        indexing[axis] = index
        np.add.at(a, tuple(indexing), alpha*src)
        return a

    @staticmethod
    def scatter(x, indices, val, /, *, axis=0):
        """未实现, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError

    @staticmethod
    def scatter_add(x, indices, val, /, *, axis=0):
        """未实现, 调用即抛 ``NotImplementedError``."""
        raise NotImplementedError

    @staticmethod
    def unique_all_(a, axis=None, **kwargs):
        """``np.unique`` 的扩展.

        Returns
        -------
        tuple
            ``(唯一值, 首次出现位置, 最后出现位置, 逆映射, 计数)``.
        """
        b, indices0, inverse, counts = np.unique(a,
                                                 return_index=True,
                                                 return_inverse=True,
                                                 return_counts=True,
                                                 axis=axis, **kwargs)
        indices1 = np.zeros_like(indices0)
        indices1[inverse] = range(inverse.shape[0]);
        return b, indices0, indices1, inverse, counts


    ### 稀疏函数 ###
    @staticmethod
    def coo_spmm(indices, values, shape, other):
        """COO 矩阵乘以一维或二维稠密数组, 调用 scipy 的 ``coo_matvec``.

        Raises
        ------
        ValueError
            ``other`` 不是一维或二维.
        NotImplementedError
            ``values`` 带批量维.
        """
        nnz = values.shape[-1]
        row = indices[0]
        col = indices[1]

        if values.ndim == 1:
            if other.ndim == 1:
                result = np.zeros((shape[0],), dtype=other.dtype)
                coo_matvec(nnz, row, col, values, other, result)
                return result
            elif other.ndim == 2:
                new_shape = (shape[0], other.shape[-1])
                result = np.zeros(new_shape, dtype=other.dtype)
                rT = result.T
                for i, acol in enumerate(other.T):
                    coo_matvec(nnz, row, col, values, acol, rT[i])
                return result
            else:
                raise ValueError("`other` must be a 1-D or 2-D array.")
        else:
            raise NotImplementedError("Batch sparse matrix multiplication has "
                                      "not been supported yet.")

    @staticmethod
    def csr_spmm(crow, col, values, shape, other):
        """CSR 矩阵乘以一维或二维稠密数组, 调用 scipy 的 ``csr_matvec(s)``.

        Raises
        ------
        ValueError
            ``other`` 不是一维或二维.
        NotImplementedError
            ``values`` 带批量维.
        """
        M, N = shape

        if values.ndim == 1:
            if other.ndim == 1:
                result = np.zeros((shape[0],), dtype=other.dtype)
                csr_matvec(M, N, crow, col, values, other, result)
                return result
            elif other.ndim == 2:
                n_vecs = other.shape[-1]
                new_shape = (shape[0], other.shape[-1])
                result = np.zeros(new_shape, dtype=other.dtype)
                csr_matvecs(M, N, n_vecs, crow, col, values, other.ravel(), result.ravel())
                return result
            else:
                raise ValueError("`other` must be a 1-D or 2-D array.")
        else:
            raise NotImplementedError("Batch sparse matrix multiplication has "
                                      "not been supported yet.")

    @staticmethod
    def csr_spspmm(crow1, col1, values1, shape1, crow2, col2, values2, shape2):
        """两个 CSR 矩阵相乘, 返回 ``(crow, col, values, shape)``."""
        from scipy.sparse import csr_matrix
        m1 = csr_matrix((values1, col1, crow1), shape=shape1)
        m2 = csr_matrix((values2, col2, crow2), shape=shape2)
        m3 = m1._matmul_sparse(m2)
        return m3.indptr, m3.indices, m3.data, m3.shape

    @staticmethod
    def coo_tocsr(indices, values, shape):
        """COO 转 CSR, 调用 scipy 的 ``coo_tocsr`` (不合并重复项), 返回 ``(crow, col, values)``."""
        M, N = shape
        idx_dtype = indices.dtype
        major, minor = indices
        nnz = len(values)
        crow = np.empty(M+1, dtype=idx_dtype)
        col = np.empty_like(minor, dtype=idx_dtype)
        data = np.empty_like(values, dtype=values.dtype)
        coo_tocsr(M, N, nnz, major, minor, values, crow, col, data)

        return crow, col, data

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
            idx_positions = np.arange(len(positions))
            bool_positions = np.ones(len(positions), dtype=bool)

            cond_5_1 = np.concatenate([(x[cond_5] + a).reshape(-1,1), (y[cond_5]).reshape(-1,1)], axis=-1) # 右侧映射
            cond_5_2 = np.concatenate([(x[cond_5]).reshape(-1,1), (y[cond_5] + b).reshape(-1,1)], axis=-1) # 上方映射 
            cond_5_3 = np.concatenate([(x[cond_5] + a).reshape(-1,1), (y[cond_5] + b).reshape(-1,1)], axis=-1) # 右上方映射
            _cond_5 = np.concatenate([cond_5_1, cond_5_2, cond_5_3], axis=0)
            idx_cond_5_1 = np.nonzero(cond_5)[0]
            idx_5 = np.concatenate([idx_cond_5_1, idx_cond_5_1, idx_cond_5_1], axis=0)
            bool_cond_5_1 = np.zeros(len(cond_5_1), dtype=bool)
            bool_5 = np.concatenate([bool_cond_5_1, bool_cond_5_1, bool_cond_5_1], axis=0)

            _cond_3 = np.concatenate([(x[cond_3]).reshape(-1,1), (y[cond_3] + b).reshape(-1,1)], axis=-1) # 上侧映射 
            idx_3 = np.nonzero(cond_3)[0]
            bool_3 = np.zeros(len(_cond_3), dtype=bool)

            cond_6_1 = np.concatenate([(x[cond_6] - a).reshape(-1,1), (y[cond_6]).reshape(-1,1)], axis=-1) # 左侧映射
            cond_6_2 = np.concatenate([(x[cond_6]).reshape(-1,1), (y[cond_6] + b).reshape(-1,1)], axis=-1) # 上方映射
            cond_6_3 = np.concatenate([(x[cond_6] - a).reshape(-1,1), (y[cond_6] + b).reshape(-1,1)], axis=-1) # 左上方映射
            _cond_6 = np.concatenate([cond_6_1, cond_6_2, cond_6_3], axis=0)
            idx_cond_6_1 = np.nonzero(cond_6)[0]
            idx_6 = np.concatenate([idx_cond_6_1, idx_cond_6_1, idx_cond_6_1], axis=0)
            bool_cond_6_1 = np.zeros(len(cond_6_1), dtype=bool)
            bool_6 = np.concatenate([bool_cond_6_1, bool_cond_6_1, bool_cond_6_1], axis=0)

            _cond_1 = np.concatenate([(x[cond_1] + a).reshape(-1,1), y[cond_1].reshape(-1,1)], axis=-1) # 右侧映射
            idx_1 = np.nonzero(cond_1)[0]
            bool_1 = np.zeros(len(_cond_1), dtype=bool)

            _cond_2 = np.concatenate([(x[cond_2] - a).reshape(-1,1), y[cond_2].reshape(-1,1)], axis=-1) # 左侧映射
            idx_2 = np.nonzero(cond_2)[0]
            bool_2 = np.zeros(len(_cond_2), dtype=bool)

            cond_7_1 = np.concatenate([(x[cond_7] + a).reshape(-1,1), (y[cond_7]).reshape(-1,1)], axis=-1) # 右侧映射
            cond_7_2 = np.concatenate([(x[cond_7]).reshape(-1,1), (y[cond_7] - b).reshape(-1,1)], axis=-1) # 下方映射
            cond_7_3 = np.concatenate([(x[cond_7] + a).reshape(-1,1), (y[cond_7] - b).reshape(-1,1)], axis=-1) # 右下方映射
            _cond_7 = np.concatenate([cond_7_1, cond_7_2, cond_7_3], axis=0)
            idx_cond_7_1 = np.nonzero(cond_7)[0]
            idx_7 = np.concatenate([idx_cond_7_1, idx_cond_7_1, idx_cond_7_1], axis=0)
            bool_cond_7_1 = np.zeros(len(cond_7_1), dtype=bool)
            bool_7 = np.concatenate([bool_cond_7_1, bool_cond_7_1, bool_cond_7_1], axis=0)

            _cond_4 = np.concatenate([(x[cond_4]).reshape(-1,1), (y[cond_4] - b).reshape(-1,1)], axis=-1) #下侧映射
            idx_4 = np.nonzero(cond_4)[0]
            bool_4 = np.zeros(len(_cond_4), dtype=bool)

            cond_8_1 = np.concatenate([(x[cond_8] - a).reshape(-1,1), (y[cond_8]).reshape(-1,1)], axis=-1) # 左侧映射
            cond_8_2 = np.concatenate([(x[cond_8]).reshape(-1,1), (y[cond_8] - b).reshape(-1,1)], axis=-1) # 下侧映射
            cond_8_3 = np.concatenate([(x[cond_8] - a).reshape(-1,1), (y[cond_8] - b).reshape(-1,1)], axis=-1) # 左下方映射
            _cond_8 = np.concatenate([cond_8_1, cond_8_2, cond_8_3], axis=0)
            idx_cond_8_1 = np.nonzero(cond_8)[0]
            idx_8 = np.concatenate([idx_cond_8_1, idx_cond_8_1, idx_cond_8_1], axis=0)
            bool_cond_8_1 = np.zeros(len(cond_8_1), dtype=bool)
            bool_8 = np.concatenate([bool_cond_8_1, bool_cond_8_1, bool_cond_8_1], axis=0)

            mapped_positions = np.concatenate([positions, _cond_5, _cond_3, _cond_6, _cond_1, _cond_2, _cond_7, _cond_4, _cond_8], axis=0)
            mapped_indices = np.concatenate([idx_positions, idx_5, idx_3, idx_6, idx_1, idx_2, idx_7, idx_4, idx_8], axis=0)
            mapped_bool = np.concatenate([bool_positions, bool_5, bool_3, bool_6, bool_1, bool_2, bool_7, bool_4, bool_8], axis=0)
            return mapped_positions, mapped_indices, mapped_bool

        if all(periodic):
            map_x, map_idx_x, map_bool_x = map_points(box_size[0], box_size[1], h, x)
            map_y , map_idx_y, map_bool_y= map_points(box_size[0], box_size[1], h, y)
            tree = KDTree(map_x)
            neighbors = tree.query_ball_point(map_y, h)
            lengths = np.array([len(sublist) for sublist in neighbors]) 
            a = np.arange(len(lengths))
            node_self = np.repeat(a, lengths)
            neighbors = np.concatenate(neighbors)
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
            lengths = np.array([len(sublist) for sublist in neighbors])  
            a = np.arange(len(lengths))
            node_self = np.repeat(a, lengths)
            neighbors = np.concatenate(neighbors)
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

    ### 函数变换 ###
    @staticmethod
    def vmap(func, /, in_axes=0, out_axes=0, *args, **kwds):
        """逐切片调用再堆叠的简易 ``vmap``.

        Raises
        ------
        ValueError
            ``in_axes != out_axes``.
        """
        if in_axes != out_axes:
            raise ValueError(f"Only support in_axes == out_axes with numpy backend")
        from functools import partial
        def vectorized(*args, **kwargs):
            """沿 ``in_axes`` 拆开所有数组参数逐个调用 ``func``, 再把结果沿同一轴堆叠."""
            arr_lists = [np.unstack(arr, axis=in_axes)
                         for arr in args if isinstance(arr, np.ndarray)]
            results = tuple(map(partial(func, **kwargs), *arr_lists))

            if isinstance(results[0], tuple):
                results = map(partial(np.stack, axis=in_axes), zip(*results))
                results = tuple(results)
            else:
                results = np.stack(results, axis=in_axes)
            return results

        return vectorized

    ### 网格与有限元专用函数 ###

    @staticmethod
    def multi_index_matrix(p: int, dim: int, *, dtype=np.int32) -> NDArray:
        """``dim`` 维单纯形上 ``p`` 次 Lagrange 插值点的多重指标.

        Returns
        -------
        TensorLike
            形状 ``(ldof, dim+1)``, 每行之和为 ``p``.
        """
        sep = np.flip(np.array(
            tuple(combinations_with_replacement(range(p+1), dim)),
            dtype=dtype
        ), axis=0)
        raw = np.zeros((sep.shape[0], dim+2), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @staticmethod
    def edge_length(edge: NDArray, node: NDArray, *, out=None) -> NDArray:
        """各边的长度."""
        assert out == None, "`out` is not supported by edge_length in NumPyBackend"
        points = node[edge, :]
        return np.linalg.norm(points[..., 0, :] - points[..., 1, :], axis=-1)

    @staticmethod
    def edge_normal(edge: NDArray, node: NDArray, unit=False, *, out=None) -> NDArray:
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
            edges /= np.linalg.norm(edges, axis=-1, keepdims=True)
        return np.stack([edges[..., 1], -edges[..., 0]], axis=-1, out=out)

    @staticmethod
    def edge_tangent(edge: NDArray, node: NDArray, unit=False, *, out=None) -> NDArray:
        """各边的切向 ``node[edge[:,1]] - node[edge[:,0]]``, ``unit`` 为 True 时单位化."""
        v = np.subtract(node[edge[:, 1], :], node[edge[:, 0], :], out=out)
        if unit:
            l = np.linalg.norm(v, axis=-1, keepdims=True)
            v /= l
        return v

    @staticmethod
    def tensorprod(*tensors: NDArray) -> NDArray:
        """多组一维重心坐标的张量积, 展平为 ``(NQ, NVC)``, ``NVC`` 为各组分量数之积."""
        num = len(tensors)
        NVC = reduce(lambda x, y: x * y.shape[-1], tensors, 1)
        desp1 = 'mnopq'
        desp2 = 'abcde'
        string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
        string += " -> " + desp1[:num] + desp2[:num]
        return np.einsum(string, *tensors).reshape(-1, NVC)

    @classmethod
    def bc_to_points(cls, bcs: Union[NDArray, Tuple[NDArray, ...]], node: NDArray, entity: NDArray) -> NDArray:
        """把重心坐标映射为各实体上的直角坐标, 形状 ``(NE, NQ, GD)``.

        Notes
        -----
        张量积重心坐标 (元组) 的分支调用 ``tensorprod(bcs)`` 时未解包, 会抛
        ``AttributeError``; pytorch 后端无此问题.
        """
        points = node[entity, :]

        if not isinstance(bcs, np.ndarray):
            bcs = cls.tensorprod(bcs)
        return np.einsum('ijk, ...j -> i...k', points, bcs)

    @staticmethod
    def barycenter(entity: NDArray, node: NDArray, loc: Optional[NDArray]=None) -> NDArray:
        """各实体顶点坐标的平均; ``loc`` 未使用."""
        return np.mean(node[entity, :], axis=1)

    @staticmethod
    def simplex_measure(entity: NDArray, node: NDArray) -> NDArray:
        """单纯形的有向测度 ``det(edges) / TD!``, 顶点逆序时为负.

        Raises
        ------
        RuntimeError
            几何维数不等于拓扑维数.
        """
        points = node[entity, :]
        TD = points.shape[-2] - 1
        if TD != points.shape[-1]:
            raise RuntimeError("The geometric dimension of points must be NVC-1"
                            "to form a simplex.")
        edges = points[..., 1:, :] - points[..., :-1, :]
        return det(edges)/(factorial(TD))

    @classmethod
    def simplex_shape_function(cls, bc: NDArray, p: int, mi=None) -> NDArray:
        """单纯形上 ``p`` 次 Lagrange 基函数在重心坐标处的值, 形状 ``(..., ldof)``."""
        if p == 1:
            return bc
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., mi, idx], axis=-1)
        return phi

    @classmethod
    def simplex_grad_shape_function(cls, bc: NDArray, p: int, mi=None) -> NDArray:
        """``p`` 次 Lagrange 基函数对重心坐标的导数, 形状 ``(..., ldof, TD+1)``."""
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = cls.multi_index_matrix(p, TD)

        ldof = mi.shape[0] # p 次 Lagrange 形函数的个数

        c = np.arange(1, p+1)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=bc.dtype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., mi, range(TD+1)]
        M = F[..., mi, range(TD+1)]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
        return R # (..., ldof, bc)

    @classmethod
    def simplex_hess_shape_function(cls,bc: NDArray, p: int, mi=None) -> NDArray:
        """``p`` 次 Lagrange 基函数对重心坐标的二阶导数, 形状 ``(..., ldof, TD+1, TD+1)``.

        由各因子的对数导数组合得到; 分母为 0 处加 ``1e-14`` 以免除零.
        """
        TD = bc.shape[-1] - 1

        if mi is None:
            try:
                mi = cls.multi_index_matrix(p, TD)
            except Exception:
                from itertools import product
                mi = np.array([m for m in product(range(p+1), repeat=TD+1) if sum(m) == p], dtype=int)

        ldof = mi.shape[0]

        # p = 0  => 常数基函数, Hessian 全 0
        if p == 0:
            shape = bc.shape[:-1] + (ldof, TD+1, TD+1)
            return np.zeros(shape, dtype=bc.dtype)

        # --- 构造 1/(p*b - r) 以及其平方的前缀和 ---
        r = np.arange(0, p, dtype=bc.dtype).reshape((-1, 1))             # (p,1)
        denom = (p * bc[..., None, :]) - r                              # (..., p, TD+1)
        # 避免除以 0
        eps = 1e-14
        denom = denom + eps * (denom == 0)
        inv = 1.0 / denom                                               # (..., p, TD+1)
        inv2 = inv * inv

        # axis_p 是 p 轴在数组中的索引 (依赖 bc 的维度)
        axis_p = bc.ndim - 1
        csum = np.cumsum(inv, axis=axis_p)      # 前缀和 sum_{r=0..m-1} 1/(p b - r)
        csum2 = np.cumsum(inv2, axis=axis_p)    # 前缀和 sum_{r=0..m-1} 1/(p b - r)^2

        # 把 m 从 0..p 表示成长度 p+1 的数组 (m=0 时和为 0)
        shape_m = bc.shape[:-1] + (p+1, TD+1)
        S1 = np.zeros(shape_m, dtype=bc.dtype)   # 一次项前缀和 sum 1/(...)
        S2 = np.zeros(shape_m, dtype=bc.dtype)   # 平方项前缀和 sum 1/(...)^2
        if p > 0:
            S1[..., 1:, :] = csum
            S2[..., 1:, :] = csum2

        # f'/f 和 f''/f 对数导数
        fprime_over_f = p * S1
        fsecond_over_f = (p * p) * (S1 * S1 - S2)

        # 计算每个因子 f_m(b) 的值
        if p > 0:
            c = np.arange(1, p+1, dtype=np.int_)
            P = 1.0 / np.multiply.accumulate(c)
            t = np.arange(0, p)
        else:
            P = np.array([], dtype=bc.dtype)
            t = np.array([], dtype=bc.dtype)

        A = np.ones(bc.shape[:-1] + (p+1, TD+1), dtype=bc.dtype)
        if p > 0:
            A[..., 1:, :] = p * bc[..., None, :] - t.reshape(-1, 1)
            np.cumprod(A, axis=axis_p, out=A)
            A[..., 1:, :] *= P.reshape(-1, 1)
        # 取出每个 dof 的对应 f_{m_j}(b_j)
        idx = np.arange(TD+1)
        Q = A[..., mi, idx]                # (..., ldof, TD+1)
        # 基函数值 phi = prod_j Q[..., j]
        phi = np.prod(Q, axis=-1)          # (..., ldof)

        # pick S 和 T 对应到 mi
        S_sel = fprime_over_f[..., mi, idx]   # (..., ldof, TD+1)
        T_sel = fsecond_over_f[..., mi, idx]  # (..., ldof, TD+1)

        # 非对角项: phi * S_i * S_j
        S_outer = S_sel[..., :, None] * S_sel[..., None, :]  # (..., ldof, TD+1, TD+1)

        # 构建 H, 先放 S_outer, 然后把对角项替换为 phi * T_i
        H = S_outer.copy()
        # 把对角项设为 T_sel
        for ii in range(TD+1):
            H[..., ii, ii] = T_sel[..., ii]

        phi_exp = phi[..., None, None]
        H *= phi_exp

        return H

    @staticmethod
    def tensor_measure(entity: NDArray, node: NDArray) -> NDArray:
        """张量积单元的测度. 尚未实现, 调用即抛 ``NotImplementedError``."""
        # TODO: 尚未实现
        raise NotImplementedError

    @staticmethod
    def interval_grad_lambda(line: NDArray, node: NDArray) -> NDArray:
        """区间单元上两个重心坐标的梯度, 形状 ``(NC, 2, GD)``."""
        points = node[line, :]
        v = points[..., 1, :] - points[..., 0, :] # (NC, GD)
        h2 = np.sum(v**2, axis=-1, keepdims=True)
        v /= h2
        return np.stack([-v, v], axis=-2)

    @staticmethod
    def triangle_area_3d(tri: NDArray, node: NDArray, out=None) -> NDArray:
        """三维空间中三角形的面积."""
        points = node[tri, :]
        edge1 = points[..., 1, :] - points[..., 0, :]
        edge2 = points[..., 2, :] - points[..., 0, :]
        cross_product = np.cross(edge1, edge2, axis=-1)
        area = 0.5 * np.linalg.norm(cross_product, axis=-1)
        if out is not None:
            out[:] = area
            return out
        return area

    @staticmethod
    def triangle_grad_lambda_2d(tri: NDArray, node: NDArray) -> NDArray:
        """二维三角形三个重心坐标的梯度, 形状 ``(NC, 3, 2)``."""
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :]
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = det(np.stack([e0, e1], axis=-2)) # 二维情形的行列式, 等价于对 2x2 矩阵调用 np.linalg.det
        e0 = np.flip(e0, axis=-1)
        e1 = np.flip(e1, axis=-1)
        e2 = np.flip(e2, axis=-1)
        result = np.stack([e0, e1, e2], axis=-2)
        result[..., 0] *= -1
        return result / np.expand_dims(nv, axis=(-1, -2))

    @staticmethod
    def triangle_grad_lambda_3d(tri: NDArray, node: NDArray) -> NDArray:
        """三维空间中三角形三个重心坐标在其所在平面内的梯度, 形状 ``(NC, 3, 3)``."""
        points = node[tri, :]
        e0 = points[..., 2, :] - points[..., 1, :]  # (..., 3)
        e1 = points[..., 0, :] - points[..., 2, :]
        e2 = points[..., 1, :] - points[..., 0, :]
        nv = np.cross(e0, e1, axis=-1)  # 法向量, (..., 3)
        length = np.linalg.norm(nv, axis=-1, keepdims=True)  # 法向量长度, (..., 1)
        n = nv / length  # 单位法向量
        return np.stack([
            np.cross(n, e0, axis=-1),
            np.cross(n, e1, axis=-1),
            np.cross(n, e2, axis=-1)
        ], axis=-2) / length[..., np.newaxis]  # 除以长度完成归一化

    @staticmethod
    def quadrangle_grad_lambda_2d(quad: NDArray, node: NDArray) -> NDArray:
        """四边形重心坐标的梯度. 尚未实现: 函数体为空, 返回 None."""
        pass

    @classmethod
    def tetrahedron_grad_lambda_3d(cls, tet: NDArray, node: NDArray, localFace: NDArray) -> NDArray:
        """四面体四个重心坐标的梯度, 形状 ``(NC, 4, 3)``.

        ``localFace[i]`` 为第 ``i`` 个顶点所对面的三个局部顶点.
        """
        NC = tet.shape[0]
        Dlambda = np.zeros((NC, 4, 3), dtype=node.dtype)
        volume = cls.simplex_measure(tet, node)
        for i in range(4):
            j, k, m = localFace[i]
            vjk = node[tet[:, k],:] - node[tet[:, j],:]
            vjm = node[tet[:, m],:] - node[tet[:, j],:]
            Dlambda[:, i, :] = np.cross(vjm, vjk) / (6*volume.reshape(-1, 1))
        return Dlambda


attribute_mapping = ATTRIBUTE_MAPPING.copy()
function_mapping = FUNCTION_MAPPING.copy()
function_mapping.update(tensor='array')

if int(np.__version__[:1]) < 2:
    attribute_mapping.update(bool='bool_')
    function_mapping.update(
        astype='asarray',
        concat='concatenate', bitwise_invert='bitwise_not', permute_dims='transpose',
        pow='power', acos='arccos', asin='arcsin', acosh='arccosh', asinh='arcsinh',
        atan='arctan', atanh='arctanh', atan2='arctan2'
    )

NumPyBackend.attach_attributes(attribute_mapping, np)
NumPyBackend.attach_methods(function_mapping, np)


##################################################
### 随机数子模块
##################################################

class NumpyRandom(ModuleProxy):
    """numpy 随机数子模块的包装, 每个线程一个 ``Generator``.

    Notes
    -----
    当前未被使用 (后端的 ``random`` 直接取 ``np.random``), 且不可实例化: ``rng`` 的
    setter 被定义成了名为 ``setter`` 的新属性, ``__init__`` 中 ``self.rng = ...`` 会抛
    ``AttributeError``.
    """
    def __init__(self):
        super().__init__()
        self._THREAD_LOCAL = threading.local()
        self.rng = np.random.default_rng()

    @property
    def rng(self) -> np.random.Generator:
        """当前线程的随机数生成器."""
        return self._THREAD_LOCAL.rng

    @rng.setter
    def setter(self, value):
        """本意为 ``rng`` 的 setter, 实际定义成了名为 ``setter`` 的属性."""
        self._THREAD_LOCAL.rng = value

    def seed(self, seed: int):
        """以 ``seed`` 重建随机数生成器."""
        self.rng = np.random.default_rng(seed)

    def rand(self, *size, dtype=None, device=None):
        """``[0, 1)`` 均匀分布随机数; ``device`` 被忽略."""
        if len(size) == 1: size = size[0]
        return self.rng.random(size=size, dtype=dtype)

    def randint(self, low, high=None, size=None, dtype=None, device=None):
        """``[low, high)`` 均匀分布随机整数; ``device`` 被忽略."""
        return self.rng.integers(low, high, size=size, dtype=dtype)

    def randn(self, *size, dtype=None, device=None):
        """标准正态分布随机数; ``device`` 被忽略."""
        if len(size) == 1: size = size[0]
        return self.rng.standard_normal(size, dtype=dtype)
