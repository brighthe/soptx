# 移植自 brighthe/fealpy ``fealpy/ml/{grad.py, sampler/functional.py, sampler/sampler.py,
# modules/module.py}`` @ f474a5775, 仅保留本示例用到的部分.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""PINN 示例的自动微分、配点采样与误差估计工具.

保留的计算路径逐句照抄 FEALPy 原文 (随机数调用顺序不变), 只裁掉本示例用不到的分支:
采样器只保留 ``mode="random"`` 与全部边界, ``Solution`` 只保留前向计算与直角坐标下的
``estimate_error``. 唯一的行为修正是 ``estimate_error`` 中的
``mesh.quadrature_formula(q, etype='cell')``: v0.4 网格的签名为
``quadrature_formula(q, name_or_topdim, qtype)``, 已改为位置参数 ``'cell'``.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import torch
from torch import Tensor
from torch.autograd import grad
from torch.nn import Module

from soptx.backend import backend_manager as bm
from soptx.typing import TensorLike

__all__ = ["gradient", "ISampler", "BoxBoundarySampler", "Solution"]


def gradient(output: Tensor, input: Tensor,
             create_graph=False, allow_unused=False, split: bool=False):
    """计算 ``output`` 对 ``input`` 的梯度.

    Parameters
    ----------
    output : Tensor
        待求导的输出张量.
    input : Tensor
        求导所对的输入张量.
    create_graph : bool, optional
        是否构建导数的计算图以便继续求高阶导数, 默认 False.
    allow_unused : bool, optional
        是否允许 ``input`` 未参与计算, 默认 False.
    split : bool, optional
        是否沿最后一维把梯度拆成元组, 默认 False.

    Returns
    -------
    Tensor or tuple of Tensor
        梯度; ``split=True`` 时为按最后一维拆分的元组.
    """
    g = grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        allow_unused=allow_unused
    )[0]
    if g is None:
        raise ValueError(f"Gradient for input '{input}' with respect to output '{output}' is None.")
    if split:
        return torch.split(g, 1, dim=-1)
    return g


def random_weights(m: int, n: int, dtype=bm.float64, device=None) -> TensorLike:
    """生成 ``m`` 组和为 1 的随机权重, 形状为 ``(m, n)``."""
    m, n = int(m), int(n)
    if n < 2:
        raise ValueError(f'Integer `n` should be larger than 1 but got {n}.')
    u = bm.zeros((m, n+1), dtype=dtype, device=device)
    u[:, n] = 1.0
    u[:, 1:n] = bm.sort(bm.random.rand(m, n-1), axis=1)
    return u[:, 1:n+1] - u[:, 0:n]


def _as_tensor(__sequence: Sequence, dtype=bm.float64, device=None):
    """把序列转为指定 dtype 与 device 的张量."""
    seq = __sequence
    if isinstance(seq, TensorLike) and (bm.backend_name == 'pytorch'):
        return seq.detach().clone().to(device=device).to(dtype=dtype)
    else:
        return bm.tensor(seq, dtype=dtype, device=device)


class ISampler:
    """超矩形区域内的逐维独立随机采样器 (仅 ``mode="random"``).

    Parameters
    ----------
    ranges : sequence or Tensor
        区域范围, 形如 ``[x_min, x_max, y_min, y_max, ...]`` 或 ``(nd, 2)``.
    mode : str, optional
        采样模式, 本示例只支持 ``"random"``.
    dtype, device
        样本的数据类型与设备.
    requires_grad : bool, optional
        样本是否需要梯度, 默认 False.
    """

    def __init__(self, ranges: Any, mode: str='random', dtype=bm.float64,
                 device=None, requires_grad: bool=False) -> None:
        if mode != 'random':
            raise ValueError(f"本示例只移植了 'random' 采样模式, 收到 {mode!r}.")
        self.dtype = dtype
        self.device = device
        self.requires_grad = bool(requires_grad)
        ranges_arr = _as_tensor(ranges, dtype=dtype, device=device)

        if ranges_arr.ndim == 1:
            _, mod = divmod(ranges_arr.shape[0], 2)
            if mod != 0:
                raise ValueError("If `ranges` is 1-dimensional, its length is"
                                 f"expected to be even, but got {mod}.")
            ranges_arr = ranges_arr.reshape(-1, 2)
        assert ranges_arr.ndim == 2
        self.nd = ranges_arr.shape[0]
        self.nodes = ranges_arr # (nd, 2)
        self.mode = mode

    def run(self, *m: int) -> TensorLike:
        """在区域内随机生成 ``m[0]`` 个样本, 形状为 ``(m[0], nd)``."""
        ruler = bm.stack(
            [random_weights(m[0], 2, dtype=self.dtype, device=self.device)
            for _ in range(self.nd)], axis=0) # (nd, m, 2)
        ret = bm.einsum('db, dmb -> md', self.nodes, ruler)

        if bm.backend_name == 'pytorch':
            ret.requires_grad_(self.requires_grad)
        return ret


class BoxBoundarySampler:
    """超矩形全部边界上的随机采样器 (仅 ``mode="random"``).

    每个面各用一个退化的 :class:`ISampler`, 面的顺序为
    ``x = x_min, x = x_max, y = y_min, y = y_max, ...``.

    Parameters
    ----------
    *args : sequence of float
        区域 ``[x_min, x_max, y_min, y_max, ...]``, 或两个对角点 ``p1, p2``.
    mode : str, optional
        采样模式, 本示例只支持 ``"random"``.
    dtype, device
        样本的数据类型与设备.
    requires_grad : bool, optional
        样本是否需要梯度, 默认 False.
    """

    def __init__(self, *args: Sequence[float], mode: str='random',
                 dtype=bm.float64, device=None, requires_grad: bool=False) -> None:
        if mode != 'random':
            raise ValueError(f"本示例只移植了 'random' 采样模式, 收到 {mode!r}.")
        if len(args) == 1:
            p1, p2 = args[0][0::2], args[0][1::2]
        elif len(args) == 2:
            p1, p2 = args
        else:
            raise ValueError( f"Expected 1 argument (domain) or 2 arguments (p1, p2) for *args, but got {len(args)}")
        t1 = _as_tensor(p1, dtype=dtype, device=device)
        t2 = _as_tensor(p2, dtype=dtype, device=device)
        if len(t1.shape) != 1:
            raise ValueError
        if t1.shape != t2.shape:
            raise ValueError("p1 and p2 should be in a same shape.")
        self.nd = int(t1.shape[0])
        self.mode = mode
        self.data = bm.stack([t1, t2]).T

        self.subs: List[ISampler] = []
        for d in range(t1.shape[0]):
            range1, range2 = bm.copy(self.data), bm.copy(self.data)
            range1[d, 1] = self.data[d, 0]
            range2[d, 0] = self.data[d, 1]
            self.subs.append(ISampler(ranges=range1, mode=mode, dtype=dtype,
                            device=device, requires_grad=requires_grad))
            self.subs.append(ISampler(ranges=range2, mode=mode, dtype=dtype,
                            device=device, requires_grad=requires_grad))

    def run(self, *mb: int) -> TensorLike:
        """在每个面上各随机生成 ``mb`` 个样本, 拼接后返回 ``(2 * nd * mb, nd)``."""
        if len(mb) == 1:
            mb *= self.nd
        assert len(mb) * 2 == len(self.subs), \
        "Number of samples per boundary must be equal to 2 times the number of boundaries."

        results: List[TensorLike] = []
        for idx, m in enumerate(mb):
            results.append(self.subs[idx*2].run(m))
            results.append(self.subs[idx*2+1].run(m))
        return bm.concat(results, axis=0)


class Solution(Module):
    """把张量函数 (通常是 ``torch.nn.Module``) 包装成可在网格上估计误差的映射.

    Parameters
    ----------
    func : callable
        接受并返回 torch 张量的函数.
    """

    def __init__(self, func) -> None:
        super().__init__()
        self.__func = func

    @property
    def net(self):
        """被包装的网络."""
        return self.__func

    def forward(self, p: Tensor) -> Tensor:
        """对输入点 ``p`` (形状 ``(..., d)``) 求值."""
        return self.__func(p)

    def get_device(self):
        """返回第一个参数所在的设备; 没有参数时返回 None."""
        for param in self.parameters():
            return param.device
        return getattr(self, '_device', None)

    def last_dim(self, p: Tensor):
        """保持除最后一维外的形状不变地求值."""
        origin_shape = p.shape[:-1]
        p = p.reshape(-1, p.shape[-1])
        val = self(p)
        return val.reshape(origin_shape + (val.shape[-1], ))

    def from_numpy(self, ps, device=None, last_dim=False) -> Tensor:
        """把 numpy 数组转成张量后求值."""
        pt = torch.from_numpy(ps)
        if device is None:
            device = self.get_device()
        if last_dim:
            return self.last_dim(pt.to(device=device))
        return self(pt.to(device=device))

    def estimate_error(self, other, mesh, power: int=2, q: int=3,
                       coordtype: str='c', device=None):
        """在网格上估计与参照函数 ``other`` 之差的 L^power 范数 (逐分量).

        Parameters
        ----------
        other : callable
            直角坐标下的参照函数.
        mesh : Mesh
            积分所用网格.
        power : int, optional
            范数阶, 默认 2.
        q : int, optional
            积分阶, 默认 3.
        coordtype : str, optional
            坐标类型, 本示例只支持直角坐标 ``"c"`` / ``"cartesian"``.
        device : torch.device, optional
            计算设备, 默认取网络参数所在设备.

        Returns
        -------
        Tensor
            各输出分量的误差.
        """
        o_coordtype = getattr(other, 'coordtype', None)
        if o_coordtype is not None:
            coordtype = o_coordtype
        if coordtype not in {'cartesian', 'c'}:
            raise ValueError(f"本示例只移植了直角坐标分支, 收到 coordtype={coordtype!r}.")

        qf = mesh.quadrature_formula(q, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        cellmeasure = mesh.entity_measure('cell')

        ps = mesh.bc_to_point(bcs)
        val = self.from_numpy(ps.detach().numpy(), device=device, last_dim=True).cpu()
        val_ture = other(ps)
        val = val.detach() if val.requires_grad else val

        # 统一形状
        ndim = len(val_ture.shape)
        if ndim == 2:  # 如果 val_ture 是 (N, M)
            val_ture = val_ture.unsqueeze(-1)  # -> (N, M, 1)
        elif ndim == 4:  # 如果 val_ture 是 (N, M, 1, 1)
            val_ture = val_ture.squeeze()  # -> (N, M, 1)

        assert val.shape == val_ture.shape, f"Shape mismatch: val {val.shape}, val_ture {val_ture.shape}"
        val = bm.real(val)
        val_ture = bm.real(val_ture)
        diff = bm.abs(val - val_ture)**power

        e = bm.einsum('q, cq..., c -> c...', ws, diff, cellmeasure)
        return bm.pow(e.sum(axis=0), 1/power)
