# 移植自 brighthe/fealpy ``fealpy/quadrature/quadrature.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""积分公式基类."""

from ..backend import bm, dtype, device, Tensor

type BCS = Tensor | tuple[Tensor, ...]


class Quadrature():
    r"""积分公式基类.

    子类实现 ``make``, 按序号生成积分点与权重.

    Parameters
    ----------
    index : int
        公式序号, 含义由子类规定 (积分点数或代数精度).
    dtype : dtype, optional
        浮点类型, 默认 ``bm.float64``.
    device : device, optional
        设备.

    Attributes
    ----------
    quadpts : Tensor or tuple of Tensor
        积分点的重心坐标, 形状 ``(NQ, TD+1)``; 张量积公式为各方向的元组.
    weights : Tensor
        权重, 总和为 1.
    """
    def __init__(
        self,
        index: int,
        *,
        dtype: dtype | None = None,
        device: device | None = None
    ) -> None:
        self.dtype = dtype if dtype else bm.float64
        self.device = device
        self.quadpts, self.weights = self.make(index)

    def __len__(self) -> int:
        return self.number_of_quadrature_points()

    def __getitem__(self, i: int) -> tuple[BCS, Tensor]:
        return self.get_quadrature_point_and_weight(i)

    def make(self, index: int) -> tuple[Tensor, Tensor]:
        """生成序号为 ``index`` 的积分点与权重, 由子类实现."""
        raise NotImplementedError

    def number_of_quadrature_points(self) -> int:
        """积分点个数, 即权重第 0 轴长度."""
        return self.weights.shape[0]

    def get_quadrature_points_and_weights(self) -> tuple[BCS, Tensor]:
        """返回全部积分点与权重.

        Returns
        -------
        tuple
            ``(积分点, 权重)``.
        """
        return self.quadpts, self.weights

    def get_quadrature_point_and_weight(self, i: int) -> tuple[BCS, Tensor]:
        """返回第 ``i`` 个积分点与权重.

        Parameters
        ----------
        i : int
            积分点序号.

        Returns
        -------
        tuple
            ``(积分点, 权重)``.

        Notes
        -----
        积分点为元组的张量积公式不支持本方法, 会抛 ``TypeError``.
        """
        return self.quadpts[i, :], self.weights[i]
