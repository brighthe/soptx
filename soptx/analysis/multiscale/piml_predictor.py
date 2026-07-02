"""T4: PIML 预测器抽象 (原型期实现 Exact / Mock 两种).

统一接口: predict(rho_local) -> SubstructureOperator. 上层组装
(InterfaceCondensedSystem) 只依赖该接口, 不感知内部是精确求解、
解析映射还是网络推断 —— TrainedPredictor (极小 MLP, T4b) 后续
以同一接口接入.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .multiscale_shape import SubstructureOperator, SubstructureTemplate


class MultiscalePredictor(ABC):
    """输入粗单元内 m=L*L 个细单元密度, 返回子结构缩聚算子."""

    def __init__(self, template: SubstructureTemplate):
        self.template = template

    @abstractmethod
    def predict(self, rho_local: np.ndarray) -> SubstructureOperator:
        ...


class ExactPredictor(MultiscalePredictor):
    """精确静力缩聚 (基线 / 对照真值 / 训练标注来源)."""

    def predict(self, rho_local: np.ndarray) -> SubstructureOperator:
        return self.template.condense(rho_local)


class MockPredictor(MultiscalePredictor):
    """解析映射 mock: 均匀密度缩放.

    K^j(rho) 对 rho 线性, 故均匀局部密度 rho_local = c*1 时
    K_s^j(c*1) = c * K_s^j(1) 且 X 不变 (精确); 非均匀输入时以
    均值缩放作近似. 仅用于打通接口与验证预测器互换性.
    """

    def __init__(self, template: SubstructureTemplate):
        super().__init__(template)
        m = template.local_c2d.shape[0]
        self._unit_op = template.condense(np.ones(m))

    def predict(self, rho_local: np.ndarray) -> SubstructureOperator:
        c = float(np.mean(rho_local))
        return SubstructureOperator(
            X=self._unit_op.X,
            K_s=c * self._unit_op.K_s,
            lu_ii=None,  # 缩放后的 K_ii 分解不再有效, mock 不支持内部载荷恢复
        )
