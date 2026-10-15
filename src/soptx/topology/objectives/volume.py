"""体积分数目标.

体积与体积梯度由 ``soptx.topology.optimizers.utils`` 的 ``compute_volume`` 与
``compute_volume_gradient`` 计算, 与 ``VolumeConstraint`` 共用; 本类只负责归一化.
"""

from typing import Dict, Literal, Optional, Union

from soptx.backend import backend_manager as bm
from soptx.core import BaseLogged
from soptx.fem.analyzers import HuZhangMFEMAnalyzer, LagrangeFEMAnalyzer
from soptx.functionspace import Function
from soptx.topology.optimizers.utils import compute_volume, compute_volume_gradient
from soptx.typing import TensorLike


class VolumeObjective(BaseLogged):
    """归一化体积分数目标 f = V(rho) / V_0.

    Parameters
    ----------
    analyzer : LagrangeFEMAnalyzer or HuZhangMFEMAnalyzer
        提供网格、插值格式与积分阶的分析器.
    diff_mode : {'manual', 'auto'}, optional
        灵敏度计算方式, 默认 'manual'. 体积对密度线性, 手动梯度即精确, 不提供自动微分;
        取 'auto' 时 ``jac`` 抛 NotImplementedError.
    enable_logging : bool, optional
        是否启用日志.
    logger_name : str, optional
        日志记录器名称.

    Notes
    -----
    ``fun`` 把当前体积缓存在 ``_v``, AL-MMA 优化器读取 ``_v`` 与 ``_v0`` 报告体积分数.
    """

    def __init__(
        self,
        analyzer: Union[LagrangeFEMAnalyzer, HuZhangMFEMAnalyzer],
        diff_mode: Literal["manual", "auto"] = "manual",
        enable_logging: bool = False,
        logger_name: Optional[str] = None,
    ) -> None:
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._analyzer = analyzer
        self._diff_mode = diff_mode

        # 体积目标只用到分析器的网格、插值格式和积分阶, 两类分析器都满足
        self._mesh = analyzer._mesh
        self._integration_order = analyzer._integration_order
        self._density_location = analyzer._interpolation_scheme._density_location

        self._v0 = bm.sum(self._mesh.entity_measure('cell'))
        self._v = None

    def fun(
        self,
        density: Union[Function, TensorLike],
        state: Optional[Dict] = None,
        **kwargs,
    ) -> float:
        """计算目标值 V(rho) / V_0.

        Parameters
        ----------
        density : Function or TensorLike
            物理密度.
        state : dict, optional
            求解状态缓存, 本方法不用到, 保留以统一签名.

        Returns
        -------
        float
            归一化体积分数.
        """
        self._v = compute_volume(density=density, mesh=self._mesh,
                                 density_location=self._density_location,
                                 integration_order=self._integration_order)
        return self._v / self._v0

    def jac(
        self,
        density: Union[Function, TensorLike],
        state: Optional[Dict] = None,
        diff_mode: Optional[Literal["manual", "auto"]] = None,
        **kwargs,
    ) -> TensorLike:
        """计算目标对物理密度的梯度 dV/drho / V_0.

        Parameters
        ----------
        density : Function or TensorLike
            物理密度.
        state : dict, optional
            求解状态缓存, 本方法不用到, 保留以统一签名.
        diff_mode : {'manual', 'auto'}, optional
            覆盖构造时的 ``diff_mode``.

        Returns
        -------
        TensorLike
            与密度自由度同形的梯度.

        Raises
        ------
        NotImplementedError
            ``diff_mode`` 为 'auto', 或密度表示不受支持.
        ValueError
            ``diff_mode`` 取值未知.
        """
        mode = diff_mode if diff_mode is not None else self._diff_mode
        if mode == "auto":
            raise NotImplementedError("体积对密度线性, 手动梯度即精确, 不提供自动微分")
        if mode != "manual":
            raise ValueError(f"未知的 diff_mode: {mode!r}")
        return compute_volume_gradient(density=density, mesh=self._mesh,
                                       density_location=self._density_location,
                                       integration_order=self._integration_order) / self._v0
