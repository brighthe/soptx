"""二维制造解线弹性问题.

完整方程与推导见 ``docs/models/manufactured-elasticity.md``。
"""

from __future__ import annotations

from math import isfinite
from typing import Callable, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

from ._base import AllDisplacementBoundaryMixin, validated_domain


class _AllDirichletElasticity2D(AllDisplacementBoundaryMixin):
    dimension = 2
    plane_type = "plane_strain"
    boundary_type = "dirichlet"
    load_type = None
    _eps = 1.0e-12

    def __init__(self, domain: Sequence[float]) -> None:
        self._domain = validated_domain(domain, self.dimension)

    @property
    def domain(self) -> tuple[float, ...]:
        return self._domain

    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        return self.disp_solution(points)

    @cartesian
    def is_dirichlet_boundary_dof_x(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        domain = self.domain
        return (
            (bm.abs(x - domain[0]) < self._eps)
            | (bm.abs(x - domain[1]) < self._eps)
            | (bm.abs(y - domain[2]) < self._eps)
            | (bm.abs(y - domain[3]) < self._eps)
        )

    @cartesian
    def is_dirichlet_boundary_dof_y(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.is_dirichlet_boundary_dof_x(points)

    def is_dirichlet_boundary(
        self,
    ) -> tuple[Callable, Callable]:
        return (
            self.is_dirichlet_boundary_dof_x,
            self.is_dirichlet_boundary_dof_y,
        )


class ExponentialSineManufacturedElasticity2D(
    _AllDirichletElasticity2D
):
    r"""指数/正弦位移的 plane strain 制造解问题.

    .. math:: -\nabla\cdot\sigma(u)=b,\quad
       u=(e^{x-y}x(1-x)y(1-y),\sin(\pi x)\sin(\pi y)).
    """

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 0.5,
    ) -> None:
        super().__init__(domain)
        if not isfinite(lame_lambda):
            raise ValueError("lame_lambda must be finite")
        if not isfinite(shear_modulus) or shear_modulus <= 0.0:
            raise ValueError("shear_modulus must be finite and positive")
        self._lam = float(lame_lambda)
        self._mu = float(shear_modulus)

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        return -self.div_stress_solution(points)

    @cartesian
    def div_stress_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)
        pi = bm.pi
        deps_xx_dx = (
            exp_xy * y * (1 - y) * (-x**2 - 3 * x)
        )
        deps_xx_dy = (
            exp_xy
            * (1 - x - x**2)
            * (1 - 3 * y + y**2)
        )
        deps_yy_dx = pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        deps_yy_dy = -pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        deps_xy_dx = 0.5 * (
            exp_xy
            * (1 - 3 * y + y**2)
            * (1 - x - x**2)
            - pi**2 * bm.sin(pi * x) * bm.sin(pi * y)
        )
        deps_xy_dy = 0.5 * (
            exp_xy * x * (1 - x) * (-4 + 5 * y - y**2)
            + pi**2 * bm.cos(pi * x) * bm.cos(pi * y)
        )
        div_x = (
            (self.lam + 2 * self.mu) * deps_xx_dx
            + self.lam * deps_yy_dx
            + 2 * self.mu * deps_xy_dy
        )
        div_y = (
            2 * self.mu * deps_xy_dx
            + self.lam * deps_xx_dy
            + (self.lam + 2 * self.mu) * deps_yy_dy
        )
        return bm.stack([div_x, div_y], axis=-1)

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return bm.stack(
            [
                bm.exp(x - y) * x * (1 - x) * y * (1 - y),
                bm.sin(bm.pi * x) * bm.sin(bm.pi * y),
            ],
            axis=-1,
        )

    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        exp_xy = bm.exp(x - y)
        du1_dx = exp_xy * y * (1 - y) * (1 - x - x**2)
        du1_dy = exp_xy * x * (1 - x) * (1 - 3 * y + y**2)
        du2_dx = (
            bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y)
        )
        du2_dy = (
            bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y)
        )
        return bm.stack(
            [
                bm.stack([du1_dx, du1_dy], axis=-1),
                bm.stack([du2_dx, du2_dy], axis=-1),
            ],
            axis=-2,
        )

    def disp_solution_gradient(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.grad_disp_solution(points)

    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
        gradient = self.grad_disp_solution(points)
        eps_xx = gradient[..., 0, 0]
        eps_yy = gradient[..., 1, 1]
        eps_xy = 0.5 * (
            gradient[..., 0, 1] + gradient[..., 1, 0]
        )
        trace = eps_xx + eps_yy
        return bm.stack(
            [
                self.lam * trace + 2 * self.mu * eps_xx,
                2 * self.mu * eps_xy,
                self.lam * trace + 2 * self.mu * eps_yy,
            ],
            axis=-1,
        )


class MixedBoundaryExponentialSineElasticity2D(
    ExponentialSineManufacturedElasticity2D
):
    """指数/正弦制造解问题的混合边界视角.

    精确位移在单位正方形边界上为零。三条边按位移边界处理, 右边按非零
    traction 边界处理, 于是同一组精确场就能检验强 traction 数据, 而不必
    引入第二套公式来源。
    """

    boundary_type = "mixed"
    load_type = "distributed"

    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        domain = self.domain
        return (
            (bm.abs(x - domain[0]) < self._eps)
            | (bm.abs(y - domain[2]) < self._eps)
            | (bm.abs(y - domain[3]) < self._eps)
        )

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """位移元强施加的边界只有 :math:`\\Gamma_D`.

        基类按全 Dirichlet 判定四条边。混合边界必须收窄到 :math:`\\Gamma_D`,
        否则 ``apply_bc`` 会把 :math:`\\Gamma_N` 的自由度也强加位移, 于是上一
        步刚加进右端项的 traction 载荷被整个覆盖 —— 而且因为强加的位移取自
        精确解, 结果依然正确、收敛阶依然是 2, 属于不报错的静默失效。

        ``is_dirichlet_boundary_dof_y`` 由基类转调本方法, 不必重复覆盖。
        """

        return self.is_displacement_boundary(points)

    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        return bm.abs(points[..., 0] - self.domain[1]) < self._eps

    def traction_bc(self, points: TensorLike) -> TensorLike:
        return self.stress_solution(points)

    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """位移元视角的自然边界数据: 法向迹 :math:`t=\\sigma\\cdot n`.

        与 ``traction_bc`` 是同一份精确应力的两种形式。混合形式要完整应力,
        由 Hu--Zhang 应力空间自行投影; 位移元的边界积分要的是已经点乘过法向
        的牵引向量。区域轴对齐, :math:`\\Gamma_N` 只有右边一条, 外法向恒为
        :math:`(1,0)`, 于是 :math:`t=(\\sigma_{xx},\\sigma_{xy})`。
        """

        stress = self.stress_solution(points)
        x = points[..., 0]

        val = bm.zeros(points.shape, **bm.context(points))
        flag_right = bm.abs(x - self.domain[1]) < self._eps
        val = bm.set_at(val, (flag_right, 0), stress[..., 0][flag_right])
        val = bm.set_at(val, (flag_right, 1), stress[..., 1][flag_right])
        return val

    def is_neumann_boundary(self) -> Callable:
        """位移元路径的牵引边界谓词.

        与 ``is_traction_boundary`` 是同一条边界, 只是位移元路径按
        ``is_neumann_boundary()`` 这个名字查找。
        """

        return self.is_traction_boundary


class MixedBoundarySinusoidalElasticity2D(
    _AllDirichletElasticity2D
):
    r"""论文制造解收敛验证使用的 plane strain 混合边界问题.

    .. math::
       u_1=u_2=\sin(\pi x)\sin(\pi y),\qquad
       \Gamma_D=\{x=0\}\cup\{y=0\},\quad
       \Gamma_N=\{x=1\}\cup\{y=1\}.

    区域固定为单位正方形. ``traction_bc`` 返回精确应力的 Voigt 向量，
    由 Hu--Zhang 应力空间在牵引边界上投影为法向迹。
    """

    dimension = 2
    plane_type = "plane_strain"
    boundary_type = "mixed"
    load_type = "distributed"

    def __init__(
        self,
        *,
        lame_lambda: float = 1.0,
        shear_modulus: float = 0.5,
    ) -> None:
        super().__init__((0.0, 1.0, 0.0, 1.0))
        if not isfinite(lame_lambda):
            raise ValueError("lame_lambda must be finite")
        if not isfinite(shear_modulus) or shear_modulus <= 0.0:
            raise ValueError("shear_modulus must be finite and positive")
        self._lam = float(lame_lambda)
        self._mu = float(shear_modulus)

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        value = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
        return bm.stack([value, value], axis=-1)

    @cartesian
    def grad_disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        derivative_x = (
            bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y)
        )
        derivative_y = (
            bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y)
        )
        row = bm.stack([derivative_x, derivative_y], axis=-1)
        return bm.stack([row, row], axis=-2)

    def disp_solution_gradient(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.grad_disp_solution(points)

    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
        gradient = self.grad_disp_solution(points)
        derivative_x = gradient[..., 0, 0]
        derivative_y = gradient[..., 0, 1]
        sigma_xx = (
            (self.lam + 2 * self.mu) * derivative_x
            + self.lam * derivative_y
        )
        sigma_xy = self.mu * (derivative_x + derivative_y)
        sigma_yy = (
            self.lam * derivative_x
            + (self.lam + 2 * self.mu) * derivative_y
        )
        return bm.stack([sigma_xx, sigma_xy, sigma_yy], axis=-1)

    @cartesian
    def div_stress_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        sine_product = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
        cosine_product = bm.cos(bm.pi * x) * bm.cos(bm.pi * y)
        value = bm.pi**2 * (
            (self.lam + self.mu) * cosine_product
            - (self.lam + 3 * self.mu) * sine_product
        )
        return bm.stack([value, value], axis=-1)

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        return -self.div_stress_solution(points)

    def is_displacement_boundary(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return (bm.abs(x) < self._eps) | (bm.abs(y) < self._eps)

    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        """位移元强施加的边界只有 :math:`\\Gamma_D`.

        基类按全 Dirichlet 判定四条边。混合边界必须收窄到 :math:`\\Gamma_D`,
        否则 ``apply_bc`` 会把 :math:`\\Gamma_N` 的自由度也强加位移, 于是上一
        步刚加进右端项的 traction 载荷被整个覆盖 —— 而且因为强加的位移取自
        精确解, 结果依然正确、收敛阶依然是 2, 属于不报错的静默失效。

        ``is_dirichlet_boundary_dof_y`` 由基类转调本方法, 不必重复覆盖。
        """

        return self.is_displacement_boundary(points)

    def is_traction_boundary(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return (
            (bm.abs(x - 1.0) < self._eps)
            | (bm.abs(y - 1.0) < self._eps)
        )

    @cartesian
    def traction_bc(self, points: TensorLike) -> TensorLike:
        return self.stress_solution(points)

    @cartesian
    def neumann_bc(self, points: TensorLike) -> TensorLike:
        """位移元视角的自然边界数据: 法向迹 :math:`t=\\sigma\\cdot n`.

        与 ``traction_bc`` 是同一份精确应力的两种形式。混合形式要完整应力,
        由 Hu--Zhang 应力空间自行投影; 位移元的边界积分要的是已经点乘过法向
        的牵引向量。区域轴对齐, :math:`\\Gamma_N` 的两条边上外法向分别是
        :math:`(1,0)` 与 :math:`(0,1)`, 于是

        .. math::
           t|_{x=1}=(\\sigma_{xx},\\sigma_{xy}),\\qquad
           t|_{y=1}=(\\sigma_{xy},\\sigma_{yy}).

        角点 :math:`(1,1)` 同时落在两条边上, 这里由后写的上边覆盖。面积分点
        取在面内部, 落不到角点上, 因此这个选择不影响装配结果。
        """

        stress = self.stress_solution(points)
        x, y = points[..., 0], points[..., 1]

        val = bm.zeros(points.shape, **bm.context(points))

        flag_right = bm.abs(x - 1.0) < self._eps
        val = bm.set_at(val, (flag_right, 0), stress[..., 0][flag_right])
        val = bm.set_at(val, (flag_right, 1), stress[..., 1][flag_right])

        flag_top = bm.abs(y - 1.0) < self._eps
        val = bm.set_at(val, (flag_top, 0), stress[..., 1][flag_top])
        val = bm.set_at(val, (flag_top, 1), stress[..., 2][flag_top])

        return val

    def is_neumann_boundary(self) -> Callable:
        """位移元路径的牵引边界谓词.

        与 ``is_traction_boundary`` 是同一条边界, 只是位移元路径按
        ``is_neumann_boundary()`` 这个名字查找。
        """

        return self.is_traction_boundary


class SinusoidalPlaneStrainElasticity2D(
    _AllDirichletElasticity2D
):
    r"""单位正方形上的 plane strain 制造解位移问题.

    .. math:: u=(\sin(\pi x)\sin(\pi y),0),\quad
       -\nabla\cdot\sigma(u)=b.
    """

    def __init__(
        self,
        domain: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
        *,
        youngs_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
    ) -> None:
        super().__init__(domain)
        youngs_modulus = float(youngs_modulus)
        poisson_ratio = float(poisson_ratio)
        if not isfinite(youngs_modulus) or youngs_modulus <= 0.0:
            raise ValueError("youngs_modulus must be finite and positive")
        if (
            not isfinite(poisson_ratio)
            or not -1.0 < poisson_ratio < 0.5
        ):
            raise ValueError(
                "poisson_ratio must be finite and satisfy -1 < nu < 0.5"
            )
        self._E = youngs_modulus
        self._nu = poisson_ratio
        self._mu = youngs_modulus / (2 * (1 + poisson_ratio))
        self._lam = (
            youngs_modulus
            * poisson_ratio
            / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        )

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def mu(self) -> float:
        return self._mu

    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        pi = bm.pi
        return bm.stack(
            [
                (self.lam + 3 * self.mu)
                * pi**2
                * bm.sin(pi * x)
                * bm.sin(pi * y),
                -(self.lam + self.mu)
                * pi**2
                * bm.cos(pi * x)
                * bm.cos(pi * y),
            ],
            axis=-1,
        )

    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        return bm.stack(
            [
                bm.sin(bm.pi * x) * bm.sin(bm.pi * y),
                bm.zeros_like(x),
            ],
            axis=-1,
        )

    @cartesian
    def disp_solution_gradient(
        self,
        points: TensorLike,
    ) -> TensorLike:
        x, y = points[..., 0], points[..., 1]
        du_x_dx = (
            bm.pi * bm.cos(bm.pi * x) * bm.sin(bm.pi * y)
        )
        du_x_dy = (
            bm.pi * bm.sin(bm.pi * x) * bm.cos(bm.pi * y)
        )
        zero = bm.zeros_like(x)
        return bm.stack(
            [
                bm.stack([du_x_dx, du_x_dy], axis=-1),
                bm.stack([zero, zero], axis=-1),
            ],
            axis=-2,
        )

    def grad_disp_solution(
        self,
        points: TensorLike,
    ) -> TensorLike:
        return self.disp_solution_gradient(points)
