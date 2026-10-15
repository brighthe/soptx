# 制造解问题

有精确解的线弹性问题：体力由精确位移反推，离散误差是唯一的误差来源，收敛阶可以直接作为判据。实现位于 `soptx.problems.manufactured`，按精确解分文件，2D 与 3D 同解放一处；公开类从 `soptx.problems` 导入，每个类一份模型文档，索引见[上级 README](../README.md)。

## 通用方程

在轴对齐区域 $\Omega\subset\mathbb{R}^d$ 上求位移 $u:\Omega\rightarrow\mathbb{R}^d$：

$$
-\nabla\cdot\sigma(u)=b,\qquad
\varepsilon(u)=\frac12(\nabla u+\nabla u^\mathsf{T}),\qquad
\sigma=2\mu\varepsilon+\lambda\operatorname{tr}(\varepsilon)I.
$$

体力 $b$ 一律由精确位移反推（$b=-\nabla\cdot\sigma(u_\mathrm{exact})$）。材料参数统一以 Lamé 常数 $\lambda$、$\mu$ 给出（`lam`、`mu`），部分类另提供换算后的 `E`、`nu`。

## 边界条件

### 边界划分

边界 $\partial\Omega$ 分成位移部分 $\Gamma_D$ 与牵引部分 $\Gamma_N$：

$$
u=u_\mathrm{exact}\ \text{ on }\Gamma_D,\qquad
\sigma(u)\cdot n=t\ \text{ on }\Gamma_N,\qquad
\Gamma_D\cup\Gamma_N=\partial\Omega,\quad
\Gamma_D\cap\Gamma_N=\varnothing.
$$

各模型的划分由类属性 `boundary_type` 声明，$\Gamma_N$ 为空即全 Dirichlet：

| 模型 | `boundary_type` | $\Gamma_D$ | $\Gamma_N$ |
| --- | --- | --- | --- |
| `ExponentialSineManufacturedElasticity2D` | `dirichlet` | 四条边全部 | 空 |
| `SinusoidalElasticity2D` | `dirichlet` | 四条边全部 | 空 |
| `HarmonicPoly2D` | `dirichlet` | 四条边全部 | 空 |
| `DivergenceFreePolynomialElasticity3D` | `dirichlet` | 六个面全部 | 空 |
| `HarmonicPoly3D` | `dirichlet` | 六个面全部 | 空 |
| `MixedBoundaryExponentialSineElasticity2D` | `mixed` | $x=0$、$y=0$、$y=1$ | $x=1$ |
| `MixedBoundarySinusoidalElasticity2D` | `mixed` | $x=0$、$y=0$ | $x=1$、$y=1$ |

边界判定一律是坐标到区域边界值的距离比较，绝对容差 `GEOMETRIC_TOL = 1e-12`（定义在 `soptx.core.numerics`，问题类以 `_eps` 持有）：$x=0$ 这条边对应 `bm.abs(x - domain[0]) < self._eps`。

### 齐次与非齐次 Dirichlet 模型的边界值

- **齐次边界模型**（`ExponentialSineManufacturedElasticity2D`、`SinusoidalElasticity2D`、`DivergenceFreePolynomialElasticity3D`）：精确位移在整个边界上恒为零，前两个靠 $\sin(\pi x)\sin(\pi y)$ 与 $x(1-x)y(1-y)$ 在边上取零，第三个靠 $\phi(0)=\phi(1)=\psi(0)=\psi(1)=0$。接口上统一采用非齐次写法：`dirichlet_bc` 返回 `disp_solution(points)` 而不是常零，使调用方代码结构保持统一。
- **非齐次边界模型**（`HarmonicPoly2D`、`HarmonicPoly3D`）：精确位移在边界上非零，整个弹性变形由边界给定位移驱动（连续体力 $b \equiv 0$），用于直接检验非零本质边界条件下的位移传递与收敛阶。

### 强施加与弱施加：同一条边界，两种角色

两类有限元形式对边界的处理正好相反，Problem 因此提供两套成员：

| | 位移元（`LagrangeFEMAnalyzer`） | Hu--Zhang 混合元（`HuZhangMFEMAnalyzer`） |
| --- | --- | --- |
| $\Gamma_D$ 上的位移 | **强施加**（本质边界，对称消元） | **弱施加**（进入变分形式） |
| $\Gamma_N$ 上的牵引 | 弱施加（自然边界，边界积分） | **强施加**（约束应力空间自由度） |
| 边界谓词 | `is_dirichlet_boundary()` | `is_displacement_boundary()` / `is_traction_boundary()` |
| 边界数据 | `dirichlet_bc` | `displacement_bc`；物理牵引由 `loads()` 提供 |

位移在混合形式下变成自然边界、牵引变成本质边界，这是混合元的固有特性，不是实现选择。

`is_dirichlet_boundary()` 返回的是逐位移分量的谓词元组（2D 两个、3D 三个），允许各分量的位移边界不同；现有模型的各分量判定相同。`is_displacement_boundary()` 与 `is_traction_boundary()` 则是单个作用于点的谓词，不区分分量。

### 全位移边界模型在混合形式下的默认实现

`AllDisplacementBoundaryMixin`（`manufactured/_all_dirichlet.py`，2D 与 3D 模型共用）为全位移边界模型补齐混合形式所需的成员，使它们无需适配层即满足 `MixedBoundaryElasticityProblem` 协议：

- `is_displacement_boundary` 恒为 `True`、`is_traction_boundary` 恒为 `False`；
- `displacement_bc` 复用 `dirichlet_bc`，保证弱施加数据与强施加完全一致；
- `loads()` 只返回 `BodyForceLoad`，不构造虚假的零牵引对象。

### 牵引数据的形式

两个 `mixed` 模型的 `loads()` 均返回 `BoundaryTractionLoad`。其 `traction()` 直接给出法向牵引向量 $\boldsymbol{t}=\boldsymbol{\sigma}\boldsymbol{n}$；`stress_solution` 仍单独提供完整应力的 Voigt 向量 $(\sigma_{xx},\sigma_{xy},\sigma_{yy})$。

## 返回 shape

对输入 shape `(..., d)`：

- `disp_solution`、`dirichlet_bc`、`displacement_bc` 以及 `BodyForce.body_force()` 返回 `(..., d)`；
- `grad_disp_solution` 返回 `(..., d, d)`，最后两维为 $(\partial u_i/\partial x_j)$；
- `stress_solution` 返回 `(..., 3)`，分量顺序为 $(\sigma_{xx},\sigma_{xy},\sigma_{yy})$；`BoundaryTraction.traction()` 返回 `(..., 2)`；
- `is_dirichlet_boundary()` 返回每个位移分量对应的边界谓词；
- `is_displacement_boundary`、`is_traction_boundary` 返回 `(...)` 的布尔量。

## Problem、Material 与 Mesh 组合

数学问题与材料参数必须显式一致，但对象彼此独立：

```python
from soptx.mesh import TriangleMesh
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import SinusoidalElasticity2D

problem = SinusoidalElasticity2D()
material = IsotropicLinearElasticMaterial(
    youngs_modulus=problem.E,
    poisson_ratio=problem.nu,
    hypothesis="plane_strain",
)
mesh = TriangleMesh.from_box(list(problem.domain), nx=16, ny=16)
```
