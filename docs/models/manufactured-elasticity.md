# 制造解线弹性模型

本文档说明 SOPTX 当前活跃制造解的方程、材料假设、边界条件和返回 shape。
实现位于 `soptx.problems.elasticity`；Problem 不创建网格，也不保存 Material 对象。

## 通用方程

在轴对齐区域 $\Omega\subset\mathbb{R}^d$ 上求位移
$u:\Omega\rightarrow\mathbb{R}^d$：

$$
-\nabla\cdot\sigma(u)=b,\qquad
\varepsilon(u)=\frac12(\nabla u+\nabla u^\mathsf{T}),\qquad
\sigma=2\mu\varepsilon+\lambda\operatorname{tr}(\varepsilon)I.
$$

体力 $b$ 一律由精确位移反推（$b=-\nabla\cdot\sigma(u_\mathrm{exact})$），因此
离散误差是唯一的误差来源，收敛阶可以直接作为判据。

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
| `SinusoidalPlaneStrainElasticity2D` | `dirichlet` | 四条边全部 | 空 |
| `DivergenceFreePolynomialElasticity3D` | `dirichlet` | 六个面全部 | 空 |
| `MixedBoundaryExponentialSineElasticity2D` | `mixed` | $x=0$、$y=0$、$y=1$ | $x=1$ |
| `MixedBoundarySinusoidalElasticity2D` | `mixed` | $x=0$、$y=0$ | $x=1$、$y=1$ |

边界判定一律是坐标到区域边界值的距离比较，容差 `_eps = 1e-12`：$x=0$ 这条边
对应 `bm.abs(x - domain[0]) < 1e-12`。

### 全 Dirichlet 模型的边界值在数值上是零

三个 `dirichlet` 模型的精确位移在整个边界上**恒为零**——前两个靠
$\sin(\pi x)\sin(\pi y)$ 与 $x(1-x)y(1-y)$ 在边上取零，第三个靠
$\phi(0)=\phi(1)=\psi(0)=\psi(1)=0$。

但接口上是**非齐次写法**：`dirichlet_bc` 返回 `disp_solution(points)` 而不是
常零，这样换一个边界上不为零的制造解时无需改动调用方。代价是这三个模型
验证不到"非零 Dirichlet 数据"这条路径——它取到的值就是零。

### 强施加与弱施加：同一条边界，两种角色

两类有限元形式对边界的处理正好相反，Problem 因此提供两套成员：

| | 位移元（`LagrangeFEMAnalyzer`） | Hu–Zhang 混合元（`HuZhangMFEMAnalyzer`） |
| --- | --- | --- |
| $\Gamma_D$ 上的位移 | **强施加**（本质边界，对称消元） | **弱施加**（进入变分形式） |
| $\Gamma_N$ 上的牵引 | 弱施加（自然边界，边界积分） | **强施加**（约束应力空间自由度） |
| 边界谓词 | `is_dirichlet_boundary()` | `is_displacement_boundary()` / `is_traction_boundary()` |
| 边界数据 | `dirichlet_bc` | `displacement_bc` / `traction_bc` |

位移在混合形式下变成自然边界、牵引变成本质边界，这是混合元的固有特性，
不是实现选择。

`is_dirichlet_boundary()` 返回的是**逐位移分量**的谓词元组（2D 两个、
3D 三个），允许各分量的位移边界不同；现有模型的各分量判定相同。
`is_displacement_boundary()` 与 `is_traction_boundary()` 则是单个作用于点的
谓词，不区分分量。

### 全 Dirichlet 模型在混合形式下的默认实现

`AllDisplacementBoundaryMixin`（`_base.py`）为三个 `dirichlet` 模型补齐混合
形式所需的成员，使它们无需适配层即满足 `MixedBoundaryElasticityProblem`
协议：

- `is_displacement_boundary` 恒为 `True`、`is_traction_boundary` 恒为 `False`；
- `displacement_bc` 复用 `dirichlet_bc`，保证弱施加数据与强施加完全一致；
- `traction_bc` **抛 `NotImplementedError`** 而不是返回零。$\Gamma_N$ 为空，
  正确的调用路径到不了这里；返回零会把"没有牵引数据"伪装成"牵引为零"，
  属于静默出错。

### 牵引数据的形式

两个 `mixed` 模型的 `traction_bc` 返回**完整应力的 Voigt 向量**
$(\sigma_{xx},\sigma_{xy},\sigma_{yy})$，即 `stress_solution` 的值，
**不是**法向牵引 $\sigma\cdot n$。法向投影由 Hu–Zhang 应力空间在牵引边界上
完成，调用方不要自行点乘法向。

## 返回 shape

对输入 shape `(..., d)`：

- `disp_solution`、`body_force`、`dirichlet_bc`、`displacement_bc`
  返回 `(..., d)`；
- `grad_disp_solution` 返回 `(..., d, d)`，最后两维为
  $(\partial u_i/\partial x_j)$；
- `stress_solution`、`traction_bc`（2D 混合模型）返回 `(..., 3)`，
  分量顺序为 $(\sigma_{xx},\sigma_{xy},\sigma_{yy})$；
- `is_dirichlet_boundary()` 返回每个位移分量对应的边界谓词；
- `is_displacement_boundary`、`is_traction_boundary` 返回 `(...)` 的布尔量。

## ExponentialSineManufacturedElasticity2D

区域为 $[0,1]^2$，采用 plane strain，默认
$\lambda=1,\mu=\tfrac12$。精确位移是

$$
u_1=e^{x-y}x(1-x)y(1-y),\qquad
u_2=\sin(\pi x)\sin(\pi y).
$$

由 $b=-\nabla\cdot\sigma(u)$ 得

$$
\begin{aligned}
b_1={}&2(x^2+3x)(y-y^2)e^{x-y}
-\tfrac12(x-x^2)(-y^2+5y-4)e^{x-y}\\
&-\tfrac32\pi^2\cos(\pi x)\cos(\pi y),\\
b_2={}&-\tfrac32(1-x-x^2)(1-3y+y^2)e^{x-y}
+\tfrac52\pi^2\sin(\pi x)\sin(\pi y).
\end{aligned}
$$

两个位移分量在整个边界上均为零。

## MixedBoundaryExponentialSineElasticity2D

继承 `ExponentialSineManufacturedElasticity2D`，精确位移、体力和材料参数
完全相同，只改变边界划分：左、下、上三条边为位移边界，右边为牵引边界。

$$
\Gamma_D=\{x=0\}\cup\{y=0\}\cup\{y=1\},\qquad
\Gamma_N=\{x=1\}.
$$

精确位移在整个边界上为零，所以 $\Gamma_D$ 上的位移数据是零，而 $\Gamma_N$
上的牵引数据非零。同一组精确场因此能检验非零 traction，不必引入第二套公式
来源。`traction_bc` 返回 `stress_solution` 的 Voigt 向量。

## MixedBoundarySinusoidalElasticity2D

该模型对应 Hu–Zhang 论文制造解收敛验证。区域固定为 $[0,1]^2$，采用
plane strain，默认 $\lambda=1,\mu=\tfrac12$。记

$$
s=\sin(\pi x)\sin(\pi y),\qquad
a=\pi\cos(\pi x)\sin(\pi y),\qquad
c=\pi\sin(\pi x)\cos(\pi y),
$$

精确位移与应力分别为

$$
u=(s,s)^\mathsf{T},
$$

$$
\sigma=
\begin{pmatrix}
(\lambda+2\mu)a+\lambda c & \mu(a+c)\\
\mu(a+c) & \lambda a+(\lambda+2\mu)c
\end{pmatrix}.
$$

应力散度的两个分量相同：

$$
(\nabla\cdot\sigma)_1=(\nabla\cdot\sigma)_2
=\pi^2\left[(\lambda+\mu)\cos(\pi x)\cos(\pi y)
-(\lambda+3\mu)\sin(\pi x)\sin(\pi y)\right],
$$

体力取 $b=-\nabla\cdot\sigma$。边界划分为

$$
\Gamma_D=\{x=0\}\cup\{y=0\},\qquad
\Gamma_N=\{x=1\}\cup\{y=1\}.
$$

$\Gamma_D$ 上施加齐次位移，$\Gamma_N$ 上施加精确应力法向迹。Problem 的
`traction_bc` 返回 Voigt 顺序 `[sigma_xx, sigma_xy, sigma_yy]`，由
Hu–Zhang 应力空间完成法向投影。

## SinusoidalPlaneStrainElasticity2D

区域为 $[0,1]^2$，固定使用 plane strain，默认 $E=1,\nu=0.3$，因此
$\mu=5/13$、$\lambda=15/26$。精确位移是

$$
u=(\sin(\pi x)\sin(\pi y),0)^\mathsf{T},
$$

体力为

$$
b=\left(
\frac{22.5}{13}\pi^2\sin(\pi x)\sin(\pi y),
-\frac{12.5}{13}\pi^2\cos(\pi x)\cos(\pi y)
\right)^\mathsf{T}.
$$

第一分量在整个边界上为零，第二分量恒为零。

## DivergenceFreePolynomialElasticity3D

区域为 $[0,1]^3$，默认 $\lambda=\mu=1$。定义

$$
\phi(t)=(t-t^2)^2,\qquad
\psi(t)=2t^3-3t^2+t.
$$

精确位移是

$$
u=\left(
200\mu\phi(x)\psi(y)\psi(z),
-100\mu\phi(y)\psi(x)\psi(z),
-100\mu\phi(z)\psi(x)\psi(y)
\right)^\mathsf{T}.
$$

直接求导可得 $\nabla\cdot u=0$，故

$$
-\nabla\cdot\sigma(u)=-\mu\Delta u=b,
$$

实现使用该式展开后的多项式体力。由于 $\phi(0)=\phi(1)=0$ 且
$\psi(0)=\psi(1)=0$，三个分量在整个边界为零。

## Problem、Material 与 Mesh 组合

数学问题与材料参数必须显式一致，但对象彼此独立：

```python
from fealpy.mesh import TriangleMesh
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import SinusoidalPlaneStrainElasticity2D

problem = SinusoidalPlaneStrainElasticity2D()
material = IsotropicLinearElasticMaterial(
    youngs_modulus=problem.E,
    poisson_ratio=problem.nu,
    hypothesis="plane_strain",
)
mesh = TriangleMesh.from_box(list(problem.domain), nx=16, ny=16)
```
