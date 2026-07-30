# 制造解线弹性模型

本文档说明 SOPTX 当前三个活跃制造解的方程、材料假设、边界条件和返回 shape。
实现位于 `soptx.problems.elasticity`；Problem 不创建网格，也不保存 Material 对象。

## 通用方程

在轴对齐区域 \(\Omega\subset\mathbb{R}^d\) 上求位移
\(u:\Omega\rightarrow\mathbb{R}^d\)：

\[
-\nabla\cdot\sigma(u)=b,\qquad
\varepsilon(u)=\frac12(\nabla u+\nabla u^\mathsf{T}),\qquad
\sigma=2\mu\varepsilon+\lambda\operatorname{tr}(\varepsilon)I.
\]

三个模型均采用全 Dirichlet 边界：

\[
u=u_\mathrm{exact}\quad\text{on }\partial\Omega.
\]

对输入 shape `(..., d)`：

- `disp_solution`、`body_force`、`dirichlet_bc` 返回 `(..., d)`；
- `grad_disp_solution` 返回 `(..., d, d)`，最后两维为
  \((\partial u_i/\partial x_j)\)；
- `is_dirichlet_boundary()` 返回每个位移分量对应的边界谓词。

## ExponentialSineManufacturedElasticity2D

区域为 \([0,1]^2\)，采用 plane strain，默认
\(\lambda=1,\mu=\tfrac12\)。精确位移是

\[
u_1=e^{x-y}x(1-x)y(1-y),\qquad
u_2=\sin(\pi x)\sin(\pi y).
\]

由 \(b=-\nabla\cdot\sigma(u)\) 得

\[
\begin{aligned}
b_1={}&2(x^2+3x)(y-y^2)e^{x-y}
-\tfrac12(x-x^2)(-y^2+5y-4)e^{x-y}\\
&-\tfrac32\pi^2\cos(\pi x)\cos(\pi y),\\
b_2={}&-\tfrac32(1-x-x^2)(1-3y+y^2)e^{x-y}
+\tfrac52\pi^2\sin(\pi x)\sin(\pi y).
\end{aligned}
\]

两个位移分量在整个边界上均为零。该模型替代旧名
`TriSolHomoDirHuZhang2d`，供 PINN 制造解验证使用。

## SinusoidalPlaneStrainElasticity2D

区域为 \([0,1]^2\)，固定使用 plane strain，默认 \(E=1,\nu=0.3\)，因此
\(\mu=5/13\)、\(\lambda=15/26\)。精确位移是

\[
u=(\sin(\pi x)\sin(\pi y),0)^\mathsf{T},
\]

体力为

\[
b=\left(
\frac{22.5}{13}\pi^2\sin(\pi x)\sin(\pi y),
-\frac{12.5}{13}\pi^2\cos(\pi x)\cos(\pi y)
\right)^\mathsf{T}.
\]

该模型替代旧名 `BoxTriLagrange2dData`，供二维 Matrix-Free 三角形网格算例使用。

## DivergenceFreePolynomialElasticity3D

区域为 \([0,1]^3\)，默认 \(\lambda=\mu=1\)。定义

\[
\phi(t)=(t-t^2)^2,\qquad
\psi(t)=2t^3-3t^2+t.
\]

精确位移是

\[
u=\left(
200\mu\phi(x)\psi(y)\psi(z),
-100\mu\phi(y)\psi(x)\psi(z),
-100\mu\phi(z)\psi(x)\psi(y)
\right)^\mathsf{T}.
\]

直接求导可得 \(\nabla\cdot u=0\)，故

\[
-\nabla\cdot\sigma(u)=-\mu\Delta u=b,
\]

实现使用该式展开后的多项式体力。由于 \(\phi(0)=\phi(1)=0\) 且
\(\psi(0)=\psi(1)=0\)，三个分量在整个边界为零。该模型替代旧名
`PolySolPureDirLagrange3d`，供三维 Matrix-Free 与 PINN 验证使用。

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
