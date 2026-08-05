# 工程基准算例

本文档说明 SOPTX 中**没有解析解**的工程基准问题。与[制造解](manufactured-elasticity.md)不同，这类问题
没有精确位移场；载荷是真实物理载荷（集中力、分布力等），而非从精确解反推的体积力。实现位于
`soptx.problems.elasticity`。

## 与制造解问题的区别

| | 制造解 | 工程基准 |
|---|---|---|
| 精确解 | 有，界面暴露 `disp_solution` / `stress_solution` | 无 |
| 体力 | 由精确解反推 ($b = -\nabla\cdot\sigma(u_\mathrm{exact})$) | 通常为零（忽略自重） |
| 验证判据 | L2 收敛阶 + 真相对残差 | 真相对残差 + 载荷等效性 |
| 用途 | 验证离散格式的正确性 | 验证载荷路径装配的正确性；演示工程问题的求解流程 |
| `boundary_type` | `dirichlet` 或 `mixed`（全 Dirichlet 或 Dirichlet + Neumann） | `mixed`（Dirichlet 约束 + 集中力/分布力） |

## 载荷类型

工程基准支持不同于制造解的载荷模型，由 `load_type` 类属性声明：

| `load_type` | 语义 | 相关接口 |
|---|---|---|
| `"concentrated"` | 节点集中力（点载荷） | `is_concentrate_load_boundary()` → 载荷标记列表；`concentrate_load_bc()` → 对应的载荷值函数列表 |

分析器在装配右端向量时检查 `load_type` 并调用对应的载荷接口，而非读取 `traction_bc`。

## HalfMBBBeamRight2d

二维 MBB 梁的对称右半域。

### 问题描述

区域为 $[0,60]\times[0,20]$ mm，采用 plane stress（默认 $E=1$ MPa，$\nu=0.3$）。
全梁左右对称，取右半域建模：

![MBB 梁对称右半域](./images/mbb-beam-half-domain.png)

- **左边界** ($x=0$)：一排滚轴支座 $\rightarrow$ $u_x = 0$（对称约束）
- **右下角** ($x=60$, $y=0$)：固定铰支座 $\rightarrow$ $u_y = 0$
- **左上角** ($x=0$, $y=20$)：竖直向下集中力 $P = -1$ N

### 接口

集中力载荷采用**标记-值分离**的列表接口，每对标记函数与载荷值函数描述一个
集中力作用区域：

```python
from soptx.problems import HalfMBBBeamRight2d

problem = HalfMBBBeamRight2d(
    domain=(0.0, 60.0, 0.0, 20.0),
    P=-1.0,
    E=1.0,
    nu=0.3,
    plane_type="plane_stress",
)

# 边界标记（按位移分量分离，与 is_dirichlet_boundary 返回格式一致）
is_dirichlet_dof_x, is_dirichlet_dof_y = problem.is_dirichlet_boundary()

# 集中力载荷：标记列表 + 载荷值列表
load_markers = problem.is_concentrate_load_boundary()   # [marker_fn, ...]
load_values  = problem.concentrate_load_bc()              # [value_fn, ...]
```

每个载荷值函数返回与输入点同 shape 的张量，集中力作用维度为非零值（这里是
分量 1 即 $y$ 方向），其余维度为零。

### 验证判据

没有解析解，验证靠两条无歧义的数值判据：

1. **真相对残差** $\|Ku - F\| / \|F\|$ — 线性系统确实解开了
2. **载荷等效性** $\left|\sum F_{\sigma_h} - P\right|$ — 等效节点力装配无丢力/多分力，且必须在 `apply_bc` *之前*测量（否则被强加自由度吞掉的载荷不可见）

### 使用示例

```python
from fealpy.mesh import TriangleMesh
from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import HalfMBBBeamRight2d

problem = HalfMBBBeamRight2d(domain=(0.0, 60.0, 0.0, 20.0))
material = IsotropicLinearElasticMaterial(
    hypothesis=problem.plane_type,
    youngs_modulus=problem.E,
    poisson_ratio=problem.nu,
    enable_logging=False,
)
mesh = TriangleMesh.from_box(list(problem.domain), nx=60, ny=20)

analyzer = LagrangeFEMAnalyzer(problem=problem, material=material, mesh=mesh)
analyzer.solve()
```

完整可运行脚本见 `examples/lagrange_elasticity/concentrated_load_demo.py`。
