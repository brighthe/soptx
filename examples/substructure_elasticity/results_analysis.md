# 子结构静力缩聚验证、契约与证据报告

本报告承载本算例的数学—代码映射契约、验证边界、验收阈值与证据产物；数值结论以同一次运行生成的 JSON 为准，避免把不同网格或边界条件下的历史数值混为一组证据。

## 当前扩展状态（2026-09-03）

本目录使用五个无 `case-id` 的直接入口：

- `verify_full_trace_convergence.py`：验证完整接口路径的位移误差收敛阶。
- `compare_full_trace_with_fa.py`：验收完整接口与 FA 等价，并比较时间和峰值内存。
- `verify_linear_corner_consistency.py`：验收角点线性迹降阶模型的内部一致性。
- `compare_linear_corner_with_fa.py`：报告角点迹相对 FA 的近似误差及性能收益。
- `compare_fa_full_trace_linear_corner.py`：在同一条件下统一比较三条路径。

共享实现位于 `_convergence.py`、`_comparison.py` 和 `_performance_process.py`；公开脚本只负责参数解析、调用和结果展示。性能结果包含 UTC 时间戳且不覆盖已有证据；`linear_corner` 一致性结果按相同配置覆盖。既有 JSON 文件保持不变，新结果分别使用 `full-trace-performance-v2`、`linear-corner-performance-v1`、`substructure-three-path-performance-v1` 和 `linear-corner-consistency-v1`。

**本次五脚本入口重构尚未运行验证。** 下文旧版实测数据原样保留，不据此宣称统一入口已通过或已有新的性能结论。三路径与重复计时的范围、密度设置不同，不能混合统计。当前命令与参数范围见 `README.md`；入口合并未解决大规模内存问题。

性能证据的口径：每个工作进程独立准备问题，分析计时止于完整位移恢复；准备时间单列，并报告准备与分析之和。峰值内存是 Linux `VmHWM`，在分析结束、诊断之前读取，包含依赖加载、问题准备及原生数值库分配，不含父进程、另一条路径及后续结果传输。试运行和正式测量均为新进程，试运行不构成后续进程的库内预热。时间不含解释器启动与导入，内存包含导入，两者范围须分别说明。

新增内存指标只用于比较当前显式全装配、全批量 Schur 实现；接口自由度更少不意味着峰值 RSS 必然更小。具体字段和单位见 `README.md`，旧版耗时不得与新鲜进程的测量合并统计。

> **能力与网格边界说明**：
> - **有限元次数**：支持任意正整数多项式次数 $p \ge 1$（一阶双线性/三线性 $p=1$，二阶高阶元 $p=2$ 等）；
> - **网格类型**：当前**仅支持规则笛卡尔结构化张量积网格**（2D `QuadrangleMesh` 与 3D `HexahedronMesh`），暂不支持非结构化三角形/四面体或非规则多边形网格。

---

## 1. 数学—代码映射契约

全部同构子结构共用一个 `SubstructurePrototype`，缩聚沿批量维 $B$ 一次完成，下表中的
`...` 表示可变前导维。

> **理论事实源**：👉 `dut-postdoc:concepts/substructural-condensation.md`

| 数学对象 | 当前代码 | 形状 | 含义 |
|---|---|---|---|
| $K^j$ | `prototype.assemble_local_stiffness_batch(density)` | `(B, n_dof, n_dof)` | SIMP 局部刚度 $K^j=(\rho^j)^p K_{\mathrm{unit}}$，一次批量装配。 |
| $K_{ii}^j$ | `K_local[..., i_dofs[:, None], i_dofs]` | `(..., n_i, n_i)` | 子结构内部自由度刚度。 |
| $K_{ib}^j$ | `K_local[..., i_dofs[:, None], b_dofs]` | `(..., n_i, n_b)` | 内部—接口耦合刚度。 |
| $N^j$ | `-bm.linalg.solve(K_ii, K_ib)` | `(..., n_i, n_b)` | 内部位移恢复映射：$u_i^j=N^j u_b^j$。 |
| $K_s^j$ | `K_bb - K_bi @ bm.linalg.solve(K_ii, K_ib)` | `(..., n_b, n_b)` | Schur 补缩聚刚度。 |
| $K_\mathcal{B}$ | `InterfaceSystem.stiffness` | `(n_I, n_I)` | 全局接口系统 Scatter-Add 装配结果（FEALPy 原生 `CSRTensor`）。 |
| $u_\mathcal{B}$ | `solve_interface_system(system, load, fixed_dofs, solver='scipy')` | `(n_I,)` | 施加位移约束后经 `fealpy.solver.spsolve` 求解的接口位移。 |

`bm.linalg.solve(K_ii, K_ib)` 与逐列施加单位接口位移、求解局部 Dirichlet 问题在代数上等价；实现不显式求逆。

### 建模假设：内部自由度不受载

当前 `FEAStaticCondensation` 只缩聚刚度，不缩聚载荷，恢复关系固定为 $u_i^j = N^j u_b^j$。

这**不是**相对 Huang 2023 的实现缺口，而是与论文一致的建模假设。论文式 (6) 把子结构平衡方程的右端项直接写成 $(f_{jb}^h,\ \mathbf 0)^{\mathsf T}$，并明言"不失一般性地假设与 $u_{ji}^h$ 相关的外部载荷为零"；由此论文式 (7) 中的"缩聚载荷" $\tilde f_{jb}^h = f_{jb}^h - K_{jbi}^h (K_{jii}^h)^{-1} f_{ji}^h$ 在 $f_{ji}^h = \mathbf 0$ 下退化为 $f_{jb}^h$ 本身，其恢复式同样没有 $+(K_{jii}^j)^{-1} f_i^j$ 项。

**因此本模块与论文同样只对内部自由度不受载的问题成立**：集中载荷、面载荷必须作用在接口自由度上。体力（自重、热载）作用在内部自由度上，$f_i \ne \mathbf 0$，此时必须真正实现
$f_s^j = f_b^j - K_{bi}^j (K_{ii}^j)^{-1} f_i^j$ 并在恢复式中补上 $(K_{ii}^j)^{-1} f_i^j$——论文未覆盖该类问题，本实现亦未覆盖。

`compare_fa_full_trace_linear_corner.py` 使用的 MBB 梁集中载荷落在接口自由度上，故不受此限制。

## 2. 验证对象与职责

| 脚本 | 验证职责 | 判定边界 |
|---|---|---|
| `verify_full_trace_convergence.py` | 制造解下的位移 $L^2$ 误差和 $H^1$ 半范误差收敛阶。 | 只验 `full_trace` 的有限元离散精度。 |
| `compare_full_trace_with_fa.py` | `full_trace` 与 FA 的位移、柔顺度等价性及时间、内存比较。 | 两项相对差均须不超过 `1e-11`。 |
| `verify_linear_corner_consistency.py` | 投影、平衡、约束、恢复、能量与虚功一致性。 | 一致性残差须不超过 `1e-9`；不要求等于 FA。 |
| `compare_linear_corner_with_fa.py` | `linear_corner` 相对 FA 的近似误差及时间、内存比较。 | 近似误差只报告，不作为失败门禁。 |
| `compare_fa_full_trace_linear_corner.py` | 相同输入和测量口径下统一比较三条路径。 | 同时执行上述 `full_trace` 等价性与 `linear_corner` 自身一致性门禁。 |

比较路径使用相同的细网格、密度场、荷载、Dirichlet 固定 DOF、求解器和线程环境。性能脚本为每条路径启动独立新进程；完整位移全部恢复后，才在父进程中计算相对 FA 的误差。

`full_trace` 保留全部接口自由度，因此与相同离散下的 FA 解等价。`linear_corner` 使用相同的精确局部 Schur 补，但把接口位移限制在角点线性迹空间中，因此通常只得到近似 FA 解。

当前实现仅支持规则结构化张量积网格。制造解入口提供 $p=1,2$；性能比较固定为 Q1。内部自由度受载、Matrix-Free、Krylov、GPU 和 PIML 均不在本目录验收范围内。

## 3. 验收契约与证据产物

### 3.1 `full_trace` 收敛阶

```bash
python examples/substructure_elasticity/verify_full_trace_convergence.py \
  --problem HarmonicPoly3D --degree 1 --levels 3
```

最终观测阶门禁为：位移 $L^2$ 阶不低于 $p+0.8$，位移 $H^1$ 半范阶不低于 $p-0.2$。该入口不求解 FA，也不提供性能结论。

### 3.2 `full_trace` 与 FA

```bash
python examples/substructure_elasticity/compare_full_trace_with_fa.py \
  --problem FullMBBBeam3d --n-sub 12 4 4 --n-fine 4 4 4
```

每个试运行和正式测量样本都必须满足位移、柔顺度相对差不超过 `1e-11`；失败时不写最终 JSON。

### 3.3 `linear_corner` 自身一致性

```bash
python examples/substructure_elasticity/verify_linear_corner_consistency.py \
  --problem FullMBBBeam3d --n-sub 6 2 2 --n-fine 4 4 4
```

所有内部一致性指标必须不超过 `1e-9`。PASS 只表示降阶模型实现自洽，不表示其位移或柔顺度与 FA 达到舍入精度一致。

### 3.4 `linear_corner` 与 FA

```bash
python examples/substructure_elasticity/compare_linear_corner_with_fa.py \
  --problem FullMBBBeam3d --n-sub 12 4 4 --n-fine 4 4 4
```

该入口以 FA 为精度参照，但只报告 `linear_corner` 的近似误差；运行成败由输入一致性和降阶模型内部一致性决定。

### 3.5 三路径统一比较

```bash
python examples/substructure_elasticity/compare_fa_full_trace_linear_corner.py \
  --problem FullMBBBeam3d --n-sub 12 4 4 --n-fine 4 4 4
```

该入口用于同一配置下的横向汇总，不替代前四个诊断入口。性能结果均记录逐次样本、分项耗时、峰值 RSS、输入指纹和环境信息；只有同一次运行中的字段才可组成一项性能结论。

## 4. 实测证据

下表逐字转录自运行生成的 `outputs/lagrange_comparison_{2d,3d}.json` 与 `outputs/substructure_convergence_{2d,3d}.json`，不做任何加工。

### 4.1 全装配 vs 缩聚求解代数等价性指标

| 指标 | 2D (`HalfMBBBeamRight2d`) | 3D (`FullMBBBeam3d`) |
|---|---|---|
| 子结构划分 `n_sub` | `6 × 2` | `6 × 2 × 2` |
| 子结构细网格 `n_fine` | `5 × 5` | `4 × 4 × 4` |
| 全尺度总自由度 | 682 | 6075 |
| Lagrange 求解自由度 (free) | 670 | 6023 |
| 接口自由度 | 298 | 4131 |
| 缩聚接口求解自由度 (free) | 286 | 4079 |
| Lagrange 结构柔度 | 406.92585239 | 228.78611503 |
| 缩聚结构柔度 | 406.92585239 | 228.78611503 |
| **柔度相对误差** | **1.7974e-12** | **5.6164e-13** |
| **位移场相对误差** | **1.9224e-12** | **7.1006e-13** |
| 验收阈值 | `1.0e-11` | `1.0e-11` |
| 判定结果 | **通过 (PASS)** | **通过 (PASS)** |

### 4.2 多层网格先验 $L_2$ 误差收敛阶实测

**2D 调和多项式制造解模型 (`HarmonicPoly2D`)**：
```text
层级 (Level)   | 网格步长 (h)   | p=1 L2 误差 (Obs Order)     | p=2 L2 误差 (Obs Order)
--------------------------------------------------------------------------------------
Level 1 (4x4)  | 2.5000e-01     | 7.8988e-03 ( -- )           | 3.0497e-03 ( -- )
Level 2 (8x8)  | 1.2500e-01     | 1.9839e-03 ( 1.993 )        | 3.8121e-04 ( 3.000 )
Level 3 (16x16)| 6.2500e-02     | 4.9659e-04 ( 1.998 )        | 4.7651e-05 ( 3.000 )
Level 4 (32x32)| 3.1250e-02     | 1.2418e-04 ( 2.000 )        | 5.9564e-06 ( 3.000 )
--------------------------------------------------------------------------------------
理论收敛阶     |                | O(h^2) (理论 2.0 阶)        | O(h^3) (理论 3.0 阶)
判定结果       |                | 通过 (2.000 >= 1.80)        | 通过 (3.000 >= 2.80)
```

**3D 调和多项式制造解模型 (`HarmonicPoly3D`)**：
```text
层级 (Level)   | 网格步长 (h)   | p=1 L2 误差 (Obs Order)     | p=2 L2 误差 (Obs Order)
--------------------------------------------------------------------------------------
Level 1 (4x4x4)| 2.5000e-01     | 4.6470e-02 ( -- )           | 3.7351e-03 ( -- )
Level 2 (8x8x8)| 1.2500e-01     | 1.6810e-02 ( 1.467 )        | 4.6689e-04 ( 3.000 )
Level 3(16x16x16)| 6.2500e-02   | 4.3349e-03 ( 1.955 )        | --
--------------------------------------------------------------------------------------
理论收敛阶     |                | O(h^2) (理论 2.0 阶)        | O(h^3) (理论 3.0 阶)
判定结果       |                | 通过 (1.955 >= 1.80)        | 通过 (3.000 >= 2.80)
```

---

## 5. 历史证据的适用范围

上述旧版记录仅支持所列模型、网格与参数下的结论：

1. **绝对代数等价性结论**：
   所列 MBB 算例的完整接口缩聚与 FA 位移、柔顺度相对差为 $10^{-12}\sim10^{-13}$，通过当次 `1e-11` 门禁；这支持这些算例下的代数等价性，不是对所有输入或全部实现的正确性证明。
2. **严格的最优有限元先验收敛性结论**：
   旧版制造解记录中，$p=1$ 的 $L^2$ 误差阶趋近 2，$p=2$ 趋近 3；旧版未记录 FA 同网格误差及 $H^1$ 半范误差，也不能由 $p=1,2$ 的实验推断任意次数均已验证。
3. **求解自由度规模缩减效益**：
   通过静力缩聚将全尺度问题转化为仅含边界的接口系统求解，2D MBB 梁自由度缩减达 **57.3%**（670 $\to$ 286），3D MBB 梁自由度缩减达 **32.3%**（6023 $\to$ 4079）。当同构子结构内部网格加密时，自由度缩减比例随网格细化进一步显著提升，为大规模拓扑优化的高效求解奠定了基础。
4. **适用边界与建模假设明确性**：
   本模块严格基于“子结构内部自由度不受外载”（载荷仅作用于子结构接口自由度）的经典建模假设（与 Huang 2023 式 (6)-(7) 完全一致）。对于内部含体力载荷的问题，须扩展包含载荷缩聚项的广义 Schur 补形式。
