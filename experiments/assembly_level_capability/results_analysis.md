# 线弹性矩阵组装层级: 内存与 MatVec 耗时实测

状态: 待跑 (占位骨架, 跑完填数后冻结)。

层级定义、存储公式与推导见 `dut-postdoc:concepts/matrix-free/assembly-levels.md`; 诊断与结论见 `dut-postdoc:entities/zang-xinyu/zang-xinyu.md`; 五方案给出同一个离散算子的验证见 [`../assembly_level_consistency/results_analysis.md`](../assembly_level_consistency/results_analysis.md)。

---

## 1. 设定

| 项 | 值 |
|---|---|
| 网格 | 单位立方体 `HexahedronMesh.from_box(n^3)`, Q1 |
| 空间 | `LagrangeFESpace(p=1)` + `TensorFunctionSpace(shape=(-1,3))`, 交错布局 |
| 材料 | 各向同性, $E = 1$, $\nu = 0.3$ |
| 积分 | $q = 2$, 8 点, $w_q = 1/8$ |
| 后端 | numpy 单线程 (`OMP/OPENBLAS/MKL_NUM_THREADS = 1`) |
| 机器 | WSL Ubuntu-24.04, 47.04 GiB 可用内存, i9-14900KF |
| 参考 $K_e$ | `LinearElasticIntegrator(method="fast")` |

每单元理论存储 (double 个数, 另加 `cell2dof` 24 个 int64):

| scheme | 组成 | 个数 |
|---|---|---|
| stored-b | $8 \times (6 \times 24) + 8 \times (6 \times 6) + 8$ | 1448 |
| ea | $24 \times 24$ | 576 |
| fa | 按实测 nnz 折算 | 见 §2.2 |
| pa | $8 \times 9 + 8$ | 80 |
| shared-ke | 全网格一份 | 0 |

---

## 2. 实测数据

### 2.1 bandwidth

| 数组 | repeats | median | 读+写 GB/s |
|---|---|---|---|
| 待填 | | | |

### 2.2 matvec, n = 48

| scheme | 常驻 GiB | build 净增 GiB | 峰值 GiB | kB/cell | build | MatVec 中位 | eff GB/s | 带宽占比 | x shared-ke |
|---|---|---|---|---|---|---|---|---|---|
| 待填 | | | | | | | | | |

### 2.3 matvec, n = 104 (约 347 万 DOF)

| scheme | 常驻 GiB | build 净增 GiB | 峰值 GiB | kB/cell | build | MatVec 中位 | eff GB/s | 带宽占比 | x shared-ke |
|---|---|---|---|---|---|---|---|---|---|
| 待填 | | | | | | | | | |

### 2.4 solve (Jacobi-PCG, shared-ke)

| n | n_dofs | iterations | iters/n | s/iter | solve 总时 |
|---|---|---|---|---|---|
| 待填 | | | | | |

---

## 3. 结论

待填。
