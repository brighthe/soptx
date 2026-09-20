# 线弹性矩阵组装层级的内存与耗时 (Assembly Level Capability)

同一 hex Q1 网格、同一材料、同一积分点、同一单线程 numpy 口径下, 测五种矩阵组装/存储层级的常驻内存、峰值内存与单次 MatVec 时间, 并用 Jacobi-PCG 记录迭代数随 n 的增长。

其中 fa / ea / pa 三个装配层级给出同一个离散算子这一前提, 由 [`../assembly_level_consistency/`](../assembly_level_consistency/) 的三条制造解收敛链逐档验证; stored-b / shared-ke 不是分类法里的层级, 那边不收, 其正确性前提由本目录自负, **眼下尚无取证**。两边共用 `../_common/assembly_levels.py` 的同一套问题设定与算子实现。

起因: 臧昕禹 (梅跃学生) 的 3D CutFEM 线弹性单元级 matrix-free 求解器 (344 万 DOF, 17.6 GB, 3996 步, 3728 s) 慢在何处。诊断与结论在 `dut-postdoc:entities/zang-xinyu/zang-xinyu.md`; 五种层级的定义与数学在 `dut-postdoc:concepts/matrix-free/assembly-levels.md`。本目录只放代码与实测数字。

---

## 五种方案

| scheme | 常驻数据 | 局部作用 | 每单元 double 数 |
|---|---|---|---|
| `stored-b` | 逐单元逐积分点物理 `B_q` (6x24)、`D_q` (6x6)、`w_q detJ_q` | 三次 `einsum` | 1448 |
| `ea` | 逐单元 `K_e` (24x24) | `einsum('cij,cj->ci')` | 576 |
| `fa` | 全局 CSR (`soptx.fem.matrix.csr_pattern` 模式先行装配) | `CSRTensor @ x` (落到 `scipy.sparse._sparsetools.csr_matvec`) | 按实测 nnz 折算 (约 81 nnz/行) |
| `pa` | 逐积分点 `J^{-1}_q` (3x3)、`w_q detJ_q`; 参考梯度与 `D` 全网格共享 | 张量形式, 不落 `B` | 80 |
| `shared-ke` | 一份 `K_e` (24x24) | `x_e @ K_e^T` (dgemm) | 0 |

`ea` / `fa` / `pa` 由 `soptx.fem.levels.create_level` 构造, 即 `ElementAssembly` / `FullAssembly` / `PartialAssembly` 三个生产层级类本身; `stored-b` (臧昕禹推断方案) 与 `shared-ke` (均匀网格共享一份 `K_e`) 没有生产对应物, 是本目录自带的对照实现。基准量的是生产栈, 改了生产类这里会跟着变。

所有单元级方案共享 `cell2dof` (24 个 int64/单元), 用同一条 gather → 局部作用 → `np.add.at` scatter 路径: `ea` / `pa` 经 `ElementRestriction.scatter_add` → `bm.index_add` 落到 `np.add.at`, 与 `stored-b` / `shared-ke` 手写的那条是同一个 numpy 调用。

`fa` 的每单元数按实测折算, `(values.nbytes + col.nbytes) / 8 / NC`。生产栈的 CSR 索引是 int64 (`build_csr_pattern` 产出 int64 的 `crow` / `col`, fealpy `CSRTensor` 原样保留), 每个非零元的索引开销与数值本身一样是 8 字节。早期原型把这三个数组交给 `scipy.sparse.csr_matrix` 构造, scipy 会把索引降到 int32, 同一个矩阵的常驻字节因此少约 25% (n = 16 下 4165 → 3116 字节/单元)。表里记的是生产口径。

`pa` 的每单元 double 数 80 只数与单元数成正比的项 (`J^{-1}_q` 72 + `w_q detJ_q` 8)。生产 `LinearElasticQFunction` 另存两个与网格规模无关的小数组 —— `strain_map` (6, 3, 3) 与 `quadratic_form` (3, 3, 3), 合计 648 字节 —— 它们计入实测 `persistent_bytes`, 不进每单元理论数。

问题设定: 单位立方体 `HexahedronMesh.from_box(n^3)`, `LagrangeFESpace(p=1)` + `TensorFunctionSpace(shape=(-1,3))` 交错布局, `E = 1`, `nu = 0.3`, `q = 2` (8 点)。参考 `K_e` 来自 `LinearElasticIntegrator(method="fast")`。

---

## 目录职责

```text
experiments/assembly_level_capability/
|-- cases.toml             # 数据点注册表: 面板 / 方案 / n / repeats
|-- config.py              # cases.toml 加载与校验, THREAD_ENV 单线程环境
|-- run.py                 # 调度器 + 子进程 Worker (五种算子、三个面板) + 看板
|-- compare.py             # 读 outputs/ 打印对比表
|-- results_analysis.md    # 实测数字与结论 (跑完冻结)
`-- outputs/               # 单点 JSON 产物
```

问题设定与五种算子的实现不在本目录, 在 `../_common/assembly_levels.py`;
`ea` / `fa` / `pa` 走 `soptx.fem.levels`, `stored-b` / `shared-ke` 是那里的本地对照实现。

---

## 面板

| panel | 工况 | 记录 |
|---|---|---|
| `bandwidth` | `np.copyto` 2 GiB float64 x 10 | 读+写 GB/s |
| `matvec` | n = 48 与 n = 104 各五方案 | 常驻字节、三时点当前 RSS (setup / build / matvec 后)、峰值 RSS、build 时间、MatVec 中位时间、有效带宽下界 |
| `solve` | `shared-ke`, n = 32 / 48 / 64 | Jacobi-PCG 迭代数、s/iter、iters/n |

每个数据点独占一个子进程 (`ru_maxrss` 是进程级高水位), 子进程环境注入 `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。内存主指标是 `persistent_bytes` (算子持有数组 nbytes 之和); `/proc/self/statm` 的三时点当前 RSS 给出 build 净增量 (`rss_after_build - rss_after_setup`), 峰值 RSS 只作参考, 小 n 下会被 import 与建网格的固定开销淹没。

---

## 常用命令

```bash
python experiments/assembly_level_capability/run.py --list
python experiments/assembly_level_capability/run.py --all --check-only
python experiments/assembly_level_capability/run.py --case bandwidth_memcpy
python experiments/assembly_level_capability/run.py --panel matvec
python experiments/assembly_level_capability/run.py --cases matvec_stored-b_n104 matvec_ea_n104 matvec_fa_n104 matvec_pa_n104 matvec_shared-ke_n104
python experiments/assembly_level_capability/run.py --panel solve
python experiments/assembly_level_capability/compare.py --case all
```

正规产物是 `outputs/<case-id>.json`, 不在 `cases.toml` 里单独声明。`--n / --scheme / --repeats` 可覆盖工况参数, 产物名随之变为 `outputs/<panel>_<scheme>_n<n>.json`, 不覆盖正规产物。
