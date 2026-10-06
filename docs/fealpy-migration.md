# FEALPy 依赖移植：完成情况与后续工作

本页是 2026-10-05 的移植汇总快照，记录 SOPTX 去除 FEALPy 依赖的做法、提交、验证结果和
约定好的后续工作。

- 来源与许可证以 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) 为准。
- fork 补丁的落点、回归保护现状与移植后遗留问题以
  [`known-issues/README.md`](known-issues/README.md) 为准（该目录唯一的可变状态源）。
- 本页的「后续工作」一节随事项完成逐条更新或删除。

---

## 一、已完成的内容

### 1.1 目标与做法

- **目标**：把 SOPTX 用到的 FEALPy 代码移植为 `soptx` 子包，此后不再依赖 FEALPy。
- **移植源**：vendor fork `brighthe/fealpy` `main` @ `f474a5775`（工作副本
  `~/codespace/fealpy`），含 fork 上全部存活补丁；全程只读，未改动 fork。
- **做法**：先分层「影子搬入」，此时 SOPTX 现有代码仍用 FEALPy；全部到位后在一个提交里
  一次性切换引用。不能逐层切换：中间状态会同时存在两个 `backend_manager` 单例与两套同名
  网格类，混用时静默出错。
- **分支**：`claude/soptx-fealpy-migration-18ef30`，基于 `main` 的 `041e424`；期间并入 `main` 上的
  `e583825` 并完成其 FEALPy 依赖切换，2026-10-05 已快进合并到 `main`（`4d9bd76`）。

### 1.2 提交清单

| 提交 | 内容 |
|---|---|
| `bf2f362` | 第 1–4 步：搬入 FEALPy 代码（87 个文件，约 1.96 万行），SOPTX 现有代码不改 |
| `aa58af5` | 修复移植前已存在的 27 条测试失败与报错，测试基线变为 903 passed、0 failed |
| `b55f265` | 第 5 步：全仓引用一次性切换到 `soptx` 子包（237 个文件） |
| `3417b28` | PINN 示例移除 `fealpy.ml` 依赖，并修复其在 v0.4 网格下的失效 |
| `6712050` | 第 6 步之一：移除 `fealpy` 依赖，CI 门禁禁止导入 `fealpy` |
| `1ce45f4` | 第 6 步之二：来源声明与规范文档 |
| `b888a20` | `viz` extra 声明 `pyevtk`（此前由 FEALPy 的 requirements 间接带入） |
| `5c0c10f` | 本汇总文档 |
| `4d9bd76` | 并入 `main` 的 `e583825`（子结构接口空间与流式精确缩聚等），解决 5 处导入区冲突，其余 12 处 FEALPy 导入切换到 `soptx`；随后 `main` 快进到此提交 |

### 1.3 搬入内容

| SOPTX 位置 | 来源 | 规模 | 与原文的差异 |
|---|---|---|---|
| `src/soptx/backend/` | `fealpy/backend/` | 6 个文件 | 只保留 numpy、pytorch 后端；logger 改为 `logging.getLogger(__name__)`；后端加载路径改为 `soptx.backend`；`manager.pyi` 的后端列表收缩为两个 |
| `src/soptx/typing.py`、`src/soptx/decorator/` | 同名模块 | 5 个文件 | 无 |
| `src/soptx/sparse/` | `fealpy/sparse/` | 8 个文件 | 无 |
| `src/soptx/quadrature/` | `fealpy/quadrature/` | 9 个文件 | 无 |
| `src/soptx/mesh/` | `fealpy/mesh/`（v0.4 网格树） | 46 个文件，约 1.12 万行 | 只保留三角形、四边形、四面体、六面体四类网格的依赖闭包；删去绘图、半边结构、局部加密与粗化（覆盖率确认 SOPTX 零执行），保留均匀加密；3 处 `fealpy.quadrature` 绝对导入改为相对导入；`__init__.py` 合并 SOPTX 原有的 `structured_box`、`structured_triangle` |
| `src/soptx/functionspace/` | `fealpy/functionspace/` 中 7 个模块 | 约 1.3k 行 | 2 处 `fealpy.decorator` 绝对导入改为相对导入；删除 1.1.x 的 HuZhang 兼容层 |
| `src/soptx/fem/` 下 `integrator.py`、`form.py`、`_bilinear_form_base.py`、`_linear_form_base.py`、`functional.py`、`coef.py` | `fealpy/fem/` 基类、`fealpy/functional.py`、`fealpy/utils/utils.py` | 6 个文件 | 两个 Form 基类加下划线改名以避开 SOPTX 同名子类文件；`coef.py` 只保留用到的 4 个函数 |
| `examples/pinn_elasticity/_pinn_support.py` | `fealpy/ml/` | 约 150 行 | 精简移植示例用到的部分；修正 `quadrature_formula` 的 `etype=` 调用 |

每个移植文件的头部都注明源文件路径、源提交 `f474a5775` 与 Huayi Wei 的
`GPL-3.0-or-later` 版权。

**明确不搬**：jax、cupy、paddle 等后端（jax 后端自身有未实现函数，SOPTX 约 329 处原地
赋值与 jax 不兼容，且从未使用）；`logs.py`；fork 的全部测试；其余 25 种函数空间；
`fealpy.ml` 中示例用不到的部分。

### 1.4 切换时的处理（`b55f265`）

- **机械替换**：220 个文件、498 处 import。替换脚本用 AST 定位 import 语句，只改模块路径，
  不做全文字符串替换。
- **逐个处理的特殊情况**：
  - `from fealpy.fem import LinearForm / BilinearForm` 改指过渡基类而非 SOPTX 子类，保证
    行为不变；删除从未使用的 `BlockForm`、`ScalarMassIntegrator`、`ScalarDiffusionIntegrator`
    导入。
  - CG 的 logger 改为 `logging.getLogger(__name__)`。
  - 测试中以字符串写的模块替身目标 `"fealpy.backend"` 改为 `"soptx.backend"`（3 处）。
  - `paper_topopt_huzhang` 的 `reproducible` 只要求本仓库干净；原判定要求 FEALPy 工作副本
    也干净，切换后会把论文证据一律判为不可复现。
  - 6 个实验与 2 个拉格朗日示例的溯源记录去掉 FEALPy 版本与路径字段。
  - evidence 的 `environment` 去掉 `fealpy` 字段，`SCHEMA_VERSION` 从 4 升到 5。
  - 两个以 FEALPy 为参照的测试改读冻结的参照数据
    `tests/unit/data/fealpy_linear_elastic_reference.npz`。

### 1.5 配置、CI 与文档（`6712050`、`1ce45f4`、`b888a20`）

- **pyproject**：删除 `fealpy>=4,<5`；`requires-python` 改为 `>=3.12`（移入代码使用 PEP 695
  泛型与 `type` 语句）；`viz` extra 加入 `vtk`、`pyevtk`；`manager.pyi` 进入 wheel；
  pyright `extraPaths` 去掉 `../fealpy`。
- **CI**：测试矩阵去掉 Python 3.10；去掉 3 个文件的 UTF-8 BOM（此前令两个检查工具解析崩溃）。
- **`check_architecture`**：LAYER 表登记 7 个新子包，支持单文件模块根；新增门禁，src、
  tests、examples、experiments、tools 中出现 `import fealpy`、`from fealpy` 或
  `import_module("fealpy")` 即失败。
- **`check_comment_style`**：新增 `PORTED_ROOTS` 豁免，移植代码暂不计入棘轮，基线数字不放松。
- **文档**：新增 `THIRD_PARTY_NOTICES.md`；改写 README 与 AGENTS.md 中的 FEALPy 约定；
  known-issues 总表改写为「FEALPy 来源与已内联补丁清单」；补丁详述页标为只读历史记录；
  `docs/index.md`、`evidence-policy.md` 与 3 份主题 README 中不再成立的说法已更新。

### 1.6 验证结果

| 验证项 | 结果 |
|---|---|
| 静态门禁 | 全仓 0 处 FEALPy 运行时引用 |
| 拦截 FEALPy 导入的全量 pytest | 903 passed、10 skipped；913 个用例逐个与移植前基线一致 |
| examples 数值基线（8 次运行） | HuZhang、`topopt_numpy` 逐行一致；拉格朗日示例只有耗时不同；`topopt_pytorch` 只有 GPU 运行间的末位波动（移植前代码连跑两次同样存在） |
| 与 FEALPy 逐项比对 | mesh 488 项、functionspace 与 fem 基类 552 项，0 处不一致 |
| fork 补丁回归用例（改为导入 `soptx`） | 35/35 通过，含缺陷 8、9 的 delta 判据（四类网格 × $p=1..4$） |
| 收敛阶基准 | 四边形 $p=3$：4.001 / 4.000；六面体 $p=2$：2.997 / 3.007；三角形 $p=2$：3.015 / 3.005；四边形 $p=1$：1.993 / 1.998 |
| 只装核心依赖 | 除 `soptx.ml`（需 torch）外，主要子包只需 numpy、scipy、sympy 即可导入 |
| 并入 `e583825` 后 | 拦截 FEALPy 导入下全量 pytest 1009 passed、1 skipped；`ihpcm` 已装 mpi4py 与 PyMUMPS，分布式与 MUMPS 用例实际运行并通过；`main` 检出处直接运行结果相同 |
| wheel | 元数据无 `fealpy`，`Requires-Python >=3.12`，含 `manager.pyi` 与全部移入子包 |
| `known-issues/fealpy-patches.md` 逐项核对 | 12 项修复全部在位，调用契约均成立；仅「运行期警告治理」在当前 torch 2.11 下不生效（文档记录环境为 2.13），只影响提示信息 |

移植前的测试基线、examples 输出与参照数据生成脚本存于
`~/codespace/soptx-baseline/aa58af5/`（仓库外）。

---

## 二、约定好的后续工作

按建议的执行顺序排列。

### 2.0 先决定的事项

| 事项 | 说明 |
|---|---|
| 缺陷 8、9 的自动化防线（可选） | 把 fork 的 `tests/mesh/unit/test_interpolation_delta.py` 放进 `tests/`，只需把导入从 `fealpy` 改为 `soptx`，22 个用例约 0.8 秒。它是唯一能抓住这两个静默错误的测试 |

合回 `main` 已于 2026-10-05 完成（`4d9bd76`）。

### 2.1 结构整理（另开分支，分两个提交）

**HuZhang 空间移入 `soptx/functionspace/`**（已完成）

- 移动 `fem/spaces/huzhang_fe_space.py`、`huzhang_fe_space_2d.py`、`huzhang_fe_space_3d.py`，
  约 1.9k 行。
- 依据：HuZhang 空间只依赖 backend、mesh、sparse 与 functionspace 基类，不依赖 fem 层的任何
  能力，现在分两处只是「FEALPy 来源 vs SOPTX 自有」的历史遗留。反过来把 functionspace 整体
  放进 fem 也不行：第 1 层的 materials（`flatten_indices`）依赖它。
- `fem/spaces/__init__.py` 保留为转发（fem 导入 functionspace 是合法方向），仓库内导入已改到
  `soptx.functionspace`；旧子模块路径（如 `soptx.fem.spaces.huzhang_fe_space_2d`）不再存在。
- 3 个文件是 SOPTX 自有代码，已加入 `tools/check_comment_style.py` 的 `PORTED_EXCEPTIONS`，
  搬进 `functionspace/` 后照常计入注释风格棘轮。
- 验证：全量测试；HuZhang 与 Lagrange 装配矩阵与 `88018e7` 逐位一致，示例输出逐行一致。

**`BilinearForm` / `LinearForm` 的基类与子类合并**（已完成）

- `_bilinear_form_base.py`、`_linear_form_base.py` 分别并入 `bilinear_form.py`、`linear_form.py`，
  两个类直接继承 `Form`，原基类的装配成为 `method='coalesce'` 路线；全文改为中文 numpydoc，
  删除从未实现的 `BilinearForm.mult()`，保留 `.T` 与 `@`（EA 实验计时用到 `@`）。
- 原先直接用基类的 `BilinearForm` 调用方（Hu--Zhang 分析器的 $A$、$B$、$J$，3 个测试，
  `assembly_methods_mesh_demo`）显式传 `method='coalesce'`，走与原来完全相同的代码；
  `test_csr_pattern` 由此仍以 coalesce 为参照，不会变成 pattern 与自身比较。
- 直接用基类的 `LinearForm` 调用方（两个分析器、`problem_adapter`、
  `verify_full_trace_convergence`）改走默认 scatter 路线：均为单列、`format='dense'`，两个
  分析器的右端项逐位一致，`problem_adapter` 由子结构测试覆盖。
- pattern 路线新增前提检查：双空间、批量装配、局部张量形状与骨架不符（面积分子、部分单元
  积分子）时抛 `ValueError`。原先前三种报含义不明的 `reshape` 错误，批量装配则静默返回不带
  批量维的矩阵。只核对形状，形状相符但自由度编号不同的积分子仍无法识别。
- 验证：同提交 1，装配矩阵与右端项与 `88018e7` 逐位一致。

### 2.2 移植遗留

明细与修法见 [`known-issues/README.md`](known-issues/README.md)「移植后遗留」一节，这里只列
条目：

- ~~PyTorch 后端下 $p \ge 2$ 的 `interpolation_points` 因 float32 / float64 混用报错~~（已修复：
  `mesh/ipoints.py` 先把 `multi_index` 转成坐标浮点类型；回归测试
  `tests/unit/test_interpolation_points_backend.py`）。
- ~~移植代码的英文 docstring 欠账~~（已完成：按子包补齐中文 numpydoc 并翻译英文 docstring 与注释，共 12 个提交；`tools/check_docstring_only.py` 逐批核对只改了 docstring 与注释，`PORTED_ROOTS` 已清空）。
- ~~Form 过渡基类~~（已随 2.1 完成）。
- ~~调用网格上不存在的方法~~（已处理）。全仓扫描 `*mesh*.<方法>(` 后共 11 类：无调用方的
  （`hess_basis`、`cell_basis_on_face`、`prolongation_matrix`、`project_solution_to_finer_mesh`、
  `HuZhangBoundarySourceIntegrator`）已删除；节点密度链路的 `cell_to_node`、`jacobi_matrix`
  已改用 `mesh.cell` 与 `entity_view('cell').jacobi_matrix`，连带修复插值格式 SIMP 节点分支的
  两处缺陷，有限差分测试 `tests/unit/test_node_density.py` 覆盖；`mesh.count` 已改用
  `mesh.entity`；同样悬空且无调用方的 `save_optimization_history`（`mesh.celldata`、
  `mesh.to_vtk`）已删除。三维跳量稳定化与 `splitter` 分块装配两项转记 known-issues。
- ~~`fem/distributed/` 中 `mesh.py`、`entity_mpi.py`、`space.py` 源自 FEALPy 但文件头未注明来源~~（已补注）。

### 2.3 移植前已存在的 CI 问题（已修复）

分支 `claude/ci-fixes`，本地按 `.github/workflows/ci.yml` 的 `fast` 任务顺序逐步通过：

| 问题 | 处理 |
|---|---|
| ~~`ml`（第 0 层）导入 `fem`（第 2 层）~~ | `ml` 调整为第 2 层，与 `fem` 同层；`ml.substructure` 与 `fem.substructure` 本就互相导入，第 0、1 层无模块导入 `ml` |
| ~~`experiments` 直接导入 `examples`（7 处）~~ | 主题隔离改为单向：`experiments` 可包式 import `examples`，反向禁止，同一父目录下主题仍互相禁止 |
| ~~注释风格超出棘轮基线~~ | 注释与 docstring 中 916 个记号的全角标点机械替换为半角，`FULLWIDTH_BASELINE` 645 → 3；补 `protocols`、`fem`、Hu–Zhang 空间 142 处 docstring，`MISSING_DOCSTRING_BASELINE` 314 → 271。两步均经 `check_docstring_only` 确认 AST 不变 |
| ~~`check_repo_layout` 缺 `results_analysis.md`~~ | 三个目录按当前脚本补写映射与验收契约；证据指向 `experiments/analysis_capability_*` 或标「待跑」 |
| ~~`docs/references/` 不存在~~ | 从 `ad95594^` 原样恢复，清单与重新生成的结果逐字节相同 |

不属于 CI 失败、移出本节的两项：

- **坏链接**：CI 不检查链接。全仓原有 67 个，恢复 `docs/references/` 修好 2 个，余下的 `docs/architecture/` 相关 8 个随 2.4 兼容层清理一并决定（迁移表本为兼容层而写），约 45 个指向不入库的 `outputs/`，约 12 个为改名后未跟进，另做一轮文档整理。
- **torch 版本**：CI 安装最新 torch，只是本地 `ihpcm` 环境旧于 known-issues 记录的验证环境。

修复过程中读代码发现的问题，已在分支 `claude/s23-fixes` 处理（每项有回归测试或运行验证）：

- ~~`collect_ood_probe_trajectory.py` 导入从未存在过的 `_common`~~：改从 `verify_stiffness_route` 取投产参数，
  `CELL_SIZE` 本地派生，与原 `deployment_config.py` 一致。
- ~~`HuZhangFESpace` 三维工厂必抛 `TypeError`，二维访问器调用即报错，三维访问器为空桩~~：工厂修正；二维
  `edge_to_dof` / `face_to_dof` 按 `index` 选边；未实现的访问器明确抛 `NotImplementedError`；二维 `basis` 接受
  布尔掩码；三维 `basis` / `value` / `div_value` 传单元子集时报错。
- ~~`benchmark_thread_scaling.py` 默认 `--rmin 2.4` 使滤波矩阵稠密~~：改为 `--rmin-cells`。同时修复该脚本一启动即
  `TypeError`（材料参数 `plane_type`）与 `assemble` 段在两种层级下取指纹报错，全部段在 `fa` / `ea` 下实测通过。
- ~~`fetch_vector_jump` 边界面逐项取绝对值；`MassIntegrator.to_global_dof` 不按 `index` 截取~~：已修。
- ~~`voigt_multiresolution`、`standard_multiresolution`、`symbolic` 变体坏掉且无调用方~~：连同
  `fem/integrators/utils.py` 与 `InterFaceSourceIntegrator` 删除。
- ~~`LagrangeFEMAnalyzer.solve_adjoint` 只适用于 `'fa'`；`n_sub` 标注~~：非 `'fa'` 明确报错；标注改为 `Optional[int]`。
- `compute_stress_state` 两种分析器返回键不同（`stress_solid` / `stress_apparent`）：语义本就不同，各消费方按分析器
  分开取用，不是缺陷。

当时只留档的问题，已在分支 `claude/followups` 处理（`sympy`、`_bcakup`、FEALPy 字样见 2.4）：

- ~~二维 Hu–Zhang `boundary_interpolate` 的报错路径与标量 `gd`~~：常值 `gd` 先转张量，标量明确报 `ValueError`
  （无可投影的分量），分量数不对时报出实际分量数；`set_tangential_traction_bc` 同步。
- ~~`HuZhangMFEMAnalyzer.__init__` 重复赋值；Hu–Zhang 积分子 `q=0` 被当作未给出~~：已修。
- ~~MMA 在 Hu–Zhang 分析器下 `is_store_stress` 报 `KeyError`~~：迭代前明确报 `NotImplementedError`；表观应力
  的记录口径待定。
- ~~三维 Hu–Zhang 标架约定与二维不一致~~：分支 `claude/huzhang-3d-verify` 先做代数验证（二维对照），表明三维空间
  本身正确（张成 $P_p(\mathbb S)$、法向迹连续、散度精确），只是 `basis_frame_of_S` 多乘 `prod(alpha!)`、边与面
  标架未归一化，使自由度系数含义与二维不同；已改为单位正交标架、基函数直接取自由度标架。$p=4$ 制造解收敛阶
  逼近理论值（末对 4.73 / 3.92 / 3.92，理论 5 / 4 / 4），见 `examples/huzhang_elasticity/results_analysis.md` §5。
- 分析器的三维边界装配（分支 `claude/huzhang-3d-bc`）：边界按面标记、位移边界项按面求积；三维空间新增
  `boundary_interpolate` / `set_tangential_traction_bc`，按格点标架张量写牵引，边界边改取边界面法向。要求边界与
  坐标轴对齐，否则明确报错；三维 $p\le3$ 构造时报错（跳量稳定化未实现）。补丁检验精确，混合边界收敛阶
  4.76 / 3.92 / 3.92，见 `examples/huzhang_elasticity/results_analysis.md` §5.3。
- 三维跳量稳定化（分支 `claude/huzhang-3d-jump`，known-issues「移植后遗留」最后一行）：面定向按几何判据；同时修复
  非齐次位移边界加稳定化时缺少数据项 $J_D(u_D, v)$ 的不相容（二维同样存在，现有算例 $u_D=0$ 未受影响）。三维
  $p=1,2,3$ 均达理论阶，见 `examples/huzhang_elasticity/results_analysis.md` §5.4；三维 $p=1$ 在原惩罚系数下稳定性不足，
  系数另乘经验因子 10（`_PHYSICAL_H_FACTOR`），$p=2,3$ 不变。另记 `vector_jump` 在 $p=1$ 时不收敛（二维同样）。
- 非坐标对齐边界上的牵引强施加（分支 `claude/huzhang-3d-oblique-traction`）：三维空间新增 `traction_face`，边界标架
  按牵引面法向对齐，折棱与角点上联合多个面求值；坐标对齐网格逐位不变。剪切立方体补丁精确，见
  `examples/huzhang_elasticity/results_analysis.md` §5.5。仍不支持：非 90° 折棱上的对称面、小平面逼近的曲面。
- 删除跳量稳定化的 `vector_jump` 变体（系数 $1/h_F$，$p=1$ 锁死不收敛）与 `matrix_jump` 的旧缩放 `gamma_hinv`（分支
  `claude/remove-vector-jump`）：二者都无调用方；分析器 `stabilization` 只余 `'none'` / `'matrix_jump'`，`stabilization_scaling`
  与积分子 `penalty_scaling` 只接受 `'physical_h'`（或 None），其余取值明确报错。

### 2.4 统一清理（已完成）

分支 `claude/compat-cleanup`，版本升至 `1.2.0.dev0`：

- ~~5 个 1.1.x 兼容层与 `soptx.model`~~：删除，连同 `test_compatibility_api.py`；`test_public_api` 改为检查这些
  路径不可导入。`LEGACY_ROOTS` 保留，防止重建旧路径。缺 docstring 基线 271 → 168。
- ~~`fem.spaces` 转发~~：与 `soptx.fem` 中的 `HuZhangFESpace`、两个结构网格生成器别名一并删除，9 个文件改从定义处导入。
- ~~指向 `docs/architecture` 的 8 个坏链接~~：删除链接；README 改列 `check_architecture.py` 强制的分层表。
- ~~残留的 FEALPy 文字~~：87 个文件改为描述当前的 soptx 子包；来源声明、历史事实与参考值比对保留。
  产物标签 `"fealpy-numpy"` 改为 `"numpy"`。
- ~~零散小项~~：`'jax'` 判断（`function.py`、`volume.py`）、`DirichletBCOperator` 现行描述、无调用方的
  `*_bcakup` / `*_inverse_backup`、运行依赖 `sympy`（证据 `environment` 仍记录该字段，未改 schema）、
  PINN 边界残差（重跑复现其余 7 项，实为 `5.3110e-04`）。

当时留待处理的事项，分支 `claude/followups` 处理结果：

- ~~`BilinearForm.__matmul__` 多列右端项散加到批量轴~~：改为 `axis=-1`，补单列 / 多列对照测试（numpy 与 pytorch
  均复现了原缺陷）。
- ~~3 个无调用方的 `*_backup` 函数~~：删除。
- ~~`core` 英文 docstring、`matrix_free_evidence` 与代码不符的说明、失效的 `krylov.weighted_cg` 链接~~：已修。
- known-issues「移植后遗留」除三维跳量外的 11 行一并修复（见 `known-issues/README.md`）。

仍未处理：

- 用户有未提交改动的文件未动：`lagrange_fem_analyzer.py`、`mesh/topology/builder.py`、`mesh/view/entity_view.py`、
  `fem/matrix/csr_pattern.py` 中的 FEALPy 现行描述，`topology/objectives/compliance.py` 中的 `'jax'` 判断。
- `experiments/fa_assembly_capability/run.py` 匹配 `"/fealpy/"` 的帧过滤分支已不会命中；VTU 元数据数组名
  `FEALPY_MESH_META` 属文件格式字段，保留。
- MMA 计算 von Mises 应力时无注释的 `/ 100.0` 缩放，用意待用户说明。

### 2.5 仓库之外的事项

- **`workstation:workspace/responsibilities.md`**：其中「不复制、vendor 算海仓库代码」的规范与
  本次移植的决定冲突；本仓库 README 已更新，规范正本需在 `workstation` 仓库同步。
- **FEALPy `COPYRIGHT.txt` 的商用条款**：「may not be sold or included in commercial products
  without a license」已在 `THIRD_PARTY_NOTICES.md` 原文照录；如有商用或再分发需求，需另行评估。
