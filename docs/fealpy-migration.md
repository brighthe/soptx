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
- 移植代码的英文 docstring 欠账（`PORTED_ROOTS`，约 596 处缺失），按子包补齐后移出豁免表。
- ~~Form 过渡基类~~（已随 2.1 完成）。
- ~~调用网格上不存在的方法~~（已处理）。全仓扫描 `*mesh*.<方法>(` 后共 11 类：无调用方的
  （`hess_basis`、`cell_basis_on_face`、`prolongation_matrix`、`project_solution_to_finer_mesh`、
  `HuZhangBoundarySourceIntegrator`）已删除；节点密度链路的 `cell_to_node`、`jacobi_matrix`
  已改用 `mesh.cell` 与 `entity_view('cell').jacobi_matrix`，连带修复插值格式 SIMP 节点分支的
  两处缺陷，有限差分测试 `tests/unit/test_node_density.py` 覆盖；`mesh.count` 已改用
  `mesh.entity`。三维跳量稳定化、`splitter` 分块装配、`save_optimization_history` 三项转记
  known-issues。
- `fem/distributed/` 中 `mesh.py`、`entity_mpi.py`、`space.py` 源自 FEALPy 但文件头未注明来源。

### 2.3 移植前已存在的 CI 问题（与移植无关）

| 问题 | 规模 |
|---|---|
| `ml`（第 0 层）导入 `fem`（第 2 层） | `ml/substructure/` 下 3 个文件；需决定调整分层还是挪动代码 |
| `experiments` 直接导入 `examples` 模块，违反主题目录隔离 | 7 处 |
| 注释风格存量超出棘轮基线 | 全角标点 1315 处（基线 645），缺 docstring 431 处（基线 314），其中 4 处是去掉 BOM 后才被统计到 |
| `check_repo_layout` 缺文档 | `parallel_execution`、`piml_substructure_elasticity`、`substructure_elasticity` 缺 `results_analysis.md` |
| `docs/references/`、`docs/architecture/` 不存在 | 导致 `generate_repository_inventory --check` 失败，以及 README 与 `docs/index.md` 中 11 个坏链接 |
| torch 版本 | `ihpcm` 现为 2.11，旧于 known-issues 记录的验证环境 2.13，稀疏不变量提示无法消除 |

### 2.4 统一清理（最后做）

- **5 个 1.1.x 兼容层**：`regularization`、`analysis`、`interpolation`、`optimization`、
  `utils` 一起删除，同步清理 `tests/unit/test_compatibility_api.py` 与 `LEGACY_ROOTS`，README
  的弃用政策改为「1.2 起移除」或直接升版本号；同时决定 `fem.spaces` 转发是否一并删除。
- **残留的 FEALPy 文字**：约 94 个代码文件的注释与 docstring、`docs/fem/*.md` 中的 FEALPy
  提及，改为描述当前状态；各主题目录的 `results_analysis.md` 是历史运行记录，保持不动。
- **零散小项**：
  - `examples/pinn_elasticity/results_analysis.md` 记录的边界残差应为 `5.311e-04`
    （现写 `5.3118e-04`，与同表总损失不自洽）；
  - `examples/linear_solvers/README.md` 仍提到早已被替换的 `DirichletBCOperator`；
  - `topology` 中两处 `'jax'` 判断；
  - 产物标签 `"fealpy-numpy"`。

### 2.5 仓库之外的事项

- **`workstation:workspace/responsibilities.md`**：其中「不复制、vendor 算海仓库代码」的规范与
  本次移植的决定冲突；本仓库 README 已更新，规范正本需在 `workstation` 仓库同步。
- **FEALPy `COPYRIGHT.txt` 的商用条款**：「may not be sold or included in commercial products
  without a license」已在 `THIRD_PARTY_NOTICES.md` 原文照录；如有商用或再分发需求，需另行评估。
