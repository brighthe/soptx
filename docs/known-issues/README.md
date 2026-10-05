# FEALPy 来源与已内联补丁清单

SOPTX 原先依赖一份长期维护的 FEALPy vendor fork。2026-10 起，SOPTX 用到的 FEALPy 代码已
全部移植为 `soptx` 子包，fork 随之退役。**本页是该目录唯一的可变状态源**：移植状态、
fork 补丁在 SOPTX 中的落点，以及移植后遗留问题都只写在这里。另一份
[fealpy-patches.md](fealpy-patches.md) 是各补丁的技术正文（缺陷机理与修改点），属于移植前
写成的历史记录。来源与许可证声明见仓库根目录的
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)。

## 当前状态

| 项 | 值 |
|---|---|
| 移植源 | `brighthe/fealpy`（私有）`main` @ `f474a5775`，含 fork 上全部存活补丁 |
| 移植提交 | 搬入 `bf2f362`、引用切换 `b55f265`、PINN 示例 `3417b28`、移除依赖 `6712050` |
| SOPTX 依赖 | `pyproject.toml` 不再声明 `fealpy`；`tools/check_architecture.py` 禁止仓库内导入 `fealpy` |
| fork | 已退役：不再安装、不再合并上游、不再打补丁；工作副本 `~/codespace/fealpy` 只读保留，不删除 |
| 移植验证 | 拦截 `fealpy` 导入下全量 pytest 903 passed / 10 skipped，逐用例与移植前基线一致；8 个 examples 数值输出与移植前一致 |
| 核对日期 | 2026-10-04 |

## 已内联补丁

fork 上存活的补丁都已随代码进入 SOPTX，下表给出落点（路径相对 `src/soptx/`）。补丁 SHA 指
fork 中的提交，需要看原始 diff 时 `git -C ~/codespace/fealpy show <SHA>`。

| 补丁 | 内容 | SOPTX 落点 | 回归保护 |
|---|---|---|---|
| `0758339` | 缺陷 1：重心坐标 tuple 的 `TD` 计算 | `functionspace/tensor_space.py` | 无 pytest（fork 测试未移植） |
| `fbfe39e` | 缺陷 5：`face_basis` / `edge_basis` 走面实体 | `functionspace/lagrange_fe_space.py` | 无 pytest（fork 测试未移植） |
| `ce8aa8ae9` | 缺陷 8：经典 `shape_function` 路径不再反置换张量积基函数列序 | `mesh/view/entity_view.py` | **无 pytest**，见下方说明 |
| `13e8ebb75` | 缺陷 9：张量积 `multi_index` 列序对齐到顶点序 | `mesh/ipoints.py`、`mesh/schema/classic/base.py`、`mesh/schema/entity_schema.py` | **无 pytest**，见下方说明 |
| `bc2ea8ecb` | 缺陷 9 的 delta 判据测试 | 未移植 | — |
| `09783f643` | PyTorch 后端对齐 numpy 语义（`split`、`cumsum`/`cumprod`、`CSRTensor.__getitem__`、`add_at`） | `backend/pytorch_backend.py`、`sparse/csr_tensor.py` | 无专门 pytest |
| `ceb0c61` | PyTorch 后端补 `take_along_axis`、`unique_counts` | `backend/pytorch_backend.py` | 无专门 pytest |
| `88cf4fa` | `COOTensor` / `CSRTensor` 暴露 `device` 属性 | `sparse/coo_tensor.py`、`sparse/csr_tensor.py`、`sparse/sparse_tensor.py` | 无专门 pytest |
| `66a040cfd` | `hstack` 列偏移修正 | `sparse/ops.py` | 无 pytest（fork 测试未移植） |
| `c33db98ae` | backend 层静态类型标注补齐 | `backend/base.py`、`backend/manager.pyi` | 纯静态 |
| `875496d` + `40016dc56` | CG 可插拔内积与真残差刷新，及其 5 条回归修复 | `solvers/cg.py`（早于本次移植） | `tests/unit/test_solvers_cg*.py` |
| `4c5887d` | `EntityMPI.dot()` 重叠修正内积工厂 | `fem/distributed/entity_mpi.py`（早于本次移植） | 需 mpi4py |
| `30ca15599` | MUMPS 直接解法暴露对称性标志 `sym` | `solvers/direct.py`（早于本次移植） | 需 PyMUMPS |
| `ffbb1fde3` | `distribute_mesh` 注释汉化 | `fem/distributed/mesh.py`（早于本次移植） | 不适用 |
| `a1ec1086d` | `linear_integral` 避免把与单元无关的基函数沿单元轴广播物化 | `fem/functional.py` | 无 pytest |

**缺陷 8 与 9 是风险最高的两条**：都是静默错误，装配出的刚度矩阵照样对称正定、CG 照样
收敛，只有张量积网格的观测收敛阶会塌。fork 内唯一能抓住它们的
`tests/mesh/unit/test_interpolation_delta.py` 没有移植。修改 `soptx.mesh` 的
`view/entity_view.py`、`ipoints.py` 或 `schema/` 后，须手工复核四边形与六面体的收敛阶：

```bash
PYTHONPATH=$PWD/src python examples/lagrange_elasticity/manufactured_convergence_demo.py --mesh-type quad --degree 2
PYTHONPATH=$PWD/src python examples/lagrange_elasticity/manufactured_convergence_demo.py --dim 3 --mesh-type hex
```

移植前的输出（L2 观测收敛阶与误差）存于 `~/codespace/soptx-baseline/aa58af5/examples/`
的 `lag2d_quad_p2.txt`、`lag3d_hex.txt`，复核时逐行比对。

## 移植后遗留

移植一律原样复制，过程中发现的问题记在这里，不在移植中顺手修；修复后删除对应行。

| 问题 | 位置 | 来源 | 根因与影响 | 修法 | 状态 |
|---|---|---|---|---|---|
| PyTorch 后端下 $p \ge 2$ 的 `interpolation_points` 抛 `RuntimeError: expected scalar type Float but found Double` | `mesh/ipoints.py:389`（fork 同一行为 `fealpy/mesh/ipoints.py:387`） | 上游 `d97a78f12`，非 fork 补丁引入 | `weights = mi / bm.sum(mi, ...)` 由整数张量相除得到，torch 给出默认精度 float32，与 float64 的节点坐标在 `einsum` 处类型不符；numpy 下整数相除即为 float64，不受影响。pytorch 后端下所有 $p \ge 2$ 的插值点不可用；gpu 系列 examples 默认 $p = 1$，至今未暴露 | 把 `weights` 转为 `points.dtype` 后再收缩 | 未修 |
| 移植代码的 docstring 为英文且大量缺失 | `tools/check_comment_style.py` 的 `PORTED_ROOTS` 所列路径 | 移植原样保留 | 豁免前约 596 处缺 docstring、16 处全角标点；豁免使其暂不计入棘轮，基线数字不放松 | 按子包补中文 numpydoc，补齐后从 `PORTED_ROOTS` 移出 | 未修 |
| Form 过渡基类 | `fem/_bilinear_form_base.py`、`fem/_linear_form_base.py` | 移植时为避开同名子类而改名 | SOPTX 的 `BilinearForm` / `LinearForm` 子类继承它们；部分调用方（Hu--Zhang 与拉格朗日分析器、子结构、若干测试与示例）直接使用基类 | 与子类合并为单个类；合并时注意直接使用基类的调用方会改用子类的装配实现 | 未修 |
| 调用网格上不存在的方法 | `functionspace/huzhang_fe_space_2d.py:1080`（`mesh.hess_shape_function`）；`functionspace/lagrange_fe_space.py:192-216, 299`（`mesh.number_of_faces_of_cells`、`mesh.update_bcs`、`mesh.prolongation_matrix`）；`fem/utils.py:40-54`（`mesh.bisect_options` 与 `isMarkedCell=` 关键字） | v0.4 网格重写后上游即已悬空，移植未使其变差 | 走到这些分支即 `AttributeError` / `TypeError`；`fem/utils.py` 中的 `project_solution_to_finer_mesh` 无任何调用方 | 删除无调用方的函数；其余按需补实现或删除分支 | 未修 |
| `fem/distributed/` 中源自 FEALPy 的文件未注明来源 | `fem/distributed/mesh.py`、`entity_mpi.py`、`space.py` | 早于本次移植（`abd0945`） | 与 fork `fealpy/distributed/` 逐行比对高度相似，`THIRD_PARTY_NOTICES.md` 已登记 | 文件头补来源说明 | 未修 |

## 记账约定

**两份文档，各有各的不变量——不要再增加第三份。**

- **移植代码就是 SOPTX 代码。** 日常修改按普通代码提交与评审，不需要在本目录记账。
  只有两种情况要回来更新本页：修改触及上表的补丁落点（例如改写了缺陷 8、9 所在的代码，
  使 `fealpy-patches.md` 的正文不再成立），或修复 / 新发现移植后遗留问题。
- **可变状态只写在本页。** `fealpy-patches.md` 是移植前的技术正文，只读保留；其中的
  「丢弃判据」「与上游比对」类段落描述的是 fork 时代的维护流程，已不再执行。
- **再从 FEALPy 取代码**时，在 `THIRD_PARTY_NOTICES.md` 登记源提交与范围，并在文件头注明
  来源；若带来新的缺陷修复，在本页「已内联补丁」加一行。
- 仓库入口（`docs/index.md`、根 `README.md`）只指向本页。
