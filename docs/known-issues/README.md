# FEALPy fork 补丁总账

SOPTX 依赖的不是上游 FEALPy，而是一份长期维护的 vendor fork：

| 项 | 值 |
| --- | --- |
| fork 位置 | `~/workspace/fealpy`（editable 安装，`import fealpy` 解析到它） |
| `origin` | `brighthe/fealpy`（私有），工作分支 `main` |
| `suanhai` | `suanhaitech/fealpy`，只读（push 已禁用） |
| 上游基线 | `suanhai/develop`，最近一次 fetch `7b76f0d`（2026-08-09） |
| 分叉量 | ahead 7 / behind 21 |

上游 `4.0.0-alpha` 存在多处回归，修复直接落在 fork 上而非通过 PR 回流，因此
fork 与上游长期分叉。**本页是这批欠账的唯一总表**：谁改了什么、有没有测试保护、
merge 上游时该检查什么、什么条件下该把补丁丢掉。缺陷的症状、根因和修复代码由各
专题文档维护，本页不复述。

---

## 补丁清单

7 个 fork-only commit，13 个文件。

| commit | 内容 | 触及文件 | 文档 | 测试 |
| --- | --- | --- | --- | --- |
| `0758339` | mesh 重构回归缺陷 1–4 | `functionspace/lagrange_fe_space.py`<br>`functionspace/tensor_space.py`<br>`mesh/schema/classic/quadrilateral.py`<br>`mesh/schema/classic/hexahedron.py` | [tensor-product-mesh](fealpy-tensor-product-mesh.md) | 仅缺陷 1 |
| `fbfe39e` | `face_basis`/`edge_basis` 走面实体（缺陷 5） | `functionspace/lagrange_fe_space.py` | [tensor-product-mesh](fealpy-tensor-product-mesh.md) | 有 |
| `ceb0c61` | PyTorch 后端补 `take_along_axis`、`unique_counts` | `backend/pytorch_backend.py` | [pytorch-backend-gpu](fealpy-pytorch-backend-gpu.md) | **无** |
| `88cf4fa` | `COOTensor`/`CSRTensor` 暴露 `device` | `sparse/{coo,csr,sparse}_tensor.py` | [pytorch-backend-gpu](fealpy-pytorch-backend-gpu.md) | **无** |
| `4c5887d` | `EntityMPI.dot()` 重叠修正内积工厂 | `distributed/entity_mpi.py` | [cg-pluggable-dot-product](fealpy-cg-pluggable-dot-product.md) | **无** |
| `875496d` | CG 可插拔内积与真残差刷新 | `solver/cg.py` | [cg-pluggable-dot-product](fealpy-cg-pluggable-dot-product.md) | **无** |
| `824dc4f` | `quadrature_formula` 接受 `etype` 旧参数名 | `mesh/view/fealpy_api.py` | [mesh-quadrature-etype-alias](fealpy-mesh-quadrature-etype-alias.md) | **无** |
| *（工作区，未提交）* | backend 层类型标注补齐（6 处缺口） | `backend/base.py`<br>`backend/manager.pyi` | [backend-static-typing](fealpy-backend-static-typing.md) | **无**（判据为诊断数） |

前两条是**修复上游缺陷**，`824dc4f` 是**向后兼容 shim**，其余四条是**能力增强**
（上游本来就没有）。三类的丢弃判据不同，见下文。

最后一行是**纯静态类型标注**，不产生任何运行期行为，单独成一类：它既不会被 pytest
捕捉，也不会与上游的功能改动冲突。目前还在 fork 工作区里没有提交，提交后回填 SHA。

### 测试覆盖现状

fork 内只有两个 pytest 文件保护这批补丁：

| 文件 | 覆盖 |
| --- | --- |
| `tests/functionspace/unit/test_barycentric_tuple_dimension.py` | 缺陷 1 |
| `tests/functionspace/unit/test_face_basis_entity_dimension.py` | 缺陷 5 |

**7 个补丁里 5 个没有任何自动化保护**：PyTorch 三处、CG 与 `EntityMPI.dot` 两处、
`etype` shim 一处，以及 `0758339` 里的缺陷 2–4。其中缺陷 2 是**静默错误**——四边形
算出来的结果是错的，但残差合格、零空间正确，只有观测收敛阶能发现。这是当前风险
最高的一格。

上游 `tests/functionspace/unit/` 下没有同名文件，所以 merge 时把这两个测试一并丢掉
**不会产生冲突提示**。

---

## merge 上游前的检查清单

按顺序执行，不要跳步：

1. `git -C ~/workspace/fealpy fetch suanhai`，重新统计 ahead/behind。
2. 对照下节「已知碰撞」，确认哪些补丁所在文件被上游改过。
3. 逐条走「丢弃判据」，先决定哪些补丁应当主动删除，再开始 merge——带着已经该丢的
   补丁去解冲突，只会解出一堆没用的三方合并。
4. merge 后跑 `tests/functionspace/unit/` 下两个测试文件。
5. 跑 [`reproduce_tensor_product_issue.py`](reproduce_tensor_product_issue.py)，
   它覆盖缺陷 1–5 全部五条判据，是缺陷 2–4 唯一的检查手段。
6. 跑 SOPTX 侧门禁（`tools/matrix_free_evidence/validate.py --dim all`
   等），确认下游没被上游的行为变化打穿。
7. 更新本页的分叉量、上游基线 SHA 和补丁清单。

`rerere.enabled = true` 已在 fork 上打开，重复 merge 的相同冲突会自动复用上次的
解法。

### 已知碰撞（截至上游 `7b76f0d`）

上游 21 个未合并提交里，有三个直接改到了 fork 补丁所在的文件：

| 上游 commit | 内容 | 碰撞的 fork 补丁 |
| --- | --- | --- |
| `1fede55` | `fix: fix TD inference for simplex elements wrapped in 1-tuple by integral()` | `0758339` 缺陷 1、`fbfe39e` 缺陷 5（同为 `lagrange_fe_space.py`） |
| `80feddb` | `fix(mesh): correct classic schema geometry integration` | `0758339` 缺陷 2–4（`quadrilateral.py`、`hexahedron.py`） |
| `9b783a1` | `feat(distributed): update EntityMPI.refs. No longer requires size when global indices are given.` | `4c5887d`（`entity_mpi.py`） |

`1fede55` 和 `80feddb` 看标题都在修同一批问题，**merge 时优先判断上游是否已经自行
修复，而不是硬保本地补丁**。`9b783a1` 改的是 `refs()` 签名，而 `EntityMPI.dot()` 和
SOPTX 的 `OverlapOperator` 都调用 `refs(local_size)`——签名放宽后本地传 size 的写法
是否仍然成立，需要单独确认。

---

## 丢弃判据

补丁不是资产，是负债。满足以下条件即应在 merge 时删除，不要保留。

| 补丁 | 判据 |
| --- | --- |
| `0758339`、`fbfe39e` | [`reproduce_tensor_product_issue.py`](reproduce_tensor_product_issue.py) 在**纯上游检出**上输出 `ALL PASSED` |
| `824dc4f` | 上游 `grep -rn "quadrature_formula(.*etype=" fealpy \| grep -v /old/ \| grep -v mesh_old` 返回空 |
| `ceb0c61` | 上游 `PyTorchBackend` 自带 `take_along_axis` 与 `unique_counts` |
| `88cf4fa` | 上游 `COOTensor`/`CSRTensor` 自带 `device` property |
| `875496d` | 上游 `cg()` 签名自带 `dot_product`（或等效的可插拔内积入口） |
| `4c5887d` | 上游 `EntityMPI` 自带 `dot()` |
| backend 类型标注 | 上游 `TensorLike` 自带位运算与标量转换 dunder，且 `manager.pyi` 已声明 `random`/`linalg` 与放宽的索引/操作数类型（判据脚本见专题文档） |

在上游代码上跑复现脚本不需要单独维护一份上游检出，从 fork 开临时 worktree 即可：

```bash
git -C ~/workspace/fealpy worktree add ~/workspace/upstream-check suanhai/develop
PYTHONPATH=~/workspace/upstream-check python ~/workspace/soptx/docs/known-issues/reproduce_tensor_product_issue.py
git -C ~/workspace/fealpy worktree remove ~/workspace/upstream-check
```

---

## 记账约定

- **各文档的版本对照表记补丁所在的提交，不记 `main` 的 HEAD。** 分支会前进，写死
  分支 revision 下一次提交就过期，而补丁 SHA 在 merge 前保持有效。
- **新增 `fealpy-<topic>.md` 时，同时在本页「补丁清单」加一行。** 仓库入口
  （`docs/index.md`、`README.md`）只指向本页，不逐份枚举专题文档。
- 文档格式仍按 `CLAUDE.md` 的要求：概要表 → 逐项详述 → 回归测试 → 环境与版本对照。
- 补丁自带的 pytest 文件放在 fork 内（`tests/` 下），不放 SOPTX；SOPTX 侧只记录它
  的路径和覆盖范围。跨仓库的复现脚本是例外，因为它只依赖 FEALPy 公开接口，放在
  本目录便于在上游检出上直接运行。
