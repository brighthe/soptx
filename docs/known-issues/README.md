# FEALPy fork 补丁总账

SOPTX 依赖的不是上游 FEALPy，而是一份长期维护的 vendor fork。**本页是这批欠账的
唯一总表，也是全目录唯一的可变状态源**：分叉量、提交 SHA、丢弃判据的当前结论都只写
在这里。另外两份文档只记技术内容与补丁 SHA（SHA 不随分支前进而过期），不各自维护
版本对照表。

## 当前状态

| 项 | 值 |
|---|---|
| fork 位置 | `~/workspace/fealpy`（editable 安装，`import fealpy` 解析到它） |
| `origin` | `brighthe/fealpy`（私有），工作分支 `main` |
| `suanhai` | `suanhaitech/fealpy`，只读（push 已禁用），无本地分支跟踪 |
| 分叉点 | `2f1753221 feat(mesh): Support meshio.`（2026-07-26） |
| 对照基线 | `suanhai/develop` @ `69b9cb5e7`，**已于 `382869aec` 全量合入 fork** |
| 本地 `main` / `origin/main` | `bc2ea8ecb` —— 已同步，双向独有均为 0 |
| 分叉量 | ahead **19** / behind **0** |
| 工作区 | 干净，无未提交或未跟踪文件 |
| 核对日期 | 2026-08-27 |

两条记账口径：

- **「对照基线」是选定的参考点，不是 fork 的来源分支。** 分叉点 `2f1753221` 同时存在于
  `suanhai/develop` 与 `suanhai/main` 上，单看它判断不出当初从哪条拉的。本目录一律以
  `develop` 作为比对基准，因为它是上游的开发主线。
- **上游只作参考，从不回推。** 因此「跟上上游」不是义务，是成本收益判断；丢弃判据
  只在**决定移植时**才需要执行，日常开发不受其约束。

## 目录结构

**全目录只有两份文档**：

| 文件 | 回答什么问题 | 内容 |
|---|---|---|
| 本页 | *和上游差在哪？能不能丢？下次移植怎么做？* | 补丁总表（一补丁一行）、当前状态、丢弃判据结论、测试覆盖、移植操作规程、记账约定 |
| [fealpy-patches.md](fealpy-patches.md) | *为什么这么改？* | 全部**存活**补丁的技术正文，按子系统分四节：<br>一 `mesh/`+`functionspace/`｜二 `backend/`+`sparse/`｜三 `solver/`+`distributed/`｜四 `functional.py` |

**已丢弃补丁的技术正文不在本目录内。** 它们的代码效果已被上游覆盖、代码对比针对的是
不存在的架构，留着只会误导下一轮移植；提交本身仍在 fork 的历史里，需要时
`git -C ~/workspace/fealpy show <SHA>` 直接看原始 diff。

---

## 补丁总表

19 个 fork-only commit，按 fealpy 子系统分组。**每个补丁只在这里出现一次**：内容、
类别、丢弃判据与当前结论都在同一行。

类别决定丢弃逻辑：**上游缺陷修复**（上游自行修好即删）、**能力增强**（上游长出等价
能力即删）、**fork 内部回归修复**（不因上游进步而失效，只能整栈保留或整栈还原）、
**纯文档**（不产生运行期行为）。

### mesh / functionspace 层 → [fealpy-patches.md](fealpy-patches.md) 第一节

| 补丁 | 内容 | 类别 | 丢弃判据 | 现状 | 测试 |
|---|---|---|---|---|---|
| `0758339` | 缺陷 1 的 `tensor_space.py` 半条：`TD = sum(item.shape[-1] - 1 ...)` | 上游缺陷 | `git show suanhai/develop:fealpy/functionspace/tensor_space.py \| grep -n "len(bc) == 1"` 非空 | ❌ 仍是 `TD = len(bc)`，**保留** | 缺陷 1 有 |
| `fbfe39e` | 缺陷 5：`face_basis`/`edge_basis` 走面实体，不再别名 `basis` | 上游缺陷 | `git show suanhai/develop:fealpy/functionspace/lagrange_fe_space.py \| grep -n "face_basis = basis"` 返回空 | ❌ 上游仍是别名，**保留**（已在 v0.4 新文件上重新落位） | 有 |
| `ce8aa8ae9` | 缺陷 8：经典 `shape_function` 路径不再反置换张量积基函数列序 | 上游缺陷（v0.4 新引入） | 上游 `_legacy_basis_indices` 对张量积 schema 整体豁免反置换，且四边形 $p=1$ 收敛阶达 2.0 | ❌ 上游未修，**保留**。不带它则四边形单元静默算错 | 收敛阶 + delta |
| `13e8ebb75` | 缺陷 9：张量积 `multi_index` 列序对齐到顶点序（新增 `multi_index_vertex_columns()` 钩子） | 上游缺陷（v0.4 新引入） | 上游 `EntitySchema.multi_index` 的列确实「one column per vertex」，或 `ipoints`/`to_ipoint_permutation` 不再假定列序即顶点序 | ❌ 上游未修，**保留**。判据直接跑下方 delta 测试，22 用例全绿才算成立 | delta + 收敛阶 |
| `bc2ea8ecb` | 缺陷 9 的 delta 判据回归测试 | 测试 | 随 `13e8ebb75` 一同存亡 | **保留** | 自身即测试 |

### backend / sparse 层 → [fealpy-patches.md](fealpy-patches.md) 第二节

| 补丁 | 内容 | 类别 | 丢弃判据 | 现状 | 测试 |
|---|---|---|---|---|---|
| `09783f643` | PyTorch 后端 numpy 语义对齐 4 条：`split` 拒收 1-D 序列、`cumsum`/`cumprod` 无轴崩溃、`CSRTensor.__getitem__` list-into-concat、`add_at` 不累加重复索引 | 上游缺陷 | 四条语义全部对齐，且纯上游检出上 `verify_matrix_free_ea_operator.py --backend pytorch` 判据全通过 | ❌ `pytorch_backend.py` 上游 0 个新提交，四条全在，**保留** | `split`、`add_at` 有；`cumsum`、`CSRTensor` **无** |
| `ceb0c61` | PyTorch 后端补 `take_along_axis`、`unique_counts` | 能力增强 | 上游 `PyTorchBackend` 自带这两个 | ❌ 均无，**保留** | **无** |
| `88cf4fa` | `COOTensor`/`CSRTensor` 暴露 `device` property | 能力增强 | 上游三个 sparse 文件自带 `device` **property**（而非仅 `device_put` 方法） | ❌ 只有 `device_put`，**保留** | **无** |
| `c33db98ae` | backend 层静态类型标注补齐（6 个缺口，消除 681 条假阳性） | 能力增强（纯静态） | 上游 `TensorLike` 自带位运算与标量转换 dunder，且 `manager.pyi` 已声明 `random`/`linalg` 与放宽的索引/操作数类型（三条 grep 判据见 [fealpy-patches.md](fealpy-patches.md) §2.3） | ❌ 两文件上游 0 个新提交，**保留** | 诊断数对照（非 pytest） |

### solver / distributed 层 → [fealpy-patches.md](fealpy-patches.md) 第三节

| 补丁 | 内容 | 类别 | 丢弃判据 | 现状 | 测试 |
|---|---|---|---|---|---|
| `875496d` | CG 支持可插拔内积 `dot_product` 与真残差刷新 | 能力增强 | 见下一行——两者是一个整栈 | ❌ 上游 `cg()` 仍无 `dot_product`，**保留** | 由 `40016dc56` 的测试覆盖 |
| `40016dc56` | 修复 `875496d` 打穿的 5 条路径：`r·z` 预条件内积、批量归约、`maxit=None`、`dot_product` 批量拦截、零右端项走 `_dot` | fork 内部回归 | **不适用**。唯一删除条件是整栈还原：`src/soptx/fem/matrix_free/krylov.py` 不再需要 `dot_product`/`residual_refresh` 时，把 `cg.py` 直接 checkout 回上游，连 `875496d` 一起丢。**不得只回退本条而保留 `875496d`** | **保留** | 有（11 函数 / 13 用例） |
| `4c5887d` | `EntityMPI.dot()` 重叠修正内积工厂 | 能力增强 | 上游 `EntityMPI` 自带 `dot()` | ❌ 只有 `refs()`，**保留** | **无** |
| `30ca15599` | MUMPS 直接解法暴露 `sym` 对称性标志（峰值 RSS 28.7 → 17.4 GiB） | 能力增强 | 上游 `_mumps_solve`/`spsolve` 自带对称性入口，且 `sym` 非零时只传下三角 | ❌ 仍是 `_mumps_solve(A, b)` + 硬编码 `DMumpsContext()`，**保留**。⚠️ 上游已把 `spsolve` 默认 solver 改为 `"scipy"`，该分支忽略 `sym`，调用侧必须显式传 `solver="mumps"`，详见 [fealpy-patches.md](fealpy-patches.md) §3.2 | 峰值 RSS（非 pytest） |
| `ffbb1fde3` | `distribute_mesh` docstring 与行注释汉化 | 纯文档 | 不适用。与上游对同一 docstring 的改动冲突时，**优先取上游正文**再重新汉化 | **保留**（自动合并，未冲突） | 不适用 |

### functional → [fealpy-patches.md](fealpy-patches.md) 第四节

| 补丁 | 内容 | 类别 | 丢弃判据 | 现状 | 测试 |
|---|---|---|---|---|---|
| `a1ec1086d` | `linear_integral` 避免把与单元无关的基函数沿单元轴广播物化（`load` 阶段 +3.2 GiB → 0） | 上游缺陷 | 上游 `linear_integral` 自身对 `basis.shape[0] == 1` 分支，**或** `NumPyBackend.einsum` 不再无条件 `optimize=True`，**或** `LagrangeFESpace.basis()` 不再返回 `C = 1` | ❌ 三条均不满足，**保留** | **无**（48 组一次性参照比对） |

### 合并提交与已丢弃的补丁

| 补丁 | 内容 | 现状 |
|---|---|---|
| `382869aec` | **合并上游 `suanhai/develop` @ `69b9cb5e7`（v0.4 mesh 架构重写，243 个提交）** | 243 个上游提交里只有 5 个文件冲突，其余补丁全部自动合并。合并后 fealpy 742 passed / 19 skipped；经验见下方[移植操作规程](#移植操作规程) |
| `0758339` 的缺陷 2、3、4 | 四边形/六面体基函数排序与 `bc_to_point` reshape | **效果已丢弃**：缺陷 4 上游 `80feddb` 自行修好（逐字相同）；缺陷 2、3 落点随 `709a8922c` 消失，症状换落点重现为缺陷 8 |
| `de7a6a8d7` | 缺陷 6：张量积高阶元 $p \ge 2$ 拓扑排列错位 | **效果已丢弃**：落点消失。症状换落点重现为缺陷 9 |
| `7fdb98500` | 缺陷 7：四边形 `grad_shape_function_barycentric` 改为显式报错 | **效果已丢弃**：上游把该方法所在的三方法组整体移出 schema 层 |
| `824dc4f39` | `quadrature_formula` 接受 `etype` 旧参数名 | **效果已丢弃**：落点文件被 `af0e52ad8` 删除。上游遗留的 10 处 `etype=` 调用是上游自己的缺口，见下方[待向上游反馈](#待向上游反馈) |

> 这四条的提交仍留在 `main` 的历史里（计入 ahead 19），但它们的代码效果在
> `382869aec` 合并时已被整边取上游覆盖。下一轮移植时不必再评估；确需看当时改了什么，
> `git -C ~/workspace/fealpy show <SHA>`。

---

## 测试覆盖现状

fork 内有六处 pytest 保护这批补丁：

| 文件 | 覆盖 |
|---|---|
| `tests/mesh/unit/test_interpolation_delta.py` | **缺陷 8、9**，兼覆盖旧缺陷 2、3、6 的症状域。22 个用例（4 种网格 × $p=1..4$，加 6 个 schema 的根因断言），0.8 秒 |
| `tests/functionspace/unit/test_barycentric_tuple_dimension.py` | 缺陷 1 |
| `tests/functionspace/unit/test_face_basis_entity_dimension.py` | 缺陷 5 |
| `test/solver/test_cg.py`（上游目录，见下方约定的例外） | `875496d` + `40016dc56` 整栈：批量右端项、`batch_first`、逐列独立求解、`maxit=None`、`maxit` 耗尽、Jacobi 预条件、自定义内积恒等式与批量拦截、真残差刷新、零右端项。11 函数 / 13 用例 |
| `test/backend/test_backends.py::test_split`（实现上游占位 stub） | numpy 语义对齐的 `split` 子项 |
| `test/backend/test_backends.py::test_add_at`（上游目录，新增） | numpy 语义对齐的 `add_at` 子项（6 组数据全部含重复索引） |

**14 个仍然存活的补丁里有 6 个受 pytest 保护。** 没有可重跑保护的是：PyTorch
`take_along_axis`/`unique_counts`、`COOTensor.device`、`EntityMPI.dot`、
`linear_integral` 广播，以及 `09783f643` 里的 `cumsum`/`CSRTensor.__getitem__` 两个
子项。其中 backend 类型标注、MUMPS `sym` 与 `linear_integral` 广播三条各自有一次性
脚本或指标核对过（诊断数 3200→2519 / 峰值 RSS / 48 组参照比对），但都没沉淀成测试
文件，换一台机器或换一次移植就不再有人跑。

**缺陷 8 与 9 是这批里风险最高的两格**：都是静默错误——四边形算出来的结果是错的，
但残差合格、零空间正确，只有观测收敛阶能发现。上游自己的 742 个测试在这两个缺陷
存在时全部通过，因此**上游测试不构成保护**。`tests/mesh/unit/test_interpolation_delta.py`
是唯一能抓住这类静默错位的自动化保护，**下一轮移植上游后必须先跑它**：

```bash
cd ~/workspace/fealpy && ~/miniconda3/envs/ihpcm/bin/python -m pytest tests/mesh/unit/test_interpolation_delta.py -q
```

`add_at` 同属静默错误（重复索引下丢贡献，直接污染节点密度的灵敏度与梯度，见
[fealpy-patches.md](fealpy-patches.md) §2.1.4），但已由 `test_add_at` 覆盖。

上游 `tests/functionspace/unit/` 目录**不存在**（`tests/functionspace/` 下只有
`test_lagrange_fe_space_p1_p2.py`），所以 fork 自建的那两个测试没有命名冲突，但也
意味着丢掉它们**不会产生任何提示**。上游那个新测试名义上覆盖缺陷 1/2/3/6 所在区域，
**但实测证明它覆盖不到基函数列序与 `cell_to_ipoint` 的一致性**（缺陷 8 存在时它照样
全绿），不要把它当作该区域的保护。

> **关于 `reproduce_tensor_product_issue.py`**：本页历史版本称「两个仓库中均无此
> 文件」，这是错的——它仍在 SOPTX 的 git HEAD 里（`docs/known-issues/`，7082 字节，
> 提交 `4875dad`），只是工作区已删除且删除尚未提交。它的职责现已由
> `tests/mesh/unit/test_interpolation_delta.py` 取代，且后者更好：只依赖
> `interpolation_points` / `cell_to_ipoint` / `bc_to_point` 三个公开接口，不碰内部
> schema API，不会随下一次架构重写而失效。**该脚本可以正式删除**，删除时把这一条
> 一并从本页移除。

---

## 移植操作规程

从 2026-08-26 那次移植（合并 `382869aec`，上游 243 个提交、v0.4 mesh 层整层重写）
沉淀下来的三条，下一轮 `git fetch suanhai` 时直接照做：

1. **先 merge 再说。** 移植前的判断是「不要 merge，从 `suanhai/develop` 开新分支重放」，
   这个判断是**错的**：243 个上游提交里只有 5 个文件产生冲突，其余 fork 补丁全部自动
   合并。「落点消失」只决定冲突时取哪一边，不构成放弃 merge 的理由；重放路线会丢掉
   上游提交与 fork 提交之间全部的自动合并成果。`merge --no-commit --no-ff` 可以随时
   `merge --abort`，成本远低于手工重放。
2. **落点消失 ≠ 问题已解决，单元测试全绿 ≠ 没问题。** 缺陷 2、3、6 的补丁确实无处安放
   并已丢弃，但合并后跑收敛阶立刻发现 Q1 观测阶塌到 0.1113——症状在新架构里换了个落点
   重现（缺陷 8、9）。这两个缺陷存在时上游 742 个测试（含它自己新增的
   `tests/functionspace/test_lagrange_fe_space_p1_p2.py`）**全部通过**，只有收敛阶算例
   能发现。手工验证收敛阶这一步不可省略。
3. **先跑 delta 判据再跑收敛阶。** 验证 $\phi_i(\mathbf{x}_j) = \delta_{ij}$ 比跑一整轮
   收敛阶快得多，且能直接定位到是哪几个基函数错位。该判据已沉淀为
   `tests/mesh/unit/test_interpolation_delta.py`，合并后第一件事就是跑它。

评估用的临时检出从 fork 开 worktree 即可，对脏工作区安全：

```bash
git -C ~/workspace/fealpy worktree add ~/workspace/upstream-check suanhai/develop
```

### 待向上游反馈

2026-08-26 合并时发现的三个上游自身问题，非本 fork 造成，均未修：

1. **上游提交里有未解决的冲突标记**：`fealpy/fem/polyharmonic_cr_fem_model.py:36-48`
   带着 `<<<<<<< HEAD` / `>>>>>>> upstream/develop` 被 `773b9fb60` 提交进了 `develop`。
   该文件不在 `import fealpy` 的导入链上，不影响使用，但它是上游 CI 缺口的直接证据。
2. **上游删了 `fealpy_api` 却漏改自己的测试**：`af0e52ad8` 删除了
   `mesh/view/fealpy_api.py` 与 `MeshView.fealpy_api()`，但上游 `test/` 下仍有 6 个文件
   在调用 `.fealpy_api()`。
3. **`quadrature_formula` 的 `etype` 兼容缺口**：上游库内仍有 10 处 `etype=` 关键字旧调用
   （`fsi/coupling_interface.py`、`functionspace/parametric_lagrange_fe_space.py`、
   `functionspace/scaled_monomial_space_2d.py`），而新的 schema 签名
   `quadrature_formula(cls, q, qtype=None, device=None)` 没有 `etype` 形参，这些路径会抛
   `TypeError`。fork 原有的 `824dc4f39` shim 因落点文件被删而丢弃；SOPTX 全部 33 处
   `quadrature_formula(q, 'cell')` 走位置参数，不受影响。

---

## 记账约定

**两份文档，各有各的不变量——不要再增加第三份。**

- **可变状态只写在本页。** 分叉量、`main`/`origin` 的 SHA、丢弃判据的当前结论都归本页；
  `fealpy-patches.md` 只写技术内容与补丁 SHA。曾经每份专题文档各带一张「环境与版本
  对照」表，同一个 `ahead N` 散在五处，一次提交就能让五处同时过期——不要再引入这种结构。
- **补丁 SHA 是不变量，`main` 的 HEAD 不是。** 分支会前进，写死分支 revision 下一次
  提交就过期，而补丁 SHA 在移植前始终有效。正文里只写前者。
- **对照基线 SHA 随每次 `git fetch suanhai` 更新。** 基线过期会让丢弃判据拿旧上游
  比对——2026-08-26 那次就查出缺陷 1、4 早已被上游修复，而判据仍写着「未满足，保留」。
- **新增补丁：往 `fealpy-patches.md` 对应子系统那一节里加，同时在本页总表加一行。**
  只有触及全新子系统时才在该文件里开新的一级节，不新建文件。仓库入口
  （`docs/index.md`、根 `README.md`）只指向本页。
- **补丁一旦「效果已丢弃」，正文从 `fealpy-patches.md` 直接删除**，本页总表保留一行
  「效果已丢弃」即可，不另存档。已死补丁的代码对比针对的是已经不存在的架构，留着会持续
  占篇幅并误导下一轮移植；提交仍在 fork 历史里，`git show <SHA>` 就是它的正本。
- **移植过程只留结论，不留流水账。** 一次 fetch/merge 的计划表、逐条碰撞预判 vs 实测
  这类内容，价值在移植完成的那天就到顶了；真正要传下去的是「下次该怎么做」，写进
  [移植操作规程](#移植操作规程)（覆盖旧条目，不追加历次记录），以及总表里各补丁现状的
  更新。
- 补丁自带的 pytest 文件放在 fork 内的 `tests/` 下，不放 SOPTX；本页只记录路径和
  覆盖范围。已有两处例外并均已在 `fealpy-patches.md` 中说明：
  `test/backend/test_backends.py`（上游已有同名 stub，就地实现以便产生冲突提示）与
  `test/solver/test_cg.py`（上游 `test/solver/` 已是既有目录，其余求解器的测试都在
  那里；上游没有 `test_cg.py`，就地新建以便上游日后补测试时能产生冲突提示）。
