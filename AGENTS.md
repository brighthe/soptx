# AGENTS.md

本文件指导 Codex 在 `soptx` 项目中工作。

## 交流语言

在本仓库工作时，一律使用简体中文与用户交流；代码、命令、文件名、专有名词可保留英文。

- 代码中的行注释、块注释和 docstring 一律使用简体中文；中文注释使用英文半角标点。

## 开始工作前

- 先确认当前分支与工作区状态；保留用户已有的未提交修改，不得回退或覆盖无关文件。
- Git 操作在 WSL 中执行。未获用户明确要求，不 commit、push、创建分支或执行破坏性 Git 操作。
- 未获用户明确要求，不运行测试、benchmark、MPI 或长时 GPU 任务；先给出可复现命令与验收条件。

## Example 目录边界

| 目录 | 职责 |
|---|---|
| `lagrange_elasticity/` | CPU 串行全装配，L2 收敛阶验证 |
| `gpu_elasticity/` | GPU 正确性对比与性能 benchmark |
| `matrix_free_elasticity/` | MPI 并行 Matrix-Free，FA/EA 双路 |
| `pinn_elasticity/` | PINN 强形式求解 |
| `huzhang_elasticity/` | Hu–Zhang 混合有限元求解 |
| `substructure_elasticity/` | 子结构静力缩聚求解与精确基线验证 |

新技术栈使用新目录；同一技术栈的不同物理问题在该目录下新增文件。不得将不同目录的验证逻辑、性能结论或运行产物混用。

## FEALPy 与实现边界

- 本地 fealpy 是一份长期维护的 vendor fork，位于 `~/workspace/fealpy`：`origin` = `brighthe/fealpy`（私有，push 开），`suanhai` = `suanhaitech/fealpy`（上游，push 禁用），工作分支 `main`。`import fealpy` 解析到的就是它；查阅上游用 `git -C ~/workspace/fealpy show suanhai/develop:<path>`。
- 通用算法能力（求解器、算子适配）优先进入该 fork；soptx 保留项目特有的物理模型与算例逻辑，避免复制通用实现。
- 修改 fealpy fork 时，必须在 `docs/known-issues/` 新增对应 `fealpy-<topic>.md`，包含概要表、逐项说明、环境与版本对照。
- GPU 代码使用 `bm.to_numpy()`，不直接调用 `.numpy()`；CPU/GPU 对比保持同一后端，仅改变 device。

## 文档与证据

- 每个主题 example 目录保持两文档结构：`README.md`（入口与文件职责）、`results_analysis.md`（代码—数学映射与验收契约、实验范围与证据解释）。
- `outputs/` 只保存运行产物，并由 `.gitignore` 忽略；结果文档不得把不同脚本、网格、边界条件或历史运行的数字混为同一证据。
- 制造解类名必须链接到 `docs/problems/manufactured-elasticity.md`（Python 使用 RST 超链接，Markdown 使用 Markdown 链接）。
- `substructure_elasticity/` 当前是精确有限元静力缩聚基线；不得将其表述为 PIML、Matrix-Free、Krylov/GPU 或拓扑优化闭环，除非相应实现与证据已实际加入。
