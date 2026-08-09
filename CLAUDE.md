# CLAUDE.md

本文件指导 Claude Code 在 `soptx` 项目中工作。

## 交流语言

在本仓库工作时，一律使用简体中文与用户交流；代码、命令、文件名、专有名词可保留英文。

## 开始工作前

先确认当前分支（`git branch --show-current`），确保在正确的分支上开发。

## 三条规则

### 1. Example 目录不能交叉污染

| 目录 | 职责 |
|------|------|
| `lagrange_elasticity/` | CPU 串行全装配，L2 收敛阶验证 |
| `gpu_elasticity/` | GPU 正确性对比 + 性能 benchmark |
| `matrix_free_elasticity/` | MPI 并行 matrix-free，FA/EA 双路 |
| `pinn_elasticity/` | PINN 强形式求解 |

新技术栈 → 新目录；同技术栈不同物理问题 → 同目录下新文件。

### 2. 改 fealpy 必须写 known-issue

本地 fealpy 是一份长期维护的 vendor fork，位于 `~/workspace/fealpy`：

| remote | 指向 | push |
|--------|------|------|
| `origin` | `brighthe/fealpy`（私有） | 开 |
| `suanhai` | `suanhaitech/fealpy`（上游） | 禁用 |

工作分支 `main`，`import fealpy` 解析到的就是它（editable install）。上游代码用
`git -C ~/workspace/fealpy show suanhai/develop:<path>` 查阅，不另开检出。

对它的任何修改，必须在 `docs/known-issues/` 下创建 `fealpy-<topic>.md`。格式参照已有的文档：概要表 → 逐项详述 → 环境与版本对照。

### 3. 通用能力进 fealpy，不复制

fealpy 缺少的能力优先在该 fork 中增强，soptx 调用增强后的接口。通用算法（求解器、算子适配）进 fealpy；soptx 特有的物理模型保留在 soptx。改动同样写入 `docs/known-issues/`。

## 三个约定

- example 文件中出现制造解类名时，必须链接到 `docs/problems/manufactured-elasticity.md`（Python 用 RST 超链接，Markdown 用 MD 链接）
- GPU 代码用 `bm.to_numpy()` 而非直接 `.numpy()`；GPU vs CPU 对比时统一后端只改 device
- 每个主题 example 目录统一两文档模板：`README.md`（入口/文件职责）、`results_analysis.md`（数学—代码映射契约 + 实验证据报告，放目录根，不放 `outputs/`）；`outputs/` 是纯运行产物，由 `.gitignore` 统一忽略
