# AGENTS.md

本文件是 `soptx` 面向所有 AI 助手的唯一指令正文；`CLAUDE.md` 与 `GEMINI.md` 只是指向本文件的入口。主题目录（`experiments/` 的各子目录）与带自述规则的目录（如 `docs/known-issues/`）各以其 `README.md` 为专项细则的权威来源，动手前先读；`src/`、`tests/`、`tools/` 不设 README，自述载体是代码 docstring。

## 开始工作前

- 先确认当前分支与工作区状态；保留用户已有的未提交修改，不得回退或覆盖无关文件。
- Git 操作在 WSL 中执行。未获用户明确要求，不 commit、push、创建分支或执行破坏性 Git 操作。
- 未获用户明确要求，不运行测试、benchmark、MPI 或长时 GPU 任务；先给出可复现命令与验收条件。
- 讨论定稿不等于编码授权：新增或修改代码前，先向用户提出方案要点并获明确同意，再动手。

## 注释与 docstring

- 注释与 docstring 采用标准 numpydoc 风格，正文一律使用简体中文；专有名词、方法名、变量保留英文，标点使用英文半角。

## 仓库特有约定

- SOPTX 不依赖 FEALPy：原先所用的 FEALPy 代码已移植为 `soptx` 子包（backend、sparse、quadrature、mesh、functionspace、fem 基类等），按普通 SOPTX 代码维护，来源与许可证见 `THIRD_PARTY_NOTICES.md`；不得再 `import fealpy`（`tools/check_architecture.py` 门禁）。修改移植代码无需另行记账，但影响已内联补丁或移植后遗留问题时同步 `docs/known-issues/README.md`；若再从 FEALPy 取代码，须在 `THIRD_PARTY_NOTICES.md` 登记来源提交并在文件头注明。
- 实现前先检索 soptx 中已有的能力，优先复用或扩展现有接口，不重复造轮子。
