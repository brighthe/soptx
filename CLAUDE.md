# CLAUDE.md

本文件指导 Claude Code 在 `soptx` 项目中工作。

## 交流语言

在本仓库工作时，一律使用简体中文与用户交流；代码、命令、文件名、专有名词可保留英文。

- 代码中的行注释、块注释和 docstring 一律使用简体中文；中文注释使用英文半角标点。

## 代码注释与 docstring 规范

- 面向外部调用的模块、类、函数和方法必须提供 docstring。docstring 优先说明职责、输入输出契约、前置条件和异常；私有实现仅在行为不直观时补充说明。
- 需要分项说明参数或返回值时，使用中文 NumPy 风格小节：`参数:`、`返回:`、`异常:`。每个条目写作 ``名称: 含义.``，续行缩进与首行描述对齐。
- Python 标识符、类型名、字面量、配置值和张量形状一律使用 reStructuredText 等宽标记 ``...``，例如 ``K_local_batch``、``None``、``float64``、``(B, n_dof, n_dof)``。不以中文引号或普通引号替代该标记。
- 张量形状按轴的实际顺序书写。批量维使用 ``B``，可变前导维使用 ``...``；必须说明各轴的物理或数据语义，例如 ``(B, n_dof, n_dof)`` 表示批量局部刚度矩阵。
- 行注释解释设计原因、物理含义、数值稳定性或约束条件，不逐字复述代码。注释与代码保持同步，过期注释应随实现删除或更新。
- docstring 中不嵌入复杂的 ``.. math::`` 块或大段推导。复杂数学定义、推导和图表放在 ``docs/``；docstring 仅保留必要的文字化契约，并链接到对应文档。
- docstring 的首句使用简短完整句并以英文句点结束。中文正文中的逗号、句号、冒号、括号均使用英文半角形式。

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
| `huzhang_elasticity/` | Hu-Zhang 混合有限元求解 |
| `substructure_elasticity/` | 子结构静力缩聚求解与精确基线验证 |
| `piml_substructure_elasticity/` | PIML 子结构静力缩聚求解与代理验证 |

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

## 代码与物理模型规范

1. **物理工程模型下沉与显式引用**：所有工程物理问题模型（如 MBB 梁、悬臂梁等）一律在 `src/soptx/problems/` 核心库中下沉封装；`examples/` 中的 Runner / Demo 脚本必须显式导入并实例化 `soptx.problems` 中的标准物理问题对象，不得在 Runner 中散装硬编码几何尺寸、材料参数或边界条件。
2. **全流程统一 FEALPy `bm` 后端规范**：数据生成（`bm.random.uniform`）、矩阵上三角/掩码提取（`bm.triu` 配合 `bm.bool` 掩码）、张量堆叠（`bm.stack`）等计算一律原生使用 FEALPy `backend_manager as bm`；不混用 NumPy 独立函数（如 `np.triu_indices`），仅在最终向第三方框架（如 PyTorch/SciPy）转换时通过 `bm.to_numpy()` 规范输出。

## 约定与文档

- example 文件中出现制造解类名时，必须链接到 `docs/problems/manufactured-elasticity.md`（Python 用 RST 超链接，Markdown 用 MD 链接）
- GPU 代码用 `bm.to_numpy()` 而非直接 `.numpy()`；GPU vs CPU 对比时统一后端只改 device
- 每个主题 example 目录统一两文档模板：`README.md`（入口/文件职责）、`results_analysis.md`（数学—代码映射契约 + 实验证据报告，放目录根，不放 `outputs/`）；`outputs/` 是纯运行产物，由 `.gitignore` 统一忽略
