# SOPTX 可执行示例索引 (Examples)

本文件是 `examples/` 下的**程序入口索引**：按主题路由到各示例目录的
`README.md`，说明每个目录做什么、怎么跑、文档状态如何。

> 职责边界：本文件只维护**程序**的查找与路由。方法知识、文献证据与长期技术路线由
> `dut-postdoc`（`concepts/piml/_index.md` 等）维护，不复制到这里。

## 主题目录一览

| 目录 | 一句话职责 | 文档状态 |
|---|---|---|
| [`lagrange_elasticity/`](lagrange_elasticity/) | CPU 串行全装配，L2 收敛阶验证 | 脚本级，缺文档模板 |
| [`gpu_elasticity/`](gpu_elasticity/) | GPU 正确性对比 + 性能 benchmark | 脚本级，缺文档模板 |
| [`matrix_free_elasticity/`](matrix_free_elasticity/) | MPI 并行 matrix-free，FA/EA 双路 | 旧三文档结构，待收敛 |
| [`substructure_elasticity/`](substructure_elasticity/) | 精确子结构静力缩聚基线（2D/3D Schur 补） | README + results_analysis（已按两文档模板） |
| [`piml_elasticity/`](piml_elasticity/) | PIML 预测子结构缩聚表示 + 精确回退（早期原型） | README + results_analysis（已按两文档模板），预测器未进全局接口 |
| [`pinn_elasticity/`](pinn_elasticity/) | 2D/3D 线弹性 PINN 强形式求解 | 旧三文档结构，待收敛 |
| [`huzhang_elasticity/`](huzhang_elasticity/) | 胡张混合有限元 2D 求解（应力—位移鞍点系统） | 旧三文档结构，待收敛 |

## PIML 主题小节

PIML（Problem-Independent Machine Learning）程序按职责链路由，从精确真值到预测器，
再到未来的全局接入：

```text
substructure_elasticity    精确真值 / Schur 补基线（已维护）
        ↓
piml_elasticity            预测器：K_s / N + 结构检查 + 精确回退（早期原型，文档已对齐）
        ↓
matrix_free_elasticity / gpu_elasticity    未来全局接入（规划，非现状）
```

- 精确标签、接口系统与细尺度恢复的数学与验收契约见 `substructure_elasticity/`；
- PIML 学习对象、路线 A/B 与统一比较契约见 `dut-postdoc/concepts/piml/`；
- 现状：`substructure_elasticity/` 提供精确基线；`piml_elasticity/` 是首个预测器原型，
  文档已按两文档模板对齐；代码已知问题见该目录 `results_analysis.md` §3，涉及 PSD 检查、能量一致性与下游评估，修复前不作为正式入口。

## 目录约定

- 每个主题目录遵循两文档模板：`README.md`（入口/文件职责）、`results_analysis.md`（数学—代码映射契约 + 实验证据报告）。
- `outputs/` 是纯运行产物，由 `.gitignore` 统一忽略。
- 新技术栈 → 新目录；同技术栈不同物理问题 → 同目录下新文件（见 `CLAUDE.md`）。
