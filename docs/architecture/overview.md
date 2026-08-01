# SOPTX v2 架构总览

SOPTX 是个人有限元分析与结构拓扑优化研究平台。稳定包采用 `src/soptx`
布局；Matrix-Free、PINN 和论文驱动程序在形成第二个真实消费者与独立单元测试前，
保持为孵化示例或实验。

## 分层与依赖方向

```text
core → materials / problems → fem → topology → visualization
```

- `core`：日志、计时、协议和基础结果类型，不依赖其他 SOPTX 层。
- `problems`：区域、载荷、边界与精确解；不创建网格，不保存 Material。
- `materials`：本构模型，不依赖 FEM workflow。
- `fem`：空间、积分子和求解 workflow，只依赖更低层。
- `topology`：插值、过滤、目标、约束、优化器和后处理。
- `visualization`：可选展示能力，基础导入不要求 `viz` extra。

依赖规则由 `tools/check_architecture.py` 静态检查。旧的
`analysis/functionspace/interpolation/model/optimization/regularization/utils`
是 `1.1.x` 公共 API 兼容命名空间，不允许被新稳定层导入。已有 maintained
替代实现的兼容模块必须是薄转发；未登记为公共 API 的历史实现不进入 wheel。

## 公共 API

根包仅公开 `soptx.__version__`，稳定对象一律从职责子包导入。这样层归属在 import
语句上直接可见，根包也不会退化成跨层的隐式聚合点。导入示例见
[仓库 README](../../README.md#公共-api)。

`Problem`、`Material` 和网格由调用方显式组合。Problem 的 shape 契约见
[制造线弹性模型](../models/manufactured-elasticity.md)。

## 问题契约（Protocol 族）

`soptx/core/protocols.py` 定义分析器对 Problem 的结构化要求。它放在 layer 0，
因此 `fem`（layer 2）不必依赖 `problems`（layer 1）或 `topology`（layer 3）就能
标注参数类型；Problem 一侧无需继承，满足成员即满足协议。

| 协议 | 语义 | 消费者 |
| --- | --- | --- |
| `ElasticityProblem` | 任何弹性问题的公共核心 | 两个分析器共用 |
| `DirichletElasticityProblem` | 主形式：Dirichlet 数据 + 按 `boundary_type`/`load_type` 分派 | `LagrangeFEMAnalyzer` |
| `MixedBoundaryElasticityProblem` | 混合形式：位移/牵引边界显式二分 + 角点 | `HuZhangMFEMAnalyzer` |

混合形式的边界划分与主形式相反：位移数据弱施加（自然），牵引数据强施加
（本质），所以两个谓词必须显式划分边界。全 Dirichlet 问题的退化实现由
`soptx/problems/elasticity/_base.py` 的 `AllDisplacementBoundaryMixin` 提供。

**规则**：往分析器里新增任何 `self._pde.xxx` 访问，必须同步扩展对应协议。
`tests/unit/test_problem_protocol_conformance.py` 用 AST 扫描强制这一点 ——
协议曾经因为无人校验而与实际需求脱节，该测试防止其重演。
