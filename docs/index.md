# SOPTX 文档

本目录维护 SOPTX 软件接口与可执行示例说明。实现默认值和运行行为以源码、测试及示例为
事实源；研究方法、文献和长期技术路线由 `dut-postdoc` 维护。

## 架构

- [`architecture/overview.md`](architecture/overview.md) — 分层、目录职责和公共 API；
- [`architecture/adr-0001-layered-src-layout.md`](architecture/adr-0001-layered-src-layout.md) —
  `src/soptx` 架构决策；
- [`architecture/migration-map.md`](architecture/migration-map.md) — `1.1.x` 旧→新路径；
- [`architecture/file-classification.md`](architecture/file-classification.md) —
  maintained、incubating、experiment、compatibility 与 archive 分类。

## 求解器

- [`solvers.md`](solvers.md) — scipy/mumps/cg 三种求解方式、各分析器的可用组合与依赖。

## 机器学习基础设施

- [`ml/mlp.md`](ml/mlp.md) — 可复用 PyTorch `MLP` 的构造契约、张量 shape、层序列与当前消费者。

## 有限元方法

- [`fem/load-handling-implementation.md`](fem/load-handling-implementation.md) —
  载荷处理架构：载荷协议与数据类分层、非体力载荷的装配门禁、四类载荷与两种离散格式的支持矩阵、
  线载荷 `degree` 要求、点力投影模式与正则化、面重心整面选取语义与 P1 迹 $L^2$ 投影。
- [`fem/linear-elastic-integrator-implementation.md`](fem/linear-elastic-integrator-implementation.md) —
  线弹性积分器实现（LinearElasticIntegrator）：三种组装方式（standard/voigt/fast）、
  同一 $\mathbf K_e$ 的三条计算路径、数学等价性、中间张量与容量差异，以及与装配层级（operator_level）的正交关系。
- [`fem/huzhang-mixed-fem-implementation.md`](fem/huzhang-mixed-fem-implementation.md) —
  胡张混合有限元实现：空间构造、次数与稳定化分支、角点松弛、2D/3D 差异、
  FEALPy 4.0 兼容要点及开放问题。
- [`fem/substructure-condensation-implementation.md`](fem/substructure-condensation-implementation.md) —
  子结构静力缩聚实现：Schur 补消元、全局接口 Scatter-Add 装配、2D/3D 统一、
  PIML 路线 A/B 接口、精确回退，以及与 MFEM 和 Huang 2023 的关系。

## 拓扑优化

- [`topology/filters.md`](topology/filters.md) —
  过滤器设计架构：定位与边界、配置/门面/矩阵构建/策略四层职责、与优化器的运行期协议
  （调用顺序、$\beta$ 延拓信号、设备约定）、`meshdata` 与 `density_location` 数据契约、
  `structured.py` 旁路、扩展点与设计决策。

## 数学模型与验证

- [`problems/manufactured-elasticity.md`](problems/manufactured-elasticity.md) —
  五个活跃线弹性制造解（含混合边界变体）；
- [`problems/engineering-benchmarks.md`](problems/engineering-benchmarks.md) —
  无解析解的工程基准算例，按载荷对象类型分四组；
- [`validation/evidence-policy.md`](validation/evidence-policy.md) —
  快速 CI、本地重验证和正式 evidence 要求；
- [`references/README.md`](references/README.md) — 第三方参考代码治理。

## 已知问题

- [`fealpy-migration.md`](fealpy-migration.md) — **FEALPy 依赖移植汇总**：做法、提交清单、
  移植范围、验证结果与约定好的后续工作（结构整理、移植遗留、既有 CI 问题、统一清理）。
- [`known-issues/README.md`](known-issues/README.md) — **FEALPy 来源与已内联补丁清单**：
  2026-10 起 SOPTX 所用的 FEALPy 代码已移植入库、fork 退役；该页记录移植状态、fork 补丁
  在 SOPTX 中的落点与回归保护现状，以及移植后遗留问题，是该目录**唯一的可变状态源**。
  该目录只有两份文档，另一份 `fealpy-patches.md`（补丁技术正文，历史记录）由该页索引，
  此处不重复枚举。来源与许可证见根目录 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

## 可执行示例

- [`examples/substructure_elasticity/README.md`](../examples/substructure_elasticity/README.md) —
  2D/3D 线弹性子结构静力缩聚精确基线；
- [`examples/piml_substructure_elasticity/README.md`](../examples/piml_substructure_elasticity/README.md) —
  PIML 增强子结构缩聚原型（神经网络代理 + 精确回退）；
- [`examples/matrix_free_elasticity/README.md`](../examples/matrix_free_elasticity/README.md) —
  2D/3D 线弹性 Matrix-Free 与 MPI 基线；
- [`examples/pinn_elasticity/README.md`](../examples/pinn_elasticity/README.md) —
  二维平面应变与三维各向同性线弹性 PINN 的标准框架、schema v3
  运行报告、训练配置、验证门禁与 clean-revision evidence 流程。
