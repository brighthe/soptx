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

## 有限元方法

- [`fem/huzhang-mixed-fem.md`](fem/huzhang-mixed-fem.md) —
  胡张混合有限元实现：空间构造、次数与稳定化分支、角点松弛、2D/3D 差异、
  FEALPy 4.0 兼容要点及开放问题。

## 数学模型与验证

- [`problems/manufactured-elasticity.md`](problems/manufactured-elasticity.md) —
  五个活跃线弹性制造解（含混合边界变体）；
- [`problems/engineering-benchmarks.md`](problems/engineering-benchmarks.md) —
  无解析解的工程基准算例（集中力等真实载荷）；
- [`validation/evidence-policy.md`](validation/evidence-policy.md) —
  快速 CI、本地重验证和正式 evidence 要求；
- [`references/README.md`](references/README.md) — 第三方参考代码治理。

## 已知问题

- [`known-issues/fealpy-tensor-product-mesh.md`](known-issues/fealpy-tensor-product-mesh.md) —
  FEALPy `4.0.0-alpha` mesh 重构的五处回归缺陷及其在 `fealpy_stable` 中的修复，
  含复现判据与 SOPTX 影响范围。

## 可执行示例

- [`examples/matrix_free_elasticity/README.md`](../examples/matrix_free_elasticity/README.md) —
  2D/3D 线弹性 Matrix-Free 与 MPI 基线；
- [`examples/pinn_elasticity/README.md`](../examples/pinn_elasticity/README.md) —
  二维平面应变与三维各向同性线弹性 PINN 的标准框架、schema v3
  运行报告、训练配置、验证门禁与 clean-revision evidence 流程。
