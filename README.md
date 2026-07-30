# SOPTX

SOPTX（Structural Optimization Topology Simulation Software）是基于
[FEALPy](https://github.com/suanhaitech/fealpy) 的个人结构拓扑优化科研软件仓库。
本仓库负责把可执行算法、数值验证和可复现实验组织为可维护的软件资产。

## 仓库职责

本仓库是以下内容的权威事实源：

- 结构拓扑优化、材料插值、正则化、目标函数、约束和优化器的实现；
- 有限元分析与拓扑优化相关的软件接口和可执行模型；
- 单元测试、等价性验证、示例程序和可复现实验；
- 软件使用文档、版本变更和发布材料。

以下内容由其他仓库维护：

- 文献、科研路线和可复用非敏感技术知识：`dut-postdoc`；
- 研究院项目任务、进度、会议和交付物索引：`dut-institute-work`；
- 身份、联系人、聊天原文和沟通上下文：`heliangos`；
- 工具配置、环境迁移和工作区自动化：`workstation`。

跨仓库只保留完成本地任务所必需的结论和
`repository:repo-relative-path#heading` 指针，不复制其他仓库的事实正文。

## FEALPy 与算海仓库边界

FEALPy 是 SOPTX 的上游数值计算依赖，其接口与实现由
`suanhaitech/fealpy` 独立维护。SOPTX 可以调用 FEALPy 的公开接口，但不复制、
vendor 或重新托管算海仓库中的代码、数据、运行日志、客户算例、凭据或内部文档。

涉及 `fealpy`、`mfleo`、`xihe` 的技术事实时，以对应算海仓库为工程事实源；本仓库
只保存属于 SOPTX 软件职责的个人实现、非敏感验证结果和事实源指针。

## 目录入口

| 路径 | 职责 |
| --- | --- |
| `src/soptx/` | 分层的软件包实现与公共 API |
| `tests/` | 快速 unit、integration 与 regression 测试 |
| `examples/` | Matrix-Free、PINN 等孵化示例和阶段验证 |
| `experiments/` | 论文矩阵、provenance 与长期运行 |
| `docs/` | 架构、数学模型、验证和引用文档 |
| `reference_code/` | 待归档且许可证未核实的第三方参考代码 |

目标依赖方向是
`core → materials/problems → fem → topology → visualization`。Problem 只表达
区域、载荷、边界与精确解；Material 独立；网格由 FEM workflow 或 example case
显式创建。详细设计见 [`docs/architecture/overview.md`](docs/architecture/overview.md)。

## 安装与公共 API

SOPTX 当前迁移版本为 `1.1.0.dev0`，Python 最低版本为 3.10：

```powershell
python -m pip install -e .
```

根包只公开 `__version__`。稳定对象从职责子包导入：

```python
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import SinusoidalPlaneStrainElasticity2D
from soptx.fem.integrators import LinearElasticIntegrator
```

迁移表中声明的旧公共路径在 `1.1.x` 发出一次 `DeprecationWarning`，迁移表见
[`docs/architecture/migration-map.md`](docs/architecture/migration-map.md)。
已经不存在的 `soptx.pde/material/solver/filter/opt` 不会重新建立。

完整的八仓库职责与内容路由规范见
`workstation:workspace/responsibilities.md#单一职责`。进入具体开发任务后，以本仓库
代码、测试和后续项目级说明为准。

## 许可证与第三方代码

SOPTX 自有代码采用 `GPL-3.0-only`。`reference_code/` 不自动适用 SOPTX
许可证；其来源和许可证确认前不可再发布，也不会进入 wheel。治理说明与 SHA-256
清单见 [`docs/references/README.md`](docs/references/README.md)。
