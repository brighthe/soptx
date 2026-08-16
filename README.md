# SOPTX

SOPTX（Structural Optimization Topology Simulation Software）是基于
[FEALPy](https://github.com/suanhaitech/fealpy) 的个人结构拓扑优化科研软件仓库。
本仓库负责把可执行算法、数值验证和可复现实验组织为可维护的软件资产。

## 快速开始

以二维线弹性 Matrix-Free EA 基线为例，从仓库根目录执行：

```powershell
conda activate ihpcm
python -m pip install -e ".[mpi,test]"
mpiexec -n 1 python .\tools\matrix_free_evidence\run.py --dim 2 --operator-level ea --p 1 --nx 8 --ny 8
```

完整参数、3D 与 FA 路径、MPI 分区约束见
[`examples/matrix_free_elasticity/README.md`](examples/matrix_free_elasticity/README.md)。
PINN 示例入口见
[`examples/pinn_elasticity/README.md`](examples/pinn_elasticity/README.md)。

## 安装与环境

SOPTX 当前迁移版本为 `1.1.0.dev0`，Python 最低版本为 3.10：

```powershell
python -m pip install -e .
```

硬依赖是 `fealpy>=4,<5`，不随本仓库分发，需单独安装。其余基础依赖为 `numpy`、
`scipy`、`sympy`。

实际使用的是一份**长期维护的 vendor fork**，位于 `~/workspace/fealpy`（`origin` =
`brighthe/fealpy`（私有），上游 `suanhai` = `suanhaitech/fealpy`，工作分支 `main`），
以 editable 方式安装，`import fealpy` 解析到的就是它。上游 `4.0.0-alpha` 存在多处
回归缺陷，修复直接落在该 fork 上而非通过 PR 回流，因此 fork 与上游长期分叉。

> **上游已知回归**：FEALPy `4.0.0-alpha` 的 mesh 重构引入四处回归，其中一处影响所有
> 网格类型，会使 SOPTX 全部算例在计算误差时抛异常；另三处使四边形、六面体静默给出
> 错误结果。这些缺陷已在 fork 中修复。缺陷细节、复现脚本与上游进展见
> [`docs/known-issues/fealpy-tensor-product-mesh.md`](docs/known-issues/fealpy-tensor-product-mesh.md)。

可选 extra 按用途划分：

| Extra | 内容 | 用途 |
| --- | --- | --- |
| `viz` | matplotlib、pillow | 可视化输出；基础导入不要求该 extra |
| `mpi` | mpi4py | Matrix-Free 分布式算子与多 rank 运行 |
| `pinn` | torch | PINN 示例训练 |
| `test` | pytest、build | 测试与 wheel 构建 |

示例目录各自绑定一个已验证的本机 conda 环境：Matrix-Free/MPI 与拉格朗日位移元
使用 `ihpcm`，PINN 使用 `soptx-gpu`。

## 公共 API

根包只公开 `__version__`。稳定对象从职责子包导入：

```python
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import SinusoidalPlaneStrainElasticity2D
from soptx.fem.integrators import LinearElasticIntegrator
```

迁移表中声明的旧公共路径在 `1.1.x` 发出一次 `DeprecationWarning`，迁移表见
[`docs/architecture/migration-map.md`](docs/architecture/migration-map.md)。
已经不存在的 `soptx.pde/material/solver/filter/opt` 不会重新建立。

## 复现与验证

本地重验证入口：

```powershell
python .\tools\matrix_free_evidence\validate.py --dim all
python .\tools\matrix_free_evidence\sync_results.py --dim all --check
python .\experiments\huzhang_topopt_paper\dry_run.py --json
```

这些命令必须在明确环境中运行，并检查退出码、预期产物和数值 acceptance criteria，
不能仅凭命令结束判断成功。dirty worktree 上的运行只能标记为开发证据。正式 evidence
的记录要求（clean revision、dirty flag、依赖版本、参数与随机种子、产物 SHA-256）见
[`docs/validation/evidence-policy.md`](docs/validation/evidence-policy.md)。

evidence 的 `environment.fealpy` 记录的不是静态版本号，而是运行时 `fealpy.__file__`
所在检出的 `path`、`git_revision`、`git_dirty` 和 `git_remote`，因此可以直接判定该次
运行用的是上游还是本地 fork、工作区是否干净。缺陷背景见
[`docs/known-issues/fealpy-tensor-product-mesh.md`](docs/known-issues/fealpy-tensor-product-mesh.md)。

## 目录入口

| 路径 | 职责 |
| --- | --- |
| `src/soptx/` | 分层的软件包实现与公共 API |
| `tests/` | 快速 unit、integration 与 regression 测试 |
| `examples/` | Matrix-Free、PINN 等孵化示例和阶段验证 |
| `experiments/` | 论文矩阵、provenance 与长期运行 |
| `docs/` | 架构、数学模型、验证和引用文档 |
| `reference_code/` | 待归档且许可证未核实的第三方参考代码 |

各路径的 maintained、incubating、experiment、compatibility 与 archive 分类及其迁移
政策见
[`docs/architecture/file-classification.md`](docs/architecture/file-classification.md)。

目标依赖方向是
`core → materials/problems → fem → topology → visualization`。Problem 只表达
区域、载荷、边界与精确解；Material 独立；网格由 FEM workflow 或 example case
显式创建。详细设计见 [`docs/architecture/overview.md`](docs/architecture/overview.md)。

## 开发门禁

提交前在本地复现 CI 的快速检查：

```powershell
python tools\check_python_syntax.py
python tools\check_architecture.py
python tools\generate_repository_inventory.py --check
python -m pytest tests -q
```

CI 另有一个 Matrix-Free fast job，装上 `mpi4py` 后重跑 `tests -q -k matrix_free`，
补上 `fast` job 里被 `importorskip` 跳过的那部分；它同样不运行 MPI benchmark
或正式 validation。

## 仓库职责与跨仓库边界

本仓库是以下内容的权威事实源：

- 结构拓扑优化、材料插值、正则化、目标函数、约束和优化器的实现；
- 有限元分析与拓扑优化相关的软件接口和可执行模型；
- 单元测试、等价性验证、示例程序和可复现实验；
- 软件使用文档。版本变更以 Git 历史和
  [`docs/architecture/migration-map.md`](docs/architecture/migration-map.md) 为准。

以下内容由其他仓库维护：

- 文献、科研路线和可复用非敏感技术知识：`dut-postdoc`；
- 研究院项目任务、进度、会议和交付物索引：`dut-institute-work`；
- 身份、联系人、聊天原文和沟通上下文：`heliangos`；
- 工具配置、环境迁移和工作区自动化：`workstation`。

跨仓库只保留完成本地任务所必需的结论和
`repository:repo-relative-path#heading` 指针，不复制其他仓库的事实正文。完整的八仓库
职责与内容路由规范见 `workstation:workspace/responsibilities.md#单一职责`。

FEALPy 是 SOPTX 的上游数值计算依赖，其接口与实现由
`suanhaitech/fealpy` 独立维护。SOPTX 可以调用 FEALPy 的公开接口，但不复制、
vendor 或重新托管算海仓库中的代码、数据、运行日志、客户算例、凭据或内部文档。
涉及 `fealpy`、`mfleo`、`xihe` 的技术事实时，以对应算海仓库为工程事实源；本仓库
只保存属于 SOPTX 软件职责的个人实现、非敏感验证结果和事实源指针。

## 许可证与第三方代码

SOPTX 自有代码采用 `GPL-3.0-only`。`reference_code/` 不自动适用 SOPTX
许可证；其来源和许可证确认前不可再发布，也不会进入 wheel。治理说明与 SHA-256
清单见 [`docs/references/README.md`](docs/references/README.md)。
