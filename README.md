# SOPTX

SOPTX（Structural Optimization Topology Simulation Software）是最初基于
[FEALPy](https://github.com/suanhaitech/fealpy) 开发的个人结构拓扑优化科研软件仓库；
所依赖的 FEALPy 代码已移植入库，来源见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
本仓库负责把可执行算法、数值验证和可复现实验组织为可维护的软件资产。

## 快速开始

本仓库位于 WSL（Ubuntu-24.04）的 `~/workspace/soptx`，所有安装、运行与 Git
操作均在该发行版的 bash 中执行。以二维线弹性 Matrix-Free EA 基线为例，从仓库
根目录执行：

```bash
conda activate ihpcm
python -m pip install -e ".[mpi,test]"
mpiexec -n 1 python tools/matrix_free_evidence/run.py --dim 2 --operator-level ea --p 1 --nx 8 --ny 8
```

完整参数、3D 与 FA 路径、MPI 分区约束见
[`examples/matrix_free_elasticity/README.md`](examples/matrix_free_elasticity/README.md)。
PINN 示例入口见
[`examples/pinn_elasticity/README.md`](examples/pinn_elasticity/README.md)。

## 安装与环境

SOPTX 当前迁移版本为 `1.1.0.dev0`，Python 最低版本为 3.12（移入的网格与积分代码使用
PEP 695 泛型与 `type` 语句）：

```bash
python -m pip install -e .
```

基础依赖为 `numpy`、`scipy`、`sympy`。

SOPTX **不依赖 FEALPy**。原先所用的 FEALPy 代码（backend、sparse、quadrature、
mesh、functionspace 与 fem 基类等）已自 vendor fork `brighthe/fealpy` `f474a5775`
移植为 `soptx` 子包，来源与许可证见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)；
fork 上的缺陷修复随代码一并内联，在 SOPTX 中的落点见
[`docs/known-issues/README.md`](docs/known-issues/README.md)。
`tools/check_architecture.py` 禁止仓库内再导入 `fealpy`。

> **张量积网格的静默错误**：FEALPy v0.4 网格重写曾使四边形、六面体单元静默给出
> 错误结果（缺陷 8、9），修复已随代码移入 `soptx.mesh`。这两处没有 pytest 级回归
> 保护，修改 `soptx.mesh` 的 `view/entity_view.py`、`ipoints.py`、`schema/` 后须手工
> 复核张量积网格收敛阶，细节见
> [`docs/known-issues/fealpy-patches.md`](docs/known-issues/fealpy-patches.md) 第一节。

可选 extra 按用途划分：

| Extra | 内容 | 用途 |
| --- | --- | --- |
| `viz` | matplotlib、pillow、vtk | 可视化与 VTU 输出；基础导入不要求该 extra |
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

```bash
python tools/matrix_free_evidence/validate.py --dim all
python tools/matrix_free_evidence/sync_results.py --dim all --check
python experiments/huzhang_topopt_paper/dry_run.py --json
```

这些命令必须在明确环境中运行，并检查退出码、预期产物和数值 acceptance criteria，
不能仅凭命令结束判断成功。dirty worktree 上的运行只能标记为开发证据。正式 evidence
的记录要求（clean revision、dirty flag、依赖版本、参数与随机种子、产物 SHA-256）见
[`docs/validation/evidence-policy.md`](docs/validation/evidence-policy.md)。

evidence 的 `environment` 自 schema 5 起不再记录 `fealpy`：全部数值代码位于本仓库，
由 `git_revision` 与 `git_dirty` 一并钉住。

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
`core/protocols/ml → materials/problems → fem → topology → postprocess`。Problem 只表达
区域、载荷、边界与精确解；Material 独立；网格由 FEM workflow 或 example case
显式创建。详细设计见 [`docs/architecture/overview.md`](docs/architecture/overview.md)。

## 开发门禁

提交前在本地复现 CI 的快速检查：

```bash
python tools/check_python_syntax.py
python tools/check_architecture.py
python tools/generate_repository_inventory.py --check
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

FEALPy 是 SOPTX 的代码来源之一：SOPTX 依赖的 FEALPy 代码已于 2026-10 自 vendor fork
`f474a5775` 按 `GPL-3.0-or-later` 移植入库（见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)），
此后以本仓库为准演化，不再依赖、也不再跟随 `suanhaitech/fealpy`。除此之外，本仓库不复制
或重新托管算海仓库中的数据、运行日志、客户算例、凭据或内部文档；涉及 `mfleo`、`xihe`
的技术事实时，以对应算海仓库为工程事实源。本仓库只保存属于 SOPTX 软件职责的个人实现、
非敏感验证结果和事实源指针。

## 许可证与第三方代码

SOPTX 自有代码采用 `GPL-3.0-only`；源自 FEALPy 的代码按 `GPL-3.0-or-later` 使用，
来源与范围见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。`reference_code/` 不自动适用 SOPTX
许可证；其来源和许可证确认前不可再发布，也不会进入 wheel。治理说明与 SHA-256
清单见 [`docs/references/README.md`](docs/references/README.md)。
