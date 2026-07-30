# 线弹性 PINN 2D/3D 基线

本目录提供 SOPTX 二维平面应变与三维各向同性线弹性 PINN 基线。两个维度共享
同一套强形式 residual、训练、checkpoint、报告和验证框架，通过显式 case 隔离
制造解、材料假设、几何维数和诊断规模。

当前能力严格限定为：

- 二维 `plane_strain` 或三维各向同性小变形静力线弹性；
- 轴对齐单位超矩形；
- 全位移 Dirichlet 边界；
- 坐标到位移的 MLP：

  $$
  (x_1,\ldots,x_d)\mapsto(u_1,\ldots,u_d),\qquad d\in\{2,3\};
  $$

- PyTorch backend 与 `float64`。

本阶段不支持 `plane_stress`、牵引或混合边界、非矩形区域、任意维数、MPI、
Neural Operator 或拓扑优化。

## 数学模型与 Case

两个维度使用相同的各向同性线弹性强形式：

$$
-\nabla\cdot\boldsymbol\sigma(\boldsymbol u)=\boldsymbol b,\qquad
\boldsymbol\varepsilon(\boldsymbol u)
=\frac12(\nabla\boldsymbol u+\nabla\boldsymbol u^{\mathsf T}),
$$

$$
\boldsymbol\sigma
=\lambda\operatorname{tr}(\boldsymbol\varepsilon)\mathbf I
+2\mu\boldsymbol\varepsilon,\qquad
\boldsymbol u=\bar{\boldsymbol u}\quad\text{on }\partial\Omega.
$$

训练 loss 为平衡 residual 与 Dirichlet residual 的加权 MSE，默认权重为
`(1, 30)`。精确位移只用于制造解一致性、固定诊断和 L2 error，不作为内部点
监督标签。

| 维数 | PDE | 材料假设 | 材料参数 | diagnostic mesh |
|---|---|---|---|---:|
| 2D | `ExponentialSineManufacturedElasticity2D` | `plane_strain` | $\lambda=1,\mu=0.5$ | 每轴 30 个节点 |
| 3D | `DivergenceFreePolynomialElasticity3D` | `3D` | $\lambda=\mu=1$ | 每轴 8 个节点 |

Matrix-Free 示例的 2D case 是
`SinusoidalPlaneStrainElasticity2D(E=1, nu=0.3)`，与这里的
2D 制造解和材料参数不同。因此两个示例在 2D 的误差、训练难度和耗时不能直接
横向比较；3D 使用相同 Problem 和材料。

完整数学模型与 shape 契约见
[`docs/models/manufactured-elasticity.md`](../../docs/models/manufactured-elasticity.md)。
Problem 不再创建诊断网格；`ElasticityCase.create_diagnostic_mesh()` 根据 domain
和 dimension 显式创建 `TriangleMesh` 或 `TetrahedronMesh`。

> 架构迁移说明：现有训练结果属于迁移前开发基线。新语义 Problem 和 `src/soptx`
> 布局尚需重新运行 `validate.py --dim all`，才能形成重构后的正式 evidence。

## 框架与调用链

- `contract.py`：schema、stage、支持范围、默认值、固定门禁和 `RunConfig`；
- `layout.py`：outputs、evidence、validation summary 与 README marker；
- `cases.py`：制造解、材料、domain、维数和诊断网格；
- `operators.py`：network、位移梯度、应变、应力和平衡/边界 residual；
- `solve.py`：network/sampler/optimizer、objective、训练、history 和 checkpoint；
- `references.py`：精确位移适配器与确定性诊断点；
- `postprocess.py`：绝对/相对 L2 error 和训练图；
- `report.py`：环境、运行门禁、schema v3 JSON 与终端摘要；
- `run.py`：CLI 和可复用 `execute(case, config)`；
- `validate.py`：真实 CLI smoke、2D/3D 正确性与完整训练门禁；
- `sync_results.py`：clean-revision evidence 和本 README 的结果同步。

`contract.py` 和 `layout.py` 不依赖 FEALPy、SOPTX 或 PyTorch。核心调用链为：

```text
run.py
  → create_case(dim)
  → RunConfig
  → execute(case, config)
      → prepare_problem
          → PINNOperator / optimizer / scheduler / diagnostic mesh
      → train_prepared_problem
          → loss_components / backward / optimizer.step
          → post-update train and fixed-validation diagnostics
          → optional schema-v3 best.pt / last.pt
      → report.local_gates / report.build_run_payload
  → optional summary JSON / training figure
```

每条 history 记录都对应参数更新完成后的同一 network 状态；`learning_rate`
记录该次更新实际使用的值。训练结束后 network 保持 final state，
`TrainingResult` 独立携带 best epoch、metrics 和 CPU state dict，可通过
`restore_best_state()` 显式恢复。

## 默认训练配置

| 参数 | 默认值 |
|---|---|
| dimension | `2` |
| network | $d\to32\to32\to16\to d$ |
| activation / optimizer | `Tanh` / Adam |
| learning rate | $10^{-3}$ |
| dtype / device | `float64` / CPU |
| sampling | `random`，每次更新重新采样 |
| interior / boundary points | 400 / 每个边界面 100 |
| fixed validation points | interior 400 / 每个边界面 100 |
| loss weights | `(1, 30)` |
| parameter updates | 2000 |
| seed / log interval | 0 / 100 |
| scheduler | 默认关闭 |
| checkpoint / summary | 默认不写 |

`sampling-mode=linspace` 时，采样参数表示每个自由坐标轴的步数，而不是总点数；
三维总点数按幂次增长。

## 环境与运行

以下命令均从 `C:\workspace\soptx`、`soptx-gpu` 环境执行。

先安装 PINN extra：

```powershell
python -m pip install -e ".[pinn,test]"
```

二维默认训练：

```powershell
python .\examples\pinn_elasticity\run.py --dim 2
```

三维训练且不打开交互图：

```powershell
python .\examples\pinn_elasticity\run.py --dim 3 --no-show
```

显式保存 checkpoint 和运行 summary：

```powershell
python .\examples\pinn_elasticity\run.py `
  --dim 3 `
  --checkpoint-dir .\examples\pinn_elasticity\outputs\checkpoints `
  --summary .\examples\pinn_elasticity\outputs\run-3d.json `
  --no-show
```

CLI 文档使用连字符形式；现有的 `--mesh_size`、`--sampling_mode`、
`--hidden_size`、`--checkpoint_dir` 等下划线参数仍作为兼容别名。

schema v3 checkpoint 保留 model/optimizer/scheduler、options、history 和 metrics，
并增加 stage、dimension、case、domain、material、environment 和 RNG state。
v2 snapshot 可由 `torch.load` 手工读取，但新验证和 evidence 工具只接受 v3。

## 测试与验证

快速测试：

```powershell
python -m pytest .\examples\pinn_elasticity\tests
```

正式验证：

```powershell
python .\examples\pinn_elasticity\validate.py --dim 2
python .\examples\pinn_elasticity\validate.py --dim 3
python .\examples\pinn_elasticity\validate.py --dim all
```

固定门禁包括：

- 精确位移梯度最大绝对误差不超过 `1e-12`；
- 精确应变非对称量不超过 `1e-12`；
- 精确平衡 residual 最大绝对值不超过 `1e-10`；
- 精确全边界 residual 最大绝对值不超过 `1e-12`；
- 非 PyTorch backend、非法维数、错误材料假设和混合边界被明确拒绝；
- schema v3 checkpoint 字段完整、可加载且预测一致；
- CLI smoke summary 通过本地运行门禁；
- 完整 history 有限且 best validation loss 优于首次记录；
- 2D best validation loss `≤5e-2`、relative L2 `≤1e-1`；
- 3D best validation loss 严格改善、relative L2 `<1.0`。

`--epochs` 只接受正数并用于开发诊断；只有完整默认配置可生成正式 evidence。
validation 总会把 per-dimension schema v3 JSON 写到已忽略的 `outputs/`；
`--dim all` 另外写 aggregate JSON，失败和异常也会被记录。只有全部门禁通过时
退出码为 0。

## 正式证据

正式 evidence 必须来自 `git_dirty=false` 的 clean revision。先分别生成 2D 和
3D validation summary，再同步精简证据与 README：

```powershell
python .\examples\pinn_elasticity\sync_results.py --dim all
python .\examples\pinn_elasticity\sync_results.py --dim all --check
```

同步工具校验 schema、stage、默认配置、全部门禁、Git 状态和源 JSON SHA-256；
不满足任一条件时拒绝写入。

### 2D CPU float64 训练基线

<!-- BEGIN GENERATED: cpu-float64-training-baseline-2d -->

尚未生成 clean-revision 正式证据。

<!-- END GENERATED: cpu-float64-training-baseline-2d -->

### 3D CPU float64 训练基线

<!-- BEGIN GENERATED: cpu-float64-training-baseline-3d -->

尚未生成 clean-revision 正式证据。

<!-- END GENERATED: cpu-float64-training-baseline-3d -->

## 历史开发验收

2026-07-30 曾在 Git base
`b5a05298c655a4743b21b99b7442b1f547f0395a` 的 dirty worktree、Python
3.12.13、PyTorch 2.13.0+cu130、CPU、`float64` 下运行旧 schema v2
`validate.py --dim all`。当时 2D/3D 门禁均通过：

| 指标 | 2D | 3D |
|---|---:|---:|
| 精确平衡 residual 最大绝对值 | $1.7764\times10^{-15}$ | $8.4377\times10^{-15}$ |
| best fixed-validation loss | 0.035441 | 0.089727 |
| best checkpoint relative displacement L2 | 0.034686 | 0.674047 |
| 固定点平衡 residual RMS | 0.091778 | 0.172532 |
| 最大边界位移误差 | 0.098866 | 0.143497 |
| 2000 次更新耗时 | 28.17 s | 69.90 s |

该记录只用于确认重构前数学基线，不属于 schema v3 clean-revision 正式证据。
