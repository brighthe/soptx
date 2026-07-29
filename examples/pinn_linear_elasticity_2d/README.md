# 二维平面应变线弹性 PINN 基线

本目录提供 SOPTX 的二维平面应变线弹性 PINN 阶段验证基线，而不是稳定公共 API。它复用 `soptx.model.linear_elasticity_2d.TriSolHomoDirHuZhang2d` 的解析位移、体力和全 Dirichlet 边界条件；PINN 的网络、自动微分 residual、训练控制和验证门禁只在本目录维护。

小变形静力线弹性的连续模型和变分形式见
`dut-postdoc:concepts/linear-elasticity.md#线弹性方程变分形式与有限元离散`；
机器学习分类原则见
`dut-postdoc:concepts/machine-learning.md#二、多维分类的组合示例`；
完整 PINN 方法流程见
`dut-postdoc:research/workflows/linear-elasticity-pinn-machine-learning-workflow.md`。
本页维护这些理论在 SOPTX 当前 PINN 算例中的问题参数、实现映射、运行方法和验证边界。

## 模块边界

- [`run.py`](run.py)：CLI 解析、模型构造、训练启动和交互式诊断图；
- [`model.py`](model.py)：位移网络、自动微分应变/应力/平衡 residual、训练 history 和 checkpoint；
- [`problem.py`](problem.py)：阶段 1 制造解的选择入口，不复制已有 SOPTX PDE 实现；
- [`validate.py`](validate.py)：独立正确性验证驱动，在终端输出逐项门禁与关键指标。

## 算例定位与输入输出

本算例是坐标型 PINN，其学习映射为

$$
(x,y)\longmapsto
\bigl(u_x(x,y),u_y(x,y)\bigr).
$$

| 项目 | 当前实现 |
|---|---|
| 输入 | 内部或边界配点坐标，shape 为 $(N,2)$ |
| 输出 | 配点处两个位移分量 $(u_x,u_y)$，shape 为 $(N,2)$ |
| 神经网络架构 | 全连接 MLP |
| 学习对象 | 函数学习：二维坐标到一个固定边值问题的位移场 |
| 训练范式 | PINN：平衡方程 residual 与全 Dirichlet 边界 residual 的加权 MSE |
| 计算角色 | 使用神经网络近似该边值问题的位移解场 |
| 计算目标 | 降低物理 residual，并使预测位移在固定诊断网格上接近解析解 |
| 非目标 | 不学习跨问题解算子，不预测局部刚度，不执行拓扑优化 |

本模型学习的是一个固定边值问题的解函数，不是学习一族函数到函数映射的 Neural Operator，也不是学习可复用局部力学表示的 Problem-Independent PIML。

## 数学模型与制造解

本节给出当前 PINN 算例完整的数学定义与参数；制造解、体力和边界数据的可执行事实源是
`soptx.model.linear_elasticity_2d.TriSolHomoDirHuZhang2d`。

求解区域为单位正方形 $\Omega=(0,1)^2$。当前问题固定为二维、小变形、静力、各向同性、平面应变线弹性：

$$
-\nabla\cdot\boldsymbol\sigma(\boldsymbol u)
=
\boldsymbol b
\quad\text{in }\Omega,
\qquad
\boldsymbol u=\boldsymbol0
\quad\text{on }\partial\Omega,
$$

$$
\boldsymbol\varepsilon(\boldsymbol u)
=
\frac12\left(
\nabla\boldsymbol u+\nabla\boldsymbol u^{\mathsf T}
\right),
\qquad
\boldsymbol\sigma
=
\lambda\operatorname{tr}(\boldsymbol\varepsilon)\mathbf I
+2\mu\boldsymbol\varepsilon.
$$

当前材料参数为：

| 材料参数 | 当前值 | 说明 |
|---|---:|---|
| Lamé 第一参数 $\lambda$ | $1.0$ | PDE 数据与 PINN 本构直接使用 |
| 剪切模量 $\mu$ | $0.5$ | PDE 数据与 PINN 本构直接使用 |
| Young 模量 $E$ | $4/3$ | 由 $\lambda,\mu$ 换算 |
| Poisson 比 $\nu$ | $1/3$ | 由 $\lambda,\mu$ 换算 |
| 分析类型 | `plane_strain` | 二维平面应变 |

其中

$$
\nu
=
\frac{\lambda}{2(\lambda+\mu)}
=
\frac13,
\qquad
E
=
2\mu(1+\nu)
=
\frac43.
$$

这是制造解验证算例采用的无量纲材料参数，未指定 Pa 等实际工程单位。代码以 $\lambda=1.0$、$\mu=0.5$ 为直接输入；$E$、$\nu$ 只作为等价工程常数说明。

制造解为

$$
u_x(x,y)
=
e^{x-y}x(1-x)y(1-y),
\qquad
u_y(x,y)
=
\sin(\pi x)\sin(\pi y).
$$

该位移在整个边界上为零。体力通过

$$
\boldsymbol b
=
-\nabla\cdot
\boldsymbol\sigma(\boldsymbol u_{\mathrm{exact}})
$$

反推，并由 `soptx.model.linear_elasticity_2d.TriSolHomoDirHuZhang2d` 提供；README 不重复其较长的分量表达式。

## PINN residual 与训练目标

网络预测位移 $\boldsymbol u_\theta$，自动微分依次构造位移梯度、小应变、应力和应力散度。内部点和位移边界上的 residual 分别为

$$
\boldsymbol r_{\mathrm{eq}}
=
-\nabla\cdot\boldsymbol\sigma(\boldsymbol u_\theta)
-\boldsymbol b,
\qquad
\boldsymbol r_D
=
\boldsymbol u_\theta-\bar{\boldsymbol u}.
$$

训练 loss 为

$$
\mathcal L
=
w_{\mathrm{eq}}
\operatorname{MSE}(\boldsymbol r_{\mathrm{eq}},\boldsymbol0)
+
w_D
\operatorname{MSE}(\boldsymbol r_D,\boldsymbol0).
$$

默认权重为 $(w_{\mathrm{eq}},w_D)=(1,30)$。解析位移用于生成边界值，并用于训练期误差诊断；它不作为内部配点的位移监督标签。

## 默认运行参数

| 参数 | 默认值 |
|---|---|
| PDE 数据 | `soptx-default`，即 `TriSolHomoDirHuZhang2d` |
| 网络 | $2\to32\to32\to16\to2$ MLP |
| 激活函数 | `Tanh` |
| optimizer | Adam |
| learning rate | $10^{-3}$ |
| dtype / device | `float64` / CPU |
| 训练采样 | `random`，每次参数更新重新采样 |
| 内部训练配点 | 400 |
| 边界训练配点 | 每条边 100 |
| 固定 validation 配点 | 内部 400、每条边 100 |
| loss 权重 | $(w_{\mathrm{eq}},w_D)=(1,30)$ |
| 参数更新次数 | 2000 |
| seed | 0 |
| 诊断网格 | 每个坐标方向 30 个节点 |
| learning-rate scheduler | 默认关闭；`step_size=0` |
| checkpoint | 默认不写；指定 `--checkpoint_dir` 后保存 `best.pt` 和 `last.pt` |
| 日志间隔 | 每 100 次参数更新 |

命令行参数可以覆盖上述训练配置。当前默认值及其合法范围以 `model.py` 为实现事实源。

## 程序调用链

一次默认训练的主调用链为：

```text
run.py
  → bm.set_backend("pytorch")
  → LinearElasticityPINNModel.get_options()
  → LinearElasticityPINNModel(...)
      → make_default_problem()
      → set_mesh()
      → set_network()
  → model.run()
      → _make_samplers() / _sample_pair()
      → _loss_components()
          → equilibrium_residual()
              → strain() / stress()
          → dirichlet_residual()
      → backward() / optimizer.step()
      → _record_diagnostics()
      → _save_checkpoint()（仅指定 checkpoint 目录时）
  → model.show()
```

`run.py` 负责把 CLI 选项交给模型并启动训练；`model.run()` 严格执行 `epochs` 次参数更新。训练采样为 `random` 时每次更新重新采样，为 `linspace` 时复用同一组训练点；validation 配点在一次运行中固定。`model.show()` 在训练完成后打开 Matplotlib 交互式诊断图。

## 理论—代码—验证对应

| 理论对象或程序契约 | 当前实现 | 验证位置 |
|---|---|---|
| 制造解、体力和全 Dirichlet 边界 | `problem.py::make_default_problem()` 返回 `TriSolHomoDirHuZhang2d` | `constructor_and_shape` |
| 坐标到位移映射 $(x,y)\mapsto(u_x,u_y)$ | `model.py::set_network()` 构造 $2\to32\to32\to16\to2$ MLP，并包装为 `Solution` | `constructor_and_shape` 检查输出 shape 为 $(N,2)$ |
| 小应变 $\boldsymbol\varepsilon=\tfrac12(\nabla\boldsymbol u+\nabla\boldsymbol u^{\mathsf T})$ | `displacement_gradient()` 与 `strain()` | `manufactured_solution_consistency` 以 $10^{-12}$ 检查应变对称性 |
| 平面应变各向同性应力 | `stress()` 使用 PDE 数据的 $\lambda$、$\mu$ | 解析位移的平衡 residual 间接检查本构与体力一致性 |
| 平衡 residual $-\nabla\cdot\boldsymbol\sigma-\boldsymbol b$ | `equilibrium_residual()` 使用 PyTorch autograd | `manufactured_solution_consistency` 以 $10^{-10}$ 检查 residual |
| 全 Dirichlet residual $\boldsymbol u_\theta-\bar{\boldsymbol u}$ | `dirichlet_residual()` | `manufactured_solution_consistency` 以 $10^{-12}$ 检查边界 residual |
| 加权训练 loss | `_loss_components()` 组合平衡与 Dirichlet MSE | `one_update_and_checkpoints` 与 `training_baseline` |
| 参数更新、固定 validation 和 checkpoint | `run()`、`_record_diagnostics()`、`_save_checkpoint()` | `one_update_and_checkpoints` 检查 history、参数变化和临时 checkpoint |
| PyTorch、二维平面应变和全 Dirichlet 能力边界 | `_require_pytorch_backend()`、`set_pde()`、`dirichlet_residual()` | `unsupported_problem_guards` |

## 环境

使用包含 SOPTX、editable FEALPy、PyTorch 和 Matplotlib 的 Python 环境；独立验证驱动不依赖 pytest。当前工作环境可以直接使用：

```powershell
conda activate soptx-gpu
```

下面的命令均从 SOPTX 仓库根目录执行。

## 运行

先运行正确性验证：

```powershell
python .\examples\pinn_linear_elasticity_2d\validate.py
```

再执行默认 2000 次参数更新并显示诊断图：

```powershell
python .\examples\pinn_linear_elasticity_2d\run.py
```

如需保存 checkpoint，可显式指定已忽略的本地输出目录：

```powershell
python .\examples\pinn_linear_elasticity_2d\run.py `
  --checkpoint_dir .\examples\pinn_linear_elasticity_2d\outputs\checkpoints
```

## 验证门禁与当前状态

`validate.py` 先执行小规模程序契约门禁，再默认运行 2000 次参数更新的训练基线：

- 模型使用 SOPTX 制造解，网络输出 shape 为 $(N,2)$；
- 精确位移的应变非对称量不超过 $10^{-12}$；
- 精确位移的平衡 residual 最大绝对值不超过 $10^{-10}$；
- 精确位移的全边界 Dirichlet residual 最大绝对值不超过 $10^{-12}$；
- 一次参数更新能记录 train/validation diagnostics，并在临时目录生成 `best.pt`、`last.pt`；
- 非 PyTorch backend、三维问题、plane stress 和非全 Dirichlet 数据必须被显式拒绝；
- best 固定 validation loss 不超过 $5\times10^{-2}$，且优于首次记录值；
- best checkpoint 的相对位移 $L^2$ error 不超过 $10^{-1}$。

默认阈值属于当前阶段的验证契约，可以通过 CLI 显式修改，但修改后产生的是不同验证语义。
`--epochs 0` 只执行结构与单步冒烟检查，不形成完整训练精度结论。

## 正确性验证结果

2026-07-29 在 Python 3.12.13、PyTorch 2.13.0+cu130、CPU、`float64` 下执行默认
2000 次参数更新，`validation status: passed`。本次使用 Git revision
`648f747185c67b2e542841afc2b28fe322fdc09e`，但运行时工作树为 `dirty=True`，因此以下
结果证明当前实现通过既定门禁，尚不构成干净 revision 下的正式可重放证据。

| 门禁 | 状态 | 实测值 | 阈值 |
|---|---|---:|---:|
| 构造、shape、失败保护、单步更新与 checkpoint | 通过 | 全部 `PASS` | 全部通过 |
| 制造解应变对称性 | 通过 | $0$ | $\le 10^{-12}$ |
| 制造解平衡 residual 最大绝对值 | 通过 | $1.7764\times10^{-15}$ | $\le 10^{-10}$ |
| 制造解 Dirichlet residual | 通过 | $0$ | $\le 10^{-12}$ |
| best 固定 validation loss | 通过 | $3.5441\times10^{-2}$ | $\le 5\times10^{-2}$ |
| best checkpoint 相对位移 $L^2$ error | 通过 | $3.4686\times10^{-2}$ | $\le 10^{-1}$ |

训练耗时为 `26.68 s`，best checkpoint 位于第 2000 次参数更新；对应绝对位移
$L^2$ error 为 $1.7387\times10^{-2}$，固定点平衡 residual RMS 为
$9.1778\times10^{-2}$，最大边界位移误差为 $9.8866\times10^{-3}$。这些补充指标用于
描述当前精度，不替代上表门禁。

当前基线可记为“既定正确性门禁已通过”。提交相关实现后，仍需在干净 revision 上以同一
命令复跑，才能升级为正式可重放基线。

## 输出

- `run.py` 将训练和 validation loss 写入控制台，并在训练结束后打开包含 loss 与位移 $L^2$ error 的交互式诊断图；
- `validate.py` 在终端输出环境、逐项门禁、关键指标和最终状态，不在仓库中写验证结果文件；
- 未指定 `--checkpoint_dir` 时不写 checkpoint；
- 指定 checkpoint 目录后保存 validation loss 最优状态 `best.pt` 和最终状态 `last.pt`；
- `validate.py` 在系统临时目录检查 checkpoint，结束后自动清理，不在仓库中保存 `.pt`。

## 当前能力边界

训练过程中记录训练 loss、固定配点 validation loss、learning rate，以及解析位移的分量和合成 $L^2$ error。固定 validation loss 用于选择可选的 `best.pt`；解析解误差只作诊断，不参与参数更新。

默认训练使用二维、小变形、静力、各向同性、平面应变和全 Dirichlet 边界。牵引边界、平面应力、三维、跨问题 Neural Operator、可复用局部力学表示以及面向 PIML 的正式复用 API 均不属于本阶段。
