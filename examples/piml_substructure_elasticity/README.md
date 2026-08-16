# PIML 二维线弹性子结构静力缩聚示例 (PIML Substructure Elasticity Demo)

本目录提供 **Problem-Independent Machine Learning (PIML)** 范式在二维线弹性子结构静力缩聚的端到端代码范例，完全基于 **FEALPy 后端管理器 (`fealpy.backend.backend_manager as bm`)** 构建，算例基准与文献 [Huang 2023](../../literature/topology-opt/translations/Huang2023-PIML-substructure-zh.md) 第 4.1 节 MBB 梁问题设定完全对齐。

---

## 核心内容

1. **统一 `bm` 后端**：阵列操作、代数求解与特征值校验全面基于 `fealpy.backend.backend_manager as bm`，在保持与 PyTorch 模型对接的同时，确保与 `soptx.fem.substructure` 核心库后端一致；
2. **对齐 Huang 2023 第 4.1 节完整 MBB 梁算例**：基于 `soptx.problems.elasticity.FullMBBBeam2d` 物理模型，几何尺寸 [$12.0 \times 2.0$]、$12 \times 2$ 子结构网格 (共 24 个子结构)、顶面中心集中荷载及完整 MBB 梁简支约束 (不利用对称性裁剪)；
3. **PIML 全局接口组装与下游评估**：PIML 预测的缩聚刚度 $\widehat{\mathbf K}_s$ 进入 `GlobalAssembler` 全局接口系统，计算结构柔度及全场位移恢复误差；
4. **变形子空间上的 Cholesky 参数化**：自由漂浮子结构的 $\mathbf K_s$ 以刚体模态为精确零空间，代理按 $\widehat{\mathbf K}_s=\mathbf R_\perp\mathbf L\mathbf L^{\mathsf T}\mathbf R_\perp^{\mathsf T}$ 重构，秩亏结构由构造保证，无需正则化，详见 [results_analysis.md](results_analysis.md) 第 1.2 节。

---

## 目录结构

```text
soptx/examples/piml_substructure_elasticity/
├── compare_exact.py        <-- [核心对比] PIML 代理缩聚 vs 精确 Schur 补缩聚
├── compare_piml_pinn.py    <-- [跨范式对比] PIML 子结构静力缩聚 vs PINN 强形式
├── results_analysis.md     <-- [契约与报告] 代码—数学映射契约、验证边界与已知问题
├── README.md               <-- [使用说明] 本文档
└── outputs/                <-- [运行产物，由 .gitignore 忽略]
    ├── piml_exact_comparison.json  <-- 代理 vs 精确的机器可读证据
    ├── piml_exact_comparison.png   <-- 位移场与 K_s 四格对比图
    └── piml_vs_pinn_comparison.png <-- PIML vs PINN 跨范式四格对比图
```

> 精确缩聚与全局组装模块 (`SubstructurePrototype`、`SubstructureMesh`、
> `FEAStaticCondensation`、`GlobalAssembler`、`solve_interface_system`)
> 由 [`src/soptx/fem/substructure/`](../../src/soptx/fem/substructure/) 提供。

---

## 调用范式

全部子结构同构，因此两个脚本都构造**一个** `SubstructurePrototype` 并让所有
`SubstructureMesh` 通过 `prototype=proto` 共享它：

- **精确路径批量化**：`prototype.assemble_local_stiffness_batch(density)` 一次得到
  `(B, n_dof, n_dof)` 局部刚度，交给**单个** `FEAStaticCondensation` 沿前导维一次缩聚。
  离线训练样本（250~300 组随机密度）同样只走一次批量装配 + 一次批量缩聚。
- **PIML 路径仍为列表**：`PIMLStaticCondensation` 的代理网络只接受单个子结构的密度输入，
  因此保持逐子结构推理，但复用已经批量装配好的 `K_local_batch[idx]`。

## 快速运行

PIML 代理缩聚与精确 Schur 补缩聚对比（MBB 梁集中载荷，报告算子层与解层两层误差，以及误差归因诊断）：

```bash
python examples/piml_substructure_elasticity/compare_exact.py
```

要求代理全程生效（在役子结构或留出集出现回退即失败）：

```bash
python examples/piml_substructure_elasticity/compare_exact.py --strict --epochs 800
```

加大训练预算与留出集，用于判定精度瓶颈在欠拟合还是训练分布：

```bash
python examples/piml_substructure_elasticity/compare_exact.py --epochs 4000 --train-samples 2000 --val-samples 200
```

随机性由 `--seed`（缺省 `2026`）统一固定，覆盖训练采样、留出采样与网络初始化；比较不同训练配置时应保持种子不变，否则观测差异混有采样噪声。

PIML vs PINN 跨范式对比：

```bash
python examples/piml_substructure_elasticity/compare_piml_pinn.py --pinn-epochs 400
```

`compare_exact.py` 的 `--output-dir` 缺省为脚本同级的 `outputs/`（按脚本位置解析，与从哪个目录发起命令无关）；传相对路径会按当前工作目录解析，可能落到 `.gitignore` 覆盖范围之外。

## 运行产物

- `outputs/piml_exact_comparison.json`：问题配置、训练配置（含 `seed`、`n_reduced_dofs`）、算子层与解层全部误差、回退计数与耗时，以及误差归因诊断（`rigid_basis_residual`、`parameterization_error_ceiling`、留出集误差、零空间伪刚度与能量归因）。
- `outputs/piml_exact_comparison.png`：上排为精确与 PIML 的 $U_y$ 场，下排为首个子结构的
  $\mathbf K_s$ 与 $\widehat{\mathbf K}_s$ 热图。
- `outputs/piml_vs_pinn_comparison.png`：上排为制造解与 PINN 位移场，下排为精确 Schur 补
  $\mathbf K_s$ 与 PIML 预测的 $\widehat{\mathbf K}_s$。
