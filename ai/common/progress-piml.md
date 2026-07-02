---
title: "SOPTX PIML Multiscale Progress"
tags:
  - ai-context
  - piml
  - multiscale
  - soptx
  - progress
status: "active"
date: 2026-07-02
---

# PIML 多尺度预测原型 Progress

本文件是 SOPTX PIML 多尺度工作线的续接入口。新窗口续接时，先读 `ai/common/status.md`，再读本文件；需要写代码时，再按本文列出的上下文和代码位置继续阅读。上位计划与数学原则见 `docs/ai_piml_context.md` 所列 dut-postdoc 文档。

## 当前结论

截至 2026-07-02，当前结论应表述为：

```text
PIML 多尺度单步前向核心管道 (路线①·子结构静力缩聚) 已在 SOPTX 打通:
宏观密度 -> 逐子结构 rho_local -> 预测器 -> 缩聚算子 K_s^j -> 接口方程组装
-> 接口求解 -> 细尺度恢复。ExactPredictor 的 K^cond 与全尺度全局 Schur 补
机器精度一致 (~1e-15, 目标 1e-10); 接口解/细尺度恢复与全尺度直解一致 (~1e-12);
全局求解规模降至接口自由度 (8x8 粗网格 L=10: 13122 -> 2754, 4.76x)。
```

当前不应表述为：

```text
已具备训练好的 PIML 预测器 / 已接入优化循环 / 已与 matrix-free 算子协同。
```

（TrainedPredictor（T4b 极小 MLP）未训练；`operator_backend="piml_multiscale"`
未接入 `LagrangeFEMAnalyzer`；K̂_s^j 喂给全局 Matrix-Free 作用属后续阶段。）

## 必跑产出实测（2026-07-02，回填 dut-postdoc 答辩帧 7）

算例：矩形悬臂 [0,2]×[0,1]，左边固支、右边中点竖向点载荷 P=-1，E=1、ν=0.3、
平面应力，Q4 均匀细网格，光滑非均匀密度场（值域约 [0.3, 0.9]），粗网格 8×8。

| 粗/细比 L | 细网格 | 全尺度 DOF | 接口 DOF | 降维 | ① V1 误差 (Exact) | ② 接口求解残差 | ③ 接口解 vs 全尺度直解 | ③′ 细尺度恢复 |
|---|---|---|---|---|---|---|---|---|
| 5×5 | 40×40 | 3,362 | 1,314 | 2.56x | 1.384e-15 | 2.142e-13 | 9.389e-14 | 9.534e-14 |
| 10×10 | 80×80 | 13,122 | 2,754 | 4.76x | 2.617e-15 | 3.382e-13 | 4.320e-12 | 4.355e-12 |

- ① V1 = ‖K^cond − S‖_F/‖S‖_F，S 为全尺度细网格刚度阵消去全部内部自由度的全局
  Schur 补——静力缩聚与全尺度数学等价，实测机器精度（目标 < 1e-10）。
- MockPredictor（均匀缩放解析映射）同场对照 V1 误差：L=5 为 2.841e-2、L=10 为
  3.117e-2——演示预测器接口互换 + 非精确预测器误差可度量（为 TrainedPredictor 预留位）。
- ② 为接口方程 ‖K_bc U_b − F_bc‖/‖F_bc‖（直解后残差）。
- 数据文件：`outputs/piml_forward_prototype.csv`（outputs/ 默认 gitignore，重跑即得）。

## 当前进度

1. T1 粗/细两级网格与映射完成：`CoarseFineMeshPair`（均匀 Q4 细网格，
   nx=ncx·L；节点 x-major、tensor DOF 节点交错、cell_to_dof 升序三条 FEALPy
   结构化约定均在构造时 assert 校验）。
2. T2 子结构静力缩聚完成：`SubstructureTemplate`（参考子结构模板：KE_ref +
   局部散射；均匀网格下细单元同构、KE_e = ρ_e·KE_ref 已实证）+
   `SubstructureOperator`（X、K_s、N、内部恢复、载荷缩聚）。
3. T3 接口缩聚组装完成：`InterfaceCondensedSystem`（K^cond 组装、含内部载荷的
   缩聚 F_b = F_b − Σ X^T F_i、Dirichlet 行列消去、接口直解、细尺度恢复）。
4. T4（部分）预测器抽象完成：`MultiscalePredictor` / `ExactPredictor` /
   `MockPredictor` 接口互换；TrainedPredictor 未做（T4b，时间富余再做）。
5. T5 单步前向闭环完成（含一般载荷的内部分量缩聚），与全尺度直解机器精度一致。
6. V1/V2 测试固化：`soptx/tests/test_equivalent_stiffness_vs_fullscale.py`
   4 个用例（V1 缩聚 vs 全尺度 Schur；刚体平移/转动再现 + K_s 零作用；
   Mock/Exact 互换；前向闭环 vs 直解），全部通过。

## 已定关键决策

1. 构造形式为路线①子结构静力缩聚（Schur 补），不是 Huang 2022 EMsFEM 角节点 +
   线性 BC（后者归长期计划 T1.3.1，答辩作奠基引用）。
2. 预测器只消费 `rho_local`（逐细单元 coef），不解释原始 rho/SIMP/过滤——与
   matrix-free 线的 rho/coef 语义边界一致。
3. 问题无关性的软件体现：所有子结构共享同一参考模板（KE_ref + 局部散射 +
   [内部;边界] 局部序），predict 不感知全局网格。
4. 原型 numpy-only、接口方程用 scipy 直解；与 matrix-free 协同（K̂_s^j 喂给全局
   matrix-free 作用、不组装 K^cond）属阶段三。
5. Dirichlet 自由度必须落在接口集合内（均匀两级网格下域边界天然在粗网格骨架上），
   代码中 assert 保护。
6. 开发分支 `codex/piml-multiscale-prototype`（与 `matrix-free-interface` 并列）。

## 重要代码位置

```text
C:\workspace\soptx_heliang\soptx\analysis\multiscale\coarse_fine_mesh.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\multiscale_shape.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\piml_predictor.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\equivalent_stiffness.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\fullscale_reference.py
C:\workspace\soptx_heliang\soptx\tests\test_equivalent_stiffness_vs_fullscale.py
C:\workspace\soptx_heliang\soptx\benchmarks\benchmark_piml_forward.py
```

## 验证命令

在 `C:\workspace\soptx_heliang` 下运行：

```powershell
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_equivalent_stiffness_vs_fullscale.py -q -p no:cacheprovider
```

当前期望结果：`4 passed`。

## Benchmark 命令（必跑产出复现）

```powershell
.\.venv\Scripts\python.exe -m soptx.benchmarks.benchmark_piml_forward `
  --coarse 8x8 --levels 5,10 --output outputs/piml_forward_prototype.csv
```

注意：PowerShell 控制台若为 GBK 编码，中文输出会乱码；CSV 为 UTF-8 正常。

## 环境备忘（2026-07-02）

`.venv` 曾被 codex runtime 重建，丢失 fealpy/scipy/pytest。当前修复方式：

- `.venv\Lib\site-packages\fealpy_local.pth` 指向 `C:\workspace\fealpy_heliang`
  （fealpy 以源码路径接入，非 pip 安装）；
- 已补装 scipy、pytest、sympy、gmsh、tqdm、matplotlib（fealpy import 依赖）。
  修复后 matrix-free 安全网 `test_matrix_free_vs_assembled.py` 恢复 6 passed。

## 下一步（按优先级）

1. **T4b 极小 TrainedPredictor**（时间富余再做）：随机局部密度采样 +
   `SubstructureTemplate.condense` 标注，极小 MLP，损失 = 算子 MSE + 缩聚刚度 MSE；
   目标预测误差入团队量级（L=5 ~1e-3，对照 Huang 2022 分档，见集成计划 V4）。
   Mock 的误差列（~3e-2）即其对照基线位。
2. **V3 出图**：`examples/piml_baseline_forward.py`，宏微映射 + 密度分布
   （`piml_baseline.pdf`，同步 dut-postdoc deck figures）。
3. **阶段二**：`operator_backend="piml_multiscale"` 接入 `LagrangeFEMAnalyzer.solve_state()`。
4. **阶段三**：K̂_s^j 直接喂给全局 Matrix-Free 作用（能力 A⊗B 咬合点，
   见 `ai/common/progress-matrix-free.md`）。

## 新窗口续接提示词

```text
按 C:\workspace\soptx_heliang\ai\common\status.md 续接「PIML 多尺度原型」。
先复述当前进度、已定关键决策、下一步与实现计划，我确认后再继续。
```
