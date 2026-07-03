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
upstream:
  guide: research/postdoc-plan/defense-sprint/direction-1-piml-matrix-free/frame7_piml_pipeline_guide.md
  principle: frame7_piml_pipeline_guide.md §3（静力缩聚构造式见 §3.3）
  plan_long_term: research/postdoc-plan/long-term/direction-1-piml-matrix-free/piml-matrix-free-execution-plan.md
  plan_id: T1.3.2
---

# PIML 多尺度预测原型 Progress

本文件是 SOPTX PIML 多尺度工作线的续接入口。新窗口续接时，先读 `ai/common/status.md`，再读本文件；需要写代码时，再按本文列出的上下文和代码位置继续阅读。

> 原 `docs/ai_piml_context.md`（旧续接入口）与 `docs/piml_multiscale_architecture_notes.md`
> （接入前架构备忘录）的仍有效内容已于 2026-07-02 并入本文件，两文件已删除；
> 帧 7 数值快照见 `docs/frame7_piml_pipeline_results.md`。

## 上位文档（dut-postdoc）

> 跨库映射集中维护于 `ai/common/status.md`「跨库映射」表；本节给细粒度指针，
> 只用「文件名 + 章节号 + 计划项 ID」，不写行号。
> **2026-07-03 核对**：dut-postdoc 已把帧 7 收敛为单帧 guide，原「总体计划」
> `soptx-piml-multiscale-integration-plan.md` 与「数学原则」`piml_multiscale_math_principles.md`
> 已删除（被单帧 guide 归并接管），其数学原理并入 guide §3。两条旧指针已移除。

- **帧级主入口 / 数学原理**（缩聚路线、实测结果、答辩口径、边界、补数方式；数学原理见
  guide §3，EMsFEM 两级网格 / 问题无关性见 §3.1–3.2、静力缩聚构造式见 §3.3）：
  `research/postdoc-plan/defense-sprint/direction-1-piml-matrix-free/frame7_piml_pipeline_guide.md`
- **24 个月长期计划**（本原型 = 其阶段一 T1.3.2 最小前向核心）：
  `research/postdoc-plan/long-term/direction-1-piml-matrix-free/piml-matrix-free-execution-plan.md`

## 当前结论

截至 2026-07-03，当前结论应表述为：

```text
PIML 多尺度单步前向核心管道 (路线①·子结构静力缩聚) 已在 SOPTX 打通:
宏观密度 -> 逐子结构 rho_local -> 预测器 -> 缩聚算子 K_s^j -> 接口方程组装
-> 接口求解 -> 细尺度恢复。ExactPredictor 的 K^cond 与全尺度全局 Schur 补
机器精度一致 (~1e-15, 目标 1e-10); 接口解/细尺度恢复与全尺度直解一致 (~1e-12);
全局求解规模降至接口自由度 (8x8 粗网格 L=10: 13122 -> 2754, 4.76x)。
T4b 极小 MLP TrainedPredictor 已训练, 在实际 64 子结构光滑场上产出真实预测误差
||K_hat_s - K_s||/||K_s|| 均值 L=5: 1.6e-3、L=10: 8.2e-3 (对照 Mock ~3e-2,
优约一个量级), 回填 deck 帧 7 证据 ④。
```

当前不应表述为：

```text
已具备结构保持 (对称正定/能量一致) 的训练预测器 / 已做训练版前向恢复 /
已接入优化循环 / 已与 matrix-free 算子协同。
```

（T4b TrainedPredictor 只学 K_s（组装 K^cond、度量 ④ 只需 K_s），不学 X/N̂、
不做训练版前向恢复——结构保持参数化与多尺度灵敏度属阶段三；
`operator_backend="piml_multiscale"` 未接入 `LagrangeFEMAnalyzer`；
K̂_s^j 喂给全局 Matrix-Free 作用属后续阶段。）

## 必跑产出实测（headline；完整数值表见 results）

> **数值单一事实源（SSOT）= `docs/frame7_piml_pipeline_results.md`**——全表、复现命令、
> deck LaTeX 均在此维护。本节只留续接需要的 headline；重跑 benchmark 后**只改 results，
> 不在此复写全表**（避免两处数字漂移）。

算例：矩形悬臂 [0,2]×[0,1]，左固支、右中点竖向载荷 P=-1，E=1/ν=0.3/平面应力，
Q4 均匀细网格、光滑非均匀密度场（约 [0.3,0.9]），粗网格 8×8、64 子结构；粗/细比 L=5、10。

- ① 求解降维：DOF 3362→1314（2.6x）/ 13122→2754（4.8x）。
- ② V1 缩聚精确性（vs 全尺度 Schur 补 S）：~1e-15（机器精度，目标 <1e-10）。
- ③ 接口解 / 细尺度恢复 vs 全尺度直解：≤4.3e-12。
- ④ TrainedPredictor（T4b 极小 MLP）逐子结构 ‖K̂_s−K_s‖/‖K_s‖ 均值 L=5 1.6e-3、
  L=10 8.2e-3（Mock 对照 ~3e-2，优约一个量级）。
- 逐档 median/max、assembled v1_trained、验证集泛化数、L=10>L=5 成因说明 → 见 results §2–§4。

## 当前进度

1. T1 粗/细两级网格与映射完成：`CoarseFineMeshPair`（均匀 Q4 细网格，
   nx=ncx·L；节点 x-major、tensor DOF 节点交错、cell_to_dof 升序三条 FEALPy
   结构化约定均在构造时 assert 校验）。
2. T2 子结构静力缩聚完成：`SubstructureTemplate`（参考子结构模板：KE_ref +
   局部散射；均匀网格下细单元同构、KE_e = ρ_e·KE_ref 已实证）+
   `SubstructureOperator`（X、K_s、N、内部恢复、载荷缩聚）。
3. T3 接口缩聚组装完成：`InterfaceCondensedSystem`（K^cond 组装、含内部载荷的
   缩聚 F_b = F_b − Σ X^T F_i、Dirichlet 行列消去、接口直解、细尺度恢复）。
4. T4 预测器抽象完成：`MultiscalePredictor` / `ExactPredictor` /
   `MockPredictor` / `TrainedPredictor` 接口互换。T4b 极小 MLP TrainedPredictor
   已训练（隐层 256×3 SiLU，输出 vech(K_s)；标注 = `SubstructureTemplate.condense`
   精确缩聚，样本 rho_local ~ U[0.3,0.9]^m i.i.d.，n_train=6e4，Adam+cosine 1500 epoch）——
   只学 K_s，返回算子 X=None（不做训练版前向恢复）。真实预测误差见「必跑产出实测」。
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

## 阶段二/三 接入设计（原架构备忘录并入，2026-07-02）

1. **接入点**：`operator_backend="piml_multiscale"` 作为 `LagrangeFEMAnalyzer` 状态方程
   求解的一种后端，与 `assembled` / `matrix_free` 并列；优化器主循环只调用
   `state = analyzer.solve_state(rho_val=rho_phys)`，不感知后端切换。
2. **路径**：`rho_val -> 材料插值 -> coef（逐细单元）-> 逐粗单元 rho_local ->
   predict -> K̂_s^j -> 接口组装（或 matrix-free 作用）-> 解 U_b -> u^fine = N̂ U_b`。
3. **灵敏度**：阶段二仍不替换（保留现有局部刚度导数路径）；多尺度灵敏度
   （含 ∂N̂/∂ρ）属阶段三。
4. **阶段三**：结构保持参数化（对称正定/能量一致）TrainedPredictor；误差传播分析
   （局部算子误差 -> 位移 -> 灵敏度 -> 拓扑偏差）；K̂_s^j 直接喂全局 Matrix-Free
   （不组装 K^cond）；GPU/多后端批量推理。
5. **命名约定**：`rho`（物理密度）/ `rho_local`（粗单元内细单元密度，预测器输入）/
   `coef`（积分器刚度系数）/ `N`、`N_hat`（精确/预测多尺度形函数）/ `K_s`、`K_s_hat`
   （精确/预测缩聚刚度）。预测器输入若已是 coef/relative_stiffness，字段名不叫 rho，
   避免误以为预测器内部负责 SIMP/材料插值。

## 重要代码位置

```text
C:\workspace\soptx_heliang\soptx\analysis\multiscale\coarse_fine_mesh.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\multiscale_shape.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\piml_predictor.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\trained_predictor.py   # T4b 极小 MLP TrainedPredictor
C:\workspace\soptx_heliang\soptx\analysis\multiscale\equivalent_stiffness.py
C:\workspace\soptx_heliang\soptx\analysis\multiscale\fullscale_reference.py
C:\workspace\soptx_heliang\soptx\tests\test_equivalent_stiffness_vs_fullscale.py
C:\workspace\soptx_heliang\soptx\tests\test_trained_predictor.py            # T4b 管道健全性 (torch 缺失跳过)
C:\workspace\soptx_heliang\soptx\benchmarks\benchmark_piml_forward.py
C:\workspace\soptx_heliang\soptx\benchmarks\train_piml_predictor.py         # T4b 训练 CLI
C:\workspace\soptx_heliang\soptx\benchmarks\benchmark_piml_trained.py       # T4b 回填 benchmark (deck ④)
```

## 验证命令

在 `C:\workspace\soptx_heliang` 下运行：

```powershell
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_equivalent_stiffness_vs_fullscale.py soptx/tests/test_trained_predictor.py -q -p no:cacheprovider
```

当前期望结果：`7 passed`（4 V1/V2 + 3 T4b；torch 缺失时后者自动跳过为 `4 passed`）。

## Benchmark 命令（必跑产出复现）

完整复现命令（`benchmark_piml_forward` / `train_piml_predictor` / `benchmark_piml_trained`）
统一维护于数值 SSOT `docs/frame7_piml_pipeline_results.md` §5，此处不复写，避免命令漂移。

注意（环境备忘）：PowerShell 控制台若为 GBK 编码，中文/组合符号（K̂）会乱码或触发
UnicodeEncodeError；两 T4b 脚本已在入口 `sys.stdout.reconfigure(encoding="utf-8")`，CSV 亦 UTF-8。

## 环境备忘（2026-07-02）

`.venv` 曾被 codex runtime 重建，丢失 fealpy/scipy/pytest。当前修复方式：

- `.venv\Lib\site-packages\fealpy_local.pth` 指向 `C:\workspace\fealpy_heliang`
  （fealpy 以源码路径接入，非 pip 安装）；
- 已补装 scipy、pytest、sympy、gmsh、tqdm、matplotlib（fealpy import 依赖）。
  修复后 matrix-free 安全网 `test_matrix_free_vs_assembled.py` 恢复 6 passed。

**T4b 训练权重与 torch（2026-07-03 补充）**：

- `torch 2.11.0+cu128` 在 `.venv` 可用，CUDA（RTX 5070 Ti）可用；训练脚本默认
  `--device cuda`（各档约 3–4 分钟），亦可 `--device cpu`（明显更慢）。
- 训练权重 `outputs/piml_trained_predictor_L{5,10}.pt` 被 `.gitignore` 忽略
  （新加 `*.pt`/`*.pth` 规则），**不入库**——与「outputs/ 重跑即得」约定一致。
  他人 checkout 后 `benchmark_piml_trained` 会因缺权重报 `FileNotFoundError`，
  **需先跑 `train_piml_predictor` 复现**（见「Benchmark 命令」）。CSV 产出
  `outputs/piml_*_prototype.csv` 同样 gitignore、重跑即得。
- `TrainedPredictor` 依赖 torch；`soptx/analysis/multiscale/__init__.py` 对 torch
  缺失做 `try/except` 跳过（不导出 TrainedPredictor 等），故 numpy-only 的
  V1/V2 前向管道与 `test_equivalent_stiffness_vs_fullscale.py` 不受 torch 有无影响；
  `test_trained_predictor.py` 则 `pytest.importorskip("torch")` 自动跳过。

## 已完成里程碑

- **✅ deck 帧 7 证据 ④ 回填（2026-07-03 完成）**：T4b 极小 MLP 真实预测误差
  `‖K̂_s − K_s‖/‖K_s‖` 均值 **L=5: 1.6e-3、L=10: 8.2e-3**（对照 Mock ~3e-2）已在
  dut-postdoc 窗口回填帧 7 ④——`talks/2026-postdoc-entry-assessment/template-8min.tex`
  ④ 行 + guide `frame7_piml_pipeline_guide.md` §5/§9.3。上游事实源与现成 LaTeX 见
  `docs/frame7_piml_pipeline_results.md` §4/§6；deck 帧 7 脚注 [2] = Huang 2023
  （子结构 PIML，*EML* 63:102041），`‖·‖` 目标量级对照沿用 Huang 2022 分档。

## 下一步（按优先级）

1. **阶段二**：`operator_backend="piml_multiscale"` 接入 `LagrangeFEMAnalyzer.solve_state()`
   （与 `assembled` / `matrix_free` 并列，优化器主循环不感知后端切换）。
2. **阶段三**：K̂_s^j 直接喂给全局 Matrix-Free 作用（能力 A⊗B 咬合点，
   见 `ai/common/progress-frame8_matrix_free.md`）；结构保持参数化 TrainedPredictor（对称
   正定/能量一致）+ 训练版前向恢复（X̂/N̂）+ 多尺度灵敏度。
3. **V3 出图（可选·答辩非阻塞，缓做）**：`examples/piml_baseline_forward.py`，宏微映射 +
   密度分布（`piml_baseline.pdf`）。帧 7 为四条文字证据链、已回填完整、无大图位，
   故本次答辩不必需；留作后续长版报告 / 技术文档配图。

## 新窗口续接提示词

```text
按 C:\workspace\soptx_heliang\ai\common\status.md 续接「PIML 多尺度原型」。
先复述当前进度、已定关键决策、下一步与实现计划，我确认后再继续。
```
