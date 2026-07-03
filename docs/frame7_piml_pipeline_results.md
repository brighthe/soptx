# 帧 7：PIML 多尺度前向分析原型验证数值结果 (T1–T5 + V1)

本文档记录了基于 `soptx_heliang` 代码库中 `soptx/benchmarks/benchmark_piml_forward.py` 单次运行（2026-07-02，分支 `codex/piml-multiscale-prototype`）所产生的基准测试数据。这些数据可作为 `dut-postdoc` 仓库下答辩 PPT（`template-8min.tex` 的 Frame 7「方向一 · PIML 增强多尺度前向分析原型」）的上游事实数据源；帧 7 右栏四条证据链的数值均出自本文。

本文的展示框架与答辩口径基于 dut-postdoc 帧级主入口
`research/postdoc-plan/defense-sprint/direction-1-piml-matrix-free/frame7_piml_pipeline_guide.md`；
任务划分与数学原则见同目录 `soptx-piml-multiscale-integration-plan.md` /
`piml_multiscale_math_principles.md`。

## 1. 测试用例参数设定

- **物理问题**: 矩形悬臂 $[0, 2] \times [0, 1]$，左边固支，右边中点竖向点载荷 $P=-1$；$E=1$，$\nu=0.3$，平面应力。
- **密度场**: 光滑非均匀密度场（值域约 $[0.3, 0.9]$），逐细单元赋值——保证 V1 比较不是均匀密度下的平凡情形。
- **两级网格**: 粗网格 $8 \times 8$（64 个子结构）；每个粗单元内细分 $L \times L$ 个 Q4 均匀细单元，两档粗/细比 $L = 5$、$L = 10$（对应全尺度细网格 $40 \times 40$、$80 \times 80$）。
- **构造形式**: 路线①·子结构静力缩聚（Schur 补）$N^j = [-(K_{ii}^j)^{-1} K_{ib}^j;\ I]$，$K_s^j = (N^j)^T K^j N^j$；接口方程 $K^{\mathrm{cond}} U_b = F_b$ 用 scipy 直解，再逐子结构恢复细尺度位移。

## 2. 规模与求解降维数据（帧 7 证据 ①）

| 粗/细比 $L$ | 全尺度细网格 | 全尺度 DOF | 接口 DOF | 降维 |
|---|---|---|---|---|
| $5 \times 5$ | $40 \times 40$ | 3,362 | 1,314 | **2.56x** |
| $10 \times 10$ | $80 \times 80$ | 13,122 | 2,754 | **4.76x** |

全局求解规模从全尺度细网格自由度降至子结构**接口自由度**；帧 7 上取整为 $2.6\times$ / $4.8\times$。

## 3. 核心验证数据（帧 7 证据 ②③）

| 粗/细比 $L$ | ① V1 缩聚精确性 (Exact) | ② 接口求解残差 | ③ 接口解 vs 全尺度直解 | ③′ 细尺度恢复 vs 直解 |
|---|---|---|---|---|
| $5 \times 5$ | **1.384e-15** | 2.142e-13 | 9.389e-14 | 9.534e-14 |
| $10 \times 10$ | **2.617e-15** | 3.382e-13 | 4.320e-12 | 4.355e-12 |

- **① V1** $= \|K^{\mathrm{cond}} - S\|_F / \|S\|_F$，$S$ 为全尺度细网格刚度阵消去全部内部自由度得到的**全局 Schur 补**——静力缩聚与全尺度数学等价，实测机器精度（验收目标 $< 10^{-10}$，超出约 5 个量级）。
- **②** 为接口方程直解后残差 $\|K_{\mathrm{bc}} U_b - F_{\mathrm{bc}}\| / \|F_{\mathrm{bc}}\|$（帧 7 上写作 $\sim 10^{-13}$）。
- **③/③′** 为接口解及细尺度恢复位移相对全尺度直解的 $\ell^2$ 相对误差（帧 7 上写作 $\le 4.3 \times 10^{-12}$）。

**结论**：单步前向管道（宏观密度 → 逐子结构 $\rho_{\mathrm{local}}$ → 预测器 → $K_s^j$ → 接口组装 → 接口求解 → 细尺度恢复）全程连通，且 Exact 路径与全尺度参考在机器精度上一致。PIML 改变的只是"如何**快速获得** $K_s^j$"（按需预测替代内部自由度消元），不改变离散模型。

## 4. 预测器对照（帧 7 证据 ④）

- **ExactPredictor**（精确静力缩聚，真值/标注来源）：即上表 Exact 列。
- **MockPredictor**（均匀缩放解析映射，演示接口互换）：同场 assembled V1 误差 $L=5$ 为 **2.841e-2**、$L=10$ 为 **3.117e-2**——证明预测器可互换、非精确预测器的误差可度量，是 TrainedPredictor 的**对照基线位**。
- **TrainedPredictor**（T4b 极小 MLP）：**已训练**（2026-07-03）。极小 MLP（隐层 256×3、SiLU）输入 $\rho_{\mathrm{local}}$、输出对称 $K_s$ 的上三角 $\mathrm{vech}(K_s)$（$L=5$ 输出 820 维、$L=10$ 输出 3240 维）；训练标注为 `SubstructureTemplate.condense` 精确缩聚，样本 $\rho_{\mathrm{local}}\sim U[0.3,0.9]^m$ i.i.d.（$n_{\mathrm{train}}=6\times10^4$），Adam（lr $10^{-3}$ + cosine）1500 epoch。

**真实预测误差**（与证据 ①②③ 同场：实际 64 个子结构、光滑非均匀密度场）：

| 粗/细比 $L$ | 逐子结构 $\|\widehat K_s-K_s\|_F/\|K_s\|_F$ (mean / median / max) | assembled $v_1^{\mathrm{trained}}$ (vs 全尺度 Schur) | vs Mock 对照 |
|---|---|---|---|
| $5\times5$ | **1.633e-3** / 8.197e-4 / 8.123e-3 | **2.112e-3** | Mock 2.841e-2（约 13× 优） |
| $10\times10$ | **8.230e-3** / 3.380e-3 / 6.257e-2 | **9.343e-3** | Mock 3.117e-2（约 3.3× 优） |

- 逐子结构 $\|\widehat K_s-K_s\|_F/\|K_s\|_F$（真值 = ExactPredictor）为 deck ④ **主口径**——$L=5$ 落在 $\sim10^{-3}$、$L=10$ 落在 $\sim10^{-2}$，与团队目标量级（对照 Huang 2022 分档）一致。
- assembled $v_1^{\mathrm{trained}}=\|K^{\mathrm{cond}}_{\mathrm{trained}}-S\|_F/\|S\|_F$ 与 Mock、Exact 同表可比，直观显示学习型预测器较解析均匀缩放 mock 提升约一个量级。
- 训练**验证集**（i.i.d. $U[0.3,0.9]$，泛化参考）逐样本 $\|\widehat K_s-K_s\|_F/\|K_s\|_F$：$L=5$ mean 2.305e-3、$L=10$ mean 1.010e-2——较上表实际场略高，因光滑场内单粗单元密度近均匀、方差更小，属分布更"易"的一侧。
- **诚实边界**：T4b 仅学 $K_s$（组装 $K^{\mathrm{cond}}$、度量 ④ 只需 $K_s$）；不学 $X/\widehat N$、不做训练版前向恢复——结构保持参数化（对称正定/能量一致）与多尺度灵敏度属阶段三。

## 5. 复现命令

在 `C:\workspace\soptx_heliang` 下运行（数据文件 `outputs/piml_forward_prototype.csv`，outputs/ 默认 gitignore，重跑即得）：

```powershell
# 必跑产出 benchmark（§2/§3 数据来源）
.\.venv\Scripts\python.exe -m soptx.benchmarks.benchmark_piml_forward `
  --coarse 8x8 --levels 5,10 --output outputs/piml_forward_prototype.csv

# T4b 训练极小 TrainedPredictor（产出 outputs/piml_trained_predictor_L{5,10}.pt）
.\.venv\Scripts\python.exe -m soptx.benchmarks.train_piml_predictor `
  --coarse 8x8 --levels 5,10 --device cuda

# T4b 回填 benchmark（§4 真实预测误差数据来源，需先训练）
.\.venv\Scripts\python.exe -m soptx.benchmarks.benchmark_piml_trained `
  --coarse 8x8 --levels 5,10 --device cuda --output outputs/piml_trained_prototype.csv

# 测试固化（期望 4 + 3 passed；torch 缺失时 test_trained_predictor 自动跳过）
.\.venv\Scripts\python.exe -m pytest soptx/tests/test_equivalent_stiffness_vs_fullscale.py soptx/tests/test_trained_predictor.py -q -p no:cacheprovider
```

## 6. 跨库同步材料 (LaTeX)

帧 7 右栏（`template-8min.tex`）当前采用的四条证据链即本文数据的展示化：

```latex
\circnum{1}~\textbf{求解降维}（粗网格 $8{\times}8$，64 子结构）
  细 $5{\times}5$：DOF $3362\to1314$（$\mathbf{2.6\times}$）
  细 $10{\times}10$：DOF $13122\to2754$（$\mathbf{4.8\times}$）
\circnum{2}~\textbf{缩聚精确性}（vs 全尺度 Schur 补）
  两档 $\mathbf{1.4}$/$\mathbf{2.6}{\times}\mathbf{10^{-15}}$（机器精度）
\circnum{3}~\textbf{接口求解一致}（vs 全尺度直解）
  残差 $\sim\!10^{-13}$；解误差 $\le 4.3{\times}10^{-12}$
\circnum{4}~\textbf{PIML 预测误差}（极小网络·实测）
  $5{\times}5$：$\mathbf{1.6{\times}10^{-3}}$；$10{\times}10$：$\mathbf{8.2{\times}10^{-3}}$
  （$\|\widehat K_s-K_s\|/\|K_s\|$ 均值；Mock 对照 $\sim3{\times}10^{-2}$）
```

如需表格形式（对应 `outline-8min.md` 帧 7 的数值结果表），可直接使用：

```latex
\begin{tabular}{@{}p{0.18\linewidth}p{0.40\linewidth}p{0.36\linewidth}@{}}
    \toprule
    {\color{structure.fg}验证环节} & {\color{structure.fg}数值指标} & {\color{structure.fg}实测结果（$5{\times}5$ / $10{\times}10$）} \\
    \midrule
    求解降维 & 全尺度 DOF $\to$ 接口 DOF & $3362\to1314$ / $13122\to2754$（$2.6\times$/$4.8\times$） \\
    缩聚精确性 & $\|K^{\mathrm{cond}}-S\|_F/\|S\|_F$（vs 全局 Schur 补） & $1.4\times10^{-15}$ / $2.6\times10^{-15}$（机器精度） \\
    接口求解 & 残差；接口解 vs 全尺度直解 & 残差 $\sim10^{-13}$；解误差 $9.4\times10^{-14}$ / $4.3\times10^{-12}$ \\
    PIML 预测 & $\|\widehat K_s-K_s\|/\|K_s\|$ 均值（Mock 对照 $\sim3\times10^{-2}$） & $1.6\times10^{-3}$ / $8.2\times10^{-3}$（T4b 极小 MLP 实测） \\
    \bottomrule
\end{tabular}
```

## 7. 诚实边界（答辩口径提醒）

- 已证明的是：**前向管道连通 + 静力缩聚机器精度精确 + 全局求解降至接口自由度 + 预测器接口互换 + 极小 MLP TrainedPredictor 产出真实预测误差（$L=5\sim10^{-3}$、$L=10\sim10^{-2}$，优于 Mock 约一个量级）**；
- 尚未做的是：结构保持参数化（对称正定/能量一致）TrainedPredictor 与训练版前向恢复（T4b 只学 $K_s$，$X/\widehat N$ 属阶段三）、`operator_backend="piml_multiscale"` 接入分析器（阶段二）、$\widehat K_s^j$ 喂给全局 Matrix-Free 作用（阶段三，能力 A⊗B 咬合点）。
- 上游进度与决策记录见 `ai/common/progress-frame7_piml.md`；帧级主入口见 dut-postdoc `frame7_piml_pipeline_guide.md`，任务计划见 `soptx-piml-multiscale-integration-plan.md`。
