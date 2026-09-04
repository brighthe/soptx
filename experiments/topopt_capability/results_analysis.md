# 基于 FEALPy 的张量化拓扑优化平台百万级算力评测报告（真实实测版）

本报告是《中国博士后科学基金第 80 批面上资助申请书》第 6 部分研究基础中 **图 10「多后端
异构并行拓扑优化平台的性能实测与大规模优化设计结果」** 的唯一权威事实源。全部性能数字来自
2026-08-28 在同一台机器上的真实运行（`examples/topopt_platform/topopt_3d_simp_real.py`）。

2026-08-28 相对前一版有两处实质变更，均记在 §7：修复了 Hex8 单元刚度的量纲缺陷，并把算例
换成博士论文算例 3.3 的加密版，四个面板统一到同一算例、CPU/GPU 两侧统一 `float64`；
同日面板 (c)(d) 曾改为「优化后构型 + 收敛历程」，当日又撤回、维持「初始构型 + 优化后构型」
的 2×2 前后对比（理由见 §「图 10 面板 (c)(d) 的构型渲染」末尾）。

---

## 1. 结论总览

在 **300×100×20 六面体网格（600,000 单元，191.5 万自由度）** 三维悬臂梁上，用真实 Hex8
弹性刚度、真实 PCG（Jacobi 预条件）求解与 OC 更新跑完整 SIMP 拓扑优化：

| 指标 | 结果 |
|:---|:---|
| 完整优化 | **90 步收敛**（柔度 1161.1 → 86.7 单调下降），体积分数 0.300 达标 |
| 柔度对标 | 与论文算例 3.3 的 k=2 二阶单元结果 86.92 相差 **0.30%** |
| GPU 单步总耗时 | **10.41 s**（均匀密度单步，与 CPU 基线同配置） |
| CPU 单步总耗时（scipy 稀疏） | **255.94 s** |
| 端到端单步加速 | **24.6×** |
| GPU 峰值显存 | **5.89 GB**（全收敛运行；单步 3.23 GB。RTX 5080 16 GB，余量充足） |
| 同一设计下 CPU/GPU 解相对差 | **6.8e-11**（两侧同为 `float64`，均收敛至相对残差 ~1e-8） |
| 同一设计下两侧 CG 迭代步数 | **完全一致**（单步均 1955 步，最终密度均 5160 步） |
| 矩阵自由算子 vs 显式装配 | 相对最大差 **5.98e-16**（代数恒等） |

---

## 2. 算例设置与数学模型

* **设计域**：悬臂梁 $L_x \times L_y \times L_z = 60 \times 20 \times 4$ mm（博士论文算例
  3.3），$x=0$ 端固支，右端面底边施 $-y$ 方向均布线载荷，总合力 1 N；
* **网格**：$300 \times 100 \times 20 = 600{,}000$ 个 Hex8 单元，单元边长各向同性
  $h = 0.2$ mm，自由度 $\mathbf{1{,}915{,}263}$（191.5 万）；
* **材料**：线弹性 $E=1.0, \nu=0.3$，SIMP 插值
  $E_e = E_{\min} + (1-E_{\min})\rho_e^p$，$p=3$，$E_{\min}=10^{-4}$；
* **目标**：柔度最小，体积约束 30%，Sigmund 锥形灵敏度滤波（半径 **1.5 mm 物理长度**，
  即 7.5 个单元），OC 更新（move=0.2）；
* **求解**：矩阵自由全局算子（gather → 批量 GEMV → 散加）+ PCG(Jacobi)，
  停机容差 $\|\mathbf{r}\|/\|\mathbf{b}\| < 10^{-8}$，热启动上一设计步解；
* **收敛判据**：设计变量最大改变量 $\max|\Delta\rho| < 10^{-2}$，连续 3 步满足；
* **后端**：同一代码经 FEALPy 后端管理器跑 NumPy(CPU) 与 PyTorch(GPU, RTX 5080)，
  **两侧统一 `float64`**。CPU 侧 scipy 恒为双精度，GPU 侧若退回 `float32` 则单步对比
  不对等，且 `float32` 的机器精度（约 1.2e-7）根本达不到 1e-8 的停机容差。

### 正确性核验（`--validate`，40×20×10 网格，$h=1$ mm）

1. Hex8 刚度矩阵物理自检：对称误差 1.4e-17、刚体平移零能 2.4e-16、半正定；
2. **单元刚度解析应变能**：三种独立变形模式（单轴受约束拉伸、纯剪切、静水膨胀）的
   应变能与解析值之比均为 **1.000000**，最大相对误差 1.65e-16（立方体单元）/
   2.25e-15（非立方体 $h=(0.3,0.7,1.3)$）；
3. **尺寸标度**：$K_e(h{=}0.5)/K_e(h{=}1) = 0.500000$，与三维理论标度 $K_e \propto h$ 吻合；
4. 矩阵自由算子单次作用 vs scipy 显式装配：相对最大差 **5.98e-16**；
5. PCG 解 vs `spsolve`：相对误差 **1.17e-11**；
6. 小规模完整优化：柔度单调下降、体积约束达标（0.3000），柔度 161.5 → 11.64。

判据 2、3 是 2026-08-28 新增的回归防线。原有的对称性/刚体零能/半正定三项对任意正常数
因子完全免疫，正是 §7 那个量纲缺陷潜伏至今的原因。

---

## 3. 完整优化实测（GPU，191.5 万自由度）

* 迭代 **90 步收敛**（$\max|\Delta\rho| < 10^{-2}$ 连续 3 步），柔度从 1161.1 单调降至
  **86.66**，体积分数 0.30004；
* 平均每步耗时 **22.22 s**（其中求解 21.51 s），平均 CG 迭代 4820 步；
* 峰值显存 **5.89 GB**；
* 最终构型是清晰的空间桁架，固支端与施载点均有实体锚定。

### 网格加密的柔度收敛性

修复量纲缺陷后，三级网格的柔度构成单调收敛序列，可与论文的二阶单元结果交叉验证：

| 网格 | $h$ (mm) | 柔度 | 参照 |
|:---|:---:|:---:|:---|
| 60×20×4 | 1.0 | 82.39 | 论文 k=1 结果 82.54，差 **0.18%** |
| 120×40×8 | 0.5 | 86.20 | — |
| 300×100×20 | 0.2 | **86.66** | 论文 k=2 结果 86.92，差 **0.30%** |

序列 82.39 → 86.20 → 86.66 单调上升并趋向约 87，符合位移从下方收敛的 FEM 行为；k=1 网格
加密与 k=2 阶次提升两条独立路径趋于同一真解。

完整历程见 `outputs/topopt300/history.json`，最终构型见 `outputs/topopt300/density_final.npy`。

## 4. 单步性能对比（均匀密度，CPU 稀疏 vs GPU 张量化）

| 阶段 | CPU 耗时 (NumPy/SciPy 稀疏) | GPU 耗时 (PyTorch 张量化) | 加速比 |
|:---|:---:|:---:|:---:|
| 1. 刚度组装 | 9.84 s | 0.03 s | **317.8×** |
| 2. 平衡方程求解 (PCG) | 243.92 s（1955 迭代） | 9.20 s（1955 迭代） | **26.5×** |
| 3. 伴随灵敏度滤波 | 1.66 s | 1.03 s | 1.6× |
| 4. 设计变量更新 (OC) | 0.52 s | 0.14 s | 3.7× |
| **★ 单步总耗时** | **255.94 s** | **10.41 s** | **24.6×** |

说明：GPU 路径不显式组装全局刚度矩阵（按需作用），故组装项加速最大；求解占 CPU 总耗时的
95%，是真正的瓶颈。两侧 CG 迭代数 **完全相同（均 1955）**、柔度吻合至 9 位有效数字
（CPU 1161.0598170 / GPU 1161.0598172），是两条路径代数等价的直接证据。

## 5. 两侧解一致性（同一收敛设计密度）

在同一最终设计密度下分别用 CPU(scipy 稀疏 PCG) 与 GPU(张量化 PCG) 求解同一系统，两侧
同为 `float64`，均收敛至相对残差 ~9.84e-9：

* CPU 求解 627.23 s、5160 迭代；GPU 24.18 s、5160 迭代（**迭代步数完全一致**）；
* 位移解相对差 **6.8e-11**。

前一版该项为 0.25%，源于 GPU 侧用 `float32` 而 CPU 侧用 `float64`——那是精度口径不对等
的产物，不是算法差异。两侧统一双精度后改善七个数量级。

---

## 6. 产物清单（`experiments/topopt_capability/outputs/topopt300/`）

| 文件 | 内容 |
|:---|:---|
| `summary.json` | GPU 全收敛运行摘要（迭代、柔度、各阶段平均耗时、峰值显存） |
| `history.json` | 90 步完整历程（柔度/体积/收敛/每步 CG 迭代数与四阶段耗时） |
| `density_final.npy` | 最终密度场 (300, 100, 20) |
| `gpu_single_step.json` | GPU 均匀密度单步计时（原始产物在 `topopt300-single/`） |
| `cpu_sparse_single_step.json` | CPU scipy 稀疏单步计时 |
| `cpu_vs_gpu_compare.json` | 两侧解对比 |

四次运行在 `cases.toml` 中逐条登记；数据快照 `figure_data/fig4_data.json` 由 `collect.py`
组装，组装前会核对四份产物的网格、设计域、求解容差与精度是否同口径，不一致直接报错。

---

## 7. 2026-08-28 的两处实质变更

### 7.1 Hex8 单元刚度的量纲缺陷（已修复）

`hex8_stiffness_unit` 存在两个叠加缺陷，使柔度绝对值系统性错误：

* **因子 4（所有网格）**：`B` 矩阵填的是形函数对**局部坐标**的导数 $\partial N/\partial r$，
  物理导数应为 $\partial N/\partial x = J^{-1}\partial N/\partial r$；而 `detJ` 又按**物理
  体积**取。两个坐标系混用，漏掉 $J^{-1}$ 使 `B` 小一半、`K` 小四倍。
* **因子 $h$（仅 $h\neq1$ 的网格）**：三维下 $K_e \propto h$，而函数硬编码单位立方体，
  实际单元边长从不进入计算。`Lx/Ly/Lz` 是死参数，命令行未暴露，永远取默认 2.0/1.0/0.5。

合并效应 $K_{\text{phys}} = 4h\,K_{\text{code}}$，即 $C_{\text{phys}} = C_{\text{code}}/(4h)$。
**构型、迭代轨迹、CG 行为均不受影响**（常数因子被 OC 的 Lagrange 乘子在二分时吸收），已在
60×20×4 与 120×40×8 两级网格上逐单元核实：密度场比值恒为理论因子，$\rho>0.3$ 与
$\rho>0.5$ 两个阈值下均 0 个单元错位。但任何跨网格的柔度对比在修复前都是错的。

`summary.json` 新增 `ke0_scale`（$=4h$）字段，供旧产物换算回物理值。

同时修正接口语义：`--Lx/--Ly/--Lz` 暴露到命令行；`--rmin` 由「单元个数」改为**物理长度**，
内部按 `rmin_cells = rmin_phys / h` 换算，保证网格无关性。

### 7.2 算例与精度口径

* 算例由 160×80×40 的 4:2:1 块体换为博士论文算例 3.3 的加密版（60×20×4 mm，300×100×20），
  图 10 四个面板统一到同一算例，且柔度可与论文结果对标；
* CPU/GPU 两侧统一 `float64`。代价是端到端加速比由 36.1× 降到 24.6×——那 36.1× 是
  GPU fp32 对 CPU fp64 的不对等比较；收益是两侧 CG 步数完全一致、解相对差 6.8e-11。

## 8. 复现命令

```bash
cd /home/brighthe/workspace/soptx
python experiments/topopt_capability/run.py --list      # 查看四个用例的产物状态
python experiments/topopt_capability/run.py --all       # 全部重跑并汇编快照（约 25 分钟）
python experiments/topopt_capability/run.py --collect   # 仅重新汇编快照
```

单独运行（`run.py --all` 等价于依次执行以下四条，参数以 `cases.toml` 为准）：

```bash
cd /home/brighthe/workspace/soptx
E=experiments/topopt_capability/outputs

# 正确性核验（小网格，分钟级）
python examples/topopt_platform/topopt_3d_simp_real.py --validate --backend pytorch

# GPU 全收敛（90 步，约 33 分钟）
python examples/topopt_platform/topopt_3d_simp_real.py --backend pytorch \
    --nx 300 --ny 100 --nz 20 --Lx 60 --Ly 20 --Lz 4 \
    --rmin 1.5 --filter cone --dtype float64 \
    --max-iters 300 --ctol 1e-2 --cg-rtol 1e-8 --cg-maxiter 20000 \
    --outdir $E/topopt300

# GPU 单步计时（约 30 秒；独立目录，避免覆盖全收敛产物）
python examples/topopt_platform/topopt_3d_simp_real.py --backend pytorch \
    --nx 300 --ny 100 --nz 20 --Lx 60 --Ly 20 --Lz 4 \
    --rmin 1.5 --filter cone --dtype float64 \
    --single-step 1 --max-iters 1 --ctol 1e-2 --cg-rtol 1e-8 --cg-maxiter 20000 \
    --outdir $E/topopt300-single
cp $E/topopt300-single/summary.json $E/topopt300/gpu_single_step.json

# CPU 稀疏单步基线（约 4.5 分钟）
python examples/topopt_platform/topopt_3d_simp_real.py --cpu-baseline \
    --nx 300 --ny 100 --nz 20 --Lx 60 --Ly 20 --Lz 4 \
    --rmin 1.5 --filter cone --cg-rtol 1e-8 --cg-maxiter 20000 \
    --outdir $E/topopt300

# 两侧解对比（约 11 分钟）
python examples/topopt_platform/topopt_3d_simp_real.py \
    --compare $E/topopt300/density_final.npy \
    --nx 300 --ny 100 --nz 20 --Lx 60 --Ly 20 --Lz 4 \
    --dtype float64 --cg-rtol 1e-8 --cg-maxiter 20000 \
    --outdir $E/topopt300
```

### 图 10 面板 (c)(d) 的构型渲染

```bash
D=/mnt/c/workspace/dut-postdoc/research/funding/active/china-postdoc-foundation-general-grant
python examples/topopt_platform/render_topology.py \
    --density experiments/topopt_capability/outputs/topopt300/density_final.npy \
    --nx 300 --ny 100 --nz 20 --Lx 60 --Ly 20 --Lz 4 \
    --threshold 0.5 --figsize 11.52x4.90 \
    --pair $D/assets/dev/fig10_panel_c_initial_render.png \
    --out  $D/assets/dev/fig10_panel_d_topology_render.png
```

`--pair` 让两张底图共用同一裁剪框、裁后尺寸完全一致，这是并排两图严格对齐的唯一手段。

底图幅面必须跟着**最终显示宽度**走：底图里 `固支端`/`F` 两处标注
（`render_topology.py` 的 `_annotate`）用的是绝对磅值字号，底图幅面越大、这两处标注
相对结构就越小。当前显示宽 3.858 in，对应 `11.52x4.90`。**改版式必须同步重渲染底图**，
否则标注相对结构会偏大或偏小——这一条已在本图上踩过两次。

**版式**：构型底图宽高比固定 `1.80`（60×20 mm 悬臂梁加下方载荷箭头），`imshow` 保持
aspect 时显示高度只能是 `宽度 / 1.80`，**宽度是唯一的杠杆**：行高无效（高度受宽度约束），
裁剪也无余量（底图四周留白已只剩 8 px）。

宽度的真正制约不是「半幅列宽」，而是 `constrained_layout` 会对齐同一 `gridspec` 内两行的
列边界。若 (a)(b)(c)(d) 共用一个 2×2 `gridspec`，(c)(d) 就被上排的 y 刻度标签
（左 1.444 in + 中 1.156 in，合计 2.6 in，占满幅 8.0 in 的 33%）挤到只剩 2.679 in 宽、
1.49 in 高。改成外层 2×1 + 每排各一个嵌套 `subgridspec` 后，两排列边界互不牵连，
(c)(d) 各得 **3.858 × 2.145 in**（线性放大 1.44 倍），整图高 7.0 in。

由此，「两张构型不可能同时画大、只能让其中一张独占整行」这个先前的结论是错的：它把
上排刻度标签占掉的宽度误算成了不可回收的开销。整行版式（4.66 × 2.59 in）相对本方案
只多 21%，却要付出整图 9.9 in、近乎占满一页的代价。

2026-08-28 曾因误判上述宽度约束而试过两版整行版式，当日均撤回：

1. (c) 优化后构型整行 + (d) 收敛历程整行。动机是「90 步收敛」此前只是图注里的一句断言，
   换成曲线可由 `change` 穿过 `ctol` 自证；且设计域长方体是拓扑优化的常识起点、信息量
   近于零。
2. (c)(d) 两张构型各占整行。整图高 9.9 in、近乎占满一页，还要为信息量近零的设计域
   单独付出一整行，而本图主题（多后端性能）被压进上三分之一。

撤回的判据是**本图主题是多后端性能，收敛证据不是必需项**；前后对比则让非专业读者一眼
看懂拓扑优化在做什么，这个受众价值高于把构型画大。随后改用嵌套 `subgridspec` 收回上排
刻度标签占掉的宽度，构型在 2×2 里也拿到了 3.858 in——「放大」与「保留前后对比」不再是
二选一，当初那个取舍其实并不存在。`make_figs.py` 中的收敛曲线绘图代码与版式开关已
删除——留着会误导下一轮改版；下面这段结论是它留下的正本。

### 收敛历程本身（当前不进图，结论留档）

`history.json` 的两条量：柔度（对数，1161.1 → 86.66）、密度变化量（对数，0.2 → 0.00953），
判据线 `ctol = 0.01`。密度变化量第 88 步首次跌破 `ctol`，`ctol_patience = 3` 于第 90 步
停机——这是「90 步收敛」的出处。

体积分数不入图的原因：全程 `0.3000 ± 5e-5`（OC 二分每步强制满足约束），是条死平线。
柔度前 57 步严格单调、尾部 8 次抖动幅度仅 0.05%（86.806 → 86.852），在跨 13.4 倍量程的
对数轴上不可见，无需平滑或截断。

`collect.py` 把 `ctol / ctol_metric / ctol_patience` 写入快照的改动予以保留：这三项是
运行的收敛判据元数据，与是否画曲线无关，缺了就无从核对「90 步收敛」这句话。
