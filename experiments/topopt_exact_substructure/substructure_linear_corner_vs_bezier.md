# 子结构 linear_corner 与三次 Bézier 接口的拓扑优化对照

> 状态：设计稿。§1、§2 先于实现起草；`bezier_cubic` 迹尚未实现，文中的选项名、类名与命令均为拟定，§3 结果待运行。

硬件与软件环境同 [substructure_full_trace_vs_linear_corner.md](substructure_full_trace_vs_linear_corner.md)，实际运行时以各输出目录的 `config.json` 为准。

## 1. 研究对象

对照相同细网格、相同子结构划分下，子结构边界位移的两种降阶表示 `linear_corner` 与 `bezier_cubic` 对 Huang2023 三维 MBB 梁的分析与优化结果；内存允许的网格上加入 `full_trace` 作为精确参考。`bezier_cubic` 取自 Guo 等 2026（CMAME，三次 Bézier 边界插值）§2.3：三维子结构的每个面用双三次 Bézier 曲面参数化边界位移。

沿用 [substructure_full_trace_vs_linear_corner.md](substructure_full_trace_vs_linear_corner.md) §1.1 的记号：子结构边界位移写为 $u_b^j=\Psi q^j$，内部位移由内部平衡方程精确回代，三种接口只差在 $\Psi$。

| 项目 | full_trace | linear_corner | bezier_cubic |
|---|---|---|---|
| 子结构边界位移 | $u_b^j=q^j$ | $u_b^j=Lq^j$，8 个角点三线性插值 | $u_b^j=Sq^j$，56 个控制点的分片双三次 Bézier 插值 |
| 子结构参数 $q^j$ 的维数 | $n_b$（$m=5$ 时 456） | 24 | 168 |
| 局部缩聚刚度 | $K_s^j$ | $L^{\mathsf T}K_s^jL\in\mathbb R^{24\times24}$ | $S^{\mathsf T}K_s^jS\in\mathbb R^{168\times168}$ |
| 全局未知量 | 边界位移 $U_\Gamma$ | 角点位移 $U_C$ | 控制点位移 $U_B$ |

### 1.1 三次 Bézier 迹空间

三次 Bernstein 基为

$$
b_s(t)=\binom{3}{s}(1-t)^{3-s}t^s,\qquad s=0,1,2,3,\quad t\in[0,1].
$$

子结构取局部坐标 $\xi=(\xi_1,\xi_2,\xi_3)\in[0,1]^3$，细网格节点位于 $\xi_k\in\{0,1/m,\dots,1\}$。在 $4\times4\times4$ 控制点格上取三个方向三次 Bernstein 基的张量积 $b_a(\xi_1)b_b(\xi_2)b_c(\xi_3)$。由于 $b_1,b_2$ 在 $t=0,1$ 处为零，格内部的 8 个点对应的函数在子结构表面恒为零，其余 56 个点对应的函数限制到表面上，在每个面上正好退化为该面的双三次 Bézier 曲面（例如 $\xi_3=0$ 面上为 $\sum_{a,b}b_a(\xi_1)b_b(\xi_2)P_{ab0}$）。于是

$$
u_b^j(\xi)=\sum_{(a,b,c)\in\mathcal B}b_a(\xi_1)\,b_b(\xi_2)\,b_c(\xi_3)\,P_{abc}^j,\qquad
\mathcal B=\{0,1,2,3\}^3\setminus\{1,2\}^3,
$$

$|\mathcal B|=56$：8 个角点、12 条棱各 2 个棱内控制点、6 个面各 4 个面内控制点。在全部边界节点上取值，得到 $S\in\mathbb R^{n_b\times168}$（每个控制点 3 个位移分量）。

- **单值定义**：上式对每个边界节点只取一次值，棱与角上的节点不会被重复计入。论文式 (19) 写作对 6 个面"散布累加"，按字面对棱、角节点求和会重复计数；本文按上式实现，与论文实现是否一致待确认。
- **`linear_corner` 是同一构造的一次情形**：把三次 Bernstein 换成一次 $\{1-t,t\}$、控制格换成 $2\times2\times2$，就得到现有的三线性插值矩阵 $L$。
- **列满秩条件**：每个面在每个方向上至少要有 4 个节点，即 $m\ge3$。$m=3$ 时每个面的节点数与控制点数相同，$S$ 为方阵且可逆，`bezier_cubic` 与 `full_trace` 张成同一空间；$m=5$ 时把 $n_b=456$ 压到 168。
- **控制点不全是节点值**：Bernstein 基只插值端点，角点控制点等于该角点的节点位移，棱内与面内控制点只是 Bézier 系数，不等于任何节点的位移。

### 1.2 局部缩聚、全局组装与恢复

把 $\Psi=S$ 代入 [对照文档](substructure_full_trace_vs_linear_corner.md) §1.1 的推导即可，计算路径沿用 `linear_corner` 的直接缩聚（README"linear_corner 直接缩聚与恢复缓存"一节），不形成完整 Schur 补：

$$
B^j=K_{ib}^jS,\qquad X^j=(K_{ii}^j)^{-1}B^j,\qquad
K_B^j=S^{\mathsf T}K_{bb}^jS-(B^j)^{\mathsf T}X^j=S^{\mathsf T}K_s^jS\in\mathbb R^{168\times168},
$$

恢复矩阵 $N_i^j=-X^j\in\mathbb R^{n_i\times168}$，装配时缓存，求解后 $u_i^j=N_i^jq^j$。局部求解的右端项为 168 个，介于 `linear_corner` 的 24 个与 `full_trace` 的 456 个之间。

**全局编号**。$n_x\times n_y\times n_z$ 个子结构的控制点落在 $(3n_x+1)\times(3n_y+1)\times(3n_z+1)$ 的格上，去掉各子结构内部的 8 个点。相邻子结构共享公共面上的 16 个控制点（共享棱上为 4 个），$q^j=A_B^jU_B$，$A_B^j$ 为布尔限制矩阵：

$$
K_B=\sum_j(A_B^j)^{\mathsf T}K_B^jA_B^j,\qquad F_B=\sum_j(A_B^j)^{\mathsf T}S^{\mathsf T}f_b^j .
$$

公共面两侧的双三次函数由同一组控制点决定，取值相同，界面位移 $C^0$ 协调。记全局插值 $U_\Gamma=P_BU_B$，协调性等价于 $A_b^jP_B=SA_B^j$，由此 $K_B=P_B^{\mathsf T}K_\Gamma P_B$、$F_B=P_B^{\mathsf T}F_\Gamma$。

**载荷与支承**。顶面集中力在细网格边界上，按 $F_B=P_B^{\mathsf T}F_\Gamma$ 投影到控制点，与 `linear_corner` 的 $F_C=P^{\mathsf T}F_\Gamma$ 同一做法。支承在梁两端底边的四个角点，这些点是子结构角点，对应的控制点就是节点位移，约束直接作用在 $U_B$ 的对应分量上。

### 1.3 嵌套关系与误差度量

**嵌套**。三次 Bernstein 基能精确表示一次函数（升阶：$(p_0,p_1)\mapsto(p_0,\tfrac{2p_0+p_1}{3},\tfrac{p_0+2p_1}{3},p_1)$），张量积后三线性函数也在 $S$ 的列空间内。记升阶矩阵为 $E\in\mathbb R^{168\times24}$，有 $L=SE$，于是

$$
\operatorname{range}L\subset\operatorname{range}S\subset\mathbb R^{n_b},\qquad
V_C\subset V_B\subset V_\Gamma .
$$

**同一密度场下的柔顺度序**。三者都是同一细网格问题在嵌套子空间上的 Galerkin 投影。对同一物理密度场、相同材料、载荷与齐次支承，且线性系统精确求解：

$$
C_C\le C_B\le C_\Gamma,\qquad
C_\Gamma-C_B=\|U^\Gamma-U^B\|_K^2,\qquad
\|U^\Gamma-U^C\|_K^2=\|U^\Gamma-U^B\|_K^2+\|U^B-U^C\|_K^2 .
$$

相对误差沿用对照文档的定义，另加一个"补回份额"，表示 `bezier_cubic` 补回了 `linear_corner` 能量误差的多少：

$$
\varepsilon_C^B=\frac{C_\Gamma-C_B}{C_\Gamma},\qquad
\varepsilon_E^B=\sqrt{\varepsilon_C^B},\qquad
r=\frac{C_B-C_C}{C_\Gamma-C_C}\in[0,1].
$$

以上只对同一密度场成立。完整优化中三条轨迹的密度各不相同，最终柔顺度不保证满足此序，须用同一密度场的重分析单独对照。

**结果与基的选法无关**。Galerkin 解只取决于 $\operatorname{range}S$。换成同一空间的任一组基（如节点 $0,1/3,2/3,1$ 上的三次 Lagrange 基），$U^B$、$C_B$ 与灵敏度都不变。因此 §3 中柔顺度、位移与拓扑的差异应归因于迹空间（分片双三次对三线性），而不是 Bézier 基本身。基的选法只影响：$K_B$ 的条件数，从而影响 Jacobi-CG 步数；$q^j$ 的含义（控制点系数还是节点值）；PIML 预测目标的形式。

**误差阶（待确认）**。对照文档 §1.3 给出 `linear_corner` 的误差对 $H$ 为一阶，依据是 $Q_1$ 粗网格的逼近性。换成三次迹后，$U^\Gamma$ 在子结构内离散调和，但密度场不光滑，其迹的正则性有限，不能据此推出更高的阶。$m=5$ 时每条棱 6 个节点、4 个控制点，压缩程度不大，改进幅度以 §3 实测为准。

### 1.4 规模预估

全局维数与非零元按控制点的拓扑计数：两个控制点同属至少一个子结构即耦合，每对耦合贡献 $3\times3$ 块，按完整对称模式计数，未扣除支承。同一方法得到的 `linear_corner` 数值与对照文档 §3.1 完全一致。恢复缓存为 $n_{\text{sub}}\times n_i\times(\text{参数维数})$ 个 float64，$n_i=192$（$m=5$）。

| 网格 | 子结构 | `linear_corner` 维数 / 非零元 | `bezier_cubic` 维数 / 非零元 | `full_trace` 维数 / 非零元 | 恢复缓存 `linear_corner` / `bezier_cubic` |
|---|---|---|---|---|---|
| $210\times35\times35$ | $42\times7\times7$ | $8\,256$ / $5.53\times10^5$ | $135\,012$ / $4.60\times10^7$ | $425\,232$ / $3.65\times10^8$ | 0.071 / 0.495 GiB |
| $240\times40\times40$ | $48\times8\times8$ | $11\,907$ / $8.16\times10^5$ | $198\,147$ / $6.85\times10^7$ | $625\,539$ / $5.43\times10^8$ | 0.105 / 0.738 GiB |
| $390\times65\times65$ | $78\times13\times13$ | $46\,452$ / $3.38\times10^6$ | $811\,632$ / $2.91\times10^8$ | $2\,578\,644$ / $2.32\times10^9$ | 0.453 / 3.168 GiB |

`full_trace` 列取自对照文档 §3.1。`bezier_cubic` 的全局维数约为 `linear_corner` 的 17 倍、`full_trace` 的 1/3，非零元约为前者的 86 倍、后者的 1/8。390 档 $K_B$ 以 CSR（float64 值 + int32 列号）存储约 3.3 GiB，加上恢复缓存约 3.2 GiB，16 GB 显存能否跑通要实测。

## 2. 实验设置

拟在 [`run_substructure_full_trace_vs_linear_corner.py`](run_substructure_full_trace_vs_linear_corner.py) 的 `--trace` 中增加 `bezier_cubic`，不另写入口；输出目录命名、线程设置、内存上限与后端规则沿用该入口。

### 2.1 模型参数

与 [对照文档](substructure_full_trace_vs_linear_corner.md) §2.1 相同：`FullMBBBeamPoint3d`，$6:1:1$，$h=1$，顶面中心单位集中力，两端底边四个角点支承，$E_0=1$，$E_{\min}=10^{-7}$，$\nu=0.3$。

### 2.2 分析参数

有限元离散与线性求解参数同对照文档 §2.2，以下只列不同之处。

| 参数 | 数值或设置 |
|---|---|
| 边界位移表示 | 新增 `bezier_cubic`，$\Psi=S$：`SubstructureAnalyzer(..., trace='bezier_cubic')`（拟定）；迹基拟实现为 `soptx.fem.substructure.traces.BezierCubicTraceBasis`，接入现有 `TraceBasis` 契约 |
| 子结构细分数 | 默认 $m=5$；`bezier_cubic` 要求 $m\ge3$，否则 $S$ 不是列满秩 |
| 预条件子 | `none` 或 `jacobi`；`gmg` 仍只支持 `full_trace` |
| 迭代初值 | 首轮为零向量；第 2 轮起取上一轮的全局控制点位移 $U_B$ |

### 2.3 优化参数

与对照文档 §2.3 相同：最小柔顺度，体积分数 $0.12$，修正 SIMP（$q=3$），密度过滤半径 $3h$，OC（移动限 0.2、阻尼 0.5），连续 5 轮柔顺度相对变化小于 $2\times10^{-4}$ 停止。

### 2.4 对照方案

| 编号 | 内容 | 网格 | 接口 |
|---|---|---|---|
| A | 同一密度场重分析：取 `linear_corner` 运行的若干轮物理密度（含第 0 轮均匀密度与末轮），三种接口各分析一次，计算 $C_\Gamma,C_B,C_C$、$\varepsilon_E^B$ 与 $r$ | 210 | 三种 |
| B | 内存占用：各接口跑 3 轮，按对照文档 §3.1 的口径记录 GPU 与 CPU 各阶段峰值 | 240、390 | `bezier_cubic`（另两种已有） |
| C | 完整优化：终止状态、轮数、最终柔顺度与体积分数、CG 步数、每轮各阶段耗时、峰值内存、最终拓扑 | 210、390 | `linear_corner` 与 `bezier_cubic`；`full_trace` 是否在 210 档跑完整优化待定 |

A 拟由 [`reanalyze_full_trace_vs_linear_corner.py`](reanalyze_full_trace_vs_linear_corner.py) 扩展。B、C 的拟定命令（`--trace bezier_cubic` 尚不可用）：

```bash
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace bezier_cubic --grid 240 40 40 --max-iter 3 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace bezier_cubic --grid 390 65 65 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
```

### 2.5 实现后先过的单元测试

运行实验前，`BezierCubicTraceBasis` 与 `bezier_cubic` 分析路径须先通过以下检查：

1. **重现角点插值**：$SE=L$，误差在舍入量级。
2. **刚体运动**：平移与线性转动场经 $S$ 精确重现（由第 1 条推出，仍单独测）。
3. **$m=3$ 退化**：`bezier_cubic` 与 `full_trace` 的位移、柔顺度与灵敏度一致。
4. **界面协调**：$A_b^jP_B=SA_B^j$ 对全部子结构成立。
5. **柔顺度序与勾股关系**：随机密度下 $C_C\le C_B\le C_\Gamma$，且 §1.3 的能量分解成立到求解容差。
6. **直接缩聚**：$S^{\mathsf T}K_{bb}S-B^{\mathsf T}X$ 与先形成 $K_s^j$ 再投影的结果一致。

## 3. 结果对照

待运行。

### 3.1 同一密度场重分析（方案 A）

待运行。

### 3.2 内存占用（方案 B）

待运行。

### 3.3 完整优化结果（方案 C）

待运行。
