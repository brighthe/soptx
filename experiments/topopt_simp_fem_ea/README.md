# 传统有限元 + EA 拓扑优化

## MBB 梁算例

采用 Huang2023 [1] 的三维 MBB 梁算例，在体积分数约束下最小化结构柔顺度。所有参数均为无量纲量。

### 模型参数

![MBB 梁的几何、载荷与支承](../topopt_piml_substructure/assets/Huang2023_Fig5.png)

梁的长、高、宽之比为 $6:1:1$，以细单元边长为长度单位（$h=1$），计算域为 $[0,n_x]\times[0,n_y]\times[0,n_z]$，首档为 $390\times65\times65$。使用整体模型，不利用对称性缩减计算域。顶面中心施加竖直向下的单位集中力 $F=1$。两端支承在侧视图中为一端铰支、另一端滚支。

长度单位的依据：文献 4.2 节粗网格 $60\times20\times40$、$m=10$，图 8 标注的尺寸正是细网格 $600\times200\times400$；过滤半径也以细单元尺寸计。若取 $6\times1\times1$ 的实际长度，实心梁柔顺度约为 $PL^3/(48EI)=54$，已大于文献体积分数 0.12 下的 $C_{\mathrm{Fine}}=9.7061$，不可能成立；取 $h=1$ 时实心梁约为 $0.83$，量级相符。图 5 的 $6$、$1$ 因而理解为长宽比。

| 材料参数 | 数值 |
|---|---|
| 实体材料杨氏模量 $E_0$ | $1$ |
| 弱材料杨氏模量 $E_{\min}$ | $10^{-7}$ |
| 泊松比 $\nu$ | $0.3$ |

### 有限元离散

| 参数 | 数值或设置 |
|---|---|
| 有限元 | 八节点六面体单元（Q1） |
| 位移插值次数 $p$ | $1$ |
| 数值积分 | 二阶高斯积分 |
| 分析网格 | $390\times65\times65$ |

### EA 与线性求解

| 参数 | 数值或设置 |
|---|---|
| 刚度算子 | EA：结构化网格上唯一一份参考单元刚度 $K^0$，乘逐单元系数 $E(\tilde\rho_e)/E_0$ |
| 边界条件 | 对称消元算子 $\Pi_IK\Pi_I+\Pi_D$，不形成矩阵 |
| 线性求解器 | 默认几何多重网格预条件共轭梯度法（MGCG），可选 Jacobi 预条件；没有全局矩阵，不能用直接法 |
| 多重网格 | 与 FA 版本相同，见 [`topopt_simp_fem_fa`](../topopt_simp_fem_fa/README.md)；最细层为 EA 算子 |
| 停机准则 | $\lVert b-Au\rVert_2/\lVert b\rVert_2\le10^{-6}$ |
| 初值 | 上一轮位移热启动，首轮取 Dirichlet 基准向量（本例为零） |
| 最大迭代数 | $20000$，逐轮记录实际迭代步数 |

停机容差取 $10^{-6}$ 的依据见 FA 版本 [`topopt_simp_fem_fa`](../topopt_simp_fem_fa/README.md)。

### 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度 |
| 体积分数 | $0.12$ |
| 体积约束 | $\frac{1}{N_F}\sum_e\tilde\rho_e=0.12$，作用于物理密度 $\tilde\rho$ |
| 过滤半径 | $r_{\min}=3$，即 $3$ 个细单元边长 |
| 设计变量更新 | OC |
| 停止条件 | 连续最后 5 次迭代的目标函数相对变化小于 $0.0002$ |

### 补定参数（文献未给出）

| 参数 | 补定值 |
|---|---|
| 支承 | 两端底边的两个角点（$z=0$ 与 $z=n_z$）：左端两角点 $u_x=u_y=u_z=0$，右端两角点 $u_y=0$（`end_corners`） |
| 集中力分配 | 顶面 $x=n_x/2$ 处，$z$ 向距中心最近的 2 个节点各施加 $-1/2$ |
| 设计对称约束 | 关于 $z$ 中面对称：每轮把过滤后的灵敏度与其 $z$ 向镜像取平均，即对称设计子空间上的梯度；$x$ 向两端支承不同，不约束 |
| 初始密度 | 均匀取 $0.12$ |
| 材料插值 | 修正 SIMP：$E(\rho)=E_{\min}+\rho^{q}(E_0-E_{\min})$，$\rho\in[0,1]$ |
| SIMP 惩罚指数 $q$ | $3$（区别于位移插值次数 $p$） |
| 过滤 | 密度过滤：$\tilde\rho_e=\sum_i H_{ei}\rho_i/\sum_i H_{ei}$，$H_{ei}=\max(0,\ r_{\min}-d_{ei})$，$d_{ei}$ 为单元形心距离；物理密度取 $\tilde\rho$，灵敏度按链式法则过滤 |
| OC 更新 | 移动限 $0.2$，阻尼指数 $0.5$；二分法求 Lagrange 乘子，初始区间 $[0,10^9]$，相对精度 $10^{-3}$ |
| 停止条件公式 | $r_k=\lvert c_k-c_{k-1}\rvert/c_k$，$r_{k-4},\dots,r_k$ 均小于 $2\times10^{-4}$ 时停止 |
| 最大迭代数 | $300$ |

支承取角点的依据亦见 FA 版本。$z$ 向对称约束的依据见 FA 版本 [`topopt_simp_fem_fa`](../topopt_simp_fem_fa/README.md)。

## 结果目录与 ParaView 帧

`run_ea.py` 的结果写到 `outputs/{装配层级}_{网格}_{nx}x{ny}x{nz}_cg/`（如 `outputs/ea_hex_390x65x65_cg/`），是计算与核查的依据：`config.json`、逐轮记录 `history.json`、`summary.json`、最终设计与位移的 `*.npy`，以及最后一帧 `result_final.vtu`。每轮分析后另写一帧 `iterations/iter_XXXX.vtu`（字段由 `--vtu-fields` 选择，默认 `density`），并更新 `evolution.pvd`。

每帧体积大（首档约 120 MB），ParaView 跨文件系统读取 WSL 很慢，运行结束后迁移到 Windows 本地目录 `C:\workspace\soptx-results\topopt_simp_fem_ea\<运行>\` 查看：

```bash
python -m soptx.postprocess.sync_visualization --source-dir outputs/ea_hex_390x65x65_cg --destination-dir /mnt/c/workspace/soptx-results/topopt_simp_fem_ea/ea_hex_390x65x65_cg --move
```

逐帧核对 SHA-256 后才删除 WSL 中的 `iterations/` 与 `evolution.pvd`，迁移位置记在结果目录的 `visualization_location.json`；`history.json` 的 `vtu_file` 相对于 Windows 目录读取。运行途中想先看已有的帧，可去掉 `--move` 做一次增量同步，计算进程不受影响。

## 参考文献

[1] Huang2023，第 4 节公共设置与第 4.1 节 MBB 梁算例，图 5–7。
