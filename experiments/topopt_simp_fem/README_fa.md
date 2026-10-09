# 传统有限元 + FA 拓扑优化

## MBB 梁算例

采用 Huang2023 [1] 的三维 MBB 梁算例，在体积分数约束下最小化结构柔顺度。所有参数均为无量纲量。

### 模型参数

![MBB 梁的几何、载荷与支承](../topopt_piml_substructure/assets/Huang2023_Fig5.png)

梁的长、高、宽之比为 $6:1:1$，以细单元边长为长度单位（$h=1$），计算域为 $[0,n_x]\times[0,n_y]\times[0,n_z]$，首档为 $390\times65\times65$。使用整体模型，不利用对称性缩减计算域。顶面中心施加竖直向下的单位集中力 $F=1$。支承按图 5 取在两端底边的四个角点：左端两角点铰支（$u_x=u_y=u_z=0$），右端两角点滚支（$u_y=0$）。

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

### FA 与线性求解

| 参数 | 数值或设置 |
|---|---|
| 刚度算子 | FA：全局 CSR 稀疏矩阵；结构化网格上只存一份实体参考单元刚度 $K^0$，乘逐单元系数 $E(\tilde\rho_e)/E_0$ 装配，不逐单元重新积分 |
| 边界条件 | 保结构对称消元：受约束行列置零、对角置 $1$，沿用原 CSR 骨架 |
| 线性求解器 | 默认几何多重网格预条件共轭梯度法（MGCG）；可选 Jacobi 预条件，或直接法 MUMPS / SciPy（仅适合小网格） |
| 多重网格 | V 循环；三线性延拓，粗网格各向单元数取 $\lceil n/2\rceil$；Galerkin 粗层算子（第 2 层按单元组合 $\sum_c s_cP_c^{\mathsf T}K^0P_c$）；加权 Jacobi 前后各 1 步，$\omega=4/(3\cdot1.1\,\hat\lambda)$；最粗层不超过 $20000$ 个自由度，直接分解 |
| 停机准则（PCG） | $\lVert b-Au\rVert_2/\lVert b\rVert_2\le10^{-6}$ |
| 初值（PCG） | 上一轮位移热启动，首轮取零 |
| 最大迭代数（PCG） | $20000$，逐轮记录实际迭代步数 |

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
| 集中力分配 | 顶面 $x=n_x/2$ 处，$z$ 向距中心最近的 2 个节点各施加 $-1/2$ |
| 设计对称约束 | 关于 $z$ 中面对称：每轮把过滤后的灵敏度与其 $z$ 向镜像取平均，即对称设计子空间上的梯度；$x$ 向两端支承不同，不约束；把舍入级不对称压为零，使 `design_symmetry_error_z` 恒为 $0$ |
| 初始密度 | 均匀取 $0.12$ |
| 材料插值 | 修正 SIMP：$E(\rho)=E_{\min}+\rho^{q}(E_0-E_{\min})$，$\rho\in[0,1]$ |
| SIMP 惩罚指数 $q$ | $3$（区别于位移插值次数 $p$） |
| 过滤 | 密度过滤：$\tilde\rho_e=\sum_i H_{ei}\rho_i/\sum_i H_{ei}$，$H_{ei}=\max(0,\ r_{\min}-d_{ei})$，$d_{ei}$ 为单元形心距离；物理密度取 $\tilde\rho$，灵敏度按链式法则过滤 |
| OC 更新 | 移动限 $0.2$，阻尼指数 $0.5$；二分法求 Lagrange 乘子，初始区间 $[0,10^9]$，相对精度 $10^{-3}$ |
| 停止条件公式 | $r_k=\lvert c_k-c_{k-1}\rvert/c_k$，$r_{k-4},\dots,r_k$ 均小于 $2\times10^{-4}$ 时停止 |
| 最大迭代数 | $300$ |

## ParaView 帧

每轮帧 `iterations/*.vtu` 与 `evolution.pvd` 体积大，运行结束后迁移到 Windows 本地目录查看，WSL 的结果目录只保留 json、npy 与 `result_final.vtu`，迁移位置记在 `visualization_location.json`：

```bash
python -m soptx.postprocess.sync_visualization --source-dir outputs/fa_hex_390x65x65_cg --destination-dir /mnt/c/workspace/soptx-results/topopt_simp_fem/fa_hex_390x65x65_cg --move
```

## 参考文献

[1] Huang2023，第 4 节公共设置与第 4.1 节 MBB 梁算例，图 5–7。
