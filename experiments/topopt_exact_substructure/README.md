# 精确子结构拓扑优化

## MBB 梁算例

采用 Huang2023 [1] 的三维 MBB 梁算例，在体积分数约束下最小化结构柔顺度。所有参数均为无量纲量。

### 模型参数

![MBB 梁的几何、载荷与支承](../topopt_piml_substructure/assets/Huang2023_Fig5.png)

梁的长、高、宽为 $6\times1\times1$，使用整体模型，不利用对称性缩减计算域。顶面中心施加竖直向下的单位集中力 $F=1$。两端支承在侧视图中为一端铰支、另一端滚支。

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

### 子结构参数

| 项目 | 数值或设置 |
|---|---|
| 接口类型 | `full_trace`、`linear_corner` |
| 子结构细分参数 $m$ | $5$，即每个子结构包含 $5\times5\times5$ 个细单元 |
| 子结构网格规模 | $78\times13\times13$ |
| 全局细网格规模 | $390\times65\times65$ |
| 细单元边长 | $h=1/65$，由几何尺寸与网格划分得到 |

两种接口共用同一内部消元流程，以迹矩阵 $\boldsymbol\Psi$ 表示边界位移：

| 接口类型 | 迹矩阵 $\boldsymbol\Psi$ | 局部接口未知量 $\boldsymbol q_j$ |
|---|---|---|
| `full_trace` | $\boldsymbol I$ | 完整边界位移 |
| `linear_corner` | $\boldsymbol L$ | 角点位移 |

其中，$\boldsymbol L$ 为由角点位移插值得到边界位移的矩阵，两种接口均满足 $\boldsymbol u_{jb}^h=\boldsymbol\Psi\boldsymbol q_j$。

在子结构内部无载荷时，求解内部位移映射：

$$
\boldsymbol K_{jii}^h\boldsymbol T_j
=-\boldsymbol K_{jib}^h\boldsymbol\Psi.
$$

$i$ 和 $b$ 分别表示内部和边界自由度。按边界和内部排列全体自由度，对应的形函数矩阵为：

$$
\boldsymbol N_j^{(\Psi)}=
\begin{bmatrix}
\boldsymbol\Psi\\
\boldsymbol T_j
\end{bmatrix},
\qquad
\begin{bmatrix}
\boldsymbol u_{jb}^h\\
\boldsymbol u_{ji}^h
\end{bmatrix}
=\boldsymbol N_j^{(\Psi)}\boldsymbol q_j.
$$

利用线弹性刚度矩阵的对称性，复用 $\boldsymbol T_j$ 计算局部接口刚度：

$$
\begin{aligned}
\boldsymbol K_{r,j}
&=(\boldsymbol N_j^{(\Psi)})^{\mathrm T}
\boldsymbol K_j^h\boldsymbol N_j^{(\Psi)}\\
&=\boldsymbol\Psi^{\mathrm T}\boldsymbol K_{jbb}^h\boldsymbol\Psi
+(\boldsymbol K_{jib}^h\boldsymbol\Psi)^{\mathrm T}\boldsymbol T_j.
\end{aligned}
$$

其中，$\boldsymbol K_j^h$ 为子结构细网格刚度矩阵。两种接口均通过一次内部线性方程求解获得 $\boldsymbol T_j$，不显式求逆。

### 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度 |
| 体积分数 | $0.12$ |
| 过滤半径 | $3$ 个细单元尺寸；对应实际长度 $r_{\min}=3h=3/65$ |
| 设计变量更新 | OC |
| 停止条件 | 连续最后 5 次迭代的目标函数相对变化小于 $0.0002$ |

### 补定参数（文献未给出）

| 参数 | 补定值 |
|---|---|
| 支承（待确认的建模选择） | 左端底边 $x=0,\ y=0$ 整条线：$u_x=u_y=u_z=0$；右端底边 $x=6,\ y=0$ 整条线：$u_y=0$ |
| 集中力分配 | 顶面 $x=3$ 处，$z$ 向距中心最近的 2 个节点各施加 $-1/2$ |
| 初始密度 | 均匀取 $0.12$ |
| 材料插值 | 修正 SIMP：$E_e=E_{\min}+\tilde\rho_e^{\,q}(E_0-E_{\min})$；设计密度 $\rho_e\in[0,1]$，物理密度为过滤后的 $\tilde\rho_e$ |
| SIMP 惩罚指数 $q$ | $3$（区别于位移插值次数 $p$） |
| 过滤 | 密度过滤：$\tilde\rho_e=\sum_i H_{ei}\rho_i/\sum_i H_{ei}$，$H_{ei}=\max(0,\ r_{\min}-d_{ei})$，$d_{ei}$ 为单元形心距离；物理密度取 $\tilde\rho$，灵敏度按链式法则过滤 |
| 体积约束 | $\sum_e v_e\tilde\rho_e/\sum_e v_e\leq0.12$，$v_e$ 为细单元体积；OC 体积判断使用物理密度，体积灵敏度对设计密度按链式法则计算 |
| OC 更新 | 移动限 $0.2$，阻尼指数 $0.5$；二分法求 Lagrange 乘子，初始区间 $[0,10^9]$，相对精度 $10^{-3}$ |
| 停止条件公式 | $r_k=\lvert c_k-c_{k-1}\rvert/c_k$，$r_{k-4},\dots,r_k$ 均小于 $2\times10^{-4}$ 时停止 |
| 最大迭代数 | $300$ |

## 执行入口

`run_linear_corner.py` 执行密度过滤、逐批局部装配与精确内部消元、接口求解、位移恢复、柔顺度及灵敏度计算和 OC 更新。支承固定为 `end_lines`，载荷固定为竖直向下的单位力。端部约束采用上文补定设置，尚未确认与论文完全一致。

在仓库根目录、已安装本地 FEALPy 和 SOPTX 的环境中执行：

```bash
# 论文第一档网格，最多分析 300 次。
python experiments/topopt_exact_substructure/run_linear_corner.py
```

可用 `--n-sub NX NY NZ` 和 `--n-fine M` 设置子结构网格与每方向细分数，默认分别为 `78 13 13` 和 `5`。全局细网格和各方向单元尺寸由几何与上述参数自动推导，并写入 `config.json`。`--n-sub` 各项须为正整数，`--n-fine` 至少为 2。过滤半径取三个 x 方向细单元尺寸；默认立方体单元下即 $3h=3/65$，若使用非等边单元，也按该长度和实际各方向间距计算过滤权重。`--chunk-size`、`--max-iter` 和 `--output-dir` 控制分块、迭代上限和输出目录。接口固定为 `linear_corner`，求解器固定为 `mumps`。

`--backend numpy|pytorch` 统一控制网格、局部装配、接口刚度累加、密度过滤、内部消元、位移恢复和 OC 的张量后端，默认 `numpy`，执行期间不切换回 NumPy。`--device cpu|cuda|cuda:N` 指定密度场、过滤、消元、恢复与 OC 的计算设备，默认 `cpu`。NumPy 仅支持 CPU；PyTorch CUDA 需要可用的设备。

网格、接口编号、局部刚度装配和全局 CSR 累加保留在 CPU，但使用所选后端的张量。PyTorch 模式下，这些阶段仍使用 PyTorch。每批刚度传至计算设备，约化矩阵传回 CPU 后累加；MUMPS／SciPy 接口与文件输出按需转换为 NumPy，求解位移再转回所选后端及设备。实际后端、设备和 CPU 阶段记录在 `config.json`。

密度过滤复用相同物理距离锥形核，NumPy 路径使用 SciPy 卷积，PyTorch 路径在输入 Tensor 的设备上使用三维卷积；伴随过滤保留先归一化再卷积的顺序。

固定工况、OC 选项和收敛判据集中定义在脚本开头；命令行参数的默认值直接定义在 `parse_args` 中。运行配置记录命令行解析后的实际值。启动时先检查 SOPTX 实际使用的 `mumps.DMumpsContext` 和 VTU 导出依赖；导入失败会在创建网格前报错，不自动切换求解器。导入检查不代表 MUMPS 分解和求解已经验证。

每次运行在独立时间戳目录保存 `config.json`、`history.json`、`summary.json`，以及最终设计密度、物理密度和全场位移的 NPY 文件。另输出 `result_final.vtu`，包含单元场 `density`、`design_density` 和节点场 `u_x`、`u_y`、`u_z`、`u_mag`，可直接用 ParaView 查看。VTU 仅在最终分析完成后生成，此时按需构建全局细网格。最终密度与柔顺度对应同一次分析；达到最大次数时不再做未经分析的 OC 更新。运行期间保留装配、接口求解、恢复和迭代结果输出。

本次统一后端改动已完成静态语法与接口核对，尚未执行数值验证、CUDA 运行或 VTU 导出验证。2026-10-05 已在 `ihpcm` 安装 PyMUMPS 与 MUMPS，并确认 `DMumpsContext` 可导入。

待授权后，可在仓库根目录分别执行以下命令，使用默认论文网格完成两次分析及一次 OC 更新：

```bash
python experiments/topopt_exact_substructure/run_linear_corner.py --backend numpy --device cpu --max-iter 2
python experiments/topopt_exact_substructure/run_linear_corner.py --backend pytorch --device cpu --max-iter 2
```

验收时核对两种后端的柔顺度、过滤结果及 OC 更新一致性，检查结果有限性、体积分数约束、接口与能量残差，并确认 VTU 字段与 NPY 对应。CUDA 路径需在上述核对后单独验证。


## 参考文献

[1] Huang2023，第 4 节公共设置与第 4.1 节 MBB 梁算例，图 5–7。
