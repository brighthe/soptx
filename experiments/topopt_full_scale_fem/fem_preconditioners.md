# 传统有限元 CG 预条件子有效性对照

硬件：AMD Ryzen 9 9950X，运行内存 45 GiB（WSL2）。软件：Ubuntu 24.04（WSL2），`ihpcm` 环境，Python 3.12.13，NumPy 2.5.3，SciPy 1.18.1。

## 1. 实验设置

脚本 [`run_fem_preconditioners.py`](run_fem_preconditioners.py)：在 Huang2023 三维 MBB 梁（`FullMBBBeamPoint3d`）的固定密度场上，从基准网格 $48\times8\times8$ 逐档加密（物理域不变，第 $k$ 档 $h=2^{-k}$），每档以所选预条件子求解一次线性系统，比较 CG 步数随 $h$ 的变化。斜率指 $\log(\text{niter})$ 对 $\log h$ 的最小二乘斜率：−1 为 $O(h^{-1})$ 的未改善行为，0 为与网格无关。

| 参数 | 设置 |
|---|---|
| 离散 | Q1 六面体，二阶高斯积分，FA 全局 CSR 矩阵：`OPERATOR_LEVELS = ('fa', )` |
| 材料 | $E_0=1$，$E_{\min}=10^{-7}$，$\nu=0.3$，修正 SIMP，$q=3$ |
| 停机 | $\lVert b-Ku\rVert_2\le10^{-6}\lVert b\rVert_2$，零初值，最多 20000 步 |
| 预条件子 | `none`；`jacobi`：$\operatorname{diag}(K)^{-1}$；`gmg`：几何多重网格 V 循环（加权 Jacobi 光滑前后各 1 步，最粗层直接分解） |
| 密度 | [`fem_preconditioners/`](fem_preconditioners/) 下由 `make_densities.py` 合成：`uniform` 处处 0.12；`binary` 为 0/1 桁架；`gray` 为 `binary` 经半径 3 单元密度过滤 |
| 环境 | NumPy / CPU，单线程 |

## 2. 结果

### 2.1 均匀密度

| 网格 | none | jacobi | gmg | none 求解耗时 | jacobi 求解耗时 | gmg 求解耗时 |
|---|---|---|---|---|---|---|
| $48\times8\times8$ | 330 | 286 | 12 | 0.10 s | 0.13 s | 0.07 s |
| $96\times16\times16$ | 629 | 581 | 12 | 1.76 s | 1.61 s | 0.65 s |
| $192\times32\times32$ | 1222 | 1177 | 12 | 25.07 s | 25.13 s | 2.54 s |
| 斜率 | −0.944 | −1.021 | 0.000 | | | |

**耗时分解（s）**：

$$
T_{\text{其他}} = T_{\text{求解}} - T_{\text{构造}} - N_{\text{iter}}\times\left(T_{\text{matvec}} + T_{\text{作用}}\right)
$$

其中 $T_{\text{求解}}$ 为实测求解耗时；$T_{\text{构造}}$ 为求解后另构造一次预条件子的耗时（GMG 层次结构已缓存，即优化中每轮重建的成本）；$T_{\text{matvec}}$ 与 $T_{\text{作用}}$ 为 CG 内部 `A @ p`、`M @ r` 单次耗时的平均；$N_{\text{iter}}$ 为 CG 步数；$T_{\text{其他}}$ 为差值，含向量运算（每步约 3 次内积与 3 次向量更新）、GMG 首次建立层次结构与调用开销。

各预条件子的构造与作用：

| 预条件子 | 构造 | 作用 $z=M^{-1}r$ |
|---|---|---|
| none | 无 | $z=r$ |
| jacobi | 取 $\operatorname{diag}(K)$ 并求倒数 | 逐元素乘法 $z_i=r_i/d_i$ |
| gmg | 重建各粗层算子，用 15 次幂迭代估计各层光滑子的阻尼系数 $\omega$，分解最粗层 | 一次 V 循环 |

拓扑优化每轮密度改变，两者每轮都要重新构造。

| 预条件子 | 网格 | $N_{\text{iter}}$ | $T_{\text{构造}}$ | $T_{\text{matvec}}$ | $T_{\text{作用}}$ | $N_{\text{iter}}(T_{\text{matvec}}+T_{\text{作用}})$ | $T_{\text{其他}}$ | $T_{\text{求解}}$ |
|---|---|---|---|---|---|---|---|---|
| none | $96\times16\times16$ | 629 | 0 | $2.31\times10^{-3}$ | 0 | 1.45 | 0.31 | 1.76 |
| none | $192\times32\times32$ | 1222 | 0 | $1.72\times10^{-2}$ | 0 | 20.96 | 4.11 | 25.07 |
| jacobi | $96\times16\times16$ | 581 | 0.009 | $2.26\times10^{-3}$ | $1.72\times10^{-5}$ | 1.32 | 0.28 | 1.61 |
| jacobi | $192\times32\times32$ | 1177 | 0.059 | $1.78\times10^{-2}$ | $1.22\times10^{-4}$ | 21.09 | 3.97 | 25.13 |
| gmg | $96\times16\times16$ | 12 | 0.391 | $2.30\times10^{-3}$ | $1.23\times10^{-2}$ | 0.18 | 0.08 | 0.65 |
| gmg | $192\times32\times32$ | 12 | 1.075 | $1.92\times10^{-2}$ | $5.51\times10^{-2}$ | 0.89 | 0.57 | 2.54 |

- none 与 jacobi 的步数均按 $O(h^{-1})$ 增长，jacobi 在常系数下几乎无收益；gmg 步数与网格无关。
- none 与 jacobi 的耗时约 $84\%$ 在迭代 matvec 上，其余 $16\%$ 为向量运算等（每步约 0.2 次 matvec 的耗时）；jacobi 的构造与作用可忽略（作用约为 matvec 的 $0.7\%$）。
- gmg 一次 V 循环约为 2.9 次 matvec（$192\times32\times32$）。步数少后构造成为主要开销：$192\times32\times32$ 上每轮重建 1.08 s，多于 12 步迭代的 0.89 s，占求解的 $42\%$；拓扑优化每轮都要重建，降低构造成本是 gmg 的主要优化方向。
- 同档三者位移最大相对差不超过 $5.8\times10^{-8}$，解的是同一个问题。
- none 另跑了第 3 档 $384\times64\times64$：2423 步，求解 591.01 s，四档斜率 −0.959（`results/preconditioner_none_uniform_48x8x8_L3/`，该次未测耗时分解）。

数据来源：`results/preconditioner_{none,jacobi,gmg}_uniform_48x8x8_L2/` 的 `results.json` 与 `summary.json`，三次在同一终端依次运行。

### 2.2 灰度密度

待测。

### 2.3 0/1 密度

待测。

## 3. 结论

待三组密度完成后填写。
