# PIML 子结构分析

## 1. 研究范围与验证对象

本文整理已有的 PIML 子结构方式，按预测对象、约束处理与刚度构造说明不同方法及其参数化方式，并将接口空间选择与网络配置分别列出。已有实现和实验结果作为对应方式的补充，验证范围包括局部缩聚、接口装配求解和内部位移恢复，不包含优化器密度更新。

### 1.1 已有方法及其构造方式

两条学习路线分别预测形函数与缩聚刚度；同一路线可以采用不同的独立分量表示。

| 方式 | 网络预测对象 | 约束处理 | 局部刚度构造 |
|---|---|---|---|
| 形函数独立分量预测与约束补全 | 内部形函数中去除非独立量后的分量 | 按 $\widehat{\mathbf{B}}_j\mathbf{R}_q^j=\boldsymbol{\Phi}_i^j$ 补全，使内部位移精确再现边界的刚体平移与转动；边界插值部分已知 | 由补全的形函数变分重构：$\widetilde{\mathbf{K}}_r^j=\widehat{\mathbf{H}}_j^{\mathsf{T}}\mathbf{K}^j\widehat{\mathbf{H}}_j$ |
| 刚度独立条目预测与约束补全 | 对称缩聚刚度中去除刚体约束相关非独立量后的条目 | 按 $\widehat{\mathbf{K}}_r^j=(\widehat{\mathbf{K}}_r^j)^{\mathsf{T}}$ 与 $\widehat{\mathbf{K}}_r^j\mathbf{R}_q^j=\mathbf{0}$ 补全对称性及刚体零空间约束 | 直接重构缩聚刚度矩阵 |
| 形函数刚体正交分解 | 变形分量 $\mathbf M_j\in\mathbb R^{n_i\times n_r}$ | 刚体响应解析给出，只学习正交补上的映射 | 由重构的内部延拓构造 $\widehat{\mathbf{H}}_j$，再计算 $\widetilde{\mathbf{K}}_r^j=\widehat{\mathbf{H}}_j^{\mathsf{T}}\mathbf{K}^j\widehat{\mathbf{H}}_j$ |
| 刚度 Cholesky 参数化 | 变形子空间上 Cholesky 因子的下三角独立条目，共 $n_r(n_r+1)/2$ 个 | 正交补保留刚体零空间，因子乘积保证对称半正定 | $\widehat{\mathbf K}_r^j=\mathbf R_\perp^j\mathbf C_j\mathbf C_j^{\mathsf T}(\mathbf R_\perp^j)^{\mathsf T}$ |


表中 $\mathbf{K}^j$ 为子结构细网格刚度，$\widehat{\mathbf{H}}_j=[\widehat{\mathbf{B}}_j;\mathbf{T}_j]$ 为完整位移延拓；$\mathbf{R}_q^j$ 表示接口刚体模态，$\boldsymbol{\Phi}_i^j$ 为对应的内部刚体响应。

**接口空间选择**

接口空间是独立于上述预测与参数化方式的选择。记 $n_i$ 为内部自由度数，$n_q$ 为保留的接口自由度数，$n_r=n_q-d(d+1)/2$ 为去除刚体模态后的维数。

| 接口空间（项目命名） | 保留的自由度 | 接口映射 $\mathbf T_j$ | Huang2023 中的对应做法 |
|---|---|---|---|
| `full_trace` | 全部边界自由度，$n_q=n_b$ | $\mathbf I$ | 完整边界上的子结构缩聚与预测 |
| `linear_corner` | 宏观角点自由度，$n_q=n_c$ | 线性边界插值矩阵 $\mathbf L_j$ | 式 (16) 的边界插值降维 |

### 1.2 本项目网络实现与可配置项

两条路线的网络类 `ShapeFunctionSurrogateNet` 与 `ReducedStiffnessSurrogateNet` 是同一个骨架 `SubstructureSurrogateNet` 的空子类，后者是 `MLP` 的薄包装（[`src/soptx/ml/substructure/nets.py`](../../src/soptx/ml/substructure/nets.py)）。构造只写在骨架里，两个子类不含实现，差别仅在 `output_dim` 的语义：

```python
class SubstructureSurrogateNet(MLP):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        activation: ActivationSpec = nn.SiLU,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
        )


class ReducedStiffnessSurrogateNet(SubstructureSurrogateNet):
    """output_dim 为 n_r * (n_r + 1) // 2."""


class ShapeFunctionSurrogateNet(SubstructureSurrogateNet):
    """output_dim 为 n_i * n_r."""
```

---



## 2. 2 隐藏层网络

本节记录本项目两种参数化方式的网络配置与已有验证结果；2 隐藏层是实验配置，不代表独立的方法类别。

### 2.1 网络结构、数据与训练配置

块内单元数 `n_fine` 是方法的自由参数：它决定网络的输入维（`n_fine` 各分量之积）、输出维与接口自由度规模，因此每个 `n_fine` 对应一族独立的网络——同一 `n_fine` 下的网络与具体问题无关，换 `n_fine` 则须重新训练。子结构划分 `n_sub` 不进网络，可自由更换，这正是问题无关性的确切边界。

设空间维数为 $d$、`n_fine` 各方向均为 $m$（一阶四边形 / 六面体单元），则各维度量有闭式：

$$n_i = d\,(m-1)^d,\qquad n_b = d\left[(m+1)^d-(m-1)^d\right],\qquad n_r = n_b - n_{\mathrm{rigid}},\qquad n_{\mathrm{rigid}}=\begin{cases}3,& d=2\\[2pt] 6,& d=3\end{cases}$$

其中 $n_b$ 为 `full_trace` 的接口自由度；`linear_corner` 下接口退化到 $2^d$ 个角点，接口自由度为 $n_c = d\,2^d$，相应地 $n_r = n_c - n_{\mathrm{rigid}}$。网络输入维为 $m^d$；形函数路线的输出维为 $n_i\times n_r$（只预测变形分量 $\mathbf{M}_j$，刚体分量解析给出），降阶刚度路线的输出维为 $n_r(n_r+1)/2$（只预测变形子空间上 Cholesky 因子的下三角独立条目）。

本节固定 $d=2$、$m=5$，两种迹空间的维度如下；降阶刚度路线的 `linear_corner` 仅列出理论维度，尚无已登记的可执行工况。

| $d$ | 迹空间 | `n_fine` | 输入维 $m^d$ | $n_i$ | 接口自由度 | $n_r$ | 形函数输出维 $n_i n_r$ | 降阶刚度输出维 $n_r(n_r+1)/2$ |
|---|---|---|---|---|---|---|---|---|
| 2 | `full_trace` | `[5, 5]` | 25 | 32 | $n_b=40$ | 37 | 1,184 | 703 |
| 2 | `linear_corner` | `[5, 5]` | 25 | 32 | $n_c=8$ | 5 | 160 | 15 |

两条路线的网络均为 2 隐藏层 SiLU MLP，`ShapeFunctionSurrogateNet` 隐层宽 256，`ReducedStiffnessSurrogateNet` 隐层宽 128。记隐层宽为 $H$、输出维为 $d_{\mathrm{out}}$，网络为三个线性层与两次 SiLU 交替，输出层不加激活：

$$
\mathbf{y} = \mathbf{W}_3\,\sigma\!\big(\mathbf{W}_2\,\sigma(\mathbf{W}_1\boldsymbol{\rho}^j+\mathbf{b}_1)+\mathbf{b}_2\big)+\mathbf{b}_3,
\qquad
\sigma(x)=\frac{x}{1+e^{-x}}
$$

其中 $\mathbf{W}_1\in\mathbb{R}^{H\times m^d}$、$\mathbf{W}_2\in\mathbb{R}^{H\times H}$、$\mathbf{W}_3\in\mathbb{R}^{d_{\mathrm{out}}\times H}$，$\mathbf{b}_1,\mathbf{b}_2\in\mathbb{R}^{H}$、$\mathbf{b}_3\in\mathbb{R}^{d_{\mathrm{out}}}$。

网络类的构造参数见第 1.2 节；两处调用点是 [`case_setup.py:263`](../../src/soptx/fem/substructure/case_setup.py#L263)（降阶刚度路线）与 [`verify_shape_function_route.py:585`](../../examples/piml_substructure_elasticity/verify_shape_function_route.py#L585)（形函数路线），两者都只传 `input_dim`、`output_dim`、`hidden_dims`：`input_dim` 统一取原型的 `n_cells`，即块内细单元总数，与空间维数无关，三维直接成立；`activation` 两处都不传，一律取骨架默认的 `nn.SiLU`。$H$ 的实际取值：形函数路线的 256 由 `cases.toml` 的标量字段 `hidden_dim` 指定，在调用点展开为 `(256, 256)`；降阶刚度路线的 `(128, 128)` 写在模块常量 `_REDUCED_STIFFNESS_HIDDEN_DIMS` 中，尚未接到 `cases.toml`。两条路线的宽度之差因此不是调优结果。

训练为全批量 Adam（`examples/piml_substructure_elasticity/verify_shape_function_route.py` 的 `fit_full_batch`）：

```python
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
for _ in range(n_epochs):
    optimizer.zero_grad()
    loss = criterion(net(X), Y)
    loss.backward()
    optimizer.step()
```



**训练与优化协议**：

| 项 | 形函数路线 | 降阶刚度路线 |
|---|---|---|
| 回归目标 | 变形分量 $\mathbf{M}_j$ | Cholesky 因子的下三角独立条目 |
| 隐藏层宽度 | `(256, 256)` | `(128, 128)` |
| 学习率 $\eta$ | $0.005$ | $0.005$ |
| 训练集 $N_{\text{train}}$ | $2000$ | $2000$ |
| 留出集 $N_{\text{eval}}$ | $200$ | $100$ |
| 训练轮数 | $4000$，全批量 | $4000$，全批量 |
| 随机种子 | $2026$ | $2026$ |

以上为登记配置；形函数 `linear_corner` 的历史产物使用 300 组 / 300 轮，不能按本表配置解读。网络输入为块内单元密度，训练目标与维度见上文；采样分布、密度范围和数据预处理应以对应产物及生成脚本为准，本报告尚未集中列出。

### 2.2 不依赖训练的构造正确性验证

#### 形函数路线：变分重构与刚体分解

先使用精确内部延拓和人工扰动验证构造本身，再评价训练网络的预测误差。


$$
\widetilde{\mathbf{K}}_r^j=(\widehat{\mathbf{H}}_j^{T})^{\mathsf{T}}\mathbf{K}^j\widehat{\mathbf{H}}_j^{T},
\qquad
\mathbf{K}_r^j=\mathbf{T}_j^{\mathsf{T}}\mathbf{K}_s^j\mathbf{T}_j=(\mathbf{H}_j^{T})^{\mathsf{T}}\mathbf{K}^j\mathbf{H}_j^{T}
$$

$$
\varepsilon_N=\frac{\lVert\widehat{\mathbf{B}}_j-\mathbf{B}_j\rVert_F}{\lVert\mathbf{B}_j\rVert_F},
\qquad
\varepsilon_K=\frac{\lVert\widetilde{\mathbf{K}}_r^j-\mathbf{K}_r^j\rVert_F}{\lVert\mathbf{K}_r^j\rVert_F}
$$

构造实现为两步：

$$
\mathbf{K}^j=\begin{bmatrix}\mathbf{K}_{ii}^j & \mathbf{K}_{ib}^j\\[2pt] (\mathbf{K}_{ib}^j)^{\mathsf{T}} & \mathbf{K}_{bb}^j\end{bmatrix}
\;\longmapsto\;
\Big(\mathbf{K}_{ii}^j,\ \mathbf{K}_{ib}^j\mathbf{T}_j,\ \mathbf{T}_j^{\mathsf{T}}\mathbf{K}_{bb}^j\mathbf{T}_j\Big)
$$

$$
\mathbf{K}_r^j=\mathbf{T}_j^{\mathsf{T}}\mathbf{K}_{bb}^j\mathbf{T}_j+\mathbf{A}+\mathbf{A}^{\mathsf{T}}+\mathbf{B}_j^{\mathsf{T}}\mathbf{K}_{ii}^j\mathbf{B}_j,
\qquad
\mathbf{A}=(\mathbf{K}_{ib}^j\mathbf{T}_j)^{\mathsf{T}}\mathbf{B}_j
$$

两条迹空间共用核心库的同一段构造（`src/soptx/fem/substructure/piml_surrogate.py`），传入网络输出 $\widehat{\mathbf{B}}_j$ 即得 $\widetilde{\mathbf{K}}_r^j$；`full_trace` 取 $\mathbf{T}_j=\mathbf{I}$，`linear_corner` 取 $\mathbf{T}_j=\mathbf{L}_j$。

```python
class ShapeFunctionCondensation(StaticCondensationBase):

    def _trace_blocks(self, K_local: Any) -> Tuple[Any, Any, Any]:
        """把局部刚度的分块映射到当前迹空间."""
        K_ii = K_local[..., self.i_dofs[:, None], self.i_dofs]
        K_ib = K_local[..., self.i_dofs[:, None], self.b_dofs]
        K_bb = K_local[..., self.b_dofs[:, None], self.b_dofs]
        if self.trace_matrix is None:
            return K_ii, K_ib, K_bb
        T = self.trace_matrix
        return K_ii, K_ib @ T, bm.matrix_transpose(T) @ K_bb @ T

    @staticmethod
    def _variational_stiffness(
        K_ii: Any, K_ib_t: Any, K_bb_t: Any, B: Any
    ) -> Any:
        """在已切片的迹空间分块上展开变分式."""
        A = bm.matrix_transpose(K_ib_t) @ B
        return (
            K_bb_t + A + bm.matrix_transpose(A)
            + bm.matrix_transpose(B) @ K_ii @ B
        )
```



| 诊断量 | `full_trace` | `linear_corner` | 读法 |
|---|---|---|---|
| 精确内部延拓 $\mathbf{B}_j$ 代入变分重构式的相对误差 | `6.33e-16` | `1.39e-15` | 变分恒等式在角点迹下同样成立 |
| 扰动扫描 $\log$-$\log$ 斜率 | `2.0030` | `2.0026` | 二阶压缩机理不依赖迹空间，与上文推导一致 |
| 拟合放大系数 | `0.84` | `7.78` | 同样的 $\varepsilon_N$ 下角点迹的刚度误差约高一个量级，对拟合质量的要求更严 |
| 刚体分量密度无关性 | `5.0e-16` | `5.3e-16` | 刚体响应是几何恒等式，两者都成立 |
| 刚体解析分量相对偏差 | `3.96e-16` | `1.193` | **待查**：角点迹下用于拆分的解析 $\boldsymbol{\Phi}_i^j$ 与真实刚体响应不符 |



`full_trace` 的变分恒等式与刚体分解偏差处于机器精度量级；12 个扰动量级扫描的拟合斜率为 `2.0030`（解析对照运行 `2.0000`，理论值为 2），支持该扫描范围内的二阶误差关系。`linear_corner` 的解析刚体分量偏差为 `1.193`，须先检查基选取与重构的一致性，尚不能据此确定唯一原因。

#### 降阶刚度路线：Cholesky 零空间参数化

$\widehat{\mathbf{K}}_r^j=\mathbf{R}_{\perp}^j\mathbf{C}_j\mathbf{C}_j^{\mathsf{T}}(\mathbf{R}_{\perp}^j)^{\mathsf{T}}$ 在代数上保证对称半正定，并在 $\mathbf{R}_{\perp}^j$ 与刚体基正交时保留刚体零空间。这一性质不依赖网络是否训练收敛。

| 指标 | 现有产物诊断值 |
|---|---|
| 参数化误差上限 | `3.84e-08` |
| 刚体零空间污染比（最大 / 均值） | `3.85e-15` / `1.11e-15` |
| 刚体基残差 | `8.77e-17` |

上述数值来自 `piml_exact_comparison.json`；零空间污染比远低于该产物中的拟合误差，但这些诊断不替代网络精度或整体求解验证。

### 2.3 网络训练结果与留出集精度

两条路线均先训练局部代理，再用留出集评价预测质量。下表集中列出已有 `full_trace` 结果；形函数产物为 `shape_function_full_trace_2d`，降阶刚度产物为 `piml_exact_comparison.json`。

| 指标 | 形函数路线 | 降阶刚度路线 |
|---|---|---|
| 训练集最终 MSE | 本报告未列出 | `2.04e-05` |
| 留出集样本数 | 200 | 100 |
| 内部延拓相对误差 $\varepsilon_N$ | 均值 $8.97\%$，最大 $14.60\%$ | 不预测内部延拓 |
| 缩聚刚度相对误差 | 均值 $0.44\%$，最大 $1.22\%$ | 均值 $4.30\%$，最大 $8.09\%$ |
| 训练耗时、收敛曲线 | 本报告未列出 | 本报告未列出 |

形函数路线的内部延拓误差经变分重构后对应较小的刚度误差，与第 2.2 节的二阶误差关系相符。两条路线的回归目标、输出维度、隐藏层宽度和留出集规模不同，现有结果是当前配置下的精度对比，不能将差距全部归因于误差抵消机理，也不能直接用训练 MSE 比较两种目标。

形函数 `linear_corner` 的历史产物 `eq17_second_order_linear_corner.json` 使用 300 组 / 300 轮，内部延拓误差均值为 $46.3\%$。该配置训练不足的影响与第 2.2 节记录的刚体分解异常尚未区分，暂不作能力判定。降阶刚度路线的 `linear_corner` 尚无已登记的可执行工况。


### 2.4 整体结构求解精度与路线对比

`FullMBBBeam2d` 的求解域为 $[0,12]\times[0,2]$，划分为 $12\times 2$ 共 24 个子结构，单块包含 $5\times 5$ Q1 单元，全尺度细网格共 682 自由度。


![图 4(c) 全局结构求解精度](figure_data/fig3_piml_panel_c.svg)

在完全一致的物理问题、网格离散与在役密度场下，对比形函数变分路线与直接预测刚度路线的解层精度：

| 评估指标 | 直接预测刚度路线 $\widehat{\mathbf{K}}_r^j$ | 预测形函数变分路线 $\widetilde{\mathbf{K}}_r^j$ | 直接预测 / 变分路线误差比 |
|:---|:---:|:---:|:---:|
| 局部刚度平均相对差 | 2.90% | 0.09% | 32.2 倍 |
| **局部刚度最大相对差** | 5.83% | **0.15%** | **38.9 倍** |
| **接口位移相对误差** | 2.05% | **0.15%** | **13.7 倍** |
| **全场回填位移相对误差** | 2.01% | **0.15%** | **13.4 倍** |
| **全局结构柔度相对误差** | 3.64% | **0.20%** | **18.2 倍** |



当前 `full_trace` 工况中，形函数路线的接口位移、全场回填位移与柔度误差为 $0.15\%$–$0.20\%$，与精确缩聚结果接近；直接预测刚度路线的对应误差为 $2.01\%$–$3.64\%$。这些数值描述本算例的实际表现，不代表任意密度场下的误差保证。两条路线的内部位移恢复方式不同：形函数路线使用预测延拓，降阶刚度路线使用精确延拓。

`linear_corner` 历史产物的全场位移误差为 $99.9\%$、柔度误差为 $96.4\%$。由于训练配置与刚体分解问题尚未厘清，不将其纳入上述路线对比。

### 2.5 当前结论与未解决问题

- `full_trace` 的变分构造与刚体分解通过现有代数诊断；训练后的形函数路线在当前算例中取得较小的局部刚度与整体求解误差。
- 降阶刚度路线保留了刚体零空间，但现有配置的留出集与解层误差高于形函数路线；路线差异与网络容量等因素尚未通过控制变量实验分离。
- `linear_corner` 需先排查解析刚体分解，再按登记配置训练与验证。当前历史结果不能单独归因为欠拟合。
- 本报告尚缺完整的训练收敛与成本记录，现有精度数据不足以评价训练效率。

---


## 3. 15 隐藏层网络

### 3.1 网络结构与预测对象

`trace_kind` 区分接口空间（`linear_corner` 或 `full_trace`），`route` 区分预测对象（`shape` 或 `stiffness`）。两种接口空间采用相同的材料输入，分别构建和训练对应的网络，不共用训练好的权重。

| 配置项 | 具体设置 |
|---|---|
| 空间维数与块内划分 | 三维，$5\times5\times5$ 个细单元 |
| 网络输入 | 125 个细单元的归一化杨氏模量（由设计密度经材料插值得到），泊松比固定为 0.3 |
| 形函数路线输出 | 内部形函数的独立分量，经平移、转动约束补全为内部形函数矩阵 |
| 直接刚度路线输出 | 独立刚度条目，经对称性与刚体零空间约束补全为缩聚刚度矩阵 |
| 网络数量与输出拆分 | 每条路线通过 `num_networks` 设置，独立输出按连续索引均衡拆分；默认形函数 4 个网络、直接刚度 1 个网络 |
| 隐藏层数 | 每个网络 15 层 |
| 逐层宽度 | `[60, 80, 100, 120, 140, 160, 180, 200, 180, 160, 140, 120, 100, 80, 60]` |
| 逐层激活函数 | `[tanh, elu, tanh, elu, tanh, elu, tanh, elu, elu, tanh, elu, tanh, elu, tanh, elu]` |

两种接口空间对应的输出维度如下；内部自由度数均为 192，刚体模态数均为 6。

| 配置项 | `linear_corner` | `full_trace` |
|---|---|---|
| 接口节点 | 8 个角点 | 全部 152 个边界节点 |
| 接口自由度数 | 24 | 456 |
| 边界位移表示 | 由角点位移线性插值 | 保留全部边界节点位移 |
| 形函数独立输出数 | $192\times(24-6)=3456$ | $192\times(456-6)=86400$ |
| 补全后的内部形函数矩阵 | $192\times24$ | $192\times456$ |
| 刚度独立输出数 | $18\times19/2=171$ | $450\times451/2=101475$ |
| 补全后的缩聚刚度矩阵 | $24\times24$ | $456\times456$ |

以下保留 [independent_training.py](../../src/soptx/ml/substructure/independent_training.py) 中 `build_network()` 的核心构造逻辑。

```python
from numbers import Integral
import torch
from torch import nn
from soptx.ml.substructure.nets import DirectStiffnessNet, SplitOutputNet

HIDDEN_DIMS = (
    60, 80, 100, 120, 140, 160, 180, 200,
    180, 160, 140, 120, 100, 80, 60,
)
ACTIVATIONS = (
    nn.Tanh, nn.ELU, nn.Tanh, nn.ELU, nn.Tanh,
    nn.ELU, nn.Tanh, nn.ELU, nn.ELU, nn.Tanh,
    nn.ELU, nn.Tanh, nn.ELU, nn.Tanh, nn.ELU,
)

def build_network(provider_metadata, *, route="shape", seed=2026, num_networks=None):
    """按所选接口空间构建单条预测路线的模型."""
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")
    # 文档省略实际函数中的元数据校验细节.
    widths = {
        "inputs": provider_metadata["n_cells"],
        "shape_targets": provider_metadata["n_shape_targets"],
        "stiffness_targets": provider_metadata["n_stiffness_targets"],
    }
    if num_networks is not None and (
        isinstance(num_networks, bool) or not isinstance(num_networks, Integral)
        or num_networks <= 0
    ):
        raise ValueError("num_networks 必须为正整数")
    count = num_networks if num_networks is not None else (4 if route == "shape" else 1)
    output_dim = widths[f"{route}_targets"]
    if count > output_dim:
        raise ValueError(f"{route} 的网络数量不能超过独立输出数")

    torch.manual_seed(seed)
    # 连续划分输出索引, 余数优先分配给前面的组.
    size, remainder = divmod(output_dim, count)
    groups = []
    start = 0
    for i in range(count):
        stop = start + size + (i < remainder)
        groups.append(tuple(range(start, stop)))
        start = stop
    kwargs = dict(
        input_dim=widths["inputs"], output_dim=output_dim,
        hidden_dims=HIDDEN_DIMS, activation=ACTIVATIONS,
    )
    # 保留原单网络刚度模型的权重键格式.
    model = (
        DirectStiffnessNet(**kwargs) if route == "stiffness" and count == 1
        else SplitOutputNet(output_groups=tuple(groups), **kwargs)
    )
    return model.to(dtype=torch.float64)
```

网络构建时，先由所选接口空间生成元数据，再调用上述函数：

```python
from soptx.fem.substructure.independent_targets import IndependentTargetProvider

trace_kind = "linear_corner"  # 当前支持 "linear_corner" 和 "full_trace", 两者需分别训练.
provider = IndependentTargetProvider(
    cell_size=(1.0, 1.0, 1.0),
    n_fine=(5, 5, 5),
    trace_kind=trace_kind,
)
shape_net = build_network(provider.metadata(), route="shape", seed=2026)
stiffness_net = build_network(provider.metadata(), route="stiffness", seed=2026)
```

### 3.2 样本生成与训练方法

形函数路线的样本规模与两条路线的损失形式沿用文献描述；本实验为两条路线采用相同的样本划分和下表训练参数。直接刚度路线的样本规模以及优化器、学习率、batch size、最大训练轮数、验证集规模、学习率调整、提前停止和随机种子为本实验选定的设置。

| 配置项 | 具体设置 |
|---|---|
| 训练样本 | 400,000 个随机样本；本次归一化杨氏模量在 $[10^{-6},1)$ 内均匀采样, 避免零刚度奇异 |
| 形函数路线监督损失 | 约束补全后的预测形函数与精确形函数之间的均方误差 |
| 直接刚度路线监督损失 | 对称性与刚体零空间约束补全后的预测刚度与精确刚度之间的均方误差 |
| 形函数路线训练后期的一致性损失 | 由预测形函数构造的刚度与直接刚度预测网络输出之间的均方误差；本次未启用 |
| 优化器 | Adam |
| 初始学习率 | $10^{-3}$ |
| batch size | 256 |
| 最大训练轮数 | 500 |
| 验证集 | 另生成 40,000 个样本，与训练集独立，用于调参和提前停止 |
| 学习率调整 | 验证损失连续 10 轮未改善时减半，最低 $10^{-6}$ |
| 提前停止 | 验证损失连续 40 轮未改善时停止，保留验证损失最低的权重 |
| 随机种子 | 2026 |

```python
from pathlib import Path
from soptx.fem.substructure.independent_targets import IndependentTargetProvider
from soptx.ml.substructure.independent_training import (
    build_network, generate_dataset, train_networks,
)
from soptx.ml.substructure.training import TrainingConfig

# 定义子结构与所选路线.
dim, n_fine = 3, 5
route = "shape"  # 可选 "shape"、"stiffness"、"both".
provider = IndependentTargetProvider(
    cell_size=(1.0,) * dim,
    n_fine=(n_fine,) * dim,
)
routes = ("shape", "stiffness") if route == "both" else (route,)
networks = {
    name: build_network(provider.metadata(), route=name, seed=2026)
    for name in routes
}

# 在本实验目录下运行; 重复运行时更换 example_run, 不覆盖已有目录.
output_root = Path("outputs/independent_15_layer")
samples_dir = output_root / "samples" / "example_run"
training_dir = output_root / "training" / "example_run"

# 生成独立的训练集与验证集, 并计算精确标签.
dataset = generate_dataset(
    provider, samples_dir,
    n_train=400_000, n_validation=40_000,
    batch_size=32, min_modulus=1e-6, seed=2026,
)

# 设置训练参数.
config = TrainingConfig(
    epochs=500, batch_size=256, learning_rate=1e-3,
    patience=40, seed=2026,
)

# 训练所选路线, 保存验证损失最低的权重.
results = train_networks(
    dataset, training_dir,
    networks=networks, codecs=provider.codecs,
    provider_metadata=provider.metadata(),
    route=route, device="cpu", config=config,
)
```

本次三维 $m=5$、`linear_corner` 工况已完成 400,000 个训练样本和 40,000 个验证样本的生成，并完成两条路线的监督训练：

| 路线 | 网络数量 | 实际训练轮数 | 最佳轮次 | 最佳验证 MSE | 停止原因 |
|---|---:|---:|---:|---:|---|
| 形函数 | 4 | 500 | 498 | $6.409\times10^{-6}$ | 达到最大训练轮数 |
| 直接刚度 | 1 | 46 | 6 | $9.765\times10^{-7}$ | 连续 40 轮未改善 |



运行命令：

```bash
# 1. 生成训练集和验证集.
python run.py --generate-samples \
  --dim 3 --n-fine 5 \
  --n-train 400000 --n-validation 40000 \
  --generation-batch-size 32 --min-modulus 1e-6 \
  --seed 2026

# 2. 读取已有数据集, 训练形函数网络.
# 将路径替换为样本生成时输出的实际目录.
DATASET="outputs/independent_15_layer/samples/20260922T065924289373Z"
python run.py --train --dataset "$DATASET" \
  --dim 3 --n-fine 5 \
  --route shape --num-networks 4 \
  --epochs 500 --batch-size 256 --lr 1e-3 \
  --patience 40 --seed 2026 --device cpu

# 3. 一次完成样本生成与形函数网络训练.
python run.py --all \
  --dim 3 --n-fine 5 \
  --n-train 400000 --n-validation 40000 \
  --generation-batch-size 32 --min-modulus 1e-6 \
  --route shape --num-networks 4 \
  --epochs 500 --batch-size 256 --lr 1e-3 \
  --patience 40 --seed 2026 --device cpu
```

数据来源：样本保存在 [samples/20260922T065924289373Z](outputs/independent_15_layer/samples/20260922T065924289373Z/)，最佳权重与训练记录保存在 [training/20260922T065924289373Z](outputs/independent_15_layer/training/20260922T065924289373Z/)，汇总见 [summary.json](outputs/independent_15_layer/training/20260922T065924289373Z/summary.json)。

### 3.3 刚度构造与结构求解

以下采用第 3.2 节保存的最佳网络权重，在 `linear_corner` 接口空间下分别构造局部刚度，再装配求解整体角点位移。两条路线均考虑子结构内部无外载的情形。

#### 形函数预测路线

网络预测内部形函数的独立分量，并按平移、转动约束补全内部延拓 $\widehat{\mathbf{B}}_j$。令 $\mathbf{L}_j$ 为角点位移到边界位移的线性插值矩阵，按内部、边界自由度顺序构造

$$
\widehat{\mathbf{H}}_j=
\begin{bmatrix}
\widehat{\mathbf{B}}_j\\
\mathbf{L}_j
\end{bmatrix},\qquad
\widetilde{\mathbf{K}}_r^j=\widehat{\mathbf{H}}_j^{\mathsf{T}}\mathbf{K}^j\widehat{\mathbf{H}}_j,
$$

其中 $\mathbf{K}^j$ 为相同自由度顺序下的局部有限元刚度矩阵。将各块刚度装配为整体角点系统，施加边界条件并求解后，取出各块角点位移 $\mathbf{q}_j$，用同一预测延拓恢复内部位移：

$$
\widehat{\mathbf{u}}_i^j=\widehat{\mathbf{B}}_j\mathbf{q}_j.
$$

该构造使角点系统的应变能与由同一预测形函数恢复的细网格位移场的应变能一致。

#### 直接刚度预测路线

刚度网络预测 171 个独立条目，补全得到 $24\times24$ 的 $\widehat{\mathbf{K}}_r^j$，满足

$$
\widehat{\mathbf{K}}_r^j=(\widehat{\mathbf{K}}_r^j)^{\mathsf{T}},\qquad
\widehat{\mathbf{K}}_r^j\mathbf{R}_q^j=\mathbf{0},
$$

其中 $\mathbf{R}_q^j$ 为角点接口上的 6 个刚体模态。将预测刚度直接装配为整体角点系统，施加相同边界条件并求解角点位移。

需要内部位移时，另调用形函数网络，补全得到 $\widehat{\mathbf{B}}_j$，并使用 $\widehat{\mathbf{u}}_i^j=\widehat{\mathbf{B}}_j\mathbf{q}_j$ 恢复。两条路线的区别在于局部刚度的来源，直接刚度路线仍可使用形函数网络恢复内部位移。

分别预测的刚度与形函数不自动满足

$$
\widehat{\mathbf{K}}_r^j=\widehat{\mathbf{H}}_j^{\mathsf{T}}\mathbf{K}^j\widehat{\mathbf{H}}_j,
$$

因此不自动保证角点系统与恢复的细网格位移场之间的应变能一致性。

#### 核心代码

以下是调用现有接口的流程示例，尚未接入 `run.py`。约束补全来自 [independent_targets.py](../../src/soptx/fem/substructure/independent_targets.py)，变分刚度构造来自 [piml_surrogate.py](../../src/soptx/fem/substructure/piml_surrogate.py)，稀疏装配与求解分别来自 [assembler.py](../../src/soptx/fem/substructure/assembler.py) 和 [solve.py](../../src/soptx/fem/substructure/solve.py)。

示例采用 NumPy 有限元后端与 CPU、float64 网络。`provider` 沿用样本生成时的配置；`networks` 已加载各路线的最佳权重。`assembler` 和 `sub_meshes` 描述待分析结构，各子结构须与训练原型的几何、材料和离散设置一致。`E` 按 `sub_meshes` 顺序排列，每行采用训练时的细单元编号；`load` 和 `fixed_dofs` 使用整体角点系统编号。直接刚度路线恢复内部位移时也需要 `networks["shape"]`。

```python
import torch
from fealpy.backend import backend_manager as bm
from soptx.fem.substructure.piml_surrogate import ShapeFunctionCondensation
from soptx.fem.substructure.solve import solve_interface_system


@torch.no_grad()
def analyze_structure(
    provider, networks, assembler, sub_meshes, E, load, fixed_dofs,
    *, route="shape",
):
    """调用已有接口构造刚度, 求解角点位移并恢复局部位移.

    Returns
    -------
    dict
        局部刚度、整体角点位移及各块内部和边界位移.
    """
    if route not in ("shape", "stiffness"):
        raise ValueError("route 必须为 shape 或 stiffness")

    prototype, trace = provider.prototype, provider.trace
    inputs = torch.as_tensor(bm.to_numpy(E), dtype=torch.float64)

    # 利用平移、转动约束, 将预测的独立分量补全为内部形函数矩阵 B.
    shape_net = networks["shape"]
    # 设置网络状态为评估模式，不进行预测.
    shape_net.eval()
    values = shape_net(inputs)               # 预测独立分量
    B = provider.shape_codec.decode(values)  # 根据平移、转动约束补全矩阵
    B = bm.asarray(B.cpu().numpy(), dtype=bm.float64)

    if route == "shape":
        rigid, deformation, interior = prototype.trace_interface_bases(trace)
        constructor = ShapeFunctionCondensation(
            prototype.i_dofs, prototype.b_dofs,
            rigid_basis=rigid,
            deformation_basis=deformation,
            rigid_interior=interior,
            trace=trace,
        )
        K_local = prototype.assemble_local_stiffness_batch(E)
        K_reduced = constructor.assemble_reduced_stiffness(K_local, B)
    else:
        stiffness_net = networks["stiffness"]
        stiffness_net.eval()
        values = stiffness_net(inputs)  # 预测独立刚度条目
        K_reduced = provider.stiffness_codec.decode(values)  # 根据对称性和刚体约束补全矩阵
        K_reduced = bm.asarray(K_reduced.cpu().numpy(), dtype=bm.float64)

    # K_reduced 已位于角点空间, 直接装配, 不再进行迹空间投影.
    system = assembler.assemble_macro_system(sub_meshes, K_reduced)
    q = solve_interface_system(system, load, fixed_dofs, solver="scipy")

    # 从整体角点位移提取各块位移, 再恢复内部和边界位移.
    corner_indices = assembler.macro_corner_indices(sub_meshes)
    q_local = q[corner_indices]
    u_internal = bm.einsum("bij,bj->bi", B, q_local)
    u_boundary = trace.expand_displacement(q_local)
    return {
        "local_stiffness": K_reduced,
        "corner_displacement": q,
        "internal_displacement": u_internal,
        "boundary_displacement": u_boundary,
    }
```

---

## 4. 产物来源与测试环境

精度结果引用 `shape_function_full_trace_2d`、`eq17_second_order_linear_corner.json` 与 `piml_exact_comparison.json`。新产物按 `outputs/<case-id>/<UTC timestamp>/` 隔离，运行配置由 `run_config.json` 记录，汇编快照为 [`figure_data/fig3_data.json`](figure_data/fig3_data.json)。现有报告未逐项列出具体时间戳路径，追溯时需核对产物配置。

GPU 批量缩聚的耗时、加速比与测试范围见 [`../piml_substructure_gpu/`](../piml_substructure_gpu/)。其三维 $4\times4\times4$ 子结构性能测试与本报告二维精度实验分别解读，不作为本节训练成本或整体求解加速的证据。

以下保留原报告记录的测试环境：


* **操作系统**：Ubuntu 24.04 LTS (WSL2)
* **计算软件栈**：Python 3.12.13, PyTorch 2.13.0 (+cu130), FEALPy, NumPy 2.2.6
* **硬件设备**：
  * CPU: 13th Gen Intel Core i9-13900K
  * GPU: NVIDIA GeForce RTX 5080 (16GB GDDR7, 标称显存带宽 960 GB/s)
