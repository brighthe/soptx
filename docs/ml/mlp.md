# MLP 实现说明

`soptx.ml.MLP` 是 SOPTX 内可复用的全连接前馈网络骨干。它不包含 PDE、有限元网格、材料模型或缩聚约束；这些物理语义由调用方定义。

源码：[`src/soptx/ml/networks.py`](../../src/soptx/ml/networks.py)。通用 MLP 数学定义与归纳偏置见 `dut-postdoc:concepts/machine-learning.md`。

## 构造契约

```python
MLP(input_dim, output_dim, hidden_dims=(), activation=nn.Tanh, *, dtype=None, device=None)
```

| 参数 | 含义 |
| --- | --- |
| `input_dim` | 单个样本输入特征数，必须为正整数。 |
| `output_dim` | 单个样本输出特征数，必须为正整数。 |
| `hidden_dims` | 每个隐藏层的宽度；空元组表示无隐藏层。 |
| `activation` | 激活层工厂，例如 `nn.Tanh`、`nn.SiLU`。代码对每个隐藏层调用一次 `activation()`。 |
| `dtype`、`device` | `Linear` 参数的精度和设备；不改变网络的映射结构。 |

输入 `x` 的最后一维必须为 `input_dim`。若批大小为 `B`，则 `(B, input_dim)` 的输入对应 `(B, output_dim)` 的输出；`nn.Linear` 也接受任意前导 batch 维。

## 从参数到执行序列

实现先拼接维度链：

```text
dimensions = (input_dim,) + hidden_dims + (output_dim,)
```

随后对每一对相邻维度创建一个 `Linear`；仅在非末层后追加激活层：

```text
for (d_in, d_out) in zip(dimensions[:-1], dimensions[1:]):
    Linear(d_in, d_out)
    activation()  # 仅非末层
```

最后由 `nn.Sequential` 保存层顺序，`forward(x)` 只执行 `self.net(x)`。因此输出层始终线性，回归范围、正定性、边界条件等输出约束必须由调用方参数化、损失函数或后处理负责。

| 实现语句 | 直接作用 |
| --- | --- |
| `dimensions = ...` | 定义所有层的输入/输出维度。 |
| `zip(dimensions[:-1], dimensions[1:])` | 枚举相邻维度对。 |
| `nn.Linear(...)` | 创建仿射变换层。 |
| `if index < len(dimensions) - 2` | 防止在输出层后添加激活。 |
| `self.net = nn.Sequential(*layers)` | 固定前向层序列。 |
| `return self.net(x)` | 执行该序列。 |

## 当前消费者

| 调用方 | 配置 | 输入和输出语义 |
| --- | --- | --- |
| [`PINNElasticityNet`](../../examples/pinn_elasticity/minimal_demo.py) | `Tanh`，`float64`，默认隐藏层 `(32, 32, 16)` | 坐标映射到位移；自动微分残差由 PINN 算例定义。 |
| [`PIMLSurrogateNet`](../../src/soptx/fem/substructure/piml_surrogate.py) | `SiLU`，隐藏层 `(128, 128)` | 子结构密度映射到变形子空间上 Cholesky 因子的下三角独立条目；秩亏刚度重构与精确回退由 `PIMLStaticCondensation` 定义。 |

例如，PIML 调用 `PIMLSurrogateNet(input_dim=n_fine_x * n_fine_y, output_dim=n_tril)` 时，实际层链为：

```text
(n_fine_x * n_fine_y) -> 128 -> 128 -> n_tril
```

若输入为 `(B, n_fine_x * n_fine_y)`，网络输出为 `(B, n_tril)`。`n_tril` 是下三角矩阵的独立条目数；其具体物理解释不属于 `MLP`。
