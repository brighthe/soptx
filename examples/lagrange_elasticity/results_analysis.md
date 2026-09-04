# 拉格朗日位移有限元 Demo 结果分析

本页汇总 [`manufactured_convergence_demo.py`](manufactured_convergence_demo.py) 与 [`concentrated_load_demo.py`](concentrated_load_demo.py) 的验证结果。运行入口和参数说明见 [`README.md`](README.md)。

## 1. 验证内容与范围

| Demo | 验证内容 | 主要指标 |
|---|---|---|
| `manufactured_convergence_demo.py` | 2D/3D 拉格朗日位移元的离散正确性 | $L_2$ 误差、观测收敛阶、真相对残差 |
| `concentrated_load_demo.py` | MBB 梁集中力装配的正确性 | 真相对残差、载荷等效性、柔顺度诊断量 |

当前结果仅适用于：

- CPU 串行全组装，即 `operator_level="fa"`；
- `float64`，FEALPy vendor fork 路径 `/home/brighthe/workspace/fealpy`；
- 2D `tri/quad`、3D `tet/hex` 规则加密网格；
- 任意正整数多项式次数 $p\ge1$，已实测 $p=1,2$；
- `scipy`、`mumps` 直接法和无预条件 `cg` 迭代法。

Matrix-Free、GPU、MPI、子结构、PIML 和拓扑优化不属于本目录的验证范围。

## 2. 制造解 Demo 结果

制造解 Demo 求解线弹性离散系统 $KU=F$，以真相对残差确认线性系统求解正确，并以

$$
\lVert u-u_h\rVert_{L_2(\Omega)}
$$

随网格加密的变化判断离散收敛性。理论预期为 $O(h^{p+1})$。

### 2.1 代表性收敛结果

| 维数与网格 | 模型与次数 | 最细网格 | 最细档 $L_2$ 误差 | 末档观测阶 | 真相对残差 |
|---|---|---:|---:|---:|---:|
| 2D `tri` | `sinusoidal`，$p=1$ | $n=64$ | `5.4027e-04` | **1.995** | `1.98e-13` |
| 2D `quad` | `sinusoidal`，$p=1$ | $n=16$ | `1.9839e-03` | **1.993** | `7.17e-15` |
| 3D `tet` | `divfree-poly`，$p=1$ | $n=64$ | `6.9927e-04` | **1.969** | `1.19e-13` |
| 3D `hex` | `divfree-poly`，$p=1$ | $n=16$ | `1.0881e-03` | **1.994** | `9.29e-15` |
| 2D `tri` | `sinusoidal`，$p=2$ | $n=16$ | `1.2184e-05` | **3.005** | `2.01e-13` |
| 2D `quad` | `sinusoidal`，$p=2$ | $n=16$ | `3.8491e-06` | **3.002** | `1.56e-13` |
| 3D `tet` | `divfree-poly`，$p=2$ | $n=8$ | `1.7488e-03` | **3.058** | `1.42e-14` |

结果表明：

1. $p=1$ 的末档观测阶接近 2，$p=2$ 的观测阶接近 3，与 $O(h^{p+1})$ 一致。
2. `tri/quad` 与 `tet/hex` 两类基函数路径均表现出预期收敛性。
3. 3D `tet` 的前三档仍处于前渐近区，观测阶由 `1.255`、`1.676` 逐步提高到 `1.891`、`1.969`；因此不能只根据前三档判断其离散质量。
4. 2D 的全 Dirichlet 与混合边界模型均已验证。$p=1$、三档网格下八组组合全部满足误差下降和残差要求，末档观测阶为 `1.814`–`1.993`。

### 2.2 直接法与 CG

2D `tri`、$p=1$、`--solver cg --rtol 1e-12` 的结果如下：

| $n$ | CG 迭代数 | 真相对残差 |
|---:|---:|---:|
| 8 | 43 | `3.58e-13` |
| 16 | 99 | `1.69e-12` |
| 32 | 206 | `2.70e-12` |

这些网格上的 $L_2$ 结果与直接法在输出精度内一致，说明迭代求解没有改变离散解。无预条件 CG 的迭代数随网格加密明显增加；3D `tet`、$n=64$ 时真相对残差为 `1.0532e-10`，略高于当前 `1e-10` 判据，因此该规模尚不能记为通过。

## 3. 集中力 Demo 结果

集中力 Demo 不使用收敛阶作为正确性判据。2D 点载荷存在应力奇异性，柔顺度随加密增长，因此这里只检查：

1. 求解后的真相对残差；
2. 施加 Dirichlet 条件前，等效节点力之和是否等于给定载荷 $P=-1$。

### 3.1 2D 半域 MBB 梁

| 网格 | $n_x$ 范围 | 真相对残差范围 | 载荷偏差 | 柔顺度范围 |
|---|---|---:|---:|---:|
| `quad` | 60–240 | `1.40e-12`–`7.19e-12` | `0.00e+00` | `125.8778`–`130.7497` |
| `tri` | 60–240 | `2.34e-12`–`1.05e-11` | `0.00e+00` | `123.5272`–`129.0135` |

两类网格上的集中力都被精确等效到节点载荷。柔顺度只作为单次计算的诊断量，不用于跨网格或跨技术栈比较。

### 3.2 3D MBB 梁

| 算例 | 网格 | 自由度 | 真相对残差 | 载荷偏差 | 柔顺度 |
|---|---|---:|---:|---:|---:|
| `mbb-half-3d` | $30\times10\times10$ | 11,253 | `5.55e-13` | `0.00e+00` | `7.382210` |
| `mbb-full-3d` | $60\times10\times10$ | 22,143 | `4.76e-13` | `0.00e+00` | `3.690937` |

这两组结果验证了 3D `hex` 路径的集中力装配。所用网格显式小于问题注册表中的缺省规模，结果不能外推为大规模性能结论。

## 4. 装配与求解配置比较

3D `tet`、$n=64$、823,875 自由度下，整个 Python 进程的峰值 RSS 如下。该指标包含装配、格式转换和求解阶段的全部临时内存，不等同于最终矩阵存储量。

| 装配方法 | 求解配置 | 峰值 RSS / GiB | 总墙钟 | $n=64$ 单档时间 |
|---|---|---:|---:|---:|
| `standard` | `cg` | 41.16 | 2:55.90 | 138.0 s |
| `standard` | `mumps, sym=0` | 41.46 | 4:05.03 | 206.0 s |
| `fast` | `cg` | 17.30 | 2:01.49 | 92.1 s |
| `fast` | `mumps, sym=0` | 28.74 | 3:08.18 | 154.9 s |
| `fast` | `mumps, sym=1` | 17.38 | 3:08.43 | 150.3 s |
| `fast` | `mumps, sym=2` | 17.32 | 4:07.41 | 205.8 s |

六种配置的相对 $L_2$ 误差均为 `6.1430e-03`，末档观测阶均为 `1.969`，说明配置变化没有改变离散结果。现有证据支持：

- `fast` 显著降低装配阶段的峰值内存；
- 对称正定位移刚度阵采用 `mumps, sym=1` 可避免一般非对称分解的额外内存；
- 当前推荐配置为 `--assembly-method fast --mumps-sym 1`；
- 只有 $n=64$ 一个内存规模点，不能据此宣称直接法已经达到内存极限。

## 5. 结果边界与复现

### 5.1 尚未覆盖

- 2D 完整域 MBB 梁；
- 非结构化网格；
- 3D 混合边界制造解；
- 预条件 CG；
- `hex` 网格的 $n=32,64$ 收敛结果；
- 自动执行全部配置的回归脚本；
- 峰值 RSS 的机器可读证据记录。

### 5.2 代表性复现命令

```bash
python examples/lagrange_elasticity/manufactured_convergence_demo.py \
    --dim 2 --mesh-type tri --model sinusoidal --base 4 --levels 5
```

```bash
python examples/lagrange_elasticity/manufactured_convergence_demo.py \
    --dim 3 --mesh-type tet --model divfree-poly --base 4 --levels 5 \
    --solver mumps --assembly-method fast --mumps-sym 1
```

```bash
python examples/lagrange_elasticity/concentrated_load_demo.py \
    --dim 3 --problem mbb-half-3d --nx 30 --ny 10 --nz 10 --levels 1
```

运行通过后，JSON 结果写入本目录的 `outputs/`。判读结果时应同时核对网格、次数、求解器、装配方法和 `fealpy_path`；更换 FEALPy 检出后，现有结论不自动成立。
