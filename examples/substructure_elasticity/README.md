# 精确子结构静力缩聚示例

本目录验证规则结构化网格上的 2D/3D 线弹性子结构静力缩聚。公开脚本按问题直接划分，不使用 `case-id`；下划线模块只存放共享实现。

本目录不包含 PIML、Matrix-Free、GPU 或拓扑优化循环。当前假设子结构内部自由度不受外载，只支持规则结构化张量积网格。

## 文件职责

| 文件 | 职责 |
|---|---|
| `verify_full_trace_convergence.py` | 用制造解验证 `full_trace` 位移误差收敛阶。 |
| `compare_full_trace_with_fa.py` | 验证 `full_trace` 与 FA 等价，并比较时间、峰值内存和求解规模。 |
| `verify_linear_corner_consistency.py` | 验证角点线性迹的投影、平衡、约束、恢复和能量一致性。 |
| `compare_linear_corner_with_fa.py` | 报告 `linear_corner` 相对 FA 的近似误差及时间、内存收益。 |
| `compare_fa_full_trace_linear_corner.py` | 在同一配置下统一比较三条路径。 |
| `_comparison.py` / `_convergence.py` | 共享数值实现，不是命令行入口。 |
| `_performance_process.py` | 独立进程时间与峰值 RSS 测量。 |
| `_common.py` | 默认值、划分校验和终端表格。 |
| `results_analysis.md` | 数学—代码契约、验收边界和历史证据。 |

## 两种接口路径

`full_trace` 保留子结构边界的全部位移自由度。局部内部自由度经 Schur 补精确消除，全局接口解与相同离散下的 FA 解等价。

`linear_corner` 先计算相同的精确局部 Schur 补，再以角点线性迹矩阵 $T$ 投影：

$$
K_L=T^{\mathsf T}K_sT .
$$

因此，`linear_corner` 的局部缩聚是精确的，但全局接口空间经过降阶，相对 FA 一般存在近似误差。其正确性验收检查降阶模型内部一致性，不使用与 FA 的舍入精度等价作为门禁。

## 运行

以下命令均在仓库根目录执行。每个公开脚本均支持 `--help` 查看可选参数、取值范围和默认值。`--problem` 使用实际 Problem 类名并自动确定维度；省略时采用相应脚本的二维问题。

### full_trace 收敛阶

```bash
python examples/substructure_elasticity/verify_full_trace_convergence.py \
  --problem HarmonicPoly3D --degree 1 --levels 3
```

可选参数为 `--problem`、`--degree`、`--levels`、`--solve-method` 和 `--output-dir`。理论阶为位移 $L^2$ 误差 $p+1$ 阶、位移 $H^1$ 半范误差 $p$ 阶。

### full_trace 与 FA

```bash
python examples/substructure_elasticity/compare_full_trace_with_fa.py \
  --problem FullMBBBeam3d --n-sub 12 4 4 --n-fine 4 4 4
```

位移和柔顺度相对差必须通过 `1e-11` 门禁。终端报告问题准备、刚度装配、内部消元、接口装配、直接求解、位移恢复和峰值 RSS。

### linear_corner 一致性

```bash
python examples/substructure_elasticity/verify_linear_corner_consistency.py \
  --problem FullMBBBeam3d --n-sub 6 2 2 --n-fine 4 4 4
```

检查 $K_L=T^{\mathsf T}K_sT$ 对应的 Galerkin 平衡、约束、支承恢复、内部平衡、能量和功的一致性。PASS 不表示与 FA 等价。

### linear_corner 与 FA

```bash
python examples/substructure_elasticity/compare_linear_corner_with_fa.py \
  --problem FullMBBBeam3d --n-sub 12 4 4 --n-fine 4 4 4
```

以 FA 为精度参照，报告 `linear_corner` 的位移、柔顺度误差以及时间、峰值内存和求解自由度，不对近似误差设置舍入精度门禁。

### 三路径统一比较

```bash
python examples/substructure_elasticity/compare_fa_full_trace_linear_corner.py \
  --problem FullMBBBeam3d --n-sub 12 4 4 --n-fine 4 4 4
```

三条路径使用同一网格、密度、载荷、约束、求解器和线程环境：

| 路径 | 精度职责 |
|---|---|
| FA | 正确性参照。 |
| `full_trace` | 必须与 FA 达到舍入精度一致。 |
| `linear_corner` | 必须通过降阶模型一致性检查，并报告相对 FA 的近似误差。 |

比较脚本的 `--warmup` 缺省为 `1`，表示不纳入统计的独立进程试运行对数；`--repeat` 缺省为 `5`，表示正式测量对数。二者通常不需要手动指定。`--density` 可取 `cell` 或 `uniform`。

## 性能口径

各路径分别在全新的 Python 子进程中串行运行，并交替先后顺序。每条路径都重新准备问题、装配并求解，不复用矩阵或分解。

终端显示正确性、分项时间中位数、峰值 RSS、求解自由度和本次配置下的结论；逐次数据及 `[Q25, Q75]` 保存在 JSON。

峰值内存使用 Linux/WSL `/proc/self/status` 的 `VmHWM`，包含依赖加载、问题准备、分析和原生数值库分配，不包含父进程及其他路径。时间不包含解释器启动、依赖导入、诊断、结果传输和 JSON 写入。精度 PASS 不等于性能优势，单次测量不能形成稳定性能结论。

结果默认写入脚本同级 `outputs/`。性能文件包含密度模式和 UTC 时间戳，不覆盖已有证据。

## 适用边界

- 只支持规则笛卡尔结构化张量积网格。
- 当前缩聚关系不含内部载荷项；体力或内部集中载荷需要补充载荷缩聚与非齐次位移恢复。
- 当前性能实现采用 CPU、显式接口矩阵和 SciPy 直接求解。
- 当前局部矩阵仍采用全批量装配，不适合直接试跑此前触发 OOM 的论文级规模。
