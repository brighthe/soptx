# Hu–Zhang topology optimization paper experiment

状态：`common-v1 configured / forward-manufactured formal-capable / evidence pending user run`。

`legacy_drivers/` 是旧论文和博士论文章节运行脚本的迁移快照，不进入新的执行路径。
旧结果只用于恢复问题定义和参数，不能作为投稿证据。

`matrix.toml` 对 7 个论文 driver 中的全部 36 个 variant selector 建立显式清单，
并为每项记录 chapter、dimension、role、method、输出名称和阻塞状态。另一个不含
selector 的字体/有限差分工具脚本被显式列为 excluded source。

`dry_run.py` 不导入、不执行 legacy driver；它只验证历史清单：

- TOML schema、stage 和 `incubating` 状态；
- 所有 legacy Python 源文件均被纳入或显式排除；
- driver SHA-256 未漂移；
- AST 中发现的 selector 与 matrix 恰好一一对应；
- case id、输出名称、chapter、dimension、role 和阻塞状态合法。

在仓库根目录运行：

```powershell
python .\experiments\huzhang_topopt_paper\dry_run.py --json
```

dry-run 通过仅表示静态清单完整，不表示任何 Hu–Zhang 数值算例已经可以运行或通过
验证。

## 公共 v1 入口

`cases.toml` 是数学与力学投稿路线共享的最小证据矩阵，固定登记：

- 带非齐次牵引的制造解；
- 低阶稳定化与角点松弛消融；
- 柔顺度和应力约束灵敏度检查；
- 两端固支梁一般柔顺度；
- 二维轴承近不可压缩问题；
- 有限宽度载荷悬臂梁应力约束问题；
- 冻结设计的统一高阶复核。

`run.py` 只使用当前 `soptx` 命名空间，并提供三类入口：

```powershell
python .\experiments\huzhang_topopt_paper\run.py --list
python .\experiments\huzhang_topopt_paper\run.py --check-only --json
python .\experiments\huzhang_topopt_paper\run.py --case forward-manufactured --output .\experiments\huzhang_topopt_paper\outputs\forward-manufactured --json
```

当前只有 `forward-manufactured` 已接入原生数值 adapter。它固定运行
`k=1,2,3,4`、`uniform-tri` checkerboard 和 `irregular-star-refined` 两类三角网格、
每类五个加密层，因此 `metrics.csv` 必须有 40 行。每行记录位移/应力 DOF、四类
误差、实际网格尺度、残差、矩阵对称性缺陷、归一化法向迹跳量、墙钟时间和 Python
traced peak；直接解法的迭代数显式为 `N/A`。

`k=3,4` 的两个网格族均执行正式门禁：最后三个细网格上的观测阶不得低于理论阶减
`0.35`。位移 $L^2$、`div` 和 $H(div)$ 的理论阶为 `k`，应力 $L^2$ 为 `k+1`。
`k=1,2` 明确只作 `characterization`，因为当前混合边界并不完全属于低阶定理的原始
设定。所有行还必须满足残差 `<=1e-8`、`J_n<=1e-10`、对称性缺陷 `<=1e-12`，且
不得出现缺失值、NaN 或求解失败。高阶依据 Hu (2015, Theorem 3.2 与 Remark 3.1)，
低阶参考 Chen–Hu–Huang (2017, Theorem 3.3)。

该 executor 的一次通过运行在 dirty revision 上仍只产生 development evidence；只有
`soptx` 与实际 `fealpy` checkout 都 clean，且 `summary.json` 的
`acceptance_status=passed` 时 runner 才会标记 formal evidence。其余六个 case 已有完整
配置、指标和 blocker；直接执行会生成 `blocked` manifest，并以退出码 2 结束，不会调用
legacy driver。

每个运行目录包含：

- `manifest.json`：case、acceptance、Git/依赖 provenance 和产物 SHA-256；
- `summary.json`：运行状态和门禁摘要；
- `metrics.csv`：逐变体指标；
- `history.csv`：优化历史；
- `figures/`：图件目录。

全局 `.gitignore` 已忽略 JSON、CSV 和 PNG，开发运行不会把产物误加入版本控制。

## Evidence 状态

- `historical`：博士论文或 legacy driver 的旧结果；
- `development`：dirty revision、环境不完整或 acceptance 未通过；
- `validated`：数值与 acceptance 通过，但尚未在 clean revision 形成正式记录；
- `formal`：`soptx` 与本地 `fealpy` 均为 clean revision，参数、环境和产物哈希完整。

当前目录不得声明为 `validated` 或 `formal`。

整个公共矩阵晋级条件：

1. 为六个 `configured` case 补齐当前命名空间的原生 adapter；
2. 由用户按明确环境逐条运行，检查退出码、40 行产物和 acceptance；
3. 在 clean revision 上以 revision 命名目录重跑，记录环境、参数和产物 SHA-256；
4. 只有全部公共门禁通过后，才根据证据选择数学或力学投稿路线。
