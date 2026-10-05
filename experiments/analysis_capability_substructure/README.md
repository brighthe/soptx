# 精确子结构分析

本目录验证 `full_trace` 与 `linear_corner` 的正确性、收敛性和计算成本。实验参数统一登记在 `cases.toml`，通过 `run.py` 执行；结果与结论写入 [`results_analysis.md`](results_analysis.md)。

精确子结构分析的主线流程（子结构与接口空间构建、局部刚度装配、静力缩聚、接口装配、边界处理与求解、完整位移恢复）见 [`results_analysis.md`](results_analysis.md) 第 2—3 节；本文件只登记工况与用法。

## 目录结构

```text
analysis_capability_substructure/
├── cases.toml               工况注册表和唯一参数入口
├── config.py                工况读取与校验
├── run.py                   统一执行入口
├── walkthrough.py           教学走查：接口装配求解、逐批位移恢复与全场拼接；同网格 FA 自检保留为注释
├── _corner_convergence.py   linear_corner 制造解一致性与收敛验证
├── _density_update.py       密度连续切换、重建对照与独立进程监控
├── _cost_measurement.py     FA、full_trace、linear_corner 独立进程成本测量
├── _fa_chunked.py           二维规则 Q1 网格的 FA 分块 pattern 装配
├── _performance_process.py  峰值 RSS、计时统计与测量环境采集（监控复用 _common/scheduler）
├── collect_cost_comparison.py  三条路径首个正式样本的精度对照
├── README.md
├── results_analysis.md
└── outputs/
    ├── <case-id>/<UTC timestamp>/  新入口生成的证据
    └── *.json                     重构前的历史证据
```

## 已注册工况

| task | case | 复用实现 | 判定边界 |
|---|---|---|---|
| `full_trace_convergence` | `full_trace_convergence_2d`、`full_trace_convergence_3d` | `verify_full_trace_convergence.run_convergence_benchmark` | 验证解析解收敛阶及与同网格 FA 的位移、应变能和平衡残差 |
| `linear_corner_convergence` | `linear_corner_consistency_2d`、`linear_corner_consistency_3d` | `_corner_convergence.run_linear_corner_convergence` | 验证同迹空间一致性，记录解析解误差、观测阶及相对 FA 的误差 |

`full_trace_convergence` 复用 `examples/substructure_elasticity/verify_full_trace_convergence.py`，并由统一入口开启同网格 FA 对照；示例脚本单独运行时保持原有收敛验证行为。`linear_corner_convergence` 由 `_corner_convergence.py` 执行：宏观外边界角点施加解析位移，其余边界位移由线性迹插值；同迹空间参照由全细网格 FA 刚度与全场延拓构造，并另解一次细网格 FA 作为精度参照。

密度更新一致性包含以下 case，task 均为 `density_update_consistency`。二维采用 `n_sub=8x4`、`n_fine=5x5`，三维采用 `n_sub=6x2x2`、`n_fine=4x4x4`，四个密度更新 case 均使用 MUMPS：

| case-id | problem | 接口 |
|---|---|---|
| `full_trace_density_update_2d` | `CantileverCorner2d` | `full_trace` |
| `full_trace_density_update_3d` | `FullMBBBeam3d` | `full_trace` |
| `linear_corner_density_update_2d` | `CantileverCorner2d` | `linear_corner` |
| `linear_corner_density_update_3d` | `FullMBBBeam3d` | `linear_corner` |

二维计算成本包含以下 case，task 均为 `route_cost`。四个 case 采用相同的 `CantileverCorner2d`、`n_sub=640x320`、`n_fine=5x5`、`density=pattern_a` 和 MUMPS：

| case-id | 路径 | 试运行 / 正式测量 |
|---|---|---|
| `fa_cost_2d` | FA | 1 / 3 |
| `full_trace_cost_2d` | `full_trace` | 1 / 3 |
| `linear_corner_cost_2d` | `linear_corner` | 1 / 3 |
| `full_trace_reference_2d` | `full_trace` | 0 / 1，仅补存与既有计时结果相同的全场位移证据 |

密度更新协议、成本测量口径与各工况的保存文件见 [`results_analysis.md`](results_analysis.md) 第 4—6 节。三维计算成本尚未接入：成本 Worker 当前仅支持二维。

## 使用方式

```bash
python run.py --list
python run.py --case full_trace_convergence_2d
python run.py --case linear_corner_consistency_3d
python run.py --case full_trace_density_update_2d --monitor
python run.py --case full_trace_cost_2d --monitor
python run.py --case fa_cost_2d --monitor
python run.py --case linear_corner_cost_2d --monitor
```

无参数调用只显示帮助，不启动实验。`--case all` 会先检查全部工况，再依次执行；`--output-dir` 可修改输出根目录。`--monitor` 适用于密度更新和性能工况，显示独立 Worker 的 CPU、当前 RSS、`VmHWM` 和系统可用内存；结果中的 `memory_peak_rss_bytes` 仍以 Worker 最终 `VmHWM` 为准。

`walkthrough.py` 独立于 `run.py`，不写输出文件，当前打印各步骤的关键形状、接口求解残差与柔顺度；接口求解后逐批恢复并拼接全场位移，打印恢复进度与耗时；同网格 FA 对照及峰值 RSS 输出尚未启用。走查先用 `StructuredSubstructureLayout` 建立整体有限元布局并创建参考子结构，再由 `GlobalAssembler` 组合同一布局完成后续接口与 trace 系统装配；旧的 `GlobalAssembler(domain_size, n_sub, n_fine, ...)` 构造方式仍保留。在仓库根目录运行 `python experiments/analysis_capability_substructure/walkthrough.py`；`--trace-kind`、`--n-sub`、`--n-fine`、`--chunk-size` 修改配置，`--mem-limit-gb` 限制进程地址空间（默认 35 GiB），超限时由 `MemoryError` 的 traceback 指出所在步骤。

走查中的接口刚度装配显式展示分块循环：局部装配后提取内部与边界刚度块，令 `B = K_ib Psi`，求解 `K_ii T_q = -B`，再利用线弹性刚度的对称性计算 `K_r = Psi^T K_bb Psi + B^T T_q`，最后通过已有 `CSRChunkAccumulator` 散加到全局接口系统。`linear_corner` 不构造完整 Schur 矩阵和完整恢复矩阵；`full_trace` 利用 `Psi = I` 跳过单位阵乘法。首个批次打印中间量形状，`T_q` 仅在当前批次存在。此流程要求子结构内部无载荷，与当前问题适配器的约束一致。

位移恢复显式采用单右端路径：从接口解提取 `q = A_q Q` 并展开 `u_b = Psi q`，按相同密度重装配局部刚度，直接求解 `K_ii u_i = -K_ib u_b`。此阶段不重新构造恢复矩阵或 Schur 补，恢复在主流程中显式逐批执行，复用 `layout.get_substructure_global_dofs` 获取编号并将边界与内部位移分别写回全场向量；共享接口位移按编号写回而不求和。恢复同样要求子结构内部无载荷。

恢复结果核对包括逐子结构内部平衡残差、全场形状与有限性、全部写回后的边界拼接误差、原始细网格支承位移，以及全场外力功、接口外力功、接口刚度二次型与局部刚度二次型之和的一致性。内部残差以两项分量力的范数之和归一化；边界误差以期望边界位移的最大绝对值归一化，支承误差以全场最大绝对位移归一化；功与能量差以比较双方绝对值的较大者归一化，零分母使用 float64 最小正规正数保护。二次型均为两倍应变能。诊断打印实际误差，不自动宣称精度通过；非有限结果明确报错。这些检查不替代独立参考解对照，原有执行进度输出保持保留。

当前实验统一采用 Q1，收敛工况的配置固定为 `degree = 1`，不提供次数覆盖参数。

每次运行先写入 `run_config.json`，随后在同一时间戳目录保存验证结果。密度更新工况保留三份实际密度场和 JSON 证据，比较用的临时矩阵在退出前清理。新运行不覆盖已有输出。
