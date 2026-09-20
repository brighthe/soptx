# 精确子结构分析

本目录验证 `full_trace` 与 `linear_corner` 的正确性、收敛性和计算成本。实验参数统一登记在 `cases.toml`，通过 `run.py` 执行；结果与结论写入 [`results_analysis.md`](results_analysis.md)。

两种迹空间统一通过 `GlobalAssembler.assemble_trace_system` 装配，由 case 构造 `FullTraceBasis` 或 `LinearCornerTraceBasis` 进行选择。旧的完整接口和宏观装配方法保留为底层兼容接口。

## 目录结构

```text
analysis_capability_substructure/
├── cases.toml               工况注册表和唯一参数入口
├── config.py                工况读取与校验
├── run.py                   统一执行入口
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

二维计算成本分别使用 fa_cost_2d、full_trace_cost_2d 和 linear_corner_cost_2d。三条路径采用相同的 CantileverCorner2d、n_sub=640x320、n_fine=5x5、density=pattern_a 和 MUMPS；各执行 1 次试运行和 3 次正式测量。full_trace_reference_2d 仅补存与既有计时结果相同的全场位移证据。

二维、三维 case 均已接入验证。验证在同一 Worker 内连续使用均匀场、非均匀场 A、非均匀场 B，再恢复均匀场。更新路径复用网格、子结构编号、迹映射及同一个 `FEAStaticCondensation` 实例，每步重新计算；更新分析结束后清空 `K_s` 和 `N`，随后，重建路径按当前密度重新构造上述对象。两条路径比较局部刚度、缩聚刚度、恢复矩阵、接口刚度、完整位移和应变能。更新结果先写入临时矩阵文件，释放后再构建参照，以免两套全量状态同时驻留内存；局部缩聚和接口装配本身仍采用全量实现。由于每步主动清空数值状态，本实验验证重复计算与重建一致性，不验证自动缓存失效。

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

当前实验统一采用 Q1，收敛工况的配置固定为 `degree = 1`，不提供次数覆盖参数。

每次运行先写入 `run_config.json`，随后在同一时间戳目录保存验证结果。密度更新工况保留三份实际密度场和 JSON 证据，比较用的临时矩阵在退出前清理。新运行不覆盖已有输出。
