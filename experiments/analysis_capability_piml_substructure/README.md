# PIML 子结构分析：正确性与精度

本目录维护 PIML 子结构力学表示的正确性与精度证据，以及申报书图件（研究基础二图 5 / 证据库图 4）的数据快照。

本目录不实现数值算法，而是通过 [`cases.toml`](cases.toml) 按研究任务调度 `examples/piml_substructure_elasticity/` 下的验证脚本，校验门禁，并汇编为带环境与 Git 溯源信息的 JSON 快照 `figure_data/fig3_data.json`。

计算成本（GPU 批量缩聚加速）不在本目录判定，见 [`../piml_substructure_gpu/`](../piml_substructure_gpu/)。实测数据与逐项结论见 [`results_analysis.md`](results_analysis.md)，本文件不转录数值。

## 目录结构

```text
experiments/analysis_capability_piml_substructure/
├── cases.toml          # 工况注册表 (统一参数与外部真值引用)
├── config.py           # 注册表加载、强类型校验与产物定位
├── provenance.py       # 环境与 Git 溯源采集
├── collect.py          # 汇编数据快照并执行门禁校验
├── run.py              # CLI 统一调度入口
├── figure_data/        # 图件数据快照与切图资产 (fig3_data.json)
├── results_analysis.md # 实测数据与逐项分析报告
└── outputs/            # 时间戳隔离的实验产物
    └── <case-id>/<UTC timestamp>/  # 包含 run_config.json 与各工况 JSON 证据
```

## 工况组织

注册表的组织键是 `task`（研究任务），不是图件分格；`panels` 只是可选的成图归属标注。训练配置（`n_train` / `n_eval` / `epochs` / `lr` / `hidden_dim` / `seed`）逐项写在注册表里，不依赖脚本缺省值。产物存储自动按 `<case-id>/<UTC timestamp>/` 隔离，每次运行写入 `run_config.json`，不互相覆盖。

| task | 学习目标 | 已登记的迹空间 |
| --- | --- | --- |
| `exact_condensation_equivalence` | 无，精确缩聚真值 | — |
| `shape_function_route` | 形函数预测，局部刚度由式 (17) 变分构造 | `full_trace`、`linear_corner` |
| `reduced_stiffness_route` | 降阶刚度直接预测 | `full_trace` |

`exact_condensation_equivalence` 的实现与结论归 [`../analysis_capability_substructure/`](../analysis_capability_substructure/)。本目录声明 `source` 引用其产物作为图件 panel (a) 的输入，`run.py` 不执行该任务，`--list` 中标记为 `EXTERNAL`。

`linear_corner` × 降阶刚度直接预测这一格尚无可执行实现（`verify_stiffness_route.py` 的 `--trace-basis` 只接受 `full_trace`），因此未登记。

## 常用命令

```bash
# 查看所有注册工况及其产物状态
python experiments/analysis_capability_piml_substructure/run.py --list

# 按研究任务执行
python experiments/analysis_capability_piml_substructure/run.py --task shape_function_route

# 执行单个工况（快速解析恒等式检查，跳过训练）
python experiments/analysis_capability_piml_substructure/run.py --case shape_function_full_trace_2d --skip-train

# 执行单个工况完整训练与解层验证
python experiments/analysis_capability_piml_substructure/run.py --case shape_function_full_trace_2d

# 从已有产物汇编快照并执行门禁校验
python experiments/analysis_capability_piml_substructure/run.py --collect
```

## 迁入的研究材料

- `legacy_examples_results.md`: 原示例报告的历史归档, 不替代当前结果报告。
- `collect_ood_probe_trajectory.py`: OOD 密度轨迹采集, 配置复用示例, 新产物默认写入本实验 outputs/。
- `prototypes/plot_local_recovery.py`: 合成预测场的云图版式原型, 不作为力学精度证据。

