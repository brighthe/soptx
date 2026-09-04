# 基于 FEALPy 的拓扑优化平台百万级算力实测与成图数据

本目录负责《中国博士后科学基金第 80 批面上资助申请书》第 6 部分研究基础中**图 10（基于 FEALPy 的张量化拓扑优化平台百万级算力与阶段性能实测）**的数据采集与成图管线。

## 目录结构

```text
experiments/topopt_capability/
├── cases.toml          # 评测用例注册表（四次运行, 指向 topopt_3d_simp_real.py）
├── config.py           # 配置与路径管理
├── provenance.py       # 环境与 Git 溯源采集
├── collect.py          # 从真实运行产物 outputs/<figure.base>/ 汇编 fig4_data.json
├── run.py              # CLI 调度入口
├── figure_data/        # 评测数据快照 (fig4_data.json)
├── results_analysis.md # 3D 悬臂梁算例定义、真实实测数据与加速机理分析报告
└── README.md           # 说明文档
```

数据来源为 `examples/topopt_platform/topopt_3d_simp_real.py` 的四次真实运行
（GPU 全收敛、GPU 单步计时、CPU 稀疏单步基线、两侧解对比），产物归档于
`outputs/topopt300/`；`fig4_data.json` 不再包含任何硬编码性能数字。

算例为博士论文算例 3.3 的加密版：60×20×4 mm 三维悬臂梁，300×100×20 网格，
191.5 万自由度，90 步收敛。CPU/GPU 两侧统一 `float64` 以构成对等对比——
`cg_rtol 1e-8` 严于 `float32` 的机器精度，GPU 侧退回单精度既无法收敛、也让
单步加速比失去意义。`collect.py` 在组装快照前会核对四份产物的网格、设计域、
求解容差与精度是否同口径，不一致直接报错。

算例定义、量纲缺陷修复始末与全部实测数字见 `results_analysis.md`，那是图 10
的唯一权威事实源。

## 常用命令

```bash
# 列出全部用例及产物状态
python experiments/topopt_capability/run.py --list

# 运行评测（四次运行全部重跑，约 50 分钟）并汇编快照
python experiments/topopt_capability/run.py --all

# 从已有产物汇编快照
python experiments/topopt_capability/run.py --collect
python experiments/topopt_capability/collect.py
```
