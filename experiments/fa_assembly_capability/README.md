# FA 显式装配能力评测

本目录用于评测 FA（Full Assembly，显式装配）在不同单刚计算方法与总刚生成路线下的内存机制与容量天花板。实验分析与结论详见 [`results_analysis.md`](results_analysis.md)。

## 文件定位

| 文件 / 目录 | 定位与职责 |
| :--- | :--- |
| [`cases.toml`](cases.toml) | 数据点注册表：声明各测量工况的参数配置与产物落盘声明 |
| [`config.py`](config.py) | 配置加载模块：负责 `cases.toml` 的解析与静态参数校验 |
| [`run.py`](run.py) | 测量执行入口：负责调度子进程 Worker 独立执行内存与耗时测量 |
| [`compare.py`](compare.py) | 数据对比入口：汇总各工况产物，生成阶段对比表与全景综合报表 |
| [`results_analysis.md`](results_analysis.md) | 实验分析报告：记录阶段内存演进、峰值归因机理与容量评估结论 |
| `outputs/` | 数据产物目录：存放各工况独立运行落盘的原始 JSON 数据 |
