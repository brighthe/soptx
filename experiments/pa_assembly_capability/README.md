# PA 部分装配能力评测

本目录用于评测 PA（Partial Assembly，部分装配无矩阵算子）在积分点数据缓存与算子乘应用下的内存机制、执行效率与容量能力。实验分析与结论详见 [`results_analysis.md`](results_analysis.md)。

## 文件定位

| 文件 / 目录 | 定位与职责 |
| :--- | :--- |
| [`cases.toml`](cases.toml) | 数据点注册表：声明各测量阶段（几何缓存、算子乘、求解等）的工况参数与产物落盘声明 |
| [`config.py`](config.py) | 配置加载模块：负责工况配置的目录级封装与静态参数校验 |
| [`run.py`](run.py) | 测量执行入口：负责调度子进程 Worker 独立执行 PA 算子构建、算子乘计时与内存测量 |
| [`compare.py`](compare.py) | 数据对比入口：汇总各工况产物，生成多规模对比表与性能看板 |
| [`results_analysis.md`](results_analysis.md) | 实验分析报告：记录 PA 内存演进、算子乘效率机理与容量评估结论 |
| `outputs/` | 数据产物目录：存放各工况独立运行落盘的原始 JSON 数据 |
