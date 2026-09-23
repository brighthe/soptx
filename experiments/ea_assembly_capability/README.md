# EA 单元装配能力评测

本目录用于评测 EA（Element Assembly，单元装配无矩阵算子）在单刚缓存与算子乘应用下的内存机制、执行效率与容量能力。实验分析与结论详见 [`results_analysis.md`](results_analysis.md)。

## 文件定位

| 文件 / 目录 | 定位与职责 |
| :--- | :--- |
| [`cases.toml`](cases.toml) | 数据点注册表：声明各测量阶段（单刚缓存、算子乘等）的工况参数与产物落盘声明 |
| [`config.py`](config.py) | 配置加载模块：负责工况配置的目录级封装与静态参数校验 |
| [`run.py`](run.py) | 测量执行入口：负责调度子进程 Worker 独立执行 EA 算子构建、算子乘计时与内存测量 |
| [`compare.py`](compare.py) | 数据对比入口：汇总各工况产物，生成多规模对比表与性能看板 |
| [`walkthrough.py`](walkthrough.py) | 教学走查脚本：手工构建单元矩阵并走一遍 $y = G^T K_e G x$，检验 $K_e$ 对称且刚体模态在其零空间，并演示单元密度（缩放 $K_e^0$ 即可）与逐点密度（须重新积分）两种 `update`；不参与测量，本地 `-m tri|quad -p <阶次>` 运行 |
| [`results_analysis.md`](results_analysis.md) | 实验分析报告：记录 EA 内存演进、算子乘效率机理与容量评估结论 |
| `outputs/` | 数据产物目录：存放各工况独立运行落盘的原始 JSON 数据 |
