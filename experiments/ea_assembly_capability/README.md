# EA 单元装配能力评测

本目录用于评测 EA（Element Assembly，单元装配无矩阵算子）在单刚缓存与算子乘应用下的内存机制、执行效率与容量能力。实验分析与结论详见 [`results_analysis.md`](results_analysis.md)。

## 文件定位

| 文件 / 目录 | 定位与职责 |
| :--- | :--- |
| [`cases.toml`](cases.toml) | 数据点注册表：声明各测量阶段（单刚缓存、算子乘等）的工况参数与产物落盘声明 |
| [`config.py`](config.py) | 配置加载模块：负责工况配置的目录级封装与静态参数校验 |
| [`run.py`](run.py) | 测量执行入口：负责调度子进程 Worker 独立执行 EA 算子构建、算子乘计时与内存测量 |
| [`compare.py`](compare.py) | 数据对比入口：汇总各工况产物，生成多规模对比表与性能看板 |
| [`walkthrough.py`](walkthrough.py) | 教学走查脚本：在同一网格上依次走标准 EA 与参考 EA 的 setup / update / apply，单元矩阵一律用 fast 装配。标准 EA 常驻 $K_e$，`update` 带新系数重新积分，apply 走 $y = \sum_e G_e^T K_e G_e x$；参考 EA 让同一平移类的单元共用一份参考单元矩阵，常驻 $N_k$ 份 $K_k^0$ 与 $s_e$，`update` 只换 $s_e$，apply 走 $y = \sum_e s_e G_e^T K_{k(e)}^0 G_e x$，依赖 `from_box` 编号约定（分析器在单元密度下用同一个类但取 $N_k = N_C$，直接引用 $K_e^0$）；每段末尾核对手工 apply 与 `@` 一致；不参与测量，本地 `--mesh tri|quad|tet|hex -p <阶次>` 运行 |
| [`results_analysis.md`](results_analysis.md) | 实验分析报告：按 setup / apply / update 记录 EA 的内存机制、算子乘效率与容量评估结论 |
| `outputs/` | 数据产物目录：存放各工况独立运行落盘的原始 JSON 数据 |
