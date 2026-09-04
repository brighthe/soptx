# PIML 子结构力学表示与缩聚加速能力验证

本目录维护 PIML 子结构力学建模与 GPU 加速能力的实验管线与成图数据快照（对应申报书研究基础二图 5 / 证据库图 4）。

本目录不直接实现数值算法，而是通过 [`cases.toml`](cases.toml) 调度底层算例脚本，校验门禁，并统一汇编为带环境与 Git 溯源信息的 JSON 数据快照 `figure_data/fig3_data.json`。

## 目录结构

```text
experiments/piml_capability/
├── cases.toml          # 评测用例注册表 (命令与产物映射)
├── config.py           # 配置与路径管理
├── provenance.py       # 环境与 Git 溯源采集
├── collect.py          # 汇编数据快照并执行门禁校验
├── run.py              # CLI 调度入口
├── figure_data/        # 评测数据快照与高清切图 (a, b, c, d)
└── results_analysis.md # 数学机理、实测数据与逐格分析报告
```

## 四格实验内容

1. **(a) 缩聚正确性（真值对不对）**：验证 2D/3D 静力缩聚求解与全装配直接求解的代数等价性（误差处于浮点舍入量级 $\sim 10^{-12}$）。
2. **(b) 变分二阶误差响应（机理通不通）**：验证基于多尺度形函数的缩聚刚度变分构造机理（Huang 2023 式 17），实测误差二次方压缩（斜率 2.00，形函数误差 $8.97\% \to$ 刚度误差 $0.44\%$）。
3. **(c) 全局结构求解精度（全局准不准）**：在 24 子结构装配系统（FullMBBBeam2d）上对比形函数路线与直接预测刚度路线的解层误差（全场位移误差 $0.15\%$ vs $2.01\%$）。
4. **(d) GPU 批量张量缩聚加速（算得快不快）**：在相同 PIML 算法下对比 CPU 与 GPU (RTX 5080) 批量推断与 GEMM 重构耗时（实测单卡硬件加速比 $22\sim 26$ 倍）。

## 常用命令

```bash
# 查看所有注册用例及其产物状态
python experiments/piml_capability/run.py --list

# 执行全套实验并生成数据快照
python experiments/piml_capability/run.py --all

# 从已有产物汇编快照并执行门禁校验
python experiments/piml_capability/run.py --collect

# 独立运行特定分格
python experiments/piml_capability/run.py --panel a
```
