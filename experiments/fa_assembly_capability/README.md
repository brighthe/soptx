# FA 显式装配内存机制与容量能力探针

本目录回答：**FA 显式装配中，不同单刚组装方式与总刚合并路线各自最多能算多大的问题（自由度天花板），以及什么决定了这个上限。** 结论见 [`results_analysis.md`](results_analysis.md)。

FA 是唯一"同层含多实现"的一级——把单刚拼成总刚这一步有三条合并路线
（`coalesce()` / `scipy.coo_tocsr` / 模式先行），内存占用差异巨大。本目录用
`run.py` 对三条路线做了跨规模实测，发现**换路线能让 FA 天花板提升约 10–20 倍**，
这是 FA 单层内"把计算规模弄到最大"的关键。

## 结论速览（受控单进程 CPU ｜ 47.04 GiB 内存）

**FA 内部三条合并路线的占比与天花板**：

| 路线与实现载体 | 装配范式 / 机制 | 是否物化全长三元组 | 每三元组内存 | 自由度天花板 (47G) |
| :--- | :--- | :---: | :---: | --: |
| **FEALPy `COOTensor.coalesce`** | 全长三元组张量排序（传统历史基准） | 是 | $77.2\text{ B}$ | **约 230 万** |
| **SciPy `coo_matrix.tocsr`** | 全长三元组 C++ 计数分桶 | 是 | $46.0\text{ B}$ | 约 382 万 |
| **SOPTX `CSRPattern` + 模式先行** | 预建 CSR 骨架 + 原地 scatter-add (当前生产默认) | **否** | **$4.9\sim 6.9\text{ B}$** | **约 3580 万（继续上抬）** |

FA 的墙在**拼总刚的过程**（三元组过渡态），不在总刚本身。详见 `results_analysis.md`。

## 与相邻目录的分工

| 目录 | 问的问题 |
|---|---|
| `examples/lagrange_elasticity/` | 解得对不对：$L_2$ 收敛阶、载荷等效性 |
| `examples/matrix_free_elasticity/` | FA 与 EA 两条路的算子层级与代数恒等 |
| `experiments/matrix_free_capability/` | GPU 求解耗时、逐档峰值显存与成图数据采集 |
| 本目录 (`experiments/fa_assembly_capability/`) | FA 单刚收缩与合并路线的内存机制、单价与容量天花板 |

## 文件职责

```text
experiments/fa_assembly_capability/
|-- cases.toml             # 数据点注册表: 声明各工况参数与产物落盘
|-- config.py              # cases.toml 的加载/校验 (Case 对象, 静态校验)
|-- run.py                 # 统一运行入口: 调度器 + 子进程 Worker 测量 + 美观看板
|-- compare.py             # 结果对比入口: 打印单刚/总刚对比表与全景综合报表
|-- results_analysis.md    # 学术结论报告: FA 单刚与合并路线两阶段内存占比与归因机理
`-- outputs/               # 单点原始产物 JSON (由 run.py 自动落盘管理，绘图直接读取)
```

底层测量逻辑已完全自包含集成于本目录的 [`run.py`](run.py)（Worker 模式），由 `run.py` 调度以独立子进程执行。

## 设计取舍

**测量与对比套件极致自包含。** 保持干净纯粹的文件架构，产物直接落盘于 `outputs/`。

**复用而不重写测量引擎，但两阶段分开测。** 阶段1（单刚）用 `build_serial_analyzer` 真组装（`--stage1 --method`），测 standard/voigt/fast 的真实每自由度内存——实测表明 `voigt` 因 `einsum` 多重中间张量堆叠而最耗内存（105.4 KB/dof），`fast` 最省（15.1 KB/dof）。阶段2（合并）用 `det_val` 假值测三条路线——因为真组装的 `_scalar_assembly` 本身会物化全长三元组，若套在 stage2 上会破坏 pattern 的"不物化"优势，无法对比"合并成本"本身。

## 命令

```bash
# 1. 一站式查看工况列表状态
python experiments/fa_assembly_capability/run.py --list

# 2. 跑全部工况 (测量执行)
python experiments/fa_assembly_capability/run.py --all

# 3. 跑单刚计算工况 (支持 --method fast/standard/voigt, --grid 32/80)
python experiments/fa_assembly_capability/run.py --case element-stiffness

# 4. 跑总刚合并工况 (支持 --route pattern/coalesce/scipy, --grid 32/80)
python experiments/fa_assembly_capability/run.py --case global-merge

# 5. 跑端到端全流程装配工况
python experiments/fa_assembly_capability/run.py --case full-assembly

# 6. 查看两阶段综合分析总报表
python experiments/fa_assembly_capability/compare.py --case all
```

## 运行须知

⚠️ 每个数据点必须独占一个进程。CPU 峰值取 `ru_maxrss`，是进程级高水位、无法按对象归因；同一进程里先跑一种路线再跑另一种，后者的峰值是被抬高过的。`run.py` 会自动保证每次测量独占子进程。

⚠️ 3D `standard` 是最重的一档（`hex8` 每自由度约 103 KB，天花板不到 50 万自由度），扫这一档时不要与其他大内存进程并行。
