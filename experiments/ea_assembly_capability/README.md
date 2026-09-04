# EA 单元装配无矩阵算子与规模能力 (EA Assembly Capability)

本目录为 SOPTX 中专门用于测量与评估 **EA (Element Assembly / Element-by-Element，单元装配无矩阵算子)** 内存机制、单价模型与规模容量极限的实验套件。

---

## 目录职责

```text
experiments/ea_assembly_capability/
|-- cases.toml             # 数据点注册表: 声明各工况参数与产物落盘
|-- config.py              # cases.toml 的加载/校验 (Case 对象, 静态校验)
|-- run.py                 # 统一运行入口: 调度器 + 子进程 Worker 测量 + 美观看板
|-- compare.py             # 结果对比入口: 打印 EA 阶段对比表与 FA vs EA 全景总报表
|-- results_analysis.md    # 学术结论报告: EA 单元缓存、MatVec 瞬态与容量天花板归因
`-- outputs/               # 单点原始产物 JSON (由 run.py 自动落盘管理，绘图直接读取)
```

底层测量逻辑已完全自包含集成于本目录的 [`run.py`](run.py)（Worker 模式），由 `run.py` 调度以独立子进程执行。

---

## 常用命令

```bash
# 1. 一站式查看已注册工况列表
python experiments/ea_assembly_capability/run.py --list

# 2. 跑全部工况 (测量执行)
python experiments/ea_assembly_capability/run.py --all

# 3. 跑阶段 1: 单元刚度张量缓存 (支持 --method fast/standard/voigt, --grid 32/80)
python experiments/ea_assembly_capability/run.py --case element-cache

# 4. 跑阶段 2: 算子乘积 MatVec 耗时与吞吐测试 (支持 --device cpu/cuda)
python experiments/ea_assembly_capability/run.py --case ea-matvec --device cuda

# 5. 跑阶段 3: 端到端 CG 线性求解端到端容量测试
python experiments/ea_assembly_capability/run.py --case ea-cg-solve --device cuda

# 6. 查看 FA vs EA 跨层级全景分析总报表
python experiments/ea_assembly_capability/compare.py --case all
```
