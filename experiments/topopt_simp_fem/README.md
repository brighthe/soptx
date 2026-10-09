# 传统有限元拓扑优化: EA 与 FA 对照

本目录验证传统有限元 (Q1 六面体, Lagrange 位移元) 下 EA 与 FA 两种刚度算子层级的拓扑优化结果一致性与计算代价。两条流程除算子层级外完全相同, 算例为 Huang2023 的三维 MBB 梁首档。

| 文件 | 内容 |
|---|---|
| [`run_ea.py`](run_ea.py), [`README_ea.md`](README_ea.md) | EA: 一份参考单元刚度乘逐单元系数, 不形成全局矩阵 |
| [`run_fa.py`](run_fa.py), [`README_fa.md`](README_fa.md) | FA: 全局 CSR 稀疏矩阵 |
| [`results_analysis.md`](results_analysis.md) | 两侧结果一致性与耗时、内存对照 |

结果写入 `outputs/{ea,fa}_hex_390x65x65_cg/`。两份 README 与脚本后续再考虑合并。
