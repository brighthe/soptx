# 子结构静力缩聚示例 (Substructure Elasticity)

本目录包含线弹性子结构静力缩聚（Schur 补）的调用 `src/soptx` 公共接口的验证脚本：

| 文件 | 说明 |
|---|---|
| [`verify_full_trace_convergence.py`](verify_full_trace_convergence.py) | **制造解收敛阶验证**：在嵌套加密网格上验证完整接口缩聚位移误差在 $L^2$ 范数与 $H^1$ 半范下的先验收敛阶。 |
| [`verify_linear_corner_consistency.py`](verify_linear_corner_consistency.py) | **角点线性迹一致性验证**：验证角点线性迹降阶系统在平衡方程、边界约束、全场恢复及能量虚功守恒上的代数自洽性。 |

> **说明**：大规模 3D 性能基准、界面局域全互联分析与学术证据报告已移至 [`experiments/analysis_capability_substructure/`](../../experiments/analysis_capability_substructure/)。

## 实现边界

子结构构造、静力缩聚、接口装配、角点投影、约束求解与位移恢复由 `src/soptx/fem/substructure/` 提供。示例负责工况设置、接口组装调用、独立误差检查与结果输出，不维护另一份投影或约束求解算法。

