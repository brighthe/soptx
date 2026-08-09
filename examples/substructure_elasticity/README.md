# 经典子结构有限元与静力缩聚范例 (Substructure FE Elasticity Demo)

本目录是 2D/3D 线弹性问题的**精确子结构静力缩聚基线**。它只包含经典有限元、SIMP 局部刚度和 Schur 补代数，**不含** PIML 预测器、Matrix-Free、Krylov/GPU 或拓扑优化迭代闭环。

## 文件职责

| 文件 | 职责 |
|---|---|
| `substructure.py` | 局部 2D/3D 网格、SIMP 刚度组装、`N` 与 `K_s` 的精确计算。 |
| `assembler.py` | 全尺度参考解、接口 DOF、`K_s` Scatter-Add、接口求解与全场恢复。 |
| `minimal_demo.py` | 局部缩聚的位移恢复验证；用全尺度参考解提供接口位移，**不**求解接口系统。 |
| `compare_lagrange.py` | 端到端缩聚接口系统与 Lagrange FEM 的 2D/3D MBB 交叉验证；通过时写出 JSON 证据。 |
| `results_analysis.md` | 数学—代码映射契约、验证边界、验收阈值与证据解释。 |

## 运行与验收

```bash
cd /home/brighthe/workspace/soptx/examples/substructure_elasticity

# 局部 N 回代的细尺度位移恢复检查（悬臂梁）
python minimal_demo.py --dim 2
python minimal_demo.py --dim 3

# 真正的全局接口缩聚交叉验证（MBB 梁）
python compare_lagrange.py --dim 2 --output-dir outputs
python compare_lagrange.py --dim 3 --output-dir outputs
```

`compare_lagrange.py` 要求柔度和全节点位移的相对误差均不超过 `1e-12`；不满足时以异常失败。通过时分别生成：

- `outputs/lagrange_comparison_2d.json`
- `outputs/lagrange_comparison_3d.json`

这些 JSON 是当前可复核的数值证据；计时仅在相同硬件、依赖版本和计时边界下可比较，不能单独用于性能结论。
