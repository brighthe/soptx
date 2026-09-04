# matrix_free_substructure_elasticity

子结构载体的 EA Matrix-Free 算子（`InterfaceOperator`）验证目录：证明子结构缩聚得到的全局接口系统可以**不经显式装配**、以算子形式进入完整迭代求解流程，且与显式装配路径代数恒等。

## 目录边界

- **载体**：子结构（批量缩聚刚度 `K_s`，来自精确 Schur 补），不是单元。单元载体的 EA/EbE 基线见 `../matrix_free_elasticity/`。
- **不含 PIML**：`K_s` 一律取精确 Schur 补；算子对 `K_s` 来源无感这一性质由扰动批量判据覆盖，但网络训练、预测与代理精度评估全部属于 `../piml_substructure_elasticity/`。
- **不含缩聚正确性验证**：精确缩聚本身（Schur 补 vs 全阶直接解）的基线在 `../substructure_elasticity/`；本目录以缩聚结果为既定输入。
- **求解器**：`fealpy.solver.cg` 直接吃 `InterfaceOperator`（duck typing，只要求 `__matmul__`）；当前为无预条件裸 CG，Jacobi 预条件（`diagonal()`）为下一步。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `verify_matrix_free_ea.py` | 统一验证脚本：算子级七项判据（单次作用与显式装配恒等，阈值相对 1e-13）+ 求解级四项判据（裸 CG 端到端一致，显式矩阵整个拿掉）。判据定义与阈值依据见脚本模块 docstring。 |
| `results_analysis.md` | 实测数值的唯一事实源：各后端运行的判据数值、迭代数与结论。 |
| `outputs/` | 脚本写出的 JSON 证据（`matrix_free_ea_{backend}.json`），不入版本控制。 |

## 运行

```bash
python examples/matrix_free_substructure_elasticity/verify_matrix_free_ea.py
python examples/matrix_free_substructure_elasticity/verify_matrix_free_ea.py --backend pytorch
```

全部判据通过时退出码 0，任一失败退出码 1。

## 数学背景

算子定义、代数恒等论证、边界条件子空间表述与对角闭式见 dut-postdoc wiki：`concepts/matrix-free/mf-ea-substructural.md`。
