# 线弹性 Matrix-Free 2D/3D 实验证据报告

本报告是 `soptx/examples/matrix_free_elasticity` 数值结果的唯一事实源。
运行入口与文件职责见 [`README.md`](README.md)；每道门禁的数学式与阈值来源见
[`math_spec.md`](math_spec.md)，阈值本身只在
[`utils/contract.py`](utils/contract.py) 定义一次。

下方带 `<!-- BEGIN/END GENERATED -->` 标记的区块由
`python utils/sync_results.py --dim {2,3}` 从 `outputs/` 下的原始 JSON 生成，
**不要手工编辑**；`--check` 用于只读比对是否漂移。

## 1. 当前证据状态

**结论：目前没有可用于验收的正式证据。** 下方两个区块只是 dirty worktree 的开发
证据，且它们是在**阶段 1b 范围**（含 2-rank 算例）下产生的，而当前默认范围已收窄
为**阶段 1a（CPU 串行 EA/FA）**。重放后区块将只含单 rank 结果。

| 项 | 实际值 |
|---|---|
| 区块源 revision | `4cd4e8da17189eb57f9a68cc316bcdf189c084ec` |
| `evidence/*.json` 的 `environment.git_dirty` | **`true`** |
| 距当前 HEAD | `4cd4e8d..HEAD` 共 9 个提交 |
| 1/2-rank 一致性正式证据 | 未固化，`evidence/` 下只有单 rank 产物 |

两点说明：

1. 这两个区块生成时，`utils/sync_results.py` 既没有校验 `git_dirty`，又把
   `git_dirty=false` 当字面量写进正文，因此区块曾错误宣称自己 clean。该缺陷已
   修复：`require_formal_environment` 现在硬性拒绝 `git_dirty != false` 的原始
   JSON，渲染时也改为读取 payload 的真实标志。**在 clean revision 上重放之前，
   `utils/sync_results.py` 会以非零状态退出**，这是预期行为。
2. `4cd4e8d` 之后，`lagrange_fem_analyzer.py`、`linear_elasticity.py` 和
   `manufactured_2d.py` 都在 evidence 依赖路径上发生过改动，所以即便忽略 dirty
   标志，这两个区块也已经不对应当前代码。

因此这两个区块**待在冻结的 clean target revision 上重放**，重放前不得作为验收
结论、对外表达或申报材料的数值来源。

## 2. 历史基线

迁移到 `src/soptx` 与语义 Problem **之前**的三维数值证据保存在
[`evidence/cpu-single-rank-fa-ea-3d-historical.json`](evidence/cpu-single-rank-fa-ea-3d-historical.json)，
只作为原三维实现的历史基线，不作为本次 2D/3D 通用化实现的验收结论。

## 3. 2D CPU 单 rank FA/EA

<!-- BEGIN GENERATED: cpu-single-rank-fa-ea-2d -->

本节由 `utils/sync_results.py --dim 2` 根据 clean-revision 原始 JSON 生成；精简证据见 `evidence/cpu-single-rank-fa-ea-2d.json`。
源 revision：`4cd4e8da17189eb57f9a68cc316bcdf189c084ec`；`git_dirty=true`。

| 网格 | EA-CG 迭代数 | 真相对残差 | 相对 L2 误差 | 边界绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| `8×8` | 38 | `5.13210e-11` | `4.61057e-02` | `0` |
| `16×16` | 89 | `8.96971e-11` | `1.20318e-02` | `0` |
| `32×32` | 188 | `8.95970e-11` | `3.04605e-03` | `0` |

| 网格 | 原始 EA/FA MatVec | Dirichlet EA/FA MatVec | EA-CG/FA 直接解 |
| --- | ---: | ---: | ---: |
| `8×8` | `1.44949e-16` | `1.40930e-16` | `8.67653e-12` |
| `16×16` | `1.62572e-16` | `1.64234e-16` | `6.57436e-12` |
| `32×32` | `1.57536e-16` | `1.56086e-16` | `3.03229e-12` |

相对 L2 误差观测阶为 `1.93809`、`1.98184`。独立 FA 粗网格 `8×8` 在 38 步收敛，真相对残差为 `4.95406e-11`。

<!-- END GENERATED: cpu-single-rank-fa-ea-2d -->

## 4. 3D CPU 单 rank FA/EA

<!-- BEGIN GENERATED: cpu-single-rank-fa-ea-3d -->

本节由 `utils/sync_results.py --dim 3` 根据 clean-revision 原始 JSON 生成；精简证据见 `evidence/cpu-single-rank-fa-ea-3d.json`。
源 revision：`4cd4e8da17189eb57f9a68cc316bcdf189c084ec`；`git_dirty=true`。

| 网格 | EA-CG 迭代数 | 真相对残差 | 相对 L2 误差 | 边界绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| `4×4×4` | 24 | `9.26796e-11` | `6.80637e-01` | `0` |
| `8×8×8` | 64 | `6.69625e-11` | `2.85095e-01` | `0` |
| `16×16×16` | 134 | `8.62632e-11` | `8.91922e-02` | `0` |

| 网格 | 原始 EA/FA MatVec | Dirichlet EA/FA MatVec | EA-CG/FA 直接解 |
| --- | ---: | ---: | ---: |
| `4×4×4` | `2.71490e-16` | `1.98982e-16` | `5.77006e-11` |
| `8×8×8` | `3.17642e-16` | `2.69544e-16` | `1.93814e-11` |
| `16×16×16` | `3.38707e-16` | `2.63232e-16` | `1.98507e-11` |

相对 L2 误差观测阶为 `1.25544`、`1.67645`。独立 FA 粗网格 `4×4×4` 在 24 步收敛，真相对残差为 `9.26796e-11`。

<!-- END GENERATED: cpu-single-rank-fa-ea-3d -->

## 5. 证据边界

本节只说明**这批数字能支持什么结论**。实现层面的能力边界（不实现 PA/QA、无预
条件、EA 的算术强度口径等）由 [`math_spec.md`](math_spec.md) 第 6 节唯一维护，
此处不复制。

- **收敛阶未进入渐近区。** 3D 相对 $L^2$ 观测阶为 `1.25544`、`1.67645`，低于 P1
  元的理论二阶。当前三档网格（`4/8/16`）尚未进入渐近区，只能说"误差随加密单调
  下降且末段观测阶过门禁"，不得宣称"收敛阶达到二阶"。2D 的 `1.93809`、`1.98184`
  已接近二阶。
- **一致性结论不等于性能结论。** MatVec 与解的机器精度级一致只支持"EA 与 FA 是
  同一个离散算子"，不支持任何关于完整 solve 时间、迭代数优劣或内存占用的结论——
  本报告目前不含任何计时或峰值内存数据。
- **无跨 rank 结论。** 阶段 1a 只跑单 rank。区块内若出现 2-rank 数字，属于重放前
  的历史遗留，不得引用。
