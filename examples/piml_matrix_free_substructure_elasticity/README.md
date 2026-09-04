# piml_matrix_free_substructure_elasticity

PIML 近似 x Matrix-Free 的**交集目录**: 验证 PIML 预测的缩聚刚度 `K_s_hat` 走
`InterfaceOperator` 的 Matrix-Free 路径求解时, 与显式装配路径恒等, 且相对精确解的
误差完全由预测误差主导 —— Matrix-Free 化不引入新的误差源。

## 边界 (本目录不做什么)

本目录**消费**两侧已各自闭环的结论, 只验证二者的组合:

- **预测精度本身** (代理 vs 精确 Schur 补, Cholesky 参数化, 回退门禁的推导) 归
  [`../piml_substructure_elasticity/`](../piml_substructure_elasticity/)
  的 `verify_stiffness_route.py`;
- **精确 `K_s` 上的算子正确性** (七项算子级恒等判据 + 裸 CG 端到端可替换) 归
  [`../matrix_free_substructure_elasticity/`](../matrix_free_substructure_elasticity/)
  的 `verify_matrix_free_ea.py`;
- 不验证缩聚本身的正确性 (归 `../substructure_elasticity/` 基线), 不做性能实测。

## 验证设计: 直接对比 + 归因 + 前提

`verify_piml_matrix_free.py` 在同一算例 (FullMBBBeam2d, 12x2 子结构, 每子结构
5x5 细网格) 上先在脚本内训练 Cholesky 因子代理 (默认 2000 样本 / 4000 轮 /
lr 0.005 / seed 2026, 与 `verify_stiffness_route.py` 一致), 然后:

1. **核心对比**: 同一个无预条件 CG、同一个 `InterfaceOperator` 容器, 唯一变量
   是精确 `K_s` → PIML 预测 `K_s_hat`。两次求解直接对比给出核心结果 —— 解的
   相对误差与迭代数增量。这两个量不设阈值 (大小由训练质量决定, 由
   `verify_stiffness_route.py` 一侧评价), 但受下方门禁保护;
2. **归因证据**: 同一批 `K_s_hat` 走显式装配路径求解, 与算子路径解互差应在
   `1e-13` 量级 (单次作用恒等阈值相对 `1e-13`, 附对称性与真残差检查) ——
   证明上述误差 100% 来自预测, Matrix-Free 化零贡献;
3. **前提 (SPD 证书)**: 门禁零回退 (`used_fallback` 计数为 0, 否则不再测
   PIML)、自由子空间算子逐列展开后最小特征值 > 0 (稠密特征值仅验证规模可行)、
   CG 全部收敛且无 breakdown —— PIML 预测算子确实能进 CG。

归因与前提合计 9 条门禁判据, 任一失败即退出码 1 且对比结果不可引用。

## 文件

- `verify_piml_matrix_free.py` — 统一验证脚本, 失败时退出码 1;
  `--backend {numpy,pytorch}`, 证据写入 `outputs/piml_matrix_free_{backend}.json`。
- `results_analysis.md` — 实测数值与结论的唯一事实来源。
- `outputs/` — JSON 证据文件。

## 运行

```bash
python examples/piml_matrix_free_substructure_elasticity/verify_piml_matrix_free.py
python examples/piml_matrix_free_substructure_elasticity/verify_piml_matrix_free.py --backend pytorch
```

## 下一步

- Jacobi 预条件 CG (消费 `InterfaceOperator.diagonal()`), 与精确 `K_s` 侧共享;
- 性能与规模实测 (两侧正确性均闭环后)。
