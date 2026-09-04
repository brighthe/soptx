# PIML 近似 Matrix-Free 验证结果分析

> 状态: **已完成** (2026-08-26, numpy 与 pytorch 双后端全部判据通过)。
> 本文件是本目录实测数值与结论的唯一事实来源; 判据设计与边界见 [README.md](README.md)。

## 1. 实验设置

- 算例: FullMBBBeam2d, domain (0, 12, 0, 2), 12x2 子结构 (共 24 个), 每子结构 5x5 细网格
- 接口自由度: 574 (含约束), 自由接口自由度 n_free = 571
- 密度场: 光滑正弦族, 取值区间 (0.3, 1.0)
- 代理训练: 2000 组随机密度 / 4000 轮 / lr 0.005 / seed 2026 (脚本内训练)
- CG 停机: rtol 1e-10, maxiter 20000, 无预条件
- 证据文件: `outputs/piml_matrix_free_numpy.json`, `outputs/piml_matrix_free_pytorch.json`

## 2. 判据结果 (全部通过)

| 判据 | 阈值 | numpy | pytorch |
|---|---|---|---|
| **SPD 证书** | | | |
| 门禁回退计数 | = 0 | 0 | 0 |
| 自由子空间最小特征值 > 0 | — | 通过 (2.439e-05) | 通过 (2.243e-05) |
| 两条 CG 收敛且无 breakdown | — | 通过 | 通过 |
| **恒等判据** | | | |
| K_s_hat 算子作用 vs 显式装配 | < 1e-13 | 3.528e-16 | 3.291e-16 |
| 对称性 x^T A y vs y^T A x | < 1e-13 | 2.622e-15 | 4.499e-15 |
| **求解级判据** | | | |
| 算子路径 CG 真残差 | ≤ 1e-9 | 8.997e-11 | 8.584e-11 |
| 显式路径 CG 真残差 | ≤ 1e-9 | 8.820e-11 | 9.011e-11 |
| 两条 CG 迭代数之差 | ≤ 1 | 1 (349 vs 348) | 0 (357 vs 357) |
| 两条 CG 解相互一致 | ≤ 1e-8 | 1.064e-13 | 7.180e-14 |

## 3. 信息量观测 (不设阈值)

| 观测量 | numpy | pytorch |
|---|---|---|
| 最终训练 MSE | 3.494e-05 | 2.256e-05 |
| PIML 算子路径解 vs 精确 Matrix-Free 解 | 3.474e-02 | 5.119e-02 |
| PIML 显式路径解 vs 精确 Matrix-Free 解 | 3.474e-02 | 5.119e-02 |
| PIML K_s_hat 下 CG 迭代数 (算子/显式) | 349 / 348 | 357 / 357 |
| 精确 Matrix-Free (K_s) CG 迭代数 | 343 | 341 |

参照解为同一无预条件 CG 在精确 K_s 算子上的解 —— 与 PIML 侧唯一的变量是
缩聚器 (K_s → K_s_hat), 求解器与算子容器完全相同。

三点解读:

1. **误差完全由预测主导**: 两条路径相对精确 Matrix-Free 解的误差在每个后端内**逐位相同**
   (numpy 3.474e-02, pytorch 5.119e-02), 互差仅 1e-13 量级 —— Matrix-Free 化对
   解的贡献为零, 整条链路的误差可全部归因于代理预测。误差为百分量级, 与
   `../piml_substructure_elasticity/results_analysis.md` §9 记录的路线 B 直接
   预测水平 (接口位移约 2%) 同量级; 两后端及与该目录的数值不严格相等是预期内的:
   训练随机源不同 (numpy 后端采样走 `bm.random`, pytorch 后端走 torch RNG,
   且各脚本训练前的随机数消耗顺序不同), 同 seed 下得到不同的网络。注意 pytorch
   侧训练 MSE 更低 (2.256e-05 vs 3.494e-05) 而部署误差更高 (5.12% vs 3.47%):
   训练损失度量随机训练分布上的拟合, 部署误差在特定的光滑正弦密度场上评估,
   两者不单调对应。
2. **SPD 证书实测拿到**: 双后端零回退, 自由子空间最小特征值约 2.2e-05 ~ 2.4e-05
   > 0, CG 无 breakdown —— Cholesky 参数化的预测算子装配加约束后确实正定,
   可直接喂迭代法。这是精确侧不需要、本目录特有的新结论。
3. **谱扰动温和**: 近似 K_s_hat 下 CG 比精确 K_s 多 6 ~ 16 步 (约 2% ~ 5%),
   预测误差对 Krylov 收敛的影响很小, 对后续预条件与性能路线是有利信号。
   精确 K_s 的参照迭代数 (343 / 341) 与
   `../matrix_free_substructure_elasticity/results_analysis.md` 的既有记录一致。

## 4. 结论

1. PIML 预测的 K_s_hat 经 `InterfaceOperator` 的 Matrix-Free 路径与显式装配路径
   **代数恒等** (单次作用 1e-16 量级, 整条 CG 链路解互差 1e-13 量级), "近似 x
   Matrix-Free" 的组合不引入任何新误差源;
2. 预测算子的 **SPD 证书**成立 (零回退 + 最小特征值为正 + CG 无 breakdown),
   PIML 输出可直接进入无预条件 CG;
3. 结合两侧前置结论 (预测精度归 `../piml_substructure_elasticity/`, 精确算子
   正确性归 `../matrix_free_substructure_elasticity/`), **PIML 近似 Matrix-Free
   的正确性验证闭环**;
4. 边界: 仅正确性, 无性能实测; 单算例、验证规模 (n_free = 571, 稠密特征值仅在
   此规模可行); 无预条件; 结论绑定本页训练配置与密度族, 换分布需重验 (回退门禁
   是运行时保险)。

## 5. 下一步

- Jacobi 预条件 CG (消费 `InterfaceOperator.diagonal()`), 与精确 K_s 侧共享;
- 性能与规模实测。

## 6. 运行环境备注

- 修复记录: `soptx.fem.substructure.case_setup.set_random_seed` 原实现无条件调用
  `bm.random.seed(seed)`, 在 pytorch 后端下崩溃 (`bm.random` 映射为
  `torch.random`, 其 `seed()` 无参); 现只在 numpy 后端调用, pytorch 后端统一由
  `torch.manual_seed` 固定。numpy 后端行为不变。
