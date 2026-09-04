# 四范式线弹性受控对比实验

本目录在同一线弹性问题、同一细网格与同一参考真解下比较四条求解路径。四条路径填满一张 2×2 定位表：

| 计算载体 | 精确数值 | 神经网络代理 |
|---|---|---|
| 全局解场 $u(x)$ | `lagrange` | `pinn` |
| 缩聚算子 $(\mathbf N, \mathbf K_s)$ | `substructure` | `piml` |

物理模型位于 `src/soptx/problems/elasticity/`，求解能力位于 `src/soptx/fem/` 与 `src/soptx/ml/`；本目录只通过 `cases.toml` 冻结算例与比较契约，并调用核心库的公共接口，不含任何求解实现。

## 为什么在 `experiments/` 而不是 `examples/`

[`CLAUDE.md`](../../CLAUDE.md) 规则 1 规定 example 目录不能交叉污染，「新技术栈 → 新目录」。本实验横跨四个技术栈，放进任何一个 `examples/` 子目录都会违规——这种污染已经发生过一次：[`examples/piml_substructure_elasticity/compare_piml_pinn.py`](../../examples/piml_substructure_elasticity/compare_piml_pinn.py) 位于 PIML 目录却自带一份 `PINNElasticityNet`，与 [`examples/pinn_elasticity/minimal_demo.py`](../../examples/pinn_elasticity/minimal_demo.py) 的同名实现重复。

`examples/` 承担单技术栈的入口演示，`experiments/` 承担论文级、多方法、冻结算例的证据流水线，目录约定对齐既有的 [`experiments/huzhang_topopt_paper/`](../huzhang_topopt_paper/)。

## 目录结构

```text
experiments/elasticity_paradigm_comparison/
|-- cases.toml            # [protocol] 共享冻结项 + [[cases]] 算例注册表
|-- config.py             # TOML 加载、契约校验与摘要格式化
|-- metrics.py            # 四条路径共用的唯一度量实现
|-- run.py                # CLI、上下文组装与调度主入口
|-- solvers/
|   |-- base.py           # ExperimentContext / SolveResult / ParadigmSolver 契约
|   |-- lagrange.py       # (全局解场, 精确)
|   |-- pinn.py           # (全局解场, 代理)
|   |-- substructure.py   # (缩聚算子, 精确)
|   `-- piml.py           # (缩聚算子, 代理)
`-- outputs/              # 运行产物, 不提交
```

## 受控比较契约

`cases.toml` 的 `[protocol]` 表冻结四条路径共享的口径，任何一项变化都使已落盘的 `manifest.json` 失效，必须同时提升 `stage` 版本号：

- **参考真解**：全部误差以同一细网格上的 `lagrange` 全装配解为基准；
- **共享细网格**：细网格由 `n_sub × n_fine` 张成，四条路径不得各用各的离散；
- **PINN 硬约束边界**：`pinn_boundary_mode = "hard"`，`config.py` 强制校验。软约束下 $w_\text{int}/w_\text{bnd}$ 权重足以改变一个数量级的误差，会把范式对比退化成调参对比；
- **计时分离**：`offline_seconds` 与 `online_seconds` 分别报告，不合并。

## 算例分档

| case id | tier | 角色 |
|---|---|---|
| `tier-a-mbb-homogeneous` | A | 均质 MBB 梁，四条路径共同可解，比较精度—成本 |
| `tier-b-mbb-simp-snapshot` | B | SIMP 密度快照，材料对比度扫描，比较适用边界与结构保持退化 |

两档必须成对报告。只跑 Tier B 时 PINN 因应力在单元界面不连续而失效，单独给出构成打靶；只跑 Tier A 则测不出 PIML 的结构保持边界。

算例参数沿用 `verify_stiffness_route.py` 已冻结的设置（`FullMBBBeam2d`，域 `12 × 2`，`n_sub = 12×2`，`n_fine = 5×5`），与既有证据可比。

## 命令

```bash
python experiments/elasticity_paradigm_comparison/run.py --list
```

```bash
python experiments/elasticity_paradigm_comparison/run.py --case tier-a-mbb-homogeneous --check-only
```

```bash
python experiments/elasticity_paradigm_comparison/run.py --case tier-a-mbb-homogeneous --method all
```

## 当前状态

框架已建立，四个适配器与上下文组装均未接线，`--list` 与 `--check-only` 可用，求解调用返回退出码 `2` 并给出阻塞原因。逐项阻塞与解除顺序见 [`results_analysis.md`](results_analysis.md) §3。
