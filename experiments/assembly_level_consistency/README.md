# 矩阵组装层级一致性

本目录验证 `fa`、`ea`、`pa`、`ua` 四个矩阵组装层级的正确性、收敛性和计算成本。实验参数统一登记在 `cases.toml`，通过 `run.py` 执行；结果与结论写入 [`results_analysis.md`](results_analysis.md)。

四个层级统一通过 `soptx.fem.levels.create_level` 按层级名构造，即 `FullAssembly`、`ElementAssembly`、`PartialAssembly`、`UnassembledAssembly` 四个生产层级类本身。`fa` 常驻全局 CSR，配 MUMPS 直接解；`ea` 常驻逐单元 `K_e`，`pa` 只常驻逐积分点的 `J^{-1}_q` 与 `w_q detJ_q`，`ua` 连这些也不留、每次作用现算，三者都不组装全局矩阵，直接解法无从分解，只能配无预条件 CG。本目录不自产数值代码，全部工况以子进程复用 `examples/lagrange_elasticity/manufactured_convergence_demo.py`，`run.py` 只是调度器。

`LA` 层级尚无工况，它要多 rank 才有意义，而本目录整套是单线程；`ua` 的 `tet` 一格也暂缺。两处原因都记在 `cases.toml` 末尾。`../_common/assembly_levels.py` 里的 `stored-b` 与 `shared-ke` 不是分类法中的层级，由 [`../assembly_level_capability/`](../assembly_level_capability/) 自行取证。

## 目录结构

```text
assembly_level_consistency/
├── cases.toml           工况注册表和唯一参数入口
├── config.py            工况读取与校验，单线程环境注入
├── run.py               统一执行入口，按注册表转调 examples 下的脚本
├── compare.py           打印逐档 L2 误差与实测阶，--strict 可作门禁
├── plot_meshes.py       绘制四族网格最粗一档的剖分图
├── README.md
├── results_analysis.md
├── figure_data/         网格剖分图的 svg 与 png
└── outputs/             单点 JSON 产物
```

## 已注册工况

行是 case id，列是 `[cases.mesh.*]` 子表；一个 (case, 网格) 组合才是一个数据点，对应一个进程、一个产物、`--list` 里的一行。

| case | `quad`（2D Q1） | `tri`（2D P1） | `hex`（3D Q1） | `tet`（3D P1） |
|---|---|---|---|---|
| `convergence_fa` | 4→64，5 档 | 4→64，5 档 | 4→32，4 档 | 2→32，5 档 |
| `convergence_ea` | 4→64，5 档 | 4→64，5 档 | 4→32，4 档 | 2→32，5 档 |
| `convergence_pa` | 4→64，5 档 | 4→64，5 档 | 4→32，4 档 | 2→32，5 档 |
| `convergence_ua` | 4→64，5 档 | 4→64，5 档 | 4→32，4 档 | — |

四条链同档、同制造解、同装配路径（`--assembly-method fast`），只换 `--operator-level`，因此可以逐档并排读。一条链同时给出两类证据：纵向是该层级自己对制造解的 $L^2$ 收敛阶，横向是同一档上各层级的解与 CG 迭代数。两类缺一不可，只有纵向则装配核出错时各条链会一致地收敛到同一个错误答案，只有横向则同源的错误也会一致通过。

判读时先横向对迭代数，再看收敛阶。`ea` 与 `pa` 是同一离散算子的两种存法，对同一初值与同一右端张成同一个 Krylov 子空间，迭代序列因而逐步相同，只差舍入，同一档上两者的迭代数应当整数相等；这比收敛阶灵敏得多，某档对不上即是被测层级的算子有问题，不是求解器的随机性。

`ua` 对 `pa` 的判据还要更硬。`ua` 每次作用现调一次 `PartialAssembly.build` 造出一个 `pa` 算子、作用完即弃，内核构造与数据流走的都是 `pa` 那一份代码，浮点运算次序字面上一致，只差在 `pa` 建表时算一次、`ua` 每次作用现算，于是 $L^2$ 误差列应当逐位相同，而不只是吻合到求解容差。哪怕末位有差，都说明两条路径已经岔开。

`hex` 从 `n = 4` 起步而不是 2。`n = 2` 时 27 个节点里 26 个在边界，只剩正中心一个内部节点，体力按对称性积分为零，`||F||` 落在舍入量级，相对残差成为 0/0；`tet` 的 `n = 2` 不退化，48 个单元打破了对称性，故保留。

## 使用方式

```bash
python run.py --list
python run.py --all --check-only
python run.py --all
python run.py --case convergence_fa --mesh hex
python compare.py --strict
```

不给 `--mesh` 就跑该工况注册的全部网格；`--output-dir` 可修改输出根目录。`compare.py --strict` 在缺产物或收敛阶不达标时返回非零退出码，可直接接进 CI。

`fa` 四条链需要 PyMUMPS。整套 15 个数据点约 7 分钟，其中 `convergence_pa` 占 4.5 分钟、`convergence_ua` 占 1.8 分钟，日常回归可先跑 `--case convergence_fa convergence_ea`。

3D 迭代链不宜再加一档。相对残差逐档涨 2~3 倍，末档 `hex` 到 `3.2e-11`、`tet` 到 `3.6e-11`，而上游脚本把 `RESIDUAL_TOLERANCE` 写死在 `1e-10`；要往 `n = 64` 走，需先启用 `--preconditioner jacobi` 或将 `--rtol` 再压一档，否则会误判成 `ea`、`pa` 出错。

产物文件名由上游脚本按维度、网格、制造解、次数、求解器、装配路径和算子层级自行拼接，本目录只能用 `--output-dir` 指定目录，因此 `cases.toml` 中必须显式写出 `artifact`，`config.py` 会强制这一点。case id 固定为 `<panel>_<scheme>` 两段，网格不进 id。
