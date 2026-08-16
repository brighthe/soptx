# Hu--Zhang 拓扑优化投稿论文复现实验

本目录保存投稿论文的可执行拓扑优化算例。物理模型位于 `src/soptx/problems/elasticity/`, 本目录只通过 `cases.toml` 选择模型、配置离散与优化参数, 并调用当前 `soptx.topology` 的公共接口。

## 多算例结构

```text
experiments/huzhang_topopt_paper/
|-- cases.toml                 # [[cases]] 算例参数注册表
|-- run.py                     # 通用 CLI 与调度主入口
|-- config.py                  # TOML 配置加载、校验与参数拍平
|-- diagnostics.py             # 能量恒等式诊断、残差计算与格式化输出
|-- fixed_fixed_beam.py        # FixedFixedBeamCenterLoad2d 专属组装器
|-- legacy/
|   `-- test_phd_section5.py   # 历史参数参照, 不作为运行入口
`-- outputs/                   # 运行产物, 不提交
```

每个 `[[cases]]` 必须包含三类独立参数:

- `[cases.model]`: `name` 和 `parameters`, 选择 `soptx.problems` 中的物理模型并给出载荷、材料、平面假设等参数。
- `[cases.discretization]`: 网格、受控比较阶次 `comparison_orders`、角点松弛和线性求解器。
- `[cases.optimization]`: 体积分数、过滤器、材料插值和 OC 迭代参数。

当前注册的模型为 `FixedFixedBeamCenterLoad2d`, 对应 case id `compliance-fixed-fixed`。新增论文算例时, 先在核心库实现模型, 再增加 `[[cases]]`，最后在 `run.py` 的 `MODEL_BUILDERS` 注册相应组装器。未注册模型会明确报错, 不会隐式复用固定梁参数。

## 命令

```bash
# 列出模型和 case id
python experiments/huzhang_topopt_paper/run.py --list

# 只校验指定 case 的配置和组合, 不运行优化
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed --method all --check-only

# 固定初始密度下的 LFEM/Hu--Zhang 单次状态对比, 不更新设计变量
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed --method all \
  --mode state-compare --solver scipy

# 单条 Hu--Zhang 计算链
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed --method huzhang --order 3

# 全部论文离散组合
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed --method all
```

临时调试可覆盖网格、迭代次数和求解器, 覆盖结果不能作为默认投稿证据:

```bash
python experiments/huzhang_topopt_paper/run.py \
  --case compliance-fixed-fixed --method huzhang --order 2 \
  --nx 8 --ny 2 --max-iterations 1 --solver scipy
```

优化模式的每个组合写入 `outputs/<case-id>/<method>-k<order>/`，包含 `density_final.vtu`、`history.json` 和 `summary.json`; 命令根目录写入 `manifest.json`。`state-compare` 写入 `outputs/<case-id>/state-comparison/state_comparison.json`，只包含一次状态分析比较数据。

## 当前固定梁参数

`compliance-fixed-fixed` 显式选择 `FixedFixedBeamCenterLoad2d`。它使用全域 `160 mm x 20 mm`、两端固定、底边中点载荷 `P=-3 N`、`E=30 MPa`、`nu=0.4`、平面应力、`Vbar=0.4`、`rmin=2.4 mm`。每个受控比较阶次取 `comparison_orders = [2, 3, 4]`（$k=1$ 因位移空间为 $P_0$ 缺失刚体旋转模态不适用于拓扑演化，详见 [`docs/fem/huzhang-mixed-fem-implementation.md`](../../docs/fem/huzhang-mixed-fem-implementation.md)），均成对运行 `LFEM p=k` 与 `Hu--Zhang k=k`，并统一使用积分阶 `q=2k+2`。

默认 `hx=1 mm`, 而 `load_width=1 mm` 的连续载荷以中点为中心，会切过相邻两条底边的半边。Hu--Zhang 的牵引强施加不能精确表达边内跳变，网格对齐也救不了（跳变点上的顶点自由度是单值的）。因此两条分析链使用的不是原始阶跃牵引，而是它在底边连续 P1 迹空间上的 L2 投影：`build_problem(parameters, n_cells=nx)` 调用 `soptx.fem.project_patch_traction_to_p1_trace` 得到该投影，再注入 `FixedFixedBeamCenterLoad2d(traction=...)`。投影精确保持合力 `P`，且能被 LFEM 的边界积分与 Hu--Zhang 的迹插值同时精确重现，所以两种方法的差异可以归因于离散格式本身。

结构合力核查（从解出的场反算真正传进结构的力：Hu--Zhang 取 $\int_{\Gamma_N}\sigma_h\cdot n$，LFEM 取支座反力）与密度无关，因此不在本目录重复：它由 [`examples/huzhang_elasticity/concentrated_load_demo.py`](../../examples/huzhang_elasticity/concentrated_load_demo.py) 在实体材料（`rho=1`、无材料插值）下承担。本目录的 `--mode state-compare` 只保留与密度相关的部分：`rho=0.4` + msimp 插值下的柔顺度对比、体积分数、真相对残差与能量恒等式诊断。