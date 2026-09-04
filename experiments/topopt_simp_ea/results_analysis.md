# EA 变密度拓扑优化实验分析

## 1. 研究对象

本模块验证 EA/EbE 精确 Matrix-Free 路径能否在规则盒状区域上完成变密度拓扑优化闭环，并在统一条件下与 FA 路径对照。材料插值与 FA 侧一致，采用带空材料刚度的 modified SIMP：

$$
E(\rho_e)=E_{\min}+\rho_e^p(E_0-E_{\min}).
$$

每次设计迭代不显式组装全局稀疏刚度矩阵：算子作用 $v\mapsto Kv$ 通过单元 gather、单元级稠密作用与 scatter-add 完成，平衡方程用 fealpy `cg` 迭代求解，柔顺度与灵敏度只依赖单元级量与解状态。离散算子与 FA 相同：$K=\sum_e R_e^{T}K_e(\rho_e)R_e$。

## 2. 验证专项与对照分层

### 2.1 统一条件强对照（Tier 1）

5 个注册工况的基准运行，两侧同用 fealpy `cg`（`rtol=atol=1e-12`、`maxiter=20000`），仅 `operator_level` 不同：

| 工况 id（两侧同名） | 网格 | 载荷路径 | Dirichlet 模式 |
|---|---|---|---|
| `cantilever_corner_2d_concentrated` | tri 160×100 | 右下角集中力 | 左端整边固支 |
| `bearing_2d_distributed` | quad 120×40 | 顶边均布面力 | 底边整边固支 |
| `half_mbb_2d_concentrated` | quad 60×20 | 左上角集中力 | 对称滚轴 + 单点铰支 |
| `cantilever_edge_3d_concentrated` | hex 30×10×2 | 右端底边集中力 | 左端面固支 |
| `simply_supported_bridge_2d_distributed` | quad 120×40 | 顶边均布面力，桥面被动区 | 左下角固定铰支 + 右下角滑动铰支 |

带 override 的运行也可进入 Tier 1：两侧用同一组 `--override` 得到同名 `run_id`，`compare.py --case ID --override ...` 在 FA 侧重放同一组 override 定位配对运行。

`compare.py` 的三段对照：

1. **参数一致性门禁**：两侧 `TopOptCase` 的 A/B/C/D 四轴字段（`config.GATE_FIELDS`）逐项相等，并核对两侧 `summary.json` 的 `config` 快照与 `run_id`，失配即拒绝比较；
2. **轨迹对照**：逐迭代柔顺度相对差、体积分数差、迭代数与 change 历史；
3. **锁步对照**：取 FA 侧密度快照（iter 1／中期／收敛），进程内分别构建 FA 与 EA analyzer，同一密度下比较位移 u（相对 $L^2$ 差）、灵敏度 dc（相对 $L^2$/$L^\infty$ 差）与两侧真残差 $\lVert Ku-f\rVert/\lVert f\rVert$。

### 2.2 闭环能力（Tier 2）

过滤（none / density / projection）与优化器（MMA）等组合不单独注册工况，以基准工况加 `--override` 运行，按第 4 节条件独立验收，只证明 EA 闭环在该组合下能够执行并收敛，不主张与 FA 轨迹恒等。EA 与 `topopt_algorithm="density_based"` 的组合此前从未被集成测试覆盖，本模块是其首个系统证据。

### 2.3 判定阈值

同一离散算子 + 同一求解器，两侧只差浮点求和顺序（稀疏 matvec vs gather/scatter einsum），预期近恒等但非逐位相同：

- 锁步 u、dc 相对差 $\leq 10^{-8}$；
- 逐迭代柔顺度相对差 $\leq 10^{-6}$；
- 优化迭代数相等；
- 阈值化（$\rho_e>0.5$）最终拓扑失配单元数为 0。

阈值以常量形式写在 `compare.py` 顶部；超限先视为待调查，不直接放宽。

## 3. 与独立 GPU 脚本的边界

`examples/topopt_platform/topopt_3d_simp_real.py` 是 EA 路径的独立 GPU 全流程脚本（申请书图 10 数据来源），保留为 GPU 性能证据；本模块是 EA 正确性与 FA 对照的唯一承载，两者口径不混用。

## 4. 单次运行验收条件

每次运行必须生成 `history.json`、`density_final.vtu` 和 `summary.json`，并通过以下检查：

1. 优化历史连续、全部标量有限；
2. 最终密度有限且满足 $\rho_{\min}\leq\rho_e\leq1$；
3. 最终体积分数误差不超过统一容限；
4. 最终柔顺度不高于初始柔顺度；
5. 密度变化量达到停止准则；
6. 最终密度场不是均匀初始场。

`summary.json` 中的 `validation.passed` 只有在全部检查通过时才为 `true`。验收逻辑集中在 `collect.py`，单次运行与结果汇总调用同一函数；`collect.py` 枚举 `outputs/` 下全部运行，在当前注册表上重放 `summary.json` 里的 `case_id` 与 `overrides`，重放结果与目录名不符的目录列为 `unclaimed_run_dirs`。

## 5. 完成条件

- Tier 1：5 个基准运行的 `outputs/compare/<run_id>.json` 全部 PASS；
- Tier 2：过滤 none / density / projection 与 MMA 的 override 运行全部通过验收（触发 EA 接口边界的组合须记录原因并显式除名，不静默跳过）。

两层都满足，才能将「EA/EbE 精确 Matrix-Free 变密度拓扑优化」记为已完成。无预条件 CG 在 msimp 刚度对比度约 $10^{6}$ 下的迭代数表现同时作为预条件子任务的动机证据记录。

## 6. 当前状态

模块结构、工况注册与验收规则已建立，工况尚未运行。当前不能填写数值结果，也不能将 EA 任务标记为已完成。
