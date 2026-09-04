# FA 变密度拓扑优化实验分析

## 1. 研究对象

本模块验证 Fully Assembled（FA）路径能否在规则盒状区域上完成变密度拓扑优化闭环，并作为 EA/EbE Matrix-Free 路径统一条件对照的参照侧。材料插值采用带空材料刚度的 modified SIMP：

$$
E(\rho_e)=E_{\min}+\rho_e^p(E_0-E_{\min}).
$$

每次设计迭代显式组装全局稀疏刚度矩阵 $K=\sum_e R_e^{T}K_e(\rho_e)R_e$，平衡方程默认用 fealpy `cg` 迭代求解，计算柔顺度与灵敏度，再通过指定 Filter 和 Optimizer 更新设计变量。

## 2. 工况与对照角色

### 2.1 注册工况

`cases.toml` 只按模型种类登记，5 个工况与 `experiments/topopt_simp_ea/` 同名，除 `operator_level` 外参数字面一致：

| 工况 id（两侧同名） | 网格 | 载荷路径 | Dirichlet 模式 |
|---|---|---|---|
| `cantilever_corner_2d_concentrated` | tri 160×100 | 右下角集中力 | 左端整边固支 |
| `bearing_2d_distributed` | quad 120×40 | 顶边均布面力 | 底边整边固支 |
| `half_mbb_2d_concentrated` | quad 60×20 | 左上角集中力 | 对称滚轴 + 单点铰支 |
| `cantilever_edge_3d_concentrated` | hex 30×10×2 | 右端底边集中力 | 左端面固支 |
| `simply_supported_bridge_2d_distributed` | quad 120×40 | 顶边均布面力，桥面被动区 | 左下角固定铰支 + 右下角滑动铰支 |

`simply_supported_bridge_2d_distributed` 对应博士论文算例 3.2。论文在单元密度、灵敏度过滤与 OC 下给出 c = 435.9132（78 步）；本工况除线性求解器外与论文参数一致，柔度直接对照，迭代数允许不同。桥面被动区由 `OCOptimizer` 经 `Filter.design_mesh` 向问题类索取掩码，每步更新后把桥面单元密度锁定为 1。

基准求解器统一为 fealpy `cg`（`rtol=atol=1e-12`、`maxiter=20000`），基准运行因此直接通过 EA 侧 `compare.py` 的参数一致性门禁。过滤、优化器、网格与求解器的变化不注册新工况，一律以 `--override` 在基准上改字段，`run_id` 由 id 与 override 标签推导，两侧同一组 override 得到同名目录。

### 2.2 与 EA 的对照角色

FA 是 Matrix-Free 拓扑优化的正确性参照，不计入 Matrix-Free 任务完成状态。对照逻辑全部在 `topopt_simp_ea/compare.py`：参数一致性门禁、逐迭代轨迹对照、锁步对照（同一密度下的 u、dc、真残差）与阈值化拓扑对照。本模块只负责按同一参数落盘可复现的产物。

### 2.3 求解器对照（FA 自身）

`--override solve_method=mumps` / `solve_method=scipy` 的运行用于核对 `cg` 在 `1e-12` 容差下与直接法的一致性，以及记录无预条件 CG 在 msimp 刚度对比度约 $10^{6}$ 下的迭代数，后者作为预条件子任务的动机证据。这类运行不与 EA 配对（EA 没有可分解的全局矩阵，只能用 `cg`）。

## 3. 与 Matrix-Free 路径的边界

- FA：每次迭代显式生成并存储全局稀疏刚度矩阵；
- EA/EbE Matrix-Free：不生成全局矩阵，通过单元 gather、局部作用和 scatter-add 完成算子作用。

两侧的离散算子相同，只差浮点求和顺序（稀疏 matvec vs gather/scatter einsum），预期近恒等但非逐位相同；判定阈值以常量形式写在 EA 侧 `compare.py` 顶部。

## 4. 单次运行验收条件

每次运行必须生成 `history.json`、`density_final.vtu`、`summary.json`、全部迭代 `.vtu` 和 `density_history.pvd`，并通过以下检查：

1. 优化历史连续、全部标量有限；
2. 最终密度有限且满足 $\rho_{\min}\leq\rho_e\leq1$；
3. 最终体积分数误差不超过统一容限；
4. 最终柔顺度不高于初始柔顺度；
5. 密度变化量达到停止准则；
6. 最终密度场不是均匀初始场；
7. `.vtu` 文件与优化迭代逐一对应，且存在 ParaView 时间序列入口。

`summary.json` 中的 `validation.passed` 只有在全部检查通过时才为 `true`。验收逻辑集中在 `collect.py`，单次运行与结果汇总调用同一函数；`collect.py` 枚举 `outputs/` 下全部运行，在当前注册表上重放 `summary.json` 里的 `case_id` 与 `overrides`，重放结果与目录名不符、或 `config` 快照与当前注册表不一致的目录列为 `unclaimed_run_dirs`。

## 5. 完成条件

- 5 个基准运行全部通过第 4 节验收；
- 作为 EA Tier 1 参照侧，5 个基准运行在 EA 侧 `outputs/compare/<run_id>.json` 全部 PASS。

FA 侧通过只说明参照侧可用，不单独构成研究结论。

## 6. 当前状态

模块结构、工况注册与验收规则已建立。`outputs/half_mbb_2d_concentrated/` 是基准改为 `cg` 之前的 `mumps` 运行，其 `config` 快照与当前注册表不符，`collect.py` 会将其列为未认领目录；5 个基准运行需按当前口径重跑。当前不能填写数值结果。
