# Exact 子结构拓扑优化

当前统一入口为 [run_exact_substructure.py](run_exact_substructure.py).
`--trace` 选择边界位移空间, `--schur-mode explicit|implicit` 选择 Schur 实现,
`--assemble-level fa|ea` 选择与之配套的装配层级. 默认 `explicit + fa`,
NumPy/PyTorch full_trace/fa 两侧现共用 Cholesky 分解与恢复缓存. 下方折叠内容保留迁移前的历史记录, 不作为当前入口说明.

## 支持的组合

| Schur mode | assemble-level | trace | 后端与求解 |
|---|---|---|---|
| explicit (默认) | fa (默认) | full_trace / linear_corner | 现有 NumPy/CPU、PyTorch/CPU 或 CUDA 路径; CG |
| implicit | ea | full_trace | NumPy/CPU, CG, none/jacobi |
| explicit | ea | — | 尚未实现局部缩聚块 EA |
| implicit | fa | full_trace | NumPy/CPU、PyTorch/CPU 或 CUDA; CG, none/jacobi; 局部稀疏刚度作用 |

显式 NumPy 的 full_trace 支持 none/jacobi/gmg, linear_corner 支持 none/jacobi;
PyTorch 仅支持 none/jacobi. bezier_cubic 尚未实现.
不支持的组合在创建输出目录前报错. implicit 默认采用 fa; 如需保留的共享参考 EA 路径, 显式给出 `--assemble-level ea`.

`assemble-level` 的含义须结合 Schur mode:
explicit/fa 是全局接口 CSR; 规划中的 explicit/ea 是局部缩聚块作用;
implicit/fa 使用已装配的局部未缩聚稀疏块; implicit/ea 在细网格上使用共享参考单元刚度. 未增加第二个装配层级参数.

## 当前 FA 控制变量对照（实现完成，尚未验证）

本次 FA 路径支持 NumPy/CPU 与 PyTorch/CPU、CUDA 的 full_trace; 四组正式对照使用 PyTorch/CUDA. explicit/fa 与 implicit/fa 共用局部装配、
内部 Cholesky 分解、稀疏 K_ib 缓存和位移恢复; 两侧恢复均不重新装配或分解。
隐式另存稀疏 K_bb, 通过局部直接求解作用 Schur 算子; 显式形成接口 CSR。
局部稀疏模式由单元连接关系建立, 不按数值阈值删项。
两侧精确 Schur 对角均按 32 列分批计算。密度更新后重建数值缓存, 复用拓扑。

隐式 FA 不形成局部 Schur 块、全局接口 CSR 或全局细网格 CSR, 不使用细网格 EA。
角点显式路径保留原实现; implicit/ea 仍仅支持 NumPy/CPU。
PyTorch 两侧采用相同的批量 Cholesky 和恢复策略, 稀疏块共享设备端拓扑索引,
数值按子结构存储; 作用与恢复均在目标设备完成, 不在迭代中转到 NumPy。
CUDA 分项计时同步设备后记录。
四组 240/390 三轮 GPU 命令见
[局部刚度 Schur 内存对照](substructure_schur_explicit_vs_implicit.md#3-四组运行命令)。

本次未运行任何测试或实验。下方 2026-10-23 的通过记录属于旧显式策略与 implicit/ea,
不能视为新局部刚度 FA 路径的验证。新输出通过
`schur_implementation=local_fa_cholesky_v1` 标记 NumPy 公共策略,
`local_fa_cholesky_torch_v1` 标记 PyTorch 公共策略; 两者均尚未实测验证。

## 隐式 Schur + 共享参考 EA（保留路径）

此路径适用于 Q1 单一平移类网格、逐单元标量材料系数、齐次支承及内部无载荷.
细网格刚度由共享参考矩阵和当前材料系数作用:

$$
K=\sum_e R_e^T(s_e K_e^0)R_e,\qquad
Sx=K_{bb}x-K_{bi}K_{ii}^{-1}K_{ib}x.
$$

每轮更新密度后, 按子结构批次生成局部刚度, 分解并缓存各内部块
`K_ii = L_i L_i^T`. 局部完整刚度随批次释放.
Schur matvec 通过细网格 EA 和缓存的 Cholesky 直接求解实现.
不形成全局细网格 CSR、全局接口 CSR 或完整局部 Schur 块;
也不常驻逐细单元刚度矩阵.

Jacobi 对角由
`diag(K_bb) - diag(K_bi solve(K_ii, K_ib))`
精确计算, 边界右端按 32 列分批处理, 再散加到接口.
这是装配准备成本, 不把完整 Schur 矩阵生成成本隐藏到预条件阶段.
当前 none 路径也计算该对角, 用于复用求解器的对角合法性检查.

分析器对外仍使用含固定分量的完整接口向量, 保持现有
`assemble_stiff_matrix → apply_bc → solve_interface → recover_displacement`
协议和热启动方式; 内部适配为自由接口算子. 恢复复用同轮的分解,
不重新装配或缩聚. 更新材料时旧系统与分解失效, 新轮重新构造.

本实现保留内部稠密分解、局部刚度批次、EA 工作向量、网格和过滤矩阵等成本.
**小规模 CPU 两轮一致性验证已通过; 尚未验证 full_trace 390 在 14 GiB 内能完成.**
显式与局部刚度隐式 FA 的 GPU 显存对照见
[显式与隐式 Schur](substructure_schur_explicit_vs_implicit.md)，该对照不属于本节的共享参考 EA 路径。

## 运行设置

在仓库根目录、已安装 SOPTX 的 WSL Python 环境中直接执行脚本.
以下为历史旧显式策略与 implicit/ea 的两轮命令, 新 FA 对照请使用上方文档中的三轮命令. 再次执行前须使用新的输出目录或自行备份旧目录; 脚本拒绝覆盖:

```bash
python experiments/topopt_exact_substructure/run_exact_substructure.py \
  --trace full_trace --schur-mode explicit --assemble-level fa \
  --backend numpy --device cpu --preconditioner jacobi \
  --grid 30 5 5 --n-fine 5 --max-iter 2 --threads 1

python experiments/topopt_exact_substructure/run_exact_substructure.py \
  --trace full_trace --schur-mode implicit --assemble-level ea \
  --backend numpy --device cpu --preconditioner jacobi \
  --grid 30 5 5 --n-fine 5 --max-iter 2 --threads 1
```

默认网格为 210 × 35 × 35, n_fine=5, chunk_size=256, 最多 300 轮.
grid、n_sub、n_fine 满足 `grid = n_sub * n_fine`, 缺项由已有参数和默认值推出;
要求整除, 每向 n_fine >= 2, MBB 网格比例为 6:1:1.
细单元边长固定为 1, 计算域随网格变化, 密度过滤半径为 3.
CG 使用 rtol=1e-6、atol=0、rtol_reference=rhs、
norm_type=unpreconditioned、verify_residual=True、maxiter=20000.
未收敛即终止. 接口残差不替代恢复后的自由全场残差验证.

`--threads N` 在导入 NumPy/SciPy 前设置 BLIS、OpenBLAS、MKL、OpenMP 线程变量;
须直接运行脚本. CPU 默认不设置内存上限;
`--cpu-memory-limit 14` 按启动后新增虚拟内存设置上限, 不等同于 RSS 上限.
CUDA 显存分配器限制默认 14 GiB, 对显式与隐式 FA 同样生效; implicit/ea 不支持 CUDA.

## 输出与计时

输出目录按 `<trace>_<schur_mode>_<assemble_level>_hex_<grid>_m<m>_cg_<precond>_t<threads>`
命名, 非 NumPy/CPU 另加后端和设备前缀. 已有目录直接报错, 不覆盖或续算.
原有输出与历史结果保持原样.

`config.json` 记录 schur_mode 与 assemble_level; `history.json` 保存逐轮阶段耗时、
峰值内存、CG 残差与迭代数, 隐式模式另记录 `schur_matvec_count`
(本轮累计, 包含真实残差复核的算子调用).
局部 FA 装配分项为 local_setup、local_stiffness、internal_factorization、local_blocks,
显式另记 schur_formation 与 global_csr, 隐式另记 schur_diagonal。EA 路径保留 fine_operator。
恢复单独计时。轮内 peak_rss_gib 在 OC 更新后计算, 覆盖全部已测阶段。
config.json 和 summary.json 记录 schur_implementation, 用于区分旧显式恢复策略。

隐式 `summary.json` 的 interface_matrix_stored=false,
interface_nnz 与 interface_index_dtype 为 null, 不为统计而触发接口 CSR 构建.
正常完成后保存 summary.json、最终密度/位移 NPY、首末 VTU、
iterations/ 和 evolution.pvd. OOM 写 failure.json, 失败不能视为完整结果.
达到最大轮数记录 max_iter, 不标记优化收敛.

## 验证结果与后续验收

2026-10-23 验证: NumPy/CPU、单线程、30 × 5 × 5 网格、m=5、chunk_size=256、
CG + Jacobi, rtol=1e-6. explicit/fa 与 implicit/ea 均完成两轮, 停止原因为
max_iter, 不代表完整优化收敛. 本轮只修复测试的 Variantmethod 包装方式,
复用此前两轮输出, 未重新运行优化.

| 一致性指标 | implicit/ea 相对 explicit/fa 的差异 |
|---|---:|
| 第 1 轮柔顺度相对差 | 2.4507e-13 |
| 第 2 轮柔顺度相对差 | 1.1915e-11 |
| 末轮位移相对 L2 差 | 8.4698e-10 |
| 末轮物理密度相对 L2 差 | 4.0939e-13 |
| 末轮物理密度灵敏度相对 L2 差 | 1.0689e-8 |

两侧 CG 步数均为 121、128. 末轮自由全场相对残差为 explicit/fa 9.4095e-7、
implicit/ea 9.2605e-7, 均满足 1e-6 门槛.
实测阶段峰值 RSS 的最大值分别为 0.158726、0.109261 GiB;
这些是小规模 CPU 记录, 不推断 390 或 GPU 的内存是否达标.
240/390 FA GPU 显存对照见 [显存结果](substructure_schur_explicit_vs_implicit.md#2-显存结果)。

原始输出:
[explicit/fa](outputs/full_trace_explicit_fa_hex_30x5x5_m5_cg_jacobi_t1/summary.json)、
[implicit/ea](outputs/full_trace_implicit_ea_hex_30x5x5_m5_cg_jacobi_t1/summary.json)、
[一致性报告](outputs/full_trace_implicit_ea_hex_30x5x5_m5_cg_jacobi_t1/consistency_check.json).
输出在本地 outputs/ 中, 不入 Git. 一致性报告中的 38 passed / 2 failed 是修复前的
测试快照, 保留不改写; 当时修复后的测试结果为 **40 passed in 1.56s**, 本次新增实现尚未复验.
修复只临时包装选中变体, 保留真实材料插值和调用次数断言,
测试结束由 monkeypatch 恢复, 未改动生产 Variantmethod.

以下相关测试命令已执行并全部通过 (WSL ihpcm Python 环境, OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1):

```bash
python -m pytest tests/unit/test_implicit_schur.py tests/unit/test_substructure_analyzer.py tests/unit/test_substructure_constrained_solve.py -q
```

已通过的回归与小规模核查覆盖非均匀密度下的显式/隐式算子作用、Jacobi 对角、全场位移、
柔顺度与灵敏度, 并与直接 FEM 对照; 检查恢复后的自由全场残差,
两轮密度更新、热启动与旧系统失效.
单元测试已确认隐式装配、求解与恢复不创建接口 CSR, 恢复不重复局部装配;
两轮运行的最终统计记录 interface_matrix_stored=false、interface_nnz=null.
小规模命令的相对误差目标为柔顺度 <= 1e-5、位移 L2 <= 1e-4,
CG 与自由全场相对残差 <= 1e-6; 超限先定位原因.
不支持参数组合的完整命令行拒绝检查仍待补充. 390 内存、耗时与完整优化另行授权后实测, 本次未运行.

<details>
<summary>迁移前的入口说明与历史验证记录 (仅供追溯, 不代表当前接口或本次验证)</summary>

# 子结构与传统有限元拓扑优化对照

用 [run_substructure_vs_fem.py](run_substructure_vs_fem.py) 每次运行一种方法, 然后比较关键耗时和内存. 例如使用 Jacobi: python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem --preconditioner jacobi. 参数和已有对照结果见 [substructure_vs_fem.md](substructure_vs_fem.md).

## 运行

在仓库根目录、已安装 SOPTX 的 WSL Python 环境中执行:

```bash
# 传统有限元, 固定 FA, 默认 MGCG.
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem

# 子结构, 固定 full_trace, 默认 MGCG; 细分数 m=5.
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method substructure
```

命令行解析必填的 --analysis-method fem|substructure, 可选的 --preconditioner (默认 gmg)、--grid NX NY NZ、--n-sub NX NY NZ、--n-fine M 与 --max-iter (默认取顶部 MAX_ITER), 以及 --help. 其余实验设置统一在脚本顶部按类别定义:

- 计算后端: BACKEND='numpy', DEVICE='cpu'.
- 模型: ELEMENT_SIZE=1.0 (细单元边长), 计算域由网格推出, 以及载荷、材料和支承.
- 离散与子结构: GRID=(210, 35, 35) (--grid 的默认值), N_FINE=5 (--n-fine 的默认值), N_SUB=GRID // N_FINE (仅用于显示, 不单独修改), CHUNK_SIZE=256, TRACE='full_trace', ASSEMBLY_METHOD='fast', OPERATOR_LEVEL='fa'.
- 线性求解: SOLVER='cg', CG_TOLERANCE=1e-6, CG_MAXITER=20000.
- 优化与输出: MAX_ITER=300 (--max-iter 的默认值), OUTPUT_DIR 为本目录下 outputs, VTU_FIELDS=('density',), 其余优化常量保持原值.

parse_args 解析分析方法、预条件子、细网格、子结构划分与最大优化步数并生成 args.output; main 使用这些参数及其余顶部配置. 输出目录名由实际网格生成并自动添加预条件子后缀, 例如 full_trace_hex_210x35x35_m5_cg_gmg、full_trace_hex_390x65x65_m5_cg_jacobi 或 fa_hex_210x35x35_cg_jacobi; 已有目录报错, 不覆盖. parse_args 检查 --grid、--n-sub 各方向与 --max-iter 为正整数, 并按下表确定子结构划分, 防止整数除法静默截断; main 检查 --grid 满足 6:1:1 (nx=6ny, ny=nz), 计算域按 ELEMENT_SIZE 由网格推出. 其余配置不在 parse_args 中逐项校验.

substructure 下 grid、n_sub、n_fine 须满足 grid = n_sub * n_fine; fem 只用 --grid, 给出 --n-sub 或 --n-fine 时报错.

| 传入 | 处理 |
|---|---|
| 都不传 | grid 取 GRID, n_fine 取 N_FINE, n_sub = grid // n_fine |
| 只传 --grid | n_fine 取 N_FINE, n_sub = grid // n_fine, 须整除 |
| 只传 --n-sub | n_fine 取 N_FINE, grid = n_sub * n_fine |
| 只传 --n-fine | grid 取 GRID, n_sub = grid // n_fine, 须整除 |
| --grid + --n-fine | n_sub = grid // n_fine, 须整除 |
| --n-sub + --n-fine | grid = n_sub * n_fine |
| --grid + --n-sub | n_fine = grid // n_sub, 须整除且三个方向相同 |
| 三者都传 | 检查 grid == n_sub * n_fine, 不满足时报错 |

各情形均要求 n_fine >= 2 (子结构内部须有节点可消元); grid 的 6:1:1 比例由 main 检查.

当前脚本按 CG 求解流程编写, 顶部 SOLVER 保持 cg; 删除参数校验不代表扩展了求解器支持范围. FEM 的 --preconditioner 可取 none/jacobi/gmg/scipy/mumps, full_trace 子结构可取 none/jacobi/gmg; none 传入分析器时转换为 None. MUMPS 预条件器依赖本机安装, 此次未验证可用性. CG 统一设置 rtol=CG_TOLERANCE、atol=0、rtol_reference=rhs、norm_type=unpreconditioned、verify_residual=True. 各自在施加边界条件后的实际系统上验收 ||b-Ax||_2 <= CG_TOLERANCE*||b||_2, 不随热启动残差变化; 子结构接口残差与 FEM 全场残差不等同, 全场误差仍须单独对照. 对照时保持两侧公共配置一致. 实际运行设置仍写入 config.json, trace 与子结构划分仅在子结构运行中记录.

run_substructure_vs_fem.py 直接包含完整流程: 统一建细网格、子结构布局与分析器、密度过滤、装配、边界条件、求解、柔顺度与灵敏度、停止判断、OC 更新、结果输出. 成本汇总由独立后处理脚本 summarize_cost.py 按需执行. --analysis-method 仅控制子结构布局与分析器构造及热启动位移空间: FEM 使用全场位移, 子结构使用接口位移. 两侧共用同一个优化循环, 直接调用 soptx 的数值接口, 不再转调用其他运行脚本. 原有独立入口保留.

各入口统一取 `ELEMENT_SIZE = 1.0`，由网格推出 `domain = (0, nx*h, 0, ny*h, 0, nz*h)`，再用 `HexahedronMesh.from_box` 创建细网格。子结构布局通过 `full_mesh=mesh` 复用同一对象，未传网格的其他调用仍按原方式惰性创建。历史上两个独立子结构入口的默认划分改为 `42 7 7`、细分数 5，与正式网格一致。

`run_substructure_vs_fem.py` 的准备阶段将公共材料与插值单独计为“材料与插值”.
子结构分支先创建布局, 再创建分析器, 分别计为“子结构布局与校验”和“分析器”;
FEM 分支只创建分析器. 这些计时块互不嵌套, “分析器”不再包含材料与插值的准备耗时.

细单元边长固定为 1，计算域随网格变化；例如 `30 5 5` 的计算域为 $30\times5\times5$，过滤半径为 3。此前按固定计算域 $210\times35\times35$ 运行的非默认网格结果（如 `outputs/*_390x65x65_*`，边长 7/13）属于旧约定，物理尺寸以各自 `config.json` 为准。新运行的 `config.json` 记录实际 `domain`、`grid`、`spacing` 与过滤半径。

## 输出与比较

当前入口不施加强制设计对称: 体积梯度和柔顺度灵敏度经过过滤后直接用于 OC, 不再作镜像平均, config.json 不再写入 symmetry 字段. 逐轮及末轮的 symmetry_error_*、design_symmetry_error_z 仍作为诊断统计保留, 不要求其为零. 历史结果包含原来的 z 向梯度对称处理, 不作为当前设置下已复现的结果; 已有文件不改写.

每种方法在 outputs/ 下有独立运行目录. full_trace 与 linear_corner 对照入口遇到已有目录直接报错, 详见下方论文规模运行说明. run_substructure_vs_fem.py 同样拒绝覆盖已有目录. outputs/ 是不入库的临时目录; 需要保留的结果先用 sync_results.py 固化到 results/ 或自行备份. 运行入口仅保存原始 config.json、history.json、summary.json 与场文件; 逐轮 ParaView 帧的去向见下文 "ParaView 帧" 一节. 运行完成后, 用 [summarize_cost.py](summarize_cost.py) 单独生成 cost_summary.json, 包含:

- 收敛状态、迭代数、最终柔顺度与体积分数.
- 准备耗时, 第 2 轮起的装配、边界条件、求解与每轮总耗时.
- 全程总耗时和峰值 RSS (GiB, 不含显存).

后处理只依赖标准库, 不导入运行入口或 soptx, 不修改原始记录. 在仓库根目录执行:

```bash
python experiments/topopt_simp_substructure/summarize_cost.py experiments/topopt_simp_substructure/outputs/fa_hex_210x35x35_cg
python experiments/topopt_simp_substructure/summarize_cost.py experiments/topopt_simp_substructure/outputs/full_trace_hex_210x35x35_m5_cg
```

参数为单次运行目录, 自定义 OUTPUT_DIR 时相应调整路径. 已有 cost_summary.json 拒绝覆盖; 如需重新生成, 先自行备份或移走旧汇总. 统计口径保持不变: 排除首轮, steady_iteration_seconds 是各阶段耗时之和的均值, 不是 history 中 iteration_seconds 的均值; 仅一轮时稳态均值为 null.

两侧在相同硬件和线程环境下分别运行, 对照各自 cost_summary.json 即可. 求解器、容差和迭代轮数须一并考虑. 原有 sync_results.py 保留为可选的历史结果固化工具, 统一入口不调用它.

## ParaView 帧

run_substructure_vs_fem.py 与 run_substructure_full_trace_vs_linear_corner.py
统一将所有结果保留在 outputs/ 下各自的运行目录, 不配置 VIEW_ROOT,
不调用自动迁移, 也不生成 visualization_location.json.

每轮帧 iterations/iter_XXXX.vtu 与 evolution.pvd 由
soptx.postprocess.iteration_frames.IterationFrameWriter 原子写出.
第 0 帧 iter_0000.vtu 是初始物理密度 (只含 density), 在迭代计时之前写出.
运行正常结束后, JSON、最终 NPY、全部逐轮帧及 PVD 均留在原目录,
同时在目录顶层保留 result_initial.vtu 和 result_final.vtu, 便于直接查看.
在 ParaView 中打开运行目录中的 evolution.pvd 即可查看迭代过程.
跨电脑查看时复制整个运行目录, 保持 PVD 与 iterations/ 的相对位置不变.

本次调整不迁回或改写历史上已经迁移的结果, 不修改已有文件.
sync_results.py 仍可用于按需固化历史结果, 不由优化入口自动调用.

## 待执行的验证

本次仅做静态核查, 未运行下列命令、测试或实验. 经授权后, 先查看帮助:

```bash
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --help
```

验收条件: 自定义命令行参数为 --analysis-method、--preconditioner、--grid、--n-sub、--n-fine 与 --max-iter; 原 --chunk-size、--cg-tolerance、--cg-maxiter、--output-dir 以及 --solver 均应被拒绝.

两轮小网格验收前, 经授权将顶部 GRID 临时设为 (30, 5, 5)、MAX_ITER=2、OUTPUT_DIR 设为未使用的验证目录; 保持 ELEMENT_SIZE、N_FINE=5 和默认 CG 参数不变. 然后运行:

```bash
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method substructure
```

验收条件: 每条命令只运行指定方法; 原始 JSON 与场文件齐全, 运行入口不生成 cost_summary.json; 随后对两个运行目录分别执行 summarize_cost.py, 确认生成 cost_summary.json; 两轮达到上限时标记 max_iter, 不标记优化收敛; 重复运行同名目录时应报错, 保留原文件. config.json、首行打印和顶部设置一致, spacing=(1, 1, 1), 过滤半径为 3; 子结构 n_sub=(6, 1, 1). 求解设置为 cg/gmg、maxiter=20000、rtol=1e-6、atol=0、rtol_reference=rhs、norm_type=unpreconditioned、verify_residual=True、cg_tolerance=1e-6. 核对 cost_summary.json 与阶段记录一致, 重复后处理须拒绝覆盖, 原始 JSON 与场文件不变; 单轮记录的稳态均值应为 null. 两种方法柔顺度相对差初步要求不超过 1e-5、位移相对 L2 差不超过 1e-4、CG 相对残差不超过 1e-6、体积分数不超过 0.12+1e-6. 超限先定位原因, 不自动放宽标准. 完成验证后恢复正式顶部配置, 保留验证输出.

另需验证: --grid 36 6 6 时, --analysis-method substructure 在创建输出目录前因不能被 n_fine=5 整除而报错; --analysis-method fem 不执行子结构整除检查, 但同时给出 --n-sub 或 --n-fine 时报错. substructure 下三者都传且 grid != n_sub * n_fine、--grid 与 --n-sub 推出的各方向 m 不同、n_fine < 2 时均应报错. 改动 GRID 使其不满足 6:1:1 时, main 的比例检查应报错. parse_args 不再承担顶部配置的通用合法性校验.

## 历史资料

已有 results/、outputs/ 及原始统计保持原样. [历史结果分析](substructure_vs_fem_history.md) 保留详细阶段表和方法误差对照; [历史目录说明](README_history.md) 保留旧工况、路径和运行记录, 不作为当前入口的参数说明.

## 网格复用改动的待执行验证

本次仅做静态核查，以下测试尚未运行；此前积分统一的测试通过记录不覆盖本次改动。

```bash
python -m pytest tests/unit/test_substructure_external_mesh.py tests/unit/test_substructure_analyzer.py tests/unit/test_substructure_field_mapping.py -q
```

验收条件：外部网格被原对象复用；二维、三维及 Q1/Q2 的自由度与接口编号和惰性构造一致；尺寸、剖分或编号不匹配被拒绝；full_trace 与 FEM 的位移、柔顺度及灵敏度等价性仍通过。两轮验收另按上方命令执行并使用新的输出目录，核对由网格推出的 `domain` 与 `spacing=(1, 1, 1)`。

## 边界条件阶段统一的待执行验证

FEM 与子结构均在 apply_bc 阶段准备受约束系统. FEM 保持显式矩阵消元, 在本轮刚度矩阵的值数组上原地进行 (apply_bc(..., inplace=True)), 不另存约束后的值数组; 子结构 CG 准备行列掩码与右端项, 直接法准备自由子矩阵与修正右端项, 一般线性约束完成一致性检查、相关行约化和乘子系统组装. solve_system 仅求解、恢复位移与验收残差, 不再隐式准备约束. 直接法的矩阵分解仍计入求解阶段.

独立 solve_constrained_system 接口保留一次完成准备与求解的用法, 内部复用 prepare_constrained_system 和 solve_prepared_constrained_system. 分步使用分析器时须先调用 apply_bc; 切换 CG 与直接法应在 apply_bc(..., solver=...) 中选择目标求解器, solve_system 使用同一求解器. solve_state(..., solver=...) 自动将该选项传入准备阶段. scipy 与 mumps 可复用同一已准备的直接法系统. 直接法若改变右端项需重新准备, 不在 solve_system 中重新消元. 分析器仍仅支持齐次 Dirichlet 条件; 通用接口保留非齐次约束支持.

历史 results/、outputs/ 和统计数据未改写. 原子结构约束准备计入求解阶段, 新运行计入边界条件阶段, 两种阶段口径不能直接混用.

本次仅做静态核查, 以下测试尚未运行:

```bash
python -m pytest tests/unit/test_substructure_constrained_solve.py tests/unit/test_substructure_analyzer.py tests/unit/test_substructure_interface_space.py -q
```

验收条件: apply_bc 已完成 CG 掩码、直接法消元或乘子矩阵构造; 子结构的原刚度与载荷不被改写 (FEM 原地消元改写本轮刚度矩阵, 下一轮重新装配); 矛盾约束在准备阶段被拒绝; solve_system 不调用约束校验、QR 约化、乘子矩阵组装或旧的约束施加入口; 无约束、全固定、非齐次固定与冗余一般约束的解析解及残差正确; CG 热启动、FEM 等价性和单次求解器切换仍通过. MUMPS 后端需在安装环境另行验证. 如需验证阶段计时, 经授权按上文两轮小网格流程使用新输出目录运行, 不覆盖历史结果.

## Huang2023 论文规模的 linear_corner 运行

直接使用 run_substructure_full_trace_vs_linear_corner.py, 不使用包装脚本.
在已安装 SOPTX 及其依赖的 Linux/WSL 电脑上, 从仓库根目录运行:

```bash
python -B -u experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py \
  --trace linear_corner --preconditioner jacobi --threads 1
```

本入口顶部 GRID=(210, 35, 35), 子结构为 42 x 7 x 7, m=5, 为 --grid、--n-sub、--n-fine 的默认值;
三者的推导与校验规则同 run_substructure_vs_fem.py (见上表), 输出目录名含实际 grid、m 与线程数后缀 _t{threads}.
计算后端与设备由 --backend numpy|pytorch (默认 numpy) 与 --device cpu|cuda (默认 cpu) 指定, 规则同 run_substructure_vs_fem.py:
cuda 须配 pytorch, pytorch 只支持 none/jacobi 预条件子; --gpu-memory-limit (cuda 下默认 14 GiB) 与 --cpu-memory-limit
分别固定 PyTorch 分配器上限与启动后新增虚拟内存上限, 内存或显存不足时把失败点写入 failure.json 并以状态 2 退出.
numpy/cpu 的目录名不带前缀, 其余组合加 <backend>_<device>_ 前缀, 例如 pytorch_cuda_linear_corner_hex_240x40x40_m5_cg_jacobi_t1.
迭代阶段记为 过滤 / 装配 / 边界条件 / 求解 / 恢复 / 灵敏度 / 输出 / OC: 求解只含接口 CG, 恢复为全场位移恢复
(full_trace 逐批重算局部组装与缩聚, linear_corner 回代装配时缓存的恢复矩阵), history.json 另记 recovery_seconds.
不提供 --output-dir. 其余参数仍在顶部配置区: CG 容差 1e-6, 细单元边长 1, 计算域默认 210 x 35 x 35,
过滤半径 3, 体积分数 0.12. 已有的 outputs/linear_corner_hex_390x65x65_m5_cg_jacobi
按旧约定运行 (计算域 210 x 35 x 35, 边长 7/13), 柔顺度为新约定的 13/7 倍.
最多 300 轮, 连续 5 轮柔顺度相对变化小于 0.0002 时停止, 不强制设计对称.
该 GRID 同时适用于本入口的 full_trace 模式, 不改变 run_substructure_vs_fem.py 的配置.

线程数由 --threads 指定, 默认 1 (主对照两侧均单线程). 脚本在导入 numpy/scipy 之前从命令行预取该值,
写入 BLIS_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、OMP_NUM_THREADS 四个变量,
覆盖外部环境中的同名设置; 不从 shell 设置, 以免漏设实际加载的 BLAS 所读的变量.
--threads 须以 --threads N 或 --threads=N 完整给出, 否则 parse_args 报错.
只在直接运行时写入: 重分析脚本 reanalyze_full_trace_vs_linear_corner.py 导入本模块取顶部常量时不写入;
该脚本自带同样的 --threads 预取与校验, 输出目录名同样带 _t{threads} 后缀.
config.json 的 threads 记录 num_threads、environment、blas_libraries 和 logical_cpu_count;
这些变量和逻辑 CPU 数均不代表运行时实际活跃线程数.


输出根据接口、网格、细分数、求解器和预条件子自动命名为
outputs/linear_corner_hex_390x65x65_m5_cg_jacobi/.
目录已存在时抛出 FileExistsError, 不清空、覆盖或自动续算.
需要再次运行时, 先自行将已有目录备份到其他名称.
主要输出与 full_trace_hex_210x35x35_m5_cg_gmg 一致:

- config.json、history.json、summary.json;
- density_final.npy、design_density_final.npy、displacement_final.npy;
- result_initial.vtu、result_final.vtu;
- iterations/iter_XXXX.vtu 和 evolution.pvd, 均保留在运行目录, 不自动迁移.

summary.json 和最终场文件仅在优化正常结束时产生.
达到 300 轮时 termination=max_iter, 不能视为已收敛.
本入口只执行所选模式的优化, 不自动启动最终密度的 full_trace 参考验证,
也不生成包装脚本原有的入口快照或资源外推文件.
outputs/ 不入 Git; 推送代码不会同步本机已有计算结果.

## linear_corner 直接缩聚与恢复缓存

SubstructureAnalyzer 的 linear_corner 分支先计算 B = K_ib P, 再求解
X = solve(K_ii, B), 直接形成 K_c = P^T K_bb P - B^T X.
不再形成完整边界 Schur 补. m=5 的三维 Q1 子结构只有 24 个右端项,
而完整边界路径有 456 个右端项; 两者采用相同的有限元刚度和角点插值空间.

每次成功装配保留 T_corner = -X, 求解后用 u_i = T_corner q 恢复内部位移,
不重新装配或缩聚局部刚度. 每次密度更新装配前先释放旧缓存;
装配失败不保留可用的旧缓存. full_trace 继续使用原完整缩聚流式路径.
恢复缓存不写入结果文件. float64 下论文规模的恢复缓存约 0.453 GiB,
210 x 35 x 35 网格约 0.071 GiB; 这不是整个进程的内存峰值.
运算顺序改变会产生舍入差异, 不要求优化轨迹逐位相同.

独立验证使用旧完整 Schur 投影路径作为基准, 对比刚度、全场位移、
柔顺度、物理密度灵敏度. 新旧各运行三次并交替顺序, 默认小算例含均匀和随机密度.
输出目录必须不存在. 从仓库根目录执行:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -u \
  experiments/topopt_simp_substructure/verify_corner_direct.py \
  --output experiments/topopt_simp_substructure/outputs/verify_corner_direct_run01
```

若本机保留原 linear_corner_hex_210x35x35_m5_cg_jacobi/iterations/iter_0094.vtu,
可以增加 --iter94 对真实非均匀密度验证. 原帧只读, 不重复过滤.
验证记录在新目录的 config.json 与 report.json, 不修改原结果.
数值验收为四项相对误差小于 1e-8, 两侧残差小于 1e-8.
计时采用 scipy 直接求解器以分离迭代次数影响, 包含装配、接口求解与位移恢复,
不包含边界条件、过滤、输出和 OC, 不能直接当作完整优化每轮的加速比.

2026-10-17 验证: 均匀及随机密度小算例 (60 x 10 x 10, m=5) 的四项相对误差
均小于 2e-12. 第 94 轮真实密度 (210 x 35 x 35, m=5) 三次结果一致:
刚度 7.41e-16, 位移 7.74e-11, 柔顺度 1.87e-13, 灵敏度 2.03e-11,
新旧求解残差均小于 6e-13. 三次中位数:

| 阶段 | 旧路径 / 秒 | 新路径 / 秒 |
|---|---:|---:|
| 装配 | 31.674 | 10.061 |
| 接口求解及恢复 | 29.411 | 0.224 |

本机固定 OpenBLAS/OMP 为 1, chunk_size=64, 另有此前授权的优化进程运行,
不是独占硬件基准. 这些数据不是 390 x 65 x 65 的完整优化计时.
报告保留在 outputs/verify_corner_direct_20261017/ 和
outputs/verify_corner_direct_iter94_20261017/, 不入 Git.
70 项直接相关回归通过; 扩展检查共 107 项通过、3 项训练测试失败.
3 项失败均为 independent_training_records_optimizer 的 KeyError: stiffness,
已在临时目录导出的修改前 HEAD 上复现, 本次未修改无关训练代码.

## 装配与边界条件分项计时

run_substructure_full_trace_vs_linear_corner.py 只在第 1 轮打印装配分项及边界分项 (与阶段行同一门控),
同时写入 history.json 每轮记录的 assembly_breakdown 和 boundary_breakdown.
summary.json 继承末轮记录. 不新增命令行参数, 不改变求解算法或输出目录规则.

linear_corner 的 assembly_breakdown 单位均为秒. 以下为互斥分项:
- local_stiffness: 逐批生成局部刚度, 含生成器初始化和结束处理;
- block_projection (分块与角点投影): 批次检查、提取 K_ii/K_ib/K_bb,
  计算 B = K_ib P 和 K_cc = P^T K_bb P;
- local_solve (内部线性系统求解): 计算 T = -solve(K_ii, B), 含结果取负;
- condensation_assembly (缩聚刚度合成): 计算 K_c = K_cc + B^T T, 并登记恢复缓存;
- global_csr: 总缩聚装配耗时扣除以上四项, 含 CSR 符号模式、散加与编排开销,
  不是纯 CSR 内核计时;
- other: 外层装配计时扣除上述项, 含密度准备、旧缓存释放及外层测量开销.

另有汇总字段 local_condensation_total (局部缩聚总耗时), 严格等于
block_projection + local_solve + condensation_assembly. 该字段与三项明细重叠,
不得参与互斥分项求和, global_csr 与 other 的差额计算均不重复扣除它.
终端在第 1 轮打印三阶段明细及单独的汇总行; history.json 的 assembly_breakdown 保存
全部字段, summary.json 继承末轮记录.

历史 block_projection 同时包含求解前投影和求解后合成; 新口径只包含求解前阶段.
对照旧数据时应将新 block_projection 与 condensation_assembly 相加, 不改写旧 JSON.
本次计时拆分仅做静态检查, 尚未运行验证, 下文历史测试记录不覆盖本次修改.

占比的分母为 stages["装配"]["seconds"]. 原 assembly_seconds 字段仍包含装配和
边界条件, 不可混作本分项的分母. full_trace 暂不拆装配分项, 记录为空字典.
采用 perf_counter 的 CPU 墙钟, 适用于当前 numpy/cpu 实验, 不作为 CUDA 异步内核计时.

boundary_breakdown 分别记录 body_force、non_body_load、dirichlet_data、
prepare_system 和 allocate_state. non_body_load 含集中力向量复制与载荷相加,
dirichlet_data 含基准位移复制, prepare_system 含接口矩阵转换、约束校验及右端复制,
allocate_state 含旧刚度引用释放和全场位移向量创建.
两个 *_cache_present_before 布尔字段仅表示调用前缓存存在, 不代表任意载荷变化时一定命中.

静态检查: 接口载荷与约束投影在分析器构造时完成; 纯集中力有指纹缓存,
Dirichlet 插值也已有缓存. 同一问题的后续迭代不会重新插值支承或重新定位集中力.
但每轮仍分配/复制载荷和位移向量, 并准备对应新刚度的约束系统;
固定约束编号的提取与一致性检查也会重复. 本次只计时和检查, 不新增缓存优化.

2026-10-17 独立两轮验证 (210 x 35 x 35, m=5, chunk_size=256, OpenBLAS/OMP=1):
首轮装配 12.33 秒, 局部刚度 21.8%、块提取与投影 53.5%、局部求解 24.6%、
CSR 及编排约 0.1%; 第二轮分别为 20.0%、55.2%、24.7%、约 0.1%.
边界条件首轮约 0.87 秒, 其中 Dirichlet 数据约 0.71 秒; 第二轮约 0.013 秒,
载荷及 Dirichlet 缓存均已存在. 这些比例不能直接代替另一台机器的测量.
独立结果位于 outputs/verify_assembly_timing_210_20261017/, 原结果未修改.
61 项缩聚与分析器回归通过; 分项之和与外层装配时间核对通过.

## 局部刚度职责分离 (待运行验证)

正式 SubstructureAnalyzer 每轮由 FEM 更新一次材料系数, 按布局提供的全局细单元
编号调用 element_stiffness, 再由 local_assembly.assemble_local_stiffness 将 KE
散加成 K^j. local_assembly 只持有尺寸和编号, 不依赖 prototype、材料或积分子.
mesh.py 的旧密度入口保留为兼容包装, 参考单元刚度改为首次使用旧入口时惰性积分.

full_trace 与 linear_corner 共用上述矩阵来源. 前者恢复位移时逐批重算完整缩聚,
不重复材料插值; 后者继续使用本轮角点恢复缓存. 更新材料或装配失败后旧系统失效,
apply_bc 与 solve_system 须使用当前成功装配及准备的系统.

verify_corner_direct.py 的两侧现在共用正式 FEM 矩阵来源; old 表示完整 Schur
后投影, new 表示直接角点缩聚, 不再代表修改前软件版本的完整执行路径.
历史测量不因此视为本次重构的验证结果. local_stiffness 仍包含获取 KE 与局部散加,
一次全局材料更新计入外层装配及 other, 不将其遗漏在总装配时间之外.

本次仅做静态检查, 未运行测试或实验. 经授权后可执行:

```bash
python -m pytest tests/unit/test_substructure_local_assembly.py tests/unit/test_substructure_analyzer.py tests/unit/test_substructure_streaming.py tests/unit/test_substructure_interface_space.py tests/unit/test_substructure_external_mesh.py tests/unit/test_substructure_field_mapping.py -q
```

验收条件: 任意单元矩阵的散加与显式参考一致; NumPy/PyTorch 兼容包装可用;
新旧局部刚度、两种迹空间的位移和灵敏度对照通过; 正式装配及恢复不调用旧密度入口;
每轮材料插值仅一次; 角点恢复不重新生成 KE; 材料更新或装配失败后拒绝旧系统.

</details>
