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

命令行解析必填的 --analysis-method fem|substructure 和可选的 --preconditioner (默认 gmg), 以及 --help. 其余实验设置统一在脚本顶部按类别定义:

- 计算后端: BACKEND='numpy', DEVICE='cpu'.
- 模型: 固定 DOMAIN=(0, 210, 0, 35, 0, 35), 以及载荷、材料和支承.
- 离散与子结构: GRID=(210, 35, 35), N_FINE=5, CHUNK_SIZE=256, TRACE='full_trace', ASSEMBLY_METHOD='fast', OPERATOR_LEVEL='fa'.
- 线性求解: SOLVER='cg', CG_TOLERANCE=1e-6, CG_MAXITER=20000.
- 优化与输出: MAX_ITER=300, OUTPUT_DIR 为本目录下 outputs, VTU_FIELDS=('density',), 其余优化常量保持原值.

parse_args 解析分析方法和预条件子并生成 args.output; main 使用所选预条件子及其余顶部配置. 输出目录自动添加预条件子后缀, 例如 full_trace_hex_210x35x35_m5_cg_gmg 或 fa_hex_210x35x35_cg_jacobi; 已有目录报错, 不覆盖. main 在创建输出目录前检查子结构网格能否被 N_FINE 整除, 防止整数除法静默截断, 并保留三个方向细单元等边长的检查. 其余配置不在 parse_args 中逐项校验.

当前脚本按 CG 求解流程编写, 顶部 SOLVER 保持 cg; 删除参数校验不代表扩展了求解器支持范围. FEM 的 --preconditioner 可取 none/jacobi/gmg/scipy/mumps, full_trace 子结构可取 none/jacobi/gmg; none 传入分析器时转换为 None. MUMPS 预条件器依赖本机安装, 此次未验证可用性. CG 统一设置 rtol=CG_TOLERANCE、atol=0、rtol_reference=rhs、norm_type=unpreconditioned、verify_residual=True. 各自在施加边界条件后的实际系统上验收 ||b-Ax||_2 <= CG_TOLERANCE*||b||_2, 不随热启动残差变化; 子结构接口残差与 FEM 全场残差不等同, 全场误差仍须单独对照. 对照时保持两侧公共配置一致. 实际运行设置仍写入 config.json, trace 与子结构划分仅在子结构运行中记录.

run_substructure_vs_fem.py 直接包含完整流程: 统一建细网格、子结构布局与分析器、密度过滤、装配、边界条件、求解、柔顺度与灵敏度、停止判断、OC 更新、结果输出. 成本汇总由独立后处理脚本 summarize_cost.py 按需执行. --analysis-method 仅控制子结构布局与分析器构造及热启动位移空间: FEM 使用全场位移, 子结构使用接口位移. 两侧共用同一个优化循环, 直接调用 soptx 的数值接口, 不再转调用其他运行脚本. 原有独立入口保留.

三个入口统一固定 `DOMAIN = (0.0, 210.0, 0.0, 35.0, 0.0, 35.0)`，再用 `HexahedronMesh.from_box` 从计算域与剖分数创建细网格。子结构布局通过 `full_mesh=mesh` 复用同一对象，未传网格的其他调用仍按原方式惰性创建。历史上两个独立子结构入口的默认划分改为 `42 7 7`、细分数 5，与正式网格一致。

`run_substructure_vs_fem.py` 的准备阶段将公共材料与插值单独计为“材料与插值”.
子结构分支先创建布局, 再创建分析器, 分别计为“子结构布局与校验”和“分析器”;
FEM 分支只创建分析器. 这些计时块互不嵌套, “分析器”不再包含材料与插值的准备耗时.

默认网格的边长仍为 1，其他剖分下由固定尺寸计算 `spacing`；例如 `30 5 5` 的边长为 7，过滤半径为 21（仍取 3 个单元边长）。原始结果不变，小网格新结果不再对应此前固定边长 1 的物理尺寸。新运行的 `config.json` 记录实际 `domain`、`grid`、`spacing` 与过滤半径。

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

每轮帧 iterations/iter_XXXX.vtu 与 evolution.pvd 运行时先写在 WSL 的运行目录, 第 0 帧 iter_0000.vtu 是初始设计经过滤与实体保留单元处理后的物理密度 (只含 density), 在计时之前写出, 便于查看固定单元; 由 soptx.postprocess.iteration_frames.IterationFrameWriter 原子写出, 跨文件系统写入不计入逐轮耗时. 运行结束 (summary.json 写出) 后, 脚本自动把帧迁移到脚本顶部 VIEW_ROOT (/mnt/c/workspace/soptx-results/topopt_simp_substructure) 下的同名子目录, 在 Windows ParaView 中打开其中的 evolution.pvd 即可; WSL 的运行目录只保留 json、npy 与末帧、初始帧的副本 result_final.vtu、result_initial.vtu, 迁移位置记在 visualization_location.json. sync_results.py 只读 json 与 npy, 不受迁移影响.

- 查看目录已存在时先清空再迁移, 此迁移规则与运行输出目录的拒绝覆盖规则不同; 其中出现 iterations/、evolution.pvd、.visualization_sync.json 以外的条目时拒绝清空并跳过迁移.
- /mnt/c/workspace/soptx-results 不存在或迁移失败时只打印提示, 帧留在 WSL 的运行目录, 运行不报错.
- 运行中断或迁移被跳过时, 在仓库根目录手动补迁, 例如:

```bash
python -m soptx.postprocess.sync_visualization --source-dir experiments/topopt_simp_substructure/outputs/full_trace_hex_210x35x35_m5_cg --destination-dir /mnt/c/workspace/soptx-results/topopt_simp_substructure/full_trace_hex_210x35x35_m5_cg --move
```

--move 要求运行目录已有 summary.json; 中断的运行没有该文件, 此时去掉 --move 只复制, 确认后自行删除 WSL 中的帧.

## 待执行的验证

本次仅做静态核查, 未运行下列命令、测试或实验. 经授权后, 先查看帮助:

```bash
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --help
```

验收条件: 自定义命令行参数只有 --analysis-method 和 --preconditioner; 原 --grid、--max-iter、--n-fine、--chunk-size、--cg-tolerance、--cg-maxiter、--output-dir 以及 --solver 均应被拒绝.

两轮小网格验收前, 经授权将顶部 GRID 临时设为 (30, 5, 5)、MAX_ITER=2、OUTPUT_DIR 设为未使用的验证目录; 保持 DOMAIN、N_FINE=5 和默认 CG 参数不变. 然后运行:

```bash
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method fem
python experiments/topopt_simp_substructure/run_substructure_vs_fem.py --analysis-method substructure
```

验收条件: 每条命令只运行指定方法; 原始 JSON 与场文件齐全, 运行入口不生成 cost_summary.json; 随后对两个运行目录分别执行 summarize_cost.py, 确认生成 cost_summary.json; 两轮达到上限时标记 max_iter, 不标记优化收敛; 重复运行同名目录时应报错, 保留原文件. config.json、首行打印和顶部设置一致, spacing=(7, 7, 7), 过滤半径为 21; 子结构 n_sub=(6, 1, 1). 求解设置为 cg/gmg、maxiter=20000、rtol=1e-6、atol=0、rtol_reference=rhs、norm_type=unpreconditioned、verify_residual=True、cg_tolerance=1e-6. 核对 cost_summary.json 与阶段记录一致, 重复后处理须拒绝覆盖, 原始 JSON 与场文件不变; 单轮记录的稳态均值应为 null. 两种方法柔顺度相对差初步要求不超过 1e-5、位移相对 L2 差不超过 1e-4、CG 相对残差不超过 1e-6、体积分数不超过 0.12+1e-6. 超限先定位原因, 不自动放宽标准. 完成验证后恢复正式顶部配置, 保留验证输出.

另需验证: GRID=(36, 6, 6)、N_FINE=5 时, --analysis-method substructure 在创建输出目录前因不能整除而报错; --analysis-method fem 不执行子结构整除检查. 改动 GRID 使三个方向细单元边长不同时, main 的等边长检查应报错. parse_args 不再承担顶部配置的通用合法性校验.

## 历史资料

已有 results/、outputs/ 及原始统计保持原样. [历史结果分析](substructure_vs_fem_history.md) 保留详细阶段表和方法误差对照; [历史目录说明](README_history.md) 保留旧工况、路径和运行记录, 不作为当前入口的参数说明.

## 网格复用改动的待执行验证

本次仅做静态核查，以下测试尚未运行；此前积分统一的测试通过记录不覆盖本次改动。

```bash
python -m pytest tests/unit/test_substructure_external_mesh.py tests/unit/test_substructure_analyzer.py tests/unit/test_substructure_field_mapping.py -q
```

验收条件：外部网格被原对象复用；二维、三维及 Q1/Q2 的自由度与接口编号和惰性构造一致；尺寸、剖分或编号不匹配被拒绝；full_trace 与 FEM 的位移、柔顺度及灵敏度等价性仍通过。两轮验收另按上方命令执行并使用新的输出目录，核对固定 `domain` 与实际 `spacing`。

## 边界条件阶段统一的待执行验证

FEM 与子结构均在 apply_bc 阶段准备受约束系统. FEM 保持显式矩阵消元; 子结构 CG 准备行列掩码与右端项, 直接法准备自由子矩阵与修正右端项, 一般线性约束完成一致性检查、相关行约化和乘子系统组装. solve_system 仅求解、恢复位移与验收残差, 不再隐式准备约束. 直接法的矩阵分解仍计入求解阶段.

独立 solve_constrained_system 接口保留一次完成准备与求解的用法, 内部复用 prepare_constrained_system 和 solve_prepared_constrained_system. 分步使用分析器时须先调用 apply_bc; 切换 CG 与直接法应在 apply_bc(..., solver=...) 中选择目标求解器, solve_system 使用同一求解器. solve_state(..., solver=...) 自动将该选项传入准备阶段. scipy 与 mumps 可复用同一已准备的直接法系统. 直接法若改变右端项需重新准备, 不在 solve_system 中重新消元. 分析器仍仅支持齐次 Dirichlet 条件; 通用接口保留非齐次约束支持.

历史 results/、outputs/ 和统计数据未改写. 原子结构约束准备计入求解阶段, 新运行计入边界条件阶段, 两种阶段口径不能直接混用.

本次仅做静态核查, 以下测试尚未运行:

```bash
python -m pytest tests/unit/test_substructure_constrained_solve.py tests/unit/test_substructure_analyzer.py tests/unit/test_substructure_interface_space.py -q
```

验收条件: apply_bc 已完成 CG 掩码、直接法消元或乘子矩阵构造; 原刚度与载荷不被改写; 矛盾约束在准备阶段被拒绝; solve_system 不调用约束校验、QR 约化、乘子矩阵组装或旧的约束施加入口; 无约束、全固定、非齐次固定与冗余一般约束的解析解及残差正确; CG 热启动、FEM 等价性和单次求解器切换仍通过. MUMPS 后端需在安装环境另行验证. 如需验证阶段计时, 经授权按上文两轮小网格流程使用新输出目录运行, 不覆盖历史结果.

## Huang2023 论文规模的 linear_corner 运行

直接使用 run_substructure_full_trace_vs_linear_corner.py, 不使用包装脚本.
在已安装 SOPTX 及其依赖的 Linux/WSL 电脑上, 从仓库根目录运行:

```bash
python -B -u experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py \
  --trace linear_corner --preconditioner jacobi
```

本入口顶部 GRID=(390, 65, 65), 子结构为 78 x 13 x 13, m=5.
仅解析 --trace 和 --preconditioner (以及 --help), 不提供 --grid 或 --output-dir.
其余参数仍在顶部配置区: CG 容差 1e-6, 物理域 210 x 35 x 35,
细单元边长 7/13, 物理过滤半径 21/13, 体积分数 0.12.
最多 300 轮, 连续 5 轮柔顺度相对变化小于 0.0002 时停止, 不强制设计对称.
该 GRID 同时适用于本入口的 full_trace 模式, 不改变 run_substructure_vs_fem.py 的配置.

本入口不再强制单线程, 不写入或清除 BLIS_NUM_THREADS、OPENBLAS_NUM_THREADS、
MKL_NUM_THREADS、OMP_NUM_THREADS. 已设置的外部环境原样保留;
未设置时由相应数值库决定默认线程数, 不保证使用全部核心或一定加速.
config.json 的 threads 记录 policy、environment、blas_libraries 和 logical_cpu_count,
不再写入固定 num_threads=1. environment 中 null 表示未设置;
这些环境变量和逻辑 CPU 数均不代表运行时实际活跃线程数.
终端不打印线程策略或环境变量, 相关信息仅保存在 config.json 中. 不新增线程解析参数.
如需指定线程数, 可在启动前从 shell 设置, 例如 OPENBLAS_NUM_THREADS=8.
修改只对新启动进程生效, 不影响已运行的优化.


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
