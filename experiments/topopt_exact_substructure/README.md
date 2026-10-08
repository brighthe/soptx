# 精确子结构拓扑优化

## MBB 梁算例

采用 Huang2023 [1] 的三维 MBB 梁算例，在体积分数约束下最小化结构柔顺度。所有参数均为无量纲量。

### 模型参数

![MBB 梁的几何、载荷与支承](../topopt_piml_substructure/assets/Huang2023_Fig5.png)

梁的长、高、宽之比为 $6:1:1$，默认以细单元边长为长度单位（$h=1$），首档计算域为 $390\times65\times65$，使用整体模型，不利用对称性缩减计算域。顶面中心施加竖直向下的单位集中力 $F=1$。两端支承在侧视图中为一端铰支、另一端滚支。

| 材料参数 | 数值 |
|---|---|
| 实体材料杨氏模量 $E_0$ | $1$ |
| 弱材料杨氏模量 $E_{\min}$ | $10^{-7}$ |
| 泊松比 $\nu$ | $0.3$ |

长度约定是根据论文其他算例的尺寸标注及柔顺度量级作出的复现推断，并非原文明示。原先 $6\times1\times1$、$h=1/65$ 的第一档结果，在相同密度、材料及总载荷下，可将位移和柔顺度除以 65 换算为当前尺度，密度不变。脚本记录的柔顺度来自 `linear_corner` 接口系统，不应未经细网格重新分析就等同于论文的 $C_{\mathrm{Fine}}$。

### 有限元离散

| 参数 | 数值或设置 |
|---|---|
| 有限元 | 八节点六面体单元（Q1） |
| 位移插值次数 $p$ | $1$ |
| 数值积分 | 二阶高斯积分 |

### 子结构参数

| 项目 | 数值或设置 |
|---|---|
| 接口类型 | `full_trace`、`linear_corner` |
| 子结构细分参数 $m$ | $5$，即每个子结构包含 $5\times5\times5$ 个细单元 |
| 子结构网格规模 | $78\times13\times13$ |
| 全局细网格规模 | $390\times65\times65$ |
| 细单元边长 | $h=1$，作为复现长度约定 |

两种接口共用同一内部消元流程，以迹矩阵 $\boldsymbol\Psi$ 表示边界位移：

| 接口类型 | 迹矩阵 $\boldsymbol\Psi$ | 局部接口未知量 $\boldsymbol q_j$ |
|---|---|---|
| `full_trace` | $\boldsymbol I$ | 完整边界位移 |
| `linear_corner` | $\boldsymbol L$ | 角点位移 |

其中，$\boldsymbol L$ 为由角点位移插值得到边界位移的矩阵，两种接口均满足 $\boldsymbol u_{jb}^h=\boldsymbol\Psi\boldsymbol q_j$。

在子结构内部无载荷时，求解内部位移映射：

$$
\boldsymbol K_{jii}^h\boldsymbol T_j
=-\boldsymbol K_{jib}^h\boldsymbol\Psi.
$$

$i$ 和 $b$ 分别表示内部和边界自由度。按边界和内部排列全体自由度，对应的形函数矩阵为：

$$
\boldsymbol N_j^{(\Psi)}=
\begin{bmatrix}
\boldsymbol\Psi\\
\boldsymbol T_j
\end{bmatrix},
\qquad
\begin{bmatrix}
\boldsymbol u_{jb}^h\\
\boldsymbol u_{ji}^h
\end{bmatrix}
=\boldsymbol N_j^{(\Psi)}\boldsymbol q_j.
$$

利用线弹性刚度矩阵的对称性，复用 $\boldsymbol T_j$ 计算局部接口刚度：

$$
\begin{aligned}
\boldsymbol K_{r,j}
&=(\boldsymbol N_j^{(\Psi)})^{\mathrm T}
\boldsymbol K_j^h\boldsymbol N_j^{(\Psi)}\\
&=\boldsymbol\Psi^{\mathrm T}\boldsymbol K_{jbb}^h\boldsymbol\Psi
+(\boldsymbol K_{jib}^h\boldsymbol\Psi)^{\mathrm T}\boldsymbol T_j.
\end{aligned}
$$

其中，$\boldsymbol K_j^h$ 为子结构细网格刚度矩阵。两种接口均通过一次内部线性方程求解获得 $\boldsymbol T_j$，不显式求逆。

### 优化参数

| 参数 | 数值或设置 |
|---|---|
| 优化目标 | 最小化柔顺度 |
| 体积分数 | $0.12$ |
| 过滤半径 | $3$ 个细单元尺寸；对应实际长度 $r_{\min}=3h=3$ |
| 设计变量更新 | OC |
| 停止条件 | 连续最后 5 次迭代的目标函数相对变化小于 $0.0002$ |

### 补定参数（文献未给出）

| 参数 | 补定值 |
|---|---|
| 支承（待确认的建模选择） | 左端底边 $x=0,\ y=0$ 整条线：$u_x=u_y=u_z=0$；右端底边 $x=L_x,\ y=0$（首档 $L_x=390$） 整条线：$u_y=0$ |
| 集中力分配 | 顶面 $x=L_x/2$ 处（首档 $x=195$），$z$ 向距中心最近的 2 个节点各施加 $-1/2$ |
| 初始密度 | 均匀取 $0.12$ |
| 材料插值 | 修正 SIMP：$E_e=E_{\min}+\tilde\rho_e^{\,q}(E_0-E_{\min})$；设计密度 $\rho_e\in[0,1]$，物理密度为过滤后的 $\tilde\rho_e$ |
| SIMP 惩罚指数 $q$ | $3$（区别于位移插值次数 $p$） |
| 过滤 | 密度过滤：$\tilde\rho_e=\sum_i H_{ei}\rho_i/\sum_i H_{ei}$，$H_{ei}=\max(0,\ r_{\min}-d_{ei})$，$d_{ei}$ 为单元形心距离；物理密度取 $\tilde\rho$，灵敏度按链式法则过滤 |
| 体积约束 | $\sum_e v_e\tilde\rho_e/\sum_e v_e\leq0.12$，$v_e$ 为细单元体积；OC 体积判断使用物理密度，体积灵敏度对设计密度按链式法则计算 |
| OC 更新 | 移动限 $0.2$，阻尼指数 $0.5$；二分法求 Lagrange 乘子，初始区间 $[0,10^9]$，相对精度 $10^{-3}$ |
| 停止条件公式 | $r_k=\lvert c_k-c_{k-1}\rvert/c_k$，$r_{k-4},\dots,r_k$ 均小于 $2\times10^{-4}$ 时停止 |
| 最大迭代数 | $300$ |

## 执行入口

两个当前入口已统一为细单元边长 h=1，计算域由网格数量推导，配置记录 domain、spacing、fine_cell_size 和 filter_radius。此次尺度同步仅完成静态检查，未重新运行优化，也未改动已有结果。当前入口与历史源码快照仍有其他差异：下文固定密度入口及对称性诊断属于历史功能说明，尚未恢复到当前入口。旧 SHA-256 清单保留用于发现版本差异。


`run_linear_corner.py` 执行密度过滤、逐批局部装配与精确内部消元、接口求解、位移恢复、柔顺度及灵敏度计算和 OC 更新。支承固定为 `end_lines`，载荷固定为竖直向下的单位力。端部约束采用上文补定设置，尚未确认与论文完全一致。

在仓库根目录、已安装 SOPTX 的环境中执行：

```bash
# 论文第一档网格，最多分析 300 次。
python experiments/topopt_exact_substructure/run_linear_corner.py
```

可用 `--n-sub NX NY NZ` 和 `--n-fine M` 设置子结构网格与每方向细分数，默认分别为 `78 13 13` 和 `5`。全局细网格由上述参数推导，细单元边长固定为 1，计算域尺寸等于各方向细单元数，并写入 `config.json`。改变网格数量也会改变域尺寸；保持 MBB 长宽高比时，应使各方向数量之比保持为 6∶1∶1。`--n-sub` 各项须为正整数，`--n-fine` 至少为 2。过滤半径固定为三个细单元边长，即 $r_{\min}=3$。`--chunk-size`、`--max-iter` 和 `--output-dir` 控制分块、迭代上限和输出目录。接口固定为 `linear_corner`，求解器通过 --solver scipy|mumps 选择，默认 mumps。scipy 为稀疏直接法，不是 CG。

`--backend numpy|pytorch` 统一控制网格、局部装配、接口刚度累加、密度过滤、内部消元、位移恢复和 OC 的张量后端，默认 `numpy`，执行期间不切换回 NumPy。`--device cpu|cuda|cuda:N` 指定密度场、过滤、消元、恢复与 OC 的计算设备，默认 `cpu`。NumPy 仅支持 CPU；PyTorch CUDA 需要可用的设备。

网格、接口编号、局部刚度装配和全局 CSR 累加保留在 CPU，但使用所选后端的张量。PyTorch 模式下，这些阶段仍使用 PyTorch。每批刚度传至计算设备，约化矩阵传回 CPU 后累加；MUMPS／SciPy 接口与文件输出按需转换为 NumPy，求解位移再转回所选后端及设备。实际后端、设备和 CPU 阶段记录在 `config.json`。

密度过滤复用相同物理距离锥形核，NumPy 路径使用 SciPy 卷积，PyTorch 路径在输入 Tensor 的设备上使用三维卷积；伴随过滤保留先归一化再卷积的顺序。

固定工况、OC 选项和收敛判据集中定义在脚本开头；命令行参数的默认值直接定义在 `parse_args` 中。运行配置记录命令行解析后的实际值。选择 mumps 时，启动前检查 SOPTX 实际使用的 `mumps.DMumpsContext` 和 VTU 导出依赖；导入失败会在创建网格前报错，不自动切换求解器。导入检查不代表 MUMPS 分解和求解已经验证。

启动前可在脚本目录执行 `sha256sum -c run_linear_corner.sha256 && python -u run_linear_corner.py`。校验清单记录本次恢复版本的字节摘要；若脚本被覆盖或再次修改，校验失败后不会启动优化。正常修改代码后，应在审查完差异后更新清单，不能为通过校验而直接接受未知内容。启动日志打印入口绝对路径与 SHA-256，`config.json` 记录 `source_path`、`source_sha256`，本次已完成运行的源码快照现归档于 `archives/source_snapshots/run_linear_corner.snapshot.py`，其 SHA-256 与 `outputs/linear_corner_mumps/config.json` 中的 `source_sha256` 一致，用于追溯。本机制用于发现版本变化，不会锁定文件或阻止其他程序写入。

通常省略 `--output-dir`，结果默认写入脚本所在目录下的 outputs/接口_求解器，例如 outputs/linear_corner_cg 或 outputs/linear_corner_mumps，不受终端当前目录影响。显式相对路径仍相对于当前工作目录。需要单独保存实验时，可通过 `--output-dir` 指定绝对路径。输出目录不追加时间戳；重复运行会覆盖同名结果文件。在该目录保存 `config.json`、`history.json`、`summary.json`，以及最终设计密度、物理密度和全场位移的 NPY 文件。另输出 `result_final.vtu`，按 --vtu-fields 选择字段，默认仅含单元物理密度 density，可直接用 ParaView 查看。每轮分析完成后、OC 更新及停止判断之前，保存 `iterations/iter_0001.vtu`、`iter_0002.vtu` 等文件，每帧均使用同一字段选择。全局细网格在首次导出时构建，后续迭代复用。最终密度与柔顺度对应同一次分析；达到最大次数时不再做未经分析的 OC 更新。运行期间保留装配、接口求解、恢复和迭代结果输出。

--vtu-fields 接受一个或多个选项：density（物理密度）、design_density（设计密度）、displacement（节点三分量向量 u）。默认仅保存 density；必要网格始终保存，最终 NPY 和迭代指标不受该参数影响。选择会写入 config.json。位移不再重复输出 u_x/u_y/u_z/u_mag，可在 ParaView 选择 u 的分量或模长。

```bash
python run_linear_corner.py --vtu-fields density
python run_linear_corner.py --vtu-fields density design_density
python run_linear_corner.py --vtu-fields density displacement
```

本次修改完成静态检查及帮助入口检查；尚未执行优化或新增向量字段的运行导出验收。

用 ParaView 打开输出目录下的 `evolution.pvd`，即可按迭代编号切换或播放拓扑演化。索引使用相对路径，VTU 完整写入后才发布对应帧；每轮通过临时文件替换索引。中途停止时，已写完并收录到 PVD 的帧仍可查看。`history.json` 中的 `vtu_file` 对应本轮文件，`export_seconds` 单独记录导出耗时，`analysis_seconds` 不包含导出。正常结束时，最后一帧另存为 `result_final.vtu`，并保留最终 NPY 和汇总文件。

重复使用输出目录时，同编号帧会被覆盖；PVD 只收录本次运行的帧，不会加入旧运行遗留的更高编号文件，也不会自动删除它们。需要保留不同运行时，请指定不同输出目录。默认网格仅保存 density 时每帧约 120 MB，300 帧约 36 GB，另有最终文件；首次导出还需要构建全局网格。当前逐轮 VTU／PVD 修改仅完成静态语法与导出接口检查，尚未运行优化或验证 ParaView 播放。

每轮 `history.json` 同时记录当前已分析的设计密度和物理密度在 x、z 中面上的对称误差。`symmetry_error_x/z` 为物理密度镜像差的最大绝对值；`symmetry_relative_error_x/z` 为镜像差的 L2 范数除以原场 L2 范数（分母下限为 $10^{-30}$）。带 `design_` 前缀的字段对应设计密度。仅记录诊断，不强制对称；x 方向指标不等于宣称当前两端支承在镜像下完全相同。最终一轮指标同时写入 `summary.json`。

长度尺度与对称性记录已完成静态检查，尚未运行数值验证。已有结果文件不作自动换算或覆盖；下次运行会按新的长度约定计算，复用输出目录前应保存需要保留的旧结果。

本次统一后端改动已完成静态语法与接口核对，尚未执行数值验证、CUDA 运行或 VTU 导出验证。2026-10-05 已在 `ihpcm` 安装 PyMUMPS 与 MUMPS，并确认 `DMumpsContext` 可导入。

待授权后，可在仓库根目录分别执行以下命令，使用默认论文网格完成两次分析及一次 OC 更新：

```bash
python experiments/topopt_exact_substructure/run_linear_corner.py --backend numpy --device cpu --max-iter 2
python experiments/topopt_exact_substructure/run_linear_corner.py --backend pytorch --device cpu --max-iter 2
```

两条命令默认使用同一输出目录；需要比较结果时，应先保存上一组结果，或为各组指定不同的绝对输出路径。验收时核对两种后端的柔顺度、过滤结果及 OC 更新一致性，检查结果有限性、体积分数约束、接口与能量残差，并确认 VTU 字段与 NPY 对应。CUDA 路径需在上述核对后单独验证。



### Windows 本地可视化副本

WSL 的 `outputs/` 是计算与验证依据；Windows 本地目录仅保存 ParaView 查看副本。`sync_visualization.py` 手动执行一次增量同步，不自动监控，不改动求解流程。默认将 PVD 已发布的 VTU 和 `evolution.pvd` 同步至 `C:\workspace\soptx-results\topopt_exact_substructure\linear_corner_mumps`，不复制配置、NPY 或验证文件，不删除目标已有文件。

在 WSL 的算例目录执行：

```bash
# 先查看待同步数量和数据量，不写目标目录。
python sync_visualization.py --dry-run

# 单次增量同步；以后有新帧时重复执行。
python sync_visualization.py
```

自定义工况目录：

```bash
python sync_visualization.py \
  --source-dir /home/brighthe/codespace/soptx/experiments/topopt_exact_substructure/outputs/linear_corner_mumps \
  --destination-dir /mnt/c/workspace/soptx-results/topopt_exact_substructure/linear_corner_mumps
```

同步完成后，在 Windows ParaView 打开目标目录下的 `evolution.pvd`。保留 `iterations/` 与 PVD 的相对位置。每帧先复制为目标临时文件，经摘要与源文件变化检查后发布；全部帧就绪后才更新 PVD。同步期间新增的帧在下次执行时纳入。失败时原 PVD 保留，已复制的完整帧可供重试复用。

增量清单 `.visualization_sync.json` 记录来源、配置摘要、文件大小、修改时间及已核对的 SHA-256。未变化帧以大小和修改时间跳过，不每次重新读取全部大文件；首次复制及未登记的已有帧核对内容。同名 VTU 内容不一致或来源配置变化时拒绝覆盖，请为不同运行指定新目标目录。不要同时启动两个同步进程；若异常退出留下 `.visualization_sync.lock`，先确认无同步进程再手动清理锁文件。首次同步现有 80 帧约占 15 GB，复制核对需读取源文件，不会改变 VTU 格式或降低精度。

2026-10-06 已按用户要求，将本次完整优化的 80 帧 VTU 和 evolution.pvd 迁移至上述 Windows 目录，所有源文件与目标文件均通过 SHA-256 一致性检查。WSL 中对应的 iterations/ 和 evolution.pvd 已移除，保留配置、收敛记录和最终 NPY；迁移位置记录在 outputs/linear_corner_mumps/visualization_location.json。history.json 中的 vtu_file 现相对于 Windows 结果目录读取。此次迁移未改变同步脚本的复制行为；当前这组结果已无源 PVD，无需再对其执行同步。尚未验证迁移后的 ParaView 播放。

### 历史结果归档

早期采用 $h=1/65$、执行两次分析的结果已从重复目录 `topopt_exact_substructure/outputs` 移至 `archives/legacy_h_1_over_65/`。归档保留原始配置、密度、位移、VTU 和收敛记录，七个文件在移动前后均通过 SHA-256 一致性检查。当前完整优化保存在 `outputs/linear_corner_mumps/`，尺度核对结果仍在 `outputs_analysis/`。

`outputs_analysis/config.json` 中的 `physical_density_source` 保留当次运行时的原始路径，作为历史来源记录；重新读取该密度时使用下方示例中的归档路径。

### 固定物理密度的单次分析

使用 `--physical-density` 读取已有 `density_final.npy`，直接进行一次装配、接口求解和位移恢复，不再次过滤密度，也不计算优化灵敏度或执行 OC。输入必须为与全局细网格同形状的有限实数数组，取值在 $[0,1]$ 内；不能用 `design_density_final.npy` 代替。`--n-sub`、`--n-fine` 应与原结果一致。

例如，在仓库根目录使用第一档网格的旧物理密度：

```bash
python experiments/topopt_exact_substructure/run_linear_corner.py \
  --backend numpy --device cpu \
  --n-sub 78 13 13 --n-fine 5 --chunk-size 256 \
  --physical-density /home/brighthe/codespace/soptx/experiments/topopt_exact_substructure/archives/legacy_h_1_over_65/density_final.npy
```

此模式忽略 `--max-iter` 的分析次数设置，只分析一次。默认输出到脚本旁的 `outputs_analysis`，显式指定目录时必须使用新的或空的目录，且不能与输入文件所在目录相同。保留逐批进度日志；输出物理密度、位移、VTU、配置和诊断结果，不输出设计密度及其对称误差。`summary.json` 的 `termination` 为 `analysis_only`，不代表优化收敛。

尺度核对时，确认旧结果使用 $6\times1\times1$ 计算域，且网格、材料、载荷和支承相同。新结果应满足 $C_{\mathrm{new}}\approx C_{\mathrm{old}}/65$、$\boldsymbol u_{\mathrm{new}}\approx\boldsymbol u_{\mathrm{old}}/65$，输入与输出物理密度应一致。可先以柔顺度相对差和位移相对 L2 差不超过 $10^{-6}$ 为核对目标，同时检查接口平衡、约束及能量残差；若不满足，应定位原因后再开始完整优化。脚本不自动读取旧柔顺度和位移进行比较。

单次分析已在 NumPy／CPU 下通过尺度核对：柔顺度缩放相对误差约为 $5.04\times10^{-12}$，位移相对 L2 误差约为 $3.56\times10^{-10}$，物理密度逐项一致。新增的逐轮 VTU／PVD 输出尚未执行运行验证；单次分析模式只生成一帧，不包含设计密度。


### 验证结果目录

已有 MUMPS 优化结果保存在 `outputs/linear_corner_mumps/`，辅助图片、局部灵敏度核对及完整细网格验证已单独移至 `validation/`，不再嵌套在优化输出中。`validation/fine_grid_check/` 保留细网格求解脚本、收敛记录、日志、位移及最终汇总。该目录的 `analyze.py` 从算例目录的 `outputs/linear_corner_mumps/` 读取原优化配置与最终物理密度。这些参考文件保留用于后续固定密度对照；新的 full_trace 优化入口不读取它们。

### full_trace 完整拓扑优化

run_full_trace.py 现为独立的完整优化入口，从均匀物理/设计初始密度 0.12 开始，逐轮执行密度过滤、完整接口算子构建、所选求解器求解、内部位移恢复、单元能量与柔顺度灵敏度、伴随过滤和 OC 更新。停止判据与角点入口一致。后续再统一两个入口；此入口不再读取原优化目录或细网格参考解。

采用 h=1，计算域由子结构数量和每方向细分数推导；默认域为 390×65×65，过滤半径为 3。材料、载荷与 end_lines 支承沿用上文设置。run_linear_corner.py 已采用相同的尺度推导方式；相同网格参数下两个入口的计算域与过滤半径一致。比较接口误差仍需使用同一物理密度、材料、载荷与支承；独立优化得到的结果不等价于固定密度对照。已完成并保存在 outputs/linear_corner_mumps 的历史结果采用 h=1。

完整优化支持两条分析路径。scipy、mumps 以及 CG 的 none／jacobi 预条件路径显式装配接口 CSR，计算完整边界 Schur 补。CG(MG) 使用隐式 Schur 作用，不创建接口 CSR 或缓存全部局部 Schur 矩阵；每轮均按当前过滤后的物理密度重新进行内部 Cholesky 分解，并更新全部 MG 粗层算子。每次 Schur 作用通过内部直接回代恢复平衡，再由共享参考单元 EA 算子提取接口力。

MG 复用 StructuredHexHierarchy 的对称 V 循环，接口预条件子为 R B R^T，其中 R 提取自由接口自由度，B 为完整细网格上的 MG 作用。MG 仅用于预条件，不改变精确 Schur 方程。几何层级和编号复用，密度相关的内部因子、粗层刚度、光滑子和粗层分解逐轮更新；每轮结束关闭粗层求解器。当前 MG 仅支持 NumPy／CPU，不自动切换后端。

支持 --solver cg|scipy|mumps（默认 cg），--cg-tol（默认 1e-6），--cg-maxiter（默认 20000），--precond none|jacobi|mg（默认 jacobi）。MG 最粗层自由度上限固定为脚本顶部 MG_COARSE_MAX_DOFS=20000。保留 --n-sub、--n-fine、--backend、--device、--chunk-size、--max-iter、--output-dir 和 --vtu-fields。已移除验证模式及 source-dir、reference-dir、comparison-tol、mg-coarse-max-dofs、estimate-only 参数。

在算例目录运行完整 CG(MG) 优化：

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u run_full_trace.py \
      --backend numpy --device cpu \
      --solver cg --precond mg \
      --n-sub 78 13 13 --n-fine 5 --chunk-size 32 \
      --max-iter 300 --vtu-fields density \
      --output-dir outputs/full_trace_cg_mg

默认输出目录为脚本旁 outputs/full_trace_cg_mg；其他求解组合仍为 outputs/full_trace_求解器。显式相对路径按终端当前目录解析。逐轮 VTU、evolution.pvd、history.json、最终 NPY／VTU 和 summary.json 保持不变，保留装配、求解、恢复及 OC 日志。不读取已有优化结果或细网格参考解。

启动时自动估算内存，超过当前可用内存的 80% 即停止。默认网格与 chunk-size=32 时，CG(MG) 内部分解缓存约 3.62 GiB，EA/MG 规划预算约 11.06 GiB；显式接口 CSR 路径的装配预算约 60.20 GiB，另需直接法分解内存。这些均非实测峰值，通过预检查不保证资源一定充足。

每轮 CG 使用上一轮接口位移热启动，按载荷范数设置绝对停止容差；恢复后重新计算接口和全场真实残差，两者均须不超过 cg-tol，否则停止，不执行 OC 更新。history.json 的 equilibrium_residual 在 MG 路径记录全场真实残差，interface_residual 记录自由接口残差，并保留 solver_iterations、solver_converged、约束残差、能量相对差及耗时。

本次仅完成静态语法与参数入口检查，尚未执行数值测试或完整优化。后续验收先核对同一密度下隐式 Schur 与显式求解的一致性，再核对多轮密度更新后的残差、体积约束和 VTU。可在仓库根目录执行 python -m pytest tests/unit/test_implicit_schur.py -q 检查已有 Schur 代数与有限元测试；测试尚未运行，不应据此宣称结果已通过验证。

## 参考文献

[1] Huang2023，第 4 节公共设置与第 4.1 节 MBB 梁算例，图 5–7。

两个优化入口均将 --solver 的实际取值写入 config.json；仅选择 mumps 时检查 PyMUMPS 依赖，选择 scipy 不要求安装 MUMPS。两种直接法的大规模内存开销均需另行验证。


### 接口迭代求解

两个入口均支持 --solver cg（默认）、scipy 和 mumps。CG 参数沿用 run_fa.py 的命名：--cg-tol 默认 1e-6，--cg-maxiter 默认 20000，--precond 支持 none 和 jacobi（默认）。直接法不接受这三个 CG 专用参数。

CG 使用齐次支承的对称行列消元；只有单自由度固定行能够覆盖全部约束列时才启用，其他一般线性约束明确报错。当前 MBB 的两种接口支承满足这一条件。接口矩阵和 CG 向量驻留 CPU，张量后端仍跟随 --backend。使用上一轮接口位移热启动，验收标准为自由自由度上真实残差二范数不超过 cg-tol 乘自由载荷二范数；失败即停止，不执行 OC 更新。

config.json 保存 CG 参数，history.json 保存 solver_iterations、solver_converged 和 equilibrium_residual。none／jacobi 路径保留显式 CSR 装配开销；full_trace 的 mg 路径采用上述隐式 Schur，按 EA/MG 预算检查内存。此项改动尚未完成数值对照验收，不应仅凭脚本可启动认定 CG 与直接法等价。


### 按接口与求解器整理结果

已有 80 轮收敛的 MUMPS 结果已归入 outputs/linear_corner_mumps，7 个文件移动前后 SHA-256 完全一致。已有结果的配置和收敛记录保持原内容。Windows VTU 仍位于原 linear_corner 目录，由 visualization_location.json 关联；该组 WSL 结果已无 PVD，不需要再运行同步。

两个优化入口默认使用 outputs/接口_求解器，full_trace 的 CG(MG) 使用 outputs/full_trace_cg_mg；CG 和 MUMPS 的结果分别保存。同步脚本默认来源为 outputs/linear_corner_mumps，其他运行应显式指定 --source-dir 和对应的新 --destination-dir。Windows 原同步清单保留历史来源记录，不因本次归档改写。

在算例目录运行 full_trace / mumps 时，可省略 --output-dir，也可显式使用 --output-dir outputs/full_trace_mumps。重复运行同一组合仍会覆盖同名文件，需要保留不同实验时应指定新的输出目录。
