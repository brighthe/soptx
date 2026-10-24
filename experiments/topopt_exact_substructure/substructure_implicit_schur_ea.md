# full_trace 显式与局部刚度隐式 Schur：实现与内存对照

本文比较 240×40×40 和 390×65×65 两个规模下，显式 Schur（explicit/fa）与基于已装配局部刚度的隐式 Schur（implicit/fa）的 GPU 显存占用。两条路径采用相同的 full_trace 空间，数学上等价，比较重点为首轮、后续两轮及全部已测阶段的内存峰值。

本次已接通 NumPy/CPU 与 PyTorch/CPU、CUDA 的局部刚度隐式路径，并统一显式侧的 Cholesky 分解和缓存恢复策略。新增实现尚未运行测试或实验；旧 implicit/ea 路径的小规模验证仅作为历史背景，不代表新实现已通过验证。模型说明见 [full_trace_vs_fem.md](full_trace_vs_fem.md)，共享参考 EA 的已有结果见 [fem_ea_vs_fa.md](../topopt_full_scale_fem/fem_ea_vs_fa.md)。

## 1. 研究对象

### 1.1 full_trace 空间与适用条件

沿用三维 MBB 梁的 Q1 六面体细网格、固定泊松比、逐单元 SIMP 模量和齐次 Dirichlet 支承。细网格划分为互不重叠的子结构，内部自由度仅属于一个子结构，公共边界自由度通过全局接口编号共享。

full_trace 保留全部自由边界位移，不施加角点线性插值限制。若一般边界降阶写为 $u_b^j=\Psi^jq^j$，本方法取 $\Psi^j=I$。改变 Schur 算子的存储方式并不减少接口未知量，也不改变边界变形空间。

假设内部块 $K_{ii}^j$ 可逆且对称正定，消除刚体运动后的全局自由系统对称正定。当前非零弱材料模量避免完全空单元，但高材料对比度仍可能造成病态，不能仅由正模量推断数值求解容易。

### 1.2 内部消元与接口方程

第 $j$ 个子结构按边界 $b$、内部 $i$ 分块：

$$
\begin{bmatrix}
K_{bb}^j&K_{bi}^j\\
K_{ib}^j&K_{ii}^j
\end{bmatrix}
\begin{bmatrix}u_b^j\\u_i^j\end{bmatrix}
=
\begin{bmatrix}f_b^j\\f_i^j\end{bmatrix}.
$$

内部平衡给出

$$
u_i^j=(K_{ii}^j)^{-1}(f_i^j-K_{ib}^ju_b^j),
$$

代入边界方程得到

$$
S^j=K_{bb}^j-K_{bi}^j(K_{ii}^j)^{-1}K_{ib}^j,\qquad
g_b^j=f_b^j-K_{bi}^j(K_{ii}^j)^{-1}f_i^j.
$$

令 $A_b^j$ 从全局接口向量提取本子结构边界位移，则

$$
K_\Gamma=\sum_j(A_b^j)^{\mathsf T}S^jA_b^j,\qquad
F_\Gamma=\sum_j(A_b^j)^{\mathsf T}g_b^j,\qquad
K_\Gamma U_\Gamma=F_\Gamma.
$$

载荷分配须保证散加后等于原全场载荷，不重复计入公共边界载荷。施加支承后在自由接口空间求解。逆号表示线性系统求解，不构造逆矩阵；上述定义也不要求显式形成 $S^j$ 或 $K_\Gamma$。

当前 MBB 算例内部载荷为零，故缩聚载荷就是原载荷的接口分量。一般内部载荷和非齐次支承的公式可推广，但不属于现有隐式实现已经支持的范围。

### 1.3 精确隐式 Schur 的作用

外层迭代只需输入 $v_\Gamma$ 后获得 $K_\Gamma v_\Gamma$。逐子结构计算

$$
v_b^j=A_b^jv_\Gamma,\qquad
K_{ii}^jz_i^j=K_{ib}^jv_b^j,\qquad
y_b^j=K_{bb}^jv_b^j-K_{bi}^jz_i^j,
$$

$$
y_\Gamma=\sum_j(A_b^j)^{\mathsf T}y_b^j=K_\Gamma v_\Gamma.
$$

因此可以完全不形成局部和全局 Schur 矩阵。每个固定密度状态下，将 $K_{ii}^j=L_i^j(L_i^j)^{\mathsf T}$ 分解一次，之后每次作用以两次三角回代求 $z_i^j$。“精确”表示使用直接内部求解而不引入有意的近似，仍存在浮点舍入与外层迭代误差。

### 1.4 共享参考单元 EA 的细网格作用（保留的另一条路径）

本规则网格上所有细单元几何仅差平移，局部自由度顺序一致；材料在每个单元内为标量模量，泊松比固定。以单位模量的 $24\times24$ 刚度为 $K^0$，可写为

$$
K_e(\rho_e)=E(\rho_e)K^0,\qquad
E(\rho_e)=E_{\min}+(E_0-E_{\min})\rho_e^p.
$$

令 $G_e$ 从全场向量提取单元位移，则未施加支承的细网格算子满足

$$
K=\sum_eG_e^{\mathsf T}\bigl(E(\rho_e)K^0\bigr)G_e,\qquad
Kx=\sum_eG_e^{\mathsf T}\bigl(E(\rho_e)K^0G_ex\bigr).
$$

共享参考 EA 按 gather、参考刚度作用、模量缩放、scatter-add 的顺序实现右式，不形成全局 $K$。常驻一份 $K^0$、逐单元系数和单元自由度映射；不同密度只改变系数，不要求各单元具有相同密度。

这一表示依赖同构几何、编号与材料缩放条件。非均匀几何或不同泊松比不能直接套用同一份 $K^0$。本路径与全量缓存每个 $K_e$ 的标准 EA 不同，但每次作用仍产生单元向量等临时数据。

齐次支承可用自由掩码投影 $P$ 表示为 $\widehat K=PKP+(I-P)$，全场作用仍可由 EA 完成；约束自由度输入为零时，结果在自由位置与原自由系统一致。

### 1.5 两次 EA 作用与一次内部求解的组合（implicit/ea）

为清晰推导，以下 $K_f$ 表示去掉固定自由度后的细网格自由系统。令 $R_\Gamma$、$R_I$ 分别从自由全场向量提取自由接口和内部向量，满足

$$
R_\Gamma^{\mathsf T}R_\Gamma+R_I^{\mathsf T}R_I=I,\qquad
K_{II}=R_IK_fR_I^{\mathsf T}
=\operatorname{diag}(K_{ii}^1,\ldots,K_{ii}^M).
$$

一次接口作用按下列步骤计算：

$$
\begin{aligned}
x^{(0)}&=R_\Gamma^{\mathsf T}v_\Gamma,\\
t&=K_fx^{(0)}, &&\text{第一次 EA 作用},\\
K_{II}w_I&=-R_It, &&\text{独立内部块回代},\\
x^{(h)}&=x^{(0)}+R_I^{\mathsf T}w_I,\\
y_\Gamma&=R_\Gamma K_fx^{(h)}, &&\text{第二次 EA 作用}.
\end{aligned}
$$

于是

$$
y_\Gamma=
\left[
R_\Gamma K_fR_\Gamma^{\mathsf T}
-R_\Gamma K_fR_I^{\mathsf T}
K_{II}^{-1}R_IK_fR_\Gamma^{\mathsf T}
\right]v_\Gamma
=K_\Gamma v_\Gamma.
$$

$x^{(h)}$ 在内部满足平衡，即 $R_IK_fx^{(h)}=0$。实际代码可保留全场长度，用零固定分量和约束掩码实现同样过程，无须另存自由子矩阵。这里不存任何全局 CSR，但仍显式装配并缓存小型内部块的分解因子。

### 1.6 恢复、等价性与验收量

求得 $U_\Gamma$ 后，内部无载荷时使用同样的内部平衡恢复：

$$
U_f=R_\Gamma^{\mathsf T}U_\Gamma
-R_I^{\mathsf T}K_{II}^{-1}R_IK_fR_\Gamma^{\mathsf T}U_\Gamma.
$$

再填回固定自由度的零位移。内部有载荷时应增加对应特解，不能沿用零载荷恢复公式。

以上消元与恢复使本方法、显式 full_trace Schur 和细网格 FEM 在精确算术下代数等价。共享参考 EA 也仅改变矩阵作用的求和顺序，不引入新的边界降阶。内部分解须与本轮材料一致；外层残差达标后，还应验收恢复后的自由全场残差

$$
r_f=F_f-K_fU_f,
$$

以及柔顺度 $C=F_f^{\mathsf T}U_f$、位移和物理密度灵敏度。固定支承处的反力不计入自由方程残差。受约束系统对称正定时可用 CG，预条件作用须满足相应条件。

### 1.7 本次对照使用的局部刚度 FA 路径

FA 在此表示局部子结构刚度已由细单元矩阵装配；不表示隐式路径形成了全局接口矩阵。两侧共用
`element_stiffness → assemble_local_stiffness`，按批次生成稠密局部刚度，再提取内部块、耦合块及边界块。

两侧均缓存内部 Cholesky 因子和稀疏 $K_{ib}^j$。稀疏模式由单元自由度连接关系构造，保留拓扑内的零值，不通过数值阈值删项。隐式另外保存稀疏 $K_{bb}^j$，每次作用按

$$
v_b^j=A_b^jv_\Gamma,\qquad
z_i^j=\operatorname{solve}(K_{ii}^j,K_{ib}^jv_b^j),\qquad
y_\Gamma=\sum_j(A_b^j)^{\mathsf T}
\left(K_{bb}^jv_b^j-(K_{ib}^j)^{\mathsf T}z_i^j\right)
$$

计算，不调用细网格 EA，也不形成局部 Schur 矩阵、全局接口 CSR 或全局细网格 CSR。
显式则按列批次求解同一内部系统，形成局部 Schur 块并装配接口 CSR，不常驻完整局部刚度或 $K_{bb}^j$。

两侧均按 32 列分批计算精确 Schur 对角，使用同一 Jacobi 数学定义；全场恢复统一为
$u_i^j=-\operatorname{solve}(K_{ii}^j,K_{ib}^ju_b^j)$，复用同轮因子，不重新装配或分解。
密度更新后数值缓存失效并重建，拓扑模式保留。公共策略用于 NumPy/CPU 与 PyTorch/CPU、CUDA 的 full_trace/fa；角点接口保留原路径，
implicit/ea 仍仅支持 NumPy/CPU。

PyTorch 路径使用共享拓扑索引和设备端批量稀疏值，按子结构批次执行
`torch.linalg.cholesky` 与 `torch.cholesky_solve`；矩阵作用通过 gather/scatter-add 完成，
不为每个子结构单独调用稀疏求解器。拓扑初始化可以使用主机，局部块值、分解因子、
接口向量和恢复结果留在目标设备，迭代过程中不转为 NumPy。GPU 的重复索引散加可能
因浮点累加顺序产生微小差异。分项计时在 CUDA 同步后取值。

## 2. 实验设置与验证背景

### 2.1 公共配置

采用三维 MBB 模型：规则 Q1 六面体、二阶积分、泊松比 0.3、$E_0=1$、$E_{\min}=10^{-7}$、SIMP 指数 3、细单元边长 1、体积分数 0.12、过滤半径 3。

四组实验统一采用 `full_trace`、PyTorch/CUDA、CG/Jacobi、`n_fine=5`、chunk_size=256、`--threads 1`、`--max-iter 3`，在同一环境下按独立进程顺序运行，使用 14 GiB 的 PyTorch 显存分配器上限。CG 使用 rtol=1e-6、atol=0、相对右端项验收、最多 20000 步，并检查真实接口残差。显式使用 `--schur-mode explicit --assemble-level fa`，隐式使用 `--schur-mode implicit --assemble-level fa`。

两种网格分别包含 384,000 和 1,647,750 个细单元；对应子结构划分为 48×8×8 和 78×13×13。两条路径从相同的均匀密度 0.12 开始，后续密度由各自优化流程更新，可能存在浮点误差引起的微小轨迹差异。

入口为 [run_exact_substructure.py](run_exact_substructure.py)。输出目录不得覆盖，保留 config.json、history.json、summary.json 和运行日志，记录环境、完成状态及失败阶段。

### 2.2 四组实验运行命令

在 WSL 中进入仓库根目录 `/home/brighthe/codespace/soptx`，激活已安装 SOPTX 和支持 CUDA 的 PyTorch 环境后执行。按下面顺序逐组运行，等待上一组退出后再启动下一组，不并发运行。以下命令尚未执行。

**240×40×40，显式 Schur：**

```bash
python -u experiments/topopt_exact_substructure/run_exact_substructure.py \
  --trace full_trace --schur-mode explicit --assemble-level fa \
  --backend pytorch --device cuda --preconditioner jacobi \
  --grid 240 40 40 --n-fine 5 --max-iter 3 --threads 1 --gpu-memory-limit 14
```

**240×40×40，隐式 Schur：**

```bash
python -u experiments/topopt_exact_substructure/run_exact_substructure.py \
  --trace full_trace --schur-mode implicit --assemble-level fa \
  --backend pytorch --device cuda --preconditioner jacobi \
  --grid 240 40 40 --n-fine 5 --max-iter 3 --threads 1 --gpu-memory-limit 14
```

**390×65×65，显式 Schur：**

```bash
python -u experiments/topopt_exact_substructure/run_exact_substructure.py \
  --trace full_trace --schur-mode explicit --assemble-level fa \
  --backend pytorch --device cuda --preconditioner jacobi \
  --grid 390 65 65 --n-fine 5 --max-iter 3 --threads 1 --gpu-memory-limit 14
```

**390×65×65，隐式 Schur：**

```bash
python -u experiments/topopt_exact_substructure/run_exact_substructure.py \
  --trace full_trace --schur-mode implicit --assemble-level fa \
  --backend pytorch --device cuda --preconditioner jacobi \
  --grid 390 65 65 --n-fine 5 --max-iter 3 --threads 1 --gpu-memory-limit 14
```

结果写入本目录的 `outputs/`，对应目录名依次为：

- `pytorch_cuda_full_trace_explicit_fa_hex_240x40x40_m5_cg_jacobi_t1`
- `pytorch_cuda_full_trace_implicit_fa_hex_240x40x40_m5_cg_jacobi_t1`
- `pytorch_cuda_full_trace_explicit_fa_hex_390x65x65_m5_cg_jacobi_t1`
- `pytorch_cuda_full_trace_implicit_fa_hex_390x65x65_m5_cg_jacobi_t1`

运行前确认这些输出目录不存在；若已有结果，先自行备份，脚本拒绝覆盖或续算。上述命令直接在终端显示进度；如需保存日志，可在各条命令末尾追加 `> 独立日志路径 2>&1`，并使用尚不存在的日志文件。`--gpu-memory-limit 14` 显式写出脚本原有默认上限，作用于 PyTorch 分配器，
不等同于 GPU 总占用或实测 allocated 峰值。启动时空闲显存不足 14 GiB 会拒绝运行；
如需调整，应对四组统一修改。390 是否能完成以实测为准。运行后检查 `history.json` 是否有完整三轮及 `summary.json` 的完成状态，失败时保留日志和已有记录，不将部分结果当作完整三轮结果。

### 2.3 验证背景与验收

历史旧显式路径与 implicit/ea 的 30×5×5、m=5、NumPy/CPU 单线程两轮验证满足数值门槛：末轮两条路径的位移相对 L2 差为 8.4698e-10，物理密度灵敏度相对 L2 差为 1.0689e-8，恢复后的自由全场相对残差均小于 1e-6。相关回归在 2026-10-23 记录为 40 passed；本次实现后尚未重跑测试或优化。小规模结果不能外推为 240 或 390 的内存结论。

原始验证记录见 [显式 summary](outputs/full_trace_explicit_fa_hex_30x5x5_m5_cg_jacobi_t1/summary.json)、[隐式 summary](outputs/full_trace_implicit_ea_hex_30x5x5_m5_cg_jacobi_t1/summary.json) 和 [一致性报告](outputs/full_trace_implicit_ea_hex_30x5x5_m5_cg_jacobi_t1/consistency_check.json)。一致性报告中的 38 passed / 2 failed 为修复前快照，未改写；修复后测试记录见 [README](README.md#验证结果与后续验收)。

本次验收需确认两侧各完成三轮、各轮 CG 收敛，并另行核查恢复后的自由全场相对残差不超过 1e-6，以及柔顺度等物理量的一致性。CG 收敛不能替代全场残差验证，三轮运行不代表完整优化收敛。失败路径保留已测数据，但不参与完整三轮的总体内存比例计算。

### 2.4 输出与实现标识

本次不运行测试或实验。新 GPU 路径只完成静态检查，不能将历史 NumPy 结果视为 GPU 验证。
运行后检查 history.json 含完整三轮且 CG 均收敛；config.json 和 summary.json 的
`schur_implementation` 应为 `local_fa_cholesky_torch_v1`。隐式输出应为
`interface_matrix_stored=false`、`interface_nnz=null`。

从 history.json 各轮 stages 中提取 `gpu_peak_gib`，以第 1 轮值和第 2–3 轮算术均值填表；
全部阶段最大峰值另覆盖 summary.json 的 setup_stages。summary.json 的
`process_peak_gpu_gib` 为上述已测阶段最大值。轮内峰值在 OC 更新后汇总。
主机 RSS 的 `peak_gib` 和 GPU 的 `gpu_reserved_gib` 另存，不混入 allocated 对比表。

<!-- schur-memory-comparison:begin -->
## 3. 结果对照

**240 与 390 三轮 GPU 显存对照尚未运行。** 数值统一标为“待测”，不代表理论估计或实测结果。

单位 GiB，统计 `torch.cuda.max_memory_allocated` 的阶段峰值，
对应 `gpu_peak_gib`；不是 reserved 或整卡总占用。前四行单元格为 **“第 1 轮阶段峰值 / 第 2–3 轮阶段峰值的算术平均值”**。

| 阶段 | 显式 240 | 隐式 240 | 显式 390 | 隐式 390 |
|---|---:|---:|---:|---:|
| 装配 | 待测 / 待测 | 待测 / 待测 | 待测 / 待测 | 待测 / 待测 |
| 边界条件 | 待测 / 待测 | 待测 / 待测 | 待测 / 待测 | 待测 / 待测 |
| 接口求解 | 待测 / 待测 | 待测 / 待测 | 待测 / 待测 | 待测 / 待测 |
| 全场恢复 | 待测 / 待测 | 待测 / 待测 | 待测 / 待测 | 待测 / 待测 |
| 三轮全部已测阶段最大峰值（含准备阶段） | 待测 | 待测 | 待测 | 待测 |
| 完成状态 / 失败阶段 | 未运行 | 未运行 | 未运行 | 未运行 |

总体比例使用完整三轮的全部已测阶段最大峰值，显式记为 $M_e$，隐式记为 $M_i$。

| 网格 | 隐式/显式峰值 $M_i/M_e$ | 显存降低比例 $1-M_i/M_e$ |
|---|---:|---:|
| 240×40×40 | 待测 | 待测 |
| 390×65×65 | 待测 | 待测 |

统计口径：

- 阶段峰值包含常驻数据和临时分配，各行不能相加；后两轮均值不是时间平均或独立重复实验的平均。
- 隐式装配含内部块分解及精确 Schur 对角准备。比较反映 explicit/fa 与 implicit/fa 两条完整实现路径的差异。
- 最大峰值遍历准备阶段及三轮实际保存的全部阶段，包含过滤、灵敏度、输出及 OC 更新，不保证覆盖测量阶段之外的瞬时峰值。
- 后续两轮记录不全时不计算其均值；任一侧未完整完成三轮时不计算总体比例。失败时注明失败阶段，未记录不等于零。
- 显存分配失败不等于实测 allocated 超过 14 GiB；需记录失败阶段和已测值，不自动填写“>14”。是否节省显存及降低多少，以实测为准。
<!-- schur-memory-comparison:end -->
