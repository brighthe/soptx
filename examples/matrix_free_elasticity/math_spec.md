# 线弹性 Matrix-Free EA/FA 算子与重叠副本代数规范 (Math Spec)

本文档规范 `soptx/examples/matrix_free_elasticity` 中 EA/FA 两级离散算子、重叠
副本 MPI 布局与加权 CG 的代数约定，并把 [`utils/contract.py`](utils/contract.py)
中的每个数值门禁写成可复核的数学式。

> **理论事实源**（本文档只做映射与判据，不复述这些页面的内容）：
>
> - 五级装配层次与框架术语映射：`dut-postdoc:concepts/matrix-free/assembly-levels.md#五级分类`
> - MPI 共享自由度、同步归约与正确性不变量：`dut-postdoc:concepts/gpu-hpc/distributed-operator-and-shared-dofs.md`
> - 线弹性变分形式与有限元离散：`dut-postdoc:concepts/linear-elasticity.md#线弹性方程变分形式与有限元离散`
> - 三个制造解的完整方程、参数与 shape 契约：[制造解文档](../../docs/problems/manufactured-elasticity.md)

---

## 1. 符号—代码映射表 (Symbol-to-Code Mapping)

设区域 $\Omega\subset\mathbb R^d$（$d\in\{2,3\}$）的一致剖分为
$\{\Omega_e\}_{e=1}^{N_e}$，一阶连续 Lagrange 向量元空间的全局自由度数为
$n=\texttt{TGDOF}$。

| 数学符号 | 代码位置 | 代数含义 |
|---|---|---|
| $\mathbf K_e\in\mathbb R^{m\times m}$ | `LinearElasticIntegrator` 计算、`BilinearForm` 在 `'ea'` 下保存的单元刚度张量 | 单元刚度矩阵，$m=(d+1)d$ |
| $\mathbf R_e\in\{0,1\}^{m\times n}$ | `space.cell_to_dof()` | 单元自由度限制算子（gather），$\mathbf R_e^{\mathsf T}$ 为 scatter-add |
| $\mathbf K\in\mathbb R^{n\times n}$ | `assemble_stiff_matrix.set('fa')` 返回的 `CSRTensor` | 全局刚度矩阵 |
| $\mathbf E_p\in\{0,1\}^{n\times n_p}$ | [`utils/distributed.py:_vector_dof_masks`](utils/distributed.py) 得到的 DOF 掩码 | rank $p$ 的局部自由度嵌入算子 |
| $r_i\in\mathbb Z_{>0}$ | `dof_comm.refs(size)` | 自由度 $i$ 的副本数（被多少个 rank 持有） |
| $w_i=1/r_i$ | `dof_comm.dot(size)` / [`solver.py`](solver.py) 内部 | 重叠内积权重 |
| $\mathcal S(\cdot)$ | `dof_comm.sync_add(...)` / [`utils/distributed.py:OverlapOperator`](utils/distributed.py) | 跨 rank 共享分量求和 |
| $\mathbf P_D,\ \mathbf P_I$ | `DirichletBCOperator.is_boundary_dof` 及其补 | Dirichlet／内部自由度上的对角投影，$\mathbf P_D+\mathbf P_I=\mathbf I$ |
| $\bar{\boldsymbol u}$ | `_prescribed_solution` / `ElasticityEAOperator.prescribed_solution` | 边界取给定值、内部取零的基准向量 |

两级算子对应同一个离散算子，差别只在保存什么、什么时候重算：

$$
\mathbf K=\sum_{e=1}^{N_e}\mathbf R_e^{\mathsf T}\mathbf K_e\mathbf R_e .
$$

---

## 2. EA 与 FA 的保存／省略对象

| 层级 | 保存对象 | 省略对象 | 每次 MatVec 执行 |
|---|---|---|---|
| `fa` | 全局 CSR $\mathbf K$（`OPERATOR_STORAGE['fa'] = "global-csr"`） | 无 | 稀疏矩阵乘 |
| `ea` | 单元矩阵集合 $\{\mathbf K_e\}$（`"cached-element-matrices"`） | 全局矩阵 $\mathbf K$ | $\boldsymbol y\leftarrow\sum_e\mathbf R_e^{\mathsf T}\bigl(\mathbf K_e(\mathbf R_e\boldsymbol x)\bigr)$ |

`ea` 每次作用重复 gather—单元乘—scatter-add，用算术换存储；$\mathbf K_e$ 本身仍
被完整形成并保存，因此按理论事实源的判定口径它属于 **EA/EbE**，而不是 PA/QA。
本阶段不实现 PA/QA 与 UA/NONE，`contract.OPERATOR_LEVELS` 相应只有
`("ea", "fa")`。

---

## 3. 重叠副本布局的向量表示与算子代数（阶段 1b）

> **本节是阶段 1b（CPU 并行 EA）的代数基础，不属于阶段 1a 的验证范围。**
> 阶段 1a 只跑单 rank，此时本节全部跨 rank 归约都退化为恒等（见 §3.3），串行
> 路径与并行路径是同一段代码。1a 的证据不包含任何跨 rank 结论。

本模块的并行分布式实现直接继承 FEALPy 的 **EMPI (Entity Message Passing Interface)** 架构（即 `fealpy.distributed.EntityMPI`）：
- **实体共享与引用计数 ($r_i$)**：使用 `dof_comm.refs(size)` 统计交界面共享实体的副本数；
- **跨 Rank 同步归约 ($\mathcal S$)**：使用 `dof_comm.sync_add(...)` 将各进程上的分量累加同步；
- **重叠点积与内积注入 ($(\cdot, \cdot)_w$)**：使用 `dof_comm.dot(size)` 构建带 $1/r_i$ 重叠修正的加权内积。

### 3.1 分区假设

`partition_cells` 产生的单元掩码是**互不相交且完全覆盖**的（代码中以
`coverage == 1` 断言），因此局部刚度算子

$$
\mathbf K^{(p)}=\sum_{e\in\Omega_p}\mathbf R_e^{\mathsf T}\mathbf K_e\mathbf R_e
\quad(\text{局部编号})
$$

满足精确分解

$$
\mathbf K=\sum_{p=0}^{P-1}\mathbf E_p\,\mathbf K^{(p)}\,\mathbf E_p^{\mathsf T}. \tag{3.1}
$$

单元不重叠、自由度重叠：交界面上的自由度在两个 rank 上各有一份副本，地位对等
（`DISTRIBUTED_REPRESENTATION = "equal-status-overlapping-copies"`），不设 owner。

### 3.2 两种向量表示

局部向量族 $\{\boldsymbol v^{(p)}\}$ 有两种约定，**必须始终分清**：

- **一致表示 (consistent)**：$\boldsymbol v^{(p)}=\mathbf E_p^{\mathsf T}\boldsymbol v$，
  每份副本持有相同的全局值；
- **加和表示 (additive)**：$\sum_p\mathbf E_p\boldsymbol v^{(p)}=\boldsymbol v$，
  每份副本只持有本 rank 的贡献。

$\mathcal S$ 把加和表示转成一致表示：

$$
\mathcal S(\boldsymbol v)^{(p)}
=\mathbf E_p^{\mathsf T}\sum_{q}\mathbf E_q\boldsymbol v^{(q)} . \tag{3.2}
$$

对已经一致的向量，$\mathcal S(\boldsymbol v)^{(p)}=r_i$ 倍的原值（逐分量），故

$$
\mathcal C(\boldsymbol v):=\mathcal S(\boldsymbol v)\oslash\boldsymbol r
$$

是**幂等投影**，且在一致表示上是恒等映射（$\oslash$ 为逐分量除）。

反向的转换出现在结果收集：[`utils/run.py`](utils/run.py) 调用
`dof_comm.gather_add(local_solution / references)`，即 $\boldsymbol x\oslash\boldsymbol r$
先把一致表示按副本数均分成加和表示，再跨 rank 求和，才能在 rank 0 得到不重复
计数的全局唯一解向量。
凡是出现 $\oslash\boldsymbol r$ 的地方，都在做「一致 ↔ 加和」的表示转换，而不是
某种加权平均。

### 3.3 `OverlapOperator` 的三步及其恒等性

[`utils/distributed.py`](utils/distributed.py) 中 `OverlapOperator.__matmul__` 为

$$
\boldsymbol x\;\longmapsto\;
\mathcal S\Bigl(\mathbf K^{(p)}\,\mathcal C(\boldsymbol x)\Bigr).
$$

代入 (3.1)(3.2)：若输入 $\boldsymbol x$ 为一致表示，则 $\mathcal C$ 是恒等，
$\mathbf K^{(p)}\mathbf E_p^{\mathsf T}\boldsymbol x$ 恰是 $\mathbf K\boldsymbol x$
的加和表示，外层 $\mathcal S$ 再把它转回一致表示。于是

$$
\boxed{\;\bigl(\texttt{OverlapOperator}\ @\ \boldsymbol x\bigr)^{(p)}
=\mathbf E_p^{\mathsf T}\,\mathbf K\boldsymbol x\;}\tag{3.3}
$$

即算子把**一致表示映到一致表示**，与全局 $\mathbf K$ 逐分量相符。前置的
$\mathcal C$ 在此约定下是冗余的恒等操作，作用是把「输入必须一致」这一前提变成
算子自身维护的不变量，而不是调用方的口头约定；CG 全程只搬运一致表示，因此
$\mathcal C$ 从不改变数值，只多一轮通信。

### 3.4 加权内积

`dof_comm.dot` 给出

$$
(\boldsymbol u,\boldsymbol v)_w
=\sum_{p}\sum_{i\in\mathcal I_p}\frac{u_i^{(p)}v_i^{(p)}}{r_i}, \tag{3.4}
$$

其中 $\mathcal I_p$ 为 rank $p$ 的局部自由度集，外层求和由 `MPI.SUM` allreduce
完成。对一致表示，全局自由度 $i$ 被计入 $r_i$ 次、每次权重 $1/r_i$，故
$(\boldsymbol u,\boldsymbol v)_w=\boldsymbol u^{\mathsf T}\boldsymbol v$ 精确成立，
$\|\cdot\|_w$ 即全局 Euclid 范数。结合 (3.3)，CG 见到的算子在该内积下对称正定：

$$
(\boldsymbol u,\mathbf K\boldsymbol v)_w=\boldsymbol u^{\mathsf T}\mathbf K\boldsymbol v
=(\mathbf K\boldsymbol u,\boldsymbol v)_w .
$$

这正是 [`solver.py:83`](solver.py:83) 的 `weighted_cg` 只向 `fealpy.solver.cg`
注入 `dot_product` 而不改写 CG 迭代本身的依据：Krylov 递推的全部并行性都集中在
内积。求解器的派发在
[`utils/analyzer.py:74`](utils/analyzer.py:74) 的 `solve_system`，它按 `solver`
名字查 `DISTRIBUTED_SOLVERS` 并把 `dof_comm` 传给对应 routine。
载荷向量由装配产生时是加和表示，`reduce_load` 用 $\mathcal S$ 把它转成一致表示
后才进入边界处理与求解。

---

## 4. Dirichlet 条件在两级算子下的施加

`fa` 走对称消元，直接改写已装配的 CSR；`ea` 不改写任何矩阵，而是把算子包进
`DirichletBCOperator`。后者的 MatVec 为「置零 → 作用 → 还原」：

$$
\tilde{\mathbf K}=\mathbf P_I\mathbf K\mathbf P_I+\mathbf P_D , \tag{4.1}
$$

右端项相应取

$$
\tilde{\boldsymbol F}
=\mathbf P_I\bigl(\boldsymbol F-\mathbf K\bar{\boldsymbol u}\bigr)
+\mathbf P_D\bar{\boldsymbol u} . \tag{4.2}
$$

$\tilde{\mathbf K}$ 显然对称；$\mathbf K$ 在内部自由度上正定时 $\tilde{\mathbf K}$
正定，故 CG 适用。(4.1)(4.2) 与 `fa` 的对称消元给出同一个线性系统，这是第 5 节
`dirichlet_matvec` 与 `explicit_solution` 两道门禁的理论依据。

迭代初值取 $\boldsymbol x_0=\bar{\boldsymbol u}$（已满足 Dirichlet 值），由
`solve_state` 显式传入而非由 `solve_system` 读实例状态。

**多 rank 下 `fa` 被拒绝**：对称消元发生在全局矩阵已经装配之后，(3.3) 的
$\mathcal S$ 没有插入点；若放行，各 rank 会在自己的局部矩阵上求解，不报错但结果
错误。单 rank 下所有 $\mathcal S$ 都是恒等，故 `fa` 携带 `dof_comm` 是安全的。

---

## 5. 数值门禁的数学定义 (Acceptance Criteria)

阈值全部来自 [`utils/contract.py`](utils/contract.py)，此处只给出对应的数学式；
两侧不得各持一份字面量。

### 5.1 单次运行门禁（`utils/run.py` 与 `utils/validate.py` 共用）

| 门禁 | 数学式 | 阈值 |
|---|---|---|
| `converged` | CG 正常退出且无 breakdown | — |
| `true_residual` | $\lVert\tilde{\mathbf K}\boldsymbol x-\tilde{\boldsymbol F}\rVert_w\le\max\bigl(\varepsilon_a,\ \varepsilon_r\lVert\tilde{\boldsymbol F}\rVert_w\bigr)$ | $\varepsilon_r=10^{-10}$，$\varepsilon_a=10^{-12}$ |
| `boundary_dofs` | $\bigl\lVert\mathbf P_D(\boldsymbol x-\bar{\boldsymbol u})\bigr\rVert_w\le\texttt{BOUNDARY\_ABSOLUTE\_TOL}$ | $10^{-12}$ |
| `raw_matvec` | $\dfrac{\lVert\mathbf K^{\mathrm{EA}}\boldsymbol\xi-\mathbf K^{\mathrm{FA}}\boldsymbol\xi\rVert_2}{\lVert\mathbf K^{\mathrm{FA}}\boldsymbol\xi\rVert_2}\le\texttt{MATVEC\_RELATIVE\_TOL}$ | $10^{-12}$ |
| `dirichlet_matvec` | 同上，把 $\mathbf K$ 换成 (4.1) 的 $\tilde{\mathbf K}$ | $10^{-12}$ |
| `operator_symmetry` | 记 $a=\boldsymbol\xi^{\mathsf T}\tilde{\mathbf K}\boldsymbol\eta$、$b=\boldsymbol\eta^{\mathsf T}\tilde{\mathbf K}\boldsymbol\xi$，要求 $\dfrac{\lvert a-b\rvert}{\max(\lvert a\rvert,\lvert b\rvert)}\le\texttt{SYMMETRY\_RELATIVE\_TOL}$ 且 $\boldsymbol\xi^{\mathsf T}\tilde{\mathbf K}\boldsymbol\xi>0$ | $10^{-12}$ |
| `explicit_solution` | $\dfrac{\lVert\boldsymbol x_{\mathrm{CG}}-\boldsymbol x_{\mathrm{direct}}\rVert_2}{\lVert\boldsymbol x_{\mathrm{direct}}\rVert_2}\le\texttt{EXPLICIT\_SOLUTION\_RELATIVE\_TOL}$ | $10^{-8}$ |

$\boldsymbol\xi,\boldsymbol\eta$ 为固定种子
`REFERENCE_RANDOM_SEED = 20260727` 生成的标准正态随机向量；
$\boldsymbol x_{\mathrm{direct}}$ 由 `spsolve` 在 `fa` 系统上独立求出。这四道带
参照的门禁只在单 rank 非 benchmark 运行下有意义，其余情形记为 `GATE_SKIPPED`
而非通过——否则 `local_passed` 会在 benchmark 模式下悄悄弱化。

对称性以随机向量的双线性配对检验，而非逐元素比较矩阵：`ea` 下根本不存在可以逐
元素比较的矩阵，这是 Matrix-Free 正确性判据必须改写的地方。

### 5.2 跨运行门禁（`utils/validate.py`）

下表中标注「1b」的两道门禁只在 `--include-parallel` 下参与判定；阶段 1a 的默认
范围不含跨 rank 项，`comparison` 里也不写入对应字段，以免空占位被误读为"已检验"。

记网格加密序列 $h_0>h_1>h_2$（`REFINEMENTS`：2D 为 $8,16,32$；3D 为 $4,8,16$），
相对 $L^2$ 位移误差

$$
E_k=\frac{\|\boldsymbol u_h^{(k)}-\boldsymbol u\|_{L^2(\Omega)}}{\|\boldsymbol u\|_{L^2(\Omega)}} ,
$$

观测阶（相邻加密恰为二等分，故取 $\log_2$）

$$
q_k=\log_2\frac{E_{k-1}}{E_k},\qquad k=1,2 .
$$

| 门禁 | 阶段 | 数学式 | 阈值 |
|---|---|---|---|
| EA/FA 解一致 | 1a | $\lVert\boldsymbol x^{\mathrm{EA}}-\boldsymbol x^{\mathrm{FA}}\rVert_2/\lVert\boldsymbol x^{\mathrm{EA}}\rVert_2\le\texttt{EA\_FA\_SOLUTION\_RELATIVE\_TOL}$ | $10^{-9}$ |
| 误差单调 | 1a | $E_0>E_1>E_2$ | — |
| 收敛阶 | 1a | $q_2\ge\texttt{MINIMUM\_FINAL\_L2\_ORDER}$ | $1.5$ |
| 1/2-rank 解一致 | **1b** | $\lVert\boldsymbol x^{(1)}-\boldsymbol x^{(2)}\rVert_2/\lVert\boldsymbol x^{(1)}\rVert_2\le\texttt{PARALLEL\_SOLUTION\_RELATIVE\_TOL}$ | $10^{-9}$ |
| 1/2-rank 误差一致 | **1b** | $\bigl\lvert E_2^{(1\,\mathrm{rank})}-E_2^{(2\,\mathrm{rank})}\bigr\rvert\le\texttt{PARALLEL\_L2\_DIFFERENCE\_TOL}$ | $10^{-10}$ |

EA/FA 解一致检验 (4.1) 与对称消元的等价性，是 1a 的核心代数判据；两道 1b 门禁
检验 (3.3) 中 $\mathcal S$ 与 $\mathcal C$ 的实现。

凡分母出现范数处一律用 $\max(\cdot,\texttt{NORM\_FLOOR})$ 兜底，
$\texttt{NORM\_FLOOR}=10^{-30}$。

---

## 6. 本阶段明确不承诺的内容

- 不实现 PA/QA、UA/NONE，不宣称任何低于 EA 的存储层级；
- 无预条件（`parameters.preconditioner` 恒为 `null`），因此迭代数只反映
  $\tilde{\mathbf K}$ 的条件数，不构成任何预条件结论；
- 只支持 $p=1$、$d\in\{2,3\}$、1/2 ranks；阶段 1a 只跑单 rank，2 ranks 属阶段
  1b 且只验证正确性，不支持任何扩展性结论；
- MatVec 一致不替代完整 solve、真残差与解误差；单 kernel 计时不替代端到端
  时间与峰值内存。

### 6.1 EA 的算术强度边界（后续 GPU 阶段的表述口径）

按 `dut-postdoc:concepts/matrix-free/assembly-levels.md#算术强度：EA 并没有解决 FA 的瓶颈`
的口径，EA 的 apply 每单元读取 $m^2$ 个 double 并执行 $2m^2$ 次浮点运算：

$$
\text{算术强度}_{\mathrm{EA}}\approx\frac{2m^2\ \text{flop}}{8m^2\ \text{byte}}
=0.25\ \text{flop/byte},
$$

与 FA 的 SpMV（$\approx 0.17$ flop/byte）**同量级**。EA 改善的是访存的规则性
（连续块读取而非随机间接寻址），不是访存的总量——每次 apply 仍要把整个
$\{\mathbf K_e\}$ 流过一遍，而这个数组比 FA 的稀疏矩阵更大。

因此，无论 1b 的并行加速还是 1c 的 GPU 加速，其收益来源都是**并行度与访存规则
性**，而**不是算术强度的提高**。这些结果不得表述为「Matrix-Free 通过提高算术
强度获得加速」——那要到 PA/QA 与 UA/NONE 才成立，而本阶段明确不实现它们。
