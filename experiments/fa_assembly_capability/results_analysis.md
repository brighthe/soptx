# FA 显式装配内存机制与容量能力

## 实验环境

| 项目 | 值 |
|:---|:---|
| 宿主 | Windows 11 Pro，64 GB 物理内存 |
| 运行环境 | WSL2 Ubuntu 24.04 LTS，内核 `6.18.33.2-microsoft-standard-WSL2` |
| CPU | Intel Core i9-14900KF，WSL 视图 32 逻辑核 |
| WSL 内存 | `.wslconfig` `memory=48GB`，来宾 `MemTotal` 47.04 GiB，swap 12 GiB |
| **峰值内存预算** | **45 GiB**：全部容量结论以进程绝对峰值 RSS 不超过 45 GiB 为界，`compare.py` 的天花板外推按此计算 |
| GPU | NVIDIA GeForce RTX 5080，16 GB 显存（仅 `_cuda` 产物，不进入 CPU 容量结论） |
| Python 栈 | Python 3.12.13，numpy 2.5.1，scipy 1.18.0，torch 2.13.0+cu130 |

## 测量口径

- 受控单进程 CPU，3D tet4 线弹性制造解问题，`n^3` 剖分，`NC = 6n^3`，`Ndof = 3(n+1)^3`。
- 每阶段记录开始时 `VmRSS`、阶段内 `VmHWM` 峰值（`/proc/self/clear_refs` 逐阶段重置）与净增；进程绝对峰值取全部阶段峰值的最大值。
- OOM 由绝对峰值决定，容量天花板以端到端绝对峰值 KB/dof 外推，并以 `.failed.json` 中内核 OOM 记录作上界。
- 所有数字均可在 `outputs/` 的产物文件中反查，产物命名 `<kind>_<method|route>_n<N>[_cuda][_bform].json`。

---

## 一、阶段 1：单刚计算内存对比

复现命令（在 `experiments/fa_assembly_capability/` 下）：

```bash
python run.py --case element-stiffness --method all --grid 32
python compare.py --case stage1
```

产物：`outputs/stage1_fast_n32.json`、`outputs/stage1_standard_n32.json`、`outputs/stage1_voigt_n32.json`。

测量对象：`LinearElasticIntegrator(material, method=<m>).assembly(vs)`，输出 $K_e$ 形状 `(NC, 12, 12)` float64。n=32：`NC` = 196,608，`Ndof` = 107,811，$K_e$ 理论体积 216.0 MiB。峰值为阶段内 `VmHWM`，含网格与空间的常驻；净增 = 峰值 − 阶段开始 RSS。

| method | 实现 | 阶段内峰值 RSS | 相对 fast | 耗时 |
|:---|:---|:---:|:---:|:---:|
| `fast` | 参考单元上预计算 `S(LDOF, LDOF, BC, BC)`，逐单元用 $\nabla\lambda$ 缩并 | 1.41 GiB | $1.00\times$ | 0.80 s |
| `standard` | 9 个 `(NC, NQ, 4, 4)` 分块梯度张量 `A_xx ... A_zz` 分别求积后按 $D$ 系数组合 | 5.84 GiB | $7.88\times$ | 6.3 s |
| `voigt` | 物化应变矩阵 `B(NC, NQ, 6, 12)`，单次 `einsum` 完成 $B^{T} D B$ | 11.90 GiB | $17.30\times$ | 9.7 s |

`fast` 的净增是 $K_e$ 理论体积的 3 倍，来自求积与缩并的中间张量；`standard` 与 `voigt` 的膨胀分别来自 9 个逐求积点分块张量和 `(NC, NQ, 6, 12)` 应变矩阵的全量物化。阶段 1 单价不单独决定容量天花板，天花板见第三部分端到端峰值。

---

## 二、阶段 2：总刚合并路线对比
 
> 复现命令：`python experiments/fa_assembly_capability/run.py --case global-merge --route all`
 
| 合并路线与载体 | 装配机制 | 是否物化全长三元组 | 每三元组内存 | 每自由度内存 $b$ | 47G 天花板 $N_{\max}$ | 生产状态 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **FEALPy `COOTensor.coalesce`** | 全长三元组张量排序去重 | 是 | $51.9\sim 77.2\text{ B}$ | $14.9\sim 22.2\text{ KB}$ | **约 230 万** | 传统历史基准 |
| **SciPy `coo_matrix.tocsr`** | 全长三元组 C++ 计数分桶去重 | 是 | $46.0\text{ B}$ | $13.3\text{ KB}$ | 约 381 万 | 未接入 |
| **SOPTX `CSRPattern` + 模式先行** | 预建 CSR 骨架 + 原地 scatter-add | **否** | **$6.9\text{ B}$** | **$2.0\text{ KB}$** | **约 3580 万** | **当前生产默认** |
 
* **归因**：传统 COO 路线必须在内存中物化全长三元组数组，排序/分桶将传统 FA 锁死在 230 万；**SOPTX 当前生产默认的模式先行路线在符号阶段锁定 CSR 骨架、数值阶段原地累加**，彻底消灭全长 COO 数组，单价暴降至 $2.0\text{ KB/dof}$。
 
---
 
## 三、实测阶梯验证与三档组合天花板
 
> 复现命令：`python experiments/fa_assembly_capability/run.py --case full-assembly --grid <32|80|96>`
 
| 网格规模 $n$ | 自由度数 $N_{\text{dof}}$ | 传统历史基准 (`fast + coalesce`) | 中间过渡 (`fast + scipy`) | 当前生产默认 (`fast + pattern`) | 运行状态与物理定论 |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$n = 32$** | 10.8 万 | $1.51\text{ GiB}$ ($15.1\text{ KB/dof}$) | $1.38\text{ GiB}$ ($13.3\text{ KB/dof}$) | **$0.20\text{ GiB}$** ($2.0\text{ KB/dof}$) | 极速完成 |
| **$n = 80$** | 159.4 万 | **$22.83\text{ GiB}$** ($15.4\text{ KB/dof}$) | $20.4\text{ GiB}$ | **$3.04\text{ GiB}$** | 传统基准安全上限（占内存 43~48%） |
| **$n = 96$** | 273.8 万 | **$> 52\text{ GiB}$ (击穿 47G 物理红线)** | $35.1\text{ GiB}$ (逼近物理红线) | **$5.22\text{ GiB}$** | **传统基准被 Linux OOM 强杀**；当前生产模式先行仍极宽裕 |
| **天花板** | — | **极限约 230 万自由度** ($n \le 90$) | **极限约 382 万自由度** ($n \le 105$) | **极限约 3580 万自由度** ($n \approx 225$) | **模式先行实现 $15.6\times$ 规模跃升** |
 
---
 
## 参考
 
* 测量执行与单点看板：`python experiments/fa_assembly_capability/run.py --case <id> [--method <m>] [--route <r>]`（全量运行：`--all`）
* 后处理综合对比报表：`python experiments/fa_assembly_capability/compare.py --case all [-n 32]`
* 产物数据读取：直接消费 `outputs/*.json` 即可进行绘图或下游分析。
