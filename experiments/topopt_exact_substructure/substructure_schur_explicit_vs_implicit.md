# 显式与局部刚度隐式 Schur 补的显存对比

## 1. 方法差异

两条路径均采用 full_trace 空间和已装配的局部刚度，使用相同的内部 Cholesky 分解、精确 Schur 对角 Jacobi 预条件器及位移恢复策略。将子结构自由度分为内部 $i$ 和边界 $b$，局部 Schur 补为

$$
S=K_{bb}-K_{bi}K_{ii}^{-1}K_{ib}.
$$

**显式 Schur** 形成局部 $S$，并装配、存储全局接口 CSR，CG 通过稀疏矩阵乘法计算接口作用。内部消元会在边界自由度之间引入新的耦合，产生填充，因此接口未知量减少并不保证矩阵存储减少；局部 Schur 形成和全局 CSR 构建还会产生临时分配，推高装配峰值。

**局部刚度隐式 Schur** 保存稀疏 $K_{bb}$、$K_{ib}$ 和内部因子，利用对称性取 $K_{bi}=K_{ib}^{\mathsf T}$，按需计算

$$
Sx=K_{bb}x-K_{bi}\operatorname{solve}(K_{ii},K_{ib}x).
$$

各子结构结果散加到全局接口，不形成局部 Schur 矩阵或全局接口 CSR。这样避开了消元填充的存储，但每次 CG 作用增加局部乘法、内部三角求解和散加。每轮内部块只分解一次，迭代中复用因子。

两者在精确算术下作用同一接口算子。当前内部无载荷、齐次支承算例中，位移恢复共用内部因子与耦合块，不重复装配、分解或另存完整恢复矩阵，因此下表合并统计算子与恢复缓存。

<!-- schur-memory-comparison:begin -->
## 2. 显存结果

单位 GiB，表中取第 1 轮。240 和 390 分别表示 240×40×40 和 390×65×65 网格。

| 部分 | 显式 240 | 隐式 240 | 显式 390 | 隐式 390 |
|---|---:|---:|---:|---:|
| 共有部分（常驻） | 0.74 | 0.74 | 3.07 | 3.07 |
| Schur 算子及恢复缓存（常驻增量） | 7.64 | 1.38 | 未完成 | 5.89 |
| 边界条件（常驻增量） | 0.06 | 0.06 | 未执行 | 0.27 |
| 装配阶段峰值 | 10.53 | 3.18 | OOM（14 GiB 上限） | 10.00 |
| 求解阶段峰值 | 8.50 | 2.30 | 未执行 | 9.56 |
| 恢复阶段峰值 | 8.47 | 2.21 | 未执行 | 9.29 |

常驻项按 gpu_after_gib 统计：共有部分取首轮过滤结束占用，算子及恢复缓存取装配结束减过滤结束，边界条件取边界阶段结束减装配结束；这些是包含辅助缓存的净驻留增量。阶段峰值取 torch.cuda.max_memory_allocated（gpu_peak_gib），包含常驻和临时分配，不能与常驻行相加，不含主机 RSS，也不等同于 reserved 或整卡占用。

第 2–3 轮装配、求解、恢复峰值的算术均值依次为：显式 240 **10.59、8.52、8.50**；隐式 240 **3.23、2.32、2.23**；隐式 390 **10.24、9.65、9.39** GiB。三轮全部已测阶段（含准备阶段）的最大峰值分别为 **10.59、3.24、10.25 GiB**，均出现在第 3 轮装配，与 summary.process_peak_gpu_gib 一致。

240 两侧共有部分相同，主要差异在算子及恢复缓存：显式 7.64 GiB，隐式 1.38 GiB。显式装配结束占用约 8.37 GiB，低于阶段峰值 10.53 GiB，体现了临时分配的成本。隐式全程已测峰值降低约 **69.4%**，但后两轮平均求解耗时从 15.149 s 增至 23.232 s，增加约 **53.4%**，反映了存储与重复计算的取舍。

显式 390 首轮装配 OOM，失败前已测峰值为 7.50 GiB，随后申请额外 17.26 GiB 失败；14 GiB 是分配器上限，不是实测峰值。隐式 390 在相同预算下完成三轮。显式未完成首轮，不计算其后两轮均值或完整运行的显存降低比例。

数据来自用户提供的 history/summary/failure JSON：无编号文件为显式 240，(1) 为隐式 240，(2) 为隐式 390，failure.json 为显式 390。三组成功记录均完成三轮、各轮 CG 接口相对残差小于 1e-6；max_iter 不表示优化收敛，恢复后的全场误差尚未核查。
<!-- schur-memory-comparison:end -->

## 3. 四组运行命令

采用三维 MBB、Q1 六面体、二阶积分、泊松比 0.3、E0=1、Emin=1e-7、SIMP 指数 3、细单元边长 1、体积分数及初始密度 0.12、过滤半径 3。两侧统一 full_trace、FA、PyTorch/CUDA、CG/Jacobi、n_fine=5、chunk_size=256、单线程和三轮；CG 使用 rtol=1e-6、atol=0、相对右端项验收、最多 20000 步。模型见 [full_trace_vs_fem.md](full_trace_vs_fem.md)。

在仓库根目录、支持 CUDA 的 SOPTX Python 环境中依次运行，不并发运行：

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

结果写入本目录 outputs/。运行前备份已有结果，脚本拒绝覆盖或续算；保留 config.json、history.json、summary.json，失败时保留 failure.json。四组使用相同的 14 GiB PyTorch 分配器上限。
