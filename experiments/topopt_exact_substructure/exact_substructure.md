# 精确子结构拓扑优化结果对比

## linear_corner 下的完整优化结果

```bash
python experiments/topopt_simp_substructure/run_substructure_full_trace_vs_linear_corner.py --trace linear_corner --grid 390 65 65 --backend pytorch --device cuda --gpu-memory-limit 14 --threads 1
```

| 指标 | linear_corner 390 |
|---|---|
| 终止状态 / 轮数 | 收敛 / 68 |
| 最终 $C$ | $6.974166$ |
| 最终物理体积分数 | $0.1199990$ |
| CG 步数，首轮 / 均值 / 最大 / 总计 | 471 / 1050 / 1633 / $71\,421$ |
| 真残差，末轮 / 最大 | $9.9\times10^{-7}$ / $1.0\times10^{-6}$ |
| 每轮耗时（s），装配 / 求解 / 恢复 / 整轮（均值） | 1.15 / 1.02 / 0.03 / 4.03 |
| 全程耗时（s），准备 / 优化循环 | 28 / 275 |
| 全程峰值显存（GiB） | 5.68 |

![linear_corner 390 最终拓扑](assets/linear_corner_hex_390x65x65_final_topology.png)

数据来源：`results/pytorch_cuda_linear_corner_hex_390x65x65_m5_cg_jacobi_t1/`（由 `outputs/` 同名目录固化，不含 VTU 帧）；拓扑图由 `tools/render_density_topology.py` 从其中的 `density_final.npy` 以默认视角渲染。
