# 数值验证与 evidence 政策

## 快速 CI

Python 3.10 和 3.12 执行安装、公共导入、unit tests、依赖方向检查、确定性清单检查、
wheel build 与 wheel 内容检查。CI 不运行 MPI、PINN 训练、论文矩阵或长期 benchmark。

## 本地重验证

- Matrix-Free：示例 pytest、`validate.py --dim all`、evidence 同步检查。
- PINN：`validate.py --dim all`。
- Hu–Zhang：`experiments/huzhang_topopt_paper/dry_run.py --json` 执行配置 schema、
  源哈希与实验矩阵完整性检查；`run.py --case forward-manufactured` 是独立的正式
  制造解矩阵入口，固定输出 40 行（4 次数、2 网格族、5 层）。

这些命令由用户在明确环境中运行。结果必须检查退出码、预期产物和数值 acceptance
criteria，不能仅凭命令结束判断成功。

## 正式 evidence

正式 evidence 必须记录：

- clean Git revision；
- dirty flag；
- Python、FEALPy、NumPy、SciPy、SymPy 及可选运行时版本；
- 完整参数和随机种子；
- 源产物 SHA-256。

dirty worktree 运行只能标记为开发证据。架构迁移前已有 3D Matrix-Free evidence 是
历史基线，不能自动声明为迁移后的验证结果。

Hu–Zhang 制造解将 `peak_python_bytes` 明确解释为 Python traced peak，不能误写为
完整进程峰值 RSS；完整成本比较需另行采集 RSS。正式收敛证据还必须保留
`summary.json`、`manifest.json`、40 行 `metrics.csv`、图件和每项产物 SHA-256。
