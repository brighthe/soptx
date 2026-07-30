# 数值验证与 evidence 政策

## 快速 CI

Python 3.10 和 3.12 执行安装、公共导入、unit tests、依赖方向检查、确定性清单检查、
wheel build 与 wheel 内容检查。CI 不运行 MPI、PINN 训练、论文矩阵或长期 benchmark。

## 本地重验证

- Matrix-Free：示例 pytest、`validate.py --dim all`、evidence 同步检查。
- PINN：`validate.py --dim all`。
- Hu–Zhang：`experiments/huzhang_topopt_paper/dry_run.py --json`
  执行配置 schema、源哈希与实验矩阵完整性检查；数值基线仍需单独选定。

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
