# 文件分类基线

| 路径/模式 | 分类 | 迁移政策 |
| --- | --- | --- |
| `src/soptx/{core,materials,problems,fem,topology,visualization}` | maintained | 稳定实现与公共 API |
| `src/soptx/{analysis,functionspace,interpolation,model,optimization,regularization,utils}` | compatibility | 仅迁移表中的旧公共 API；有稳定替代时使用薄转发，`2.0.0` 删除 |
| `src/soptx/{old,demo,tests}` | archive candidate | 从 wheel/CI 隔离，归档 tag 后删除 |
| `experiments/legacy/package_archive` | archive candidate | `*_old.py`、`*_backup.py` 和被薄转发替代的完整旧实现，不进入 wheel |
| `tests/` | maintained | 自动化验证，不进入 wheel |
| `examples/matrix_free_elasticity`、`examples/pinn_elasticity` | incubating | 严格验证的孵化示例 |
| `experiments/` | experiment | 论文与长期运行，不进入 wheel |
| `reference_code/` | archive candidate / unredistributable | 只保留哈希，授权 tag 后退出 main |
| `docs/`、`tools/`、`.github/` | maintained | 治理、检查与文档 |

`docs/architecture/current-python-files.sha256` 给出当前 Python 文件级清单。
compatibility 只保存必要接口；完整 archive 实体位于 `experiments/legacy`，避免与
maintained 实现形成两个可安装来源。
