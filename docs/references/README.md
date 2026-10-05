# 第三方参考代码治理

`reference_code/` 当前包含 214 个 tracked 文件，约 52 MB。仓库内未找到统一的
README、LICENSE、COPYING 或 NOTICE，来源和许可证状态尚未逐项核实。因此：

- 不把这些文件声明为 SOPTX GPL-3.0 自有代码；
- 不随 wheel、source distribution 或文档发布物分发；
- 在来源和许可证确认前视为不可再发布；
- 只在 `reference-code-manifest.sha256` 保存路径与内容校验值；
- 获得 `archive/pre-v2` tag 的明确授权后，实体才从 main 删除。

清单可用以下命令验证：

```powershell
python tools/generate_repository_inventory.py --check
```
