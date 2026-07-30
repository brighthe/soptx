# Hu–Zhang topology optimization paper experiment

状态：`incubating / inventory-complete / not yet reproducible`。

`legacy_drivers/` 是旧论文和博士论文章节运行脚本的迁移快照。当前工作树没有一个
已验证、统一的 Hu–Zhang 数值 `run.py` 和 acceptance baseline，因此本目录不能声明
为可复现实验。

`matrix.toml` 对 7 个论文 driver 中的全部 36 个 variant selector 建立显式清单，
并为每项记录 chapter、dimension、role、method、输出名称和阻塞状态。另一个不含
selector 的字体/有限差分工具脚本被显式列为 excluded source。

`dry_run.py` 不导入、不执行 legacy driver；它只验证：

- TOML schema、stage 和 `incubating` 状态；
- 所有 legacy Python 源文件均被纳入或显式排除；
- driver SHA-256 未漂移；
- AST 中发现的 selector 与 matrix 恰好一一对应；
- case id、输出名称、chapter、dimension、role 和阻塞状态合法。

在仓库根目录运行：

```powershell
python .\experiments\huzhang_topopt_paper\dry_run.py --json
```

dry-run 通过仅表示静态清单完整，不表示任何 Hu–Zhang 数值算例已经可以运行或通过
验证。

晋级条件：

1. 将每个阻塞 case 的内嵌参数迁为可执行配置；
2. 删除对 compatibility 命名空间的依赖；
3. 选择并重跑数值基线；
4. 记录 clean revision、环境、参数和产物 SHA-256；
5. 建立 acceptance criteria 后再把 case 状态从 `blocked` 晋级。
