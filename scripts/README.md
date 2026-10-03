# scripts

开发期用的零散脚本：跑资产生成器、跑基准集、一键重装 bundle。

**这里只放胶水。** 有逻辑的东西应该进 `src/ark/`，否则它会成为第二个没人管的代码库。

- [check_docs.py](check_docs.py)：文档树的自检（状态行 · 取值 · 断链 · 已废术语）。
- [filehash.py](filehash.py)：核规格文件的哈希，口径是**按 LF 折算**的 sha256（[D-全局-18](../docs/20-决策记录/全局.md)）。派任务与交回时用它，不要用 `Get-FileHash`。
