# RAG Studio v0.6.1

这是面向 Windows PowerShell 5.1 的兼容性修正版。

- 所有根目录 PowerShell 脚本统一保存为 UTF-8 with BOM。
- 所有 PowerShell 脚本统一使用 CRLF 换行。
- 增加发布回归测试，验证脚本字节头、换行与 UTF-8 解码。
- 修复中文 Windows 环境中脚本乱码、字符串引号被误解析以及随后的 `UnexpectedToken` 报错。

安装流程不变：首次运行 `setup.ps1`，以后运行 `start.ps1`。
