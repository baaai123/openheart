# Proto: Window Context Attribution — 验证结论

**日期**: 2026-05-16

## 问题
窗口归属能否过滤终端噪声？

## 验证结果

### Q1: PowerShell EnumWindows 能在 WSL2 跨层工作吗？
**答案: 否（当前方案）**

- C# inline 代码通过 PowerShell stdin (Popen) 在 WSL2 跨层无法枚举窗口
- `Add-Type` 中的 `DllImport user32.dll` 调用不报错但返回空结果
- Model/OCR 正常：139 图标 + 147 文字

### 替代方案

| 方案 | 可行性 | 代价 |
|---|---|---|
| A: 写 .ps1 文件到磁盘再运行 | 可能可行（绕过 stdin 问题） | 需验证 |
| B: .NET Process.GetProcesses() | MainWindowTitle 可用，但无窗口 rect | 只返回标题和 PID，无位置 |
| C: 视觉窗口检测 | 用边缘检测/颜色采样找窗口边界 | 需额外计算 |
| D: 坐标启发式 | 屏幕左半=桌面区，y<50=标题栏 | 零成本，已在用 |

### 建议
在 PowerShell 修复前，先用**方案 D**（坐标启发式：y>50 过滤 + 左半区域 + conf>0.3），生产管线已经部分实施。

PowerShell 写 .ps1 文件到磁盘的方案可以在非原型阶段尝试。
