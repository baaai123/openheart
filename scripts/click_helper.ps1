<#
.SYNOPSIS
    Click at screen coordinates via Win32 API — used by MouseChannel.click_at().
.DESCRIPTION
    Calls SetProcessDPIAware (for HiDPI physical-pixel correctness),
    then SetCursorPos + mouse_event for the click action.
.PARAMETER x
    Screen X coordinate (physical pixels).
.PARAMETER y
    Screen Y coordinate (physical pixels).
.PARAMETER right
    Switch: use right mouse button instead of left.
.EXAMPLE
    .\click_helper.ps1 -x 100 -y 200
    .\click_helper.ps1 -x 100 -y 200 -right
#>

param(
    [int]$x,
    [int]$y,
    [switch]$right
)

# DPI awareness: ensure coordinates are interpreted as physical pixels
# (same DPI context as SetProcessDPIAware used in screenshot capture).
Add-Type -Name NativeDPI -Namespace Temp -MemberDefinition @"
[DllImport("user32.dll")]
public static extern bool SetProcessDPIAware();
"@
[Temp.NativeDPI]::SetProcessDPIAware() | Out-Null

Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point($x, $y)

# Register mouse_event P/Invoke once (shared by left/right click paths)
Add-Type -MemberDefinition @"
[DllImport("user32.dll")]
public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, int dwExtraInfo);
"@ -Name Win32Mouse -Namespace Win32

if ($right) {
    # MOUSEEVENTF_RIGHTDOWN = 0x0008, MOUSEEVENTF_RIGHTUP = 0x0010
    Start-Sleep -Milliseconds 50
    [Win32.Win32Mouse]::mouse_event(0x0008, $x, $y, 0, 0)
    Start-Sleep -Milliseconds 50
    [Win32.Win32Mouse]::mouse_event(0x0010, $x, $y, 0, 0)
} else {
    # MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004
    Start-Sleep -Milliseconds 50
    [Win32.Win32Mouse]::mouse_event(0x0002, $x, $y, 0, 0)
    Start-Sleep -Milliseconds 50
    [Win32.Win32Mouse]::mouse_event(0x0004, $x, $y, 0, 0)
}
