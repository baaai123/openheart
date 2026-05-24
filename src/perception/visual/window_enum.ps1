Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections.Generic;

public class WinEnum {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll")] static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
    [DllImport("user32.dll")] static extern int GetClassName(IntPtr hWnd, StringBuilder text, int count);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [DllImport("user32.dll")] static extern IntPtr GetWindow(IntPtr hWnd, uint uCmd);
    [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();

    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left,Top,Right,Bottom; }
    const uint GW_OWNER = 4;
    const int MIN_FULLSCREEN_AREA = 2000000; // ~2M px: catches fullscreen games on any monitor
    
    static List<Dictionary<string,object>> _windows = new List<Dictionary<string,object>>();
    static IntPtr _foregroundHwnd = IntPtr.Zero;
    
    static string GetWindowClass(IntPtr hWnd) {
        var sb = new StringBuilder(256);
        GetClassName(hWnd, sb, 256);
        return sb.ToString().Trim();
    }
    
    static bool Callback(IntPtr hWnd, IntPtr lParam) {
        RECT r;
        if (!GetWindowRect(hWnd, out r)) return true;
        int w = r.Right - r.Left, h = r.Bottom - r.Top;
        if (w <= 0 || h <= 0) return true;
        long area = (long)w * h;
        
        // v5.x FIX: IsWindowVisible returns false for DirectX exclusive-fullscreen
        // games (they don't set WS_VISIBLE in the traditional sense).  Large
        // windows (>2M px) are included regardless of IsWindowVisible status.
        bool isVisible = IsWindowVisible(hWnd);
        if (!isVisible && area < MIN_FULLSCREEN_AREA) return true;
        
        var titleSb = new StringBuilder(256);
        GetWindowText(hWnd, titleSb, 256);
        var title = titleSb.ToString().Trim();
        if (string.IsNullOrEmpty(title) || title == "Program Manager") return true;
        if (GetWindow(hWnd, GW_OWNER) != IntPtr.Zero) return true;
        
        var className = GetWindowClass(hWnd);
        
        _windows.Add(new Dictionary<string,object>{
            {"title",title},
            {"class_name",className},
            {"left",r.Left}, {"top",r.Top},
            {"width",w}, {"height",h},
            {"area",area},
            {"visible",isVisible},
            {"foreground", hWnd == _foregroundHwnd}
        });
        return true;
    }
    
    public static string GetWindows() { 
        _windows.Clear();
        _foregroundHwnd = GetForegroundWindow();
        SetProcessDPIAware();
        EnumWindows(Callback, IntPtr.Zero);
        
        // v5.x FIX: Sort by foreground first, then area descending, then by
        // original EnumWindows order.  Foreground window always gets z=0.
        // Fullscreen games (area >50% of virtual screen or >MIN_FULLSCREEN_AREA)
        // get priority over smaller windows even without foreground flag.
        _windows.Sort((a,b) => {
            bool aFg = (bool)a["foreground"];
            bool bFg = (bool)b["foreground"];
            if (aFg != bFg) return aFg ? -1 : 1;
            long aArea = (long)a["area"];
            long bArea = (long)b["area"];
            // Fullscreen-size windows (>2M px) rank above normal windows
            bool aLarge = aArea >= MIN_FULLSCREEN_AREA;
            bool bLarge = bArea >= MIN_FULLSCREEN_AREA;
            if (aLarge != bLarge) return aLarge ? -1 : 1;
            return bArea.CompareTo(aArea); // larger area first
        });
        for (int i = 0; i < _windows.Count; i++) {
            _windows[i]["z"] = i;
        }
        
        var sb = new StringBuilder("[");
        for(int i=0;i<_windows.Count;i++) {
            var w = _windows[i];
            if(i>0) sb.Append(",");
            sb.Append("{\"title\":\"").Append(w["title"].ToString().Replace("\\","\\\\").Replace("\"","\\\"")).Append("\",");
            sb.Append("\"class_name\":\"").Append(w["class_name"].ToString().Replace("\\","\\\\").Replace("\"","\\\"")).Append("\",");
            sb.Append("\"left\":").Append(w["left"]).Append(",\"top\":").Append(w["top"]).Append(",");
            sb.Append("\"width\":").Append(w["width"]).Append(",\"height\":").Append(w["height"]).Append(",");
            sb.Append("\"z\":").Append(w["z"]).Append(",");
            sb.Append("\"foreground\":").Append((bool)w["foreground"] ? "true" : "false").Append("}");
        }
        sb.Append("]"); return sb.ToString();
    }
}
'@
[WinEnum]::GetWindows()
