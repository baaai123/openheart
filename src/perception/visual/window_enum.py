"""Windows window hierarchy enumeration via PowerShell .ps1 file."""
import json, os, subprocess

_PS1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "window_enum.ps1")

def get_window_hierarchy() -> list[dict[str, object]]:
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", _PS1_PATH],
            capture_output=True, timeout=15,
        )
        if r.returncode != 0 or not r.stdout:
            return []
        stdout_bytes = r.stdout
        text = None
        for enc in ["utf-16", "utf-8", "gbk", "cp936", "latin-1"]:
            try:
                decoded = stdout_bytes.decode(enc)
                if decoded.strip().startswith("[") and decoded.strip().endswith("]"):
                    text = decoded
                    break
            except (UnicodeDecodeError, LookupError):
                continue
        if text is None:
            return []
        return json.loads(text.strip())
    except Exception:
        return []
