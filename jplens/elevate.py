"""系統管理員權限：遊戲（絕區零、鳴潮、星塔旅人的啟動器…）以系統管理員身分執行時，
Windows 的 UIPI 不讓一般權限程式的低階鍵盤／滑鼠掛鉤收到在那個視窗上的按鍵，查字工具就完全沒反應
（AutoHotkey 也有同樣的限制）。解法是查字工具也用系統管理員身分執行。

- is_admin()：自己是不是系統管理員身分。
- foreground_elevated()：前景視窗的程式是不是比自己高權限（用來在遊戲裡提醒一次）。
- relaunch_as_admin()：用 UAC（runas）重新啟動自己；使用者按「否」回傳 False。
"""

import ctypes
import ctypes.wintypes as wt
import os
import sys
from pathlib import Path

from .config import FROZEN, ROOT

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_ELEVATION = 20  # TOKEN_INFORMATION_CLASS.TokenElevation
WAIT_FLAG = "--wait-previous"  # 重新啟動時帶上：等舊的程序結束、放掉「只能開一個」的鎖

kernel32.OpenProcess.restype = wt.HANDLE
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
advapi32.OpenProcessToken.argtypes = [wt.HANDLE, wt.DWORD, ctypes.POINTER(wt.HANDLE)]
advapi32.GetTokenInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD)]
user32.GetForegroundWindow.restype = wt.HWND
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
shell32.ShellExecuteW.restype = ctypes.c_void_p
shell32.ShellExecuteW.argtypes = [wt.HWND, wt.LPCWSTR, wt.LPCWSTR, wt.LPCWSTR, wt.LPCWSTR, ctypes.c_int]


def is_admin() -> bool:
    try:
        return bool(shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001
        return False


def _elevated(hproc) -> bool | None:
    """程序的 token 是否提升過；開不了 token（權限比自己高）回傳 None。"""
    tok = wt.HANDLE()
    if not advapi32.OpenProcessToken(hproc, TOKEN_QUERY, ctypes.byref(tok)):
        return None
    try:
        val, size = wt.DWORD(), wt.DWORD()
        if not advapi32.GetTokenInformation(tok, TOKEN_ELEVATION, ctypes.byref(val), ctypes.sizeof(val),
                                            ctypes.byref(size)):
            return None
        return bool(val.value)
    finally:
        kernel32.CloseHandle(tok)


def foreground_elevated() -> tuple[bool, str]:
    """(前景視窗的程式權限比自己高?, 執行檔名稱)。查不到（權限不夠開程序或 token）也當成較高權限。"""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return False, ""
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value or pid.value == os.getpid():
        return False, ""
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h:
        return True, ""
    try:
        buf, n = ctypes.create_unicode_buffer(1024), wt.DWORD(1024)
        name = Path(buf.value).name if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else ""
        el = _elevated(h)
        return (True if el is None else el), name
    finally:
        kernel32.CloseHandle(h)


def relaunch_as_admin() -> bool:
    """用 UAC 以系統管理員身分重新啟動查字工具（新的程序會等這個結束）；成功送出回傳 True。"""
    exe = Path(sys.executable)
    if FROZEN:  # 打包的 jplens.exe
        params = WAIT_FLAG
    else:
        pyw = exe.with_name("pythonw.exe")
        exe = pyw if pyw.exists() else exe
        params = f"-m jplens {WAIT_FLAG}"
    r = shell32.ShellExecuteW(None, "runas", str(exe), params, str(ROOT), 1)
    return (r or 0) > 32
