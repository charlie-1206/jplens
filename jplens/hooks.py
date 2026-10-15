"""全域低階滑鼠／鍵盤掛鉤（Win32 SetWindowsHookEx，不需要系統管理員權限）。

按住設定的鍵（預設 Alt）時按下滑鼠左鍵 → 開始框選；放開左鍵 → 框選完成。
框選期間的左鍵按下／放開會被攔截，底下的程式（遊戲、瀏覽器）收不到，所以不會誤點。
座標是實體像素（程序需設為 per-monitor DPI aware，Qt 6 預設就是）。

掛鉤回呼必須很快回傳（太慢 Windows 會把掛鉤移除），所以這裡只記錄狀態並轉發事件，
實際工作交給 Qt 主執行緒。
"""

import ctypes
import ctypes.wintypes as wt
import threading

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
WM_QUIT = 0x0012
WM_KEYDOWN, WM_SYSKEYDOWN, WM_KEYUP, WM_SYSKEYUP = 0x0100, 0x0104, 0x0101, 0x0105
WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP = 0x0200, 0x0201, 0x0202
WM_RBUTTONDOWN, WM_RBUTTONUP, WM_MBUTTONDOWN = 0x0204, 0x0205, 0x0207
LLKHF_INJECTED = 0x10
VK_ESCAPE = 0x1B
VK_MASK = 0xE8        # 未指派的虛擬鍵：框選時送出，避免放開 Alt 後程式跳到選單列（AutoHotkey 同樣做法）
KEYEVENTF_KEYUP = 0x2
MIN_DRAG = 6          # 拖曳兩個方向都小於這個像素數視為點擊（取消）
STUCK_MS = 1500       # 熱鍵兩次「按下」事件間隔超過這麼久，表示中間漏收了放開
# 低階鍵盤掛鉤回報的是分左右的鍵碼，換成設定檔用的通用鍵碼（Shift／Ctrl／Alt）
_GENERIC_VK = {0xA0: 0x10, 0xA1: 0x10, 0x10: 0x10, 0xA2: 0x11, 0xA3: 0x11, 0x11: 0x11,
               0xA4: 0x12, 0xA5: 0x12, 0x12: 0x12}

LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wt.WPARAM, wt.LPARAM)


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wt.DWORD), ("scanCode", wt.DWORD), ("flags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", wt.POINT), ("mouseData", wt.DWORD), ("flags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wt.HINSTANCE, wt.DWORD]
user32.SetWindowsHookExW.restype = wt.HHOOK
user32.CallNextHookEx.argtypes = [wt.HHOOK, ctypes.c_int, wt.WPARAM, wt.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wt.HHOOK]
user32.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
user32.PostThreadMessageW.argtypes = [wt.DWORD, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.keybd_event.argtypes = [wt.BYTE, wt.BYTE, wt.DWORD, ctypes.c_size_t]


def _down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


class SelectionHook:
    """callbacks（都在掛鉤執行緒呼叫，請轉發到 Qt 主執行緒）：
    on_start(x, y)、on_move(x, y)、on_end(起點x, 起點y, 終點x, 終點y)、on_cancel()、
    on_click(x, y)：沒按住熱鍵時的一般滑鼠點擊（用來關閉結果視窗）、on_escape()、
    on_hotkey()：熱鍵剛被按下（還沒點滑鼠；反白模式用來提早偵測版面）、
    on_tap(x, y)：按住熱鍵點一下、幾乎沒有移動（漫畫模式用來判斷雙擊；會先呼叫 on_cancel）、
    on_mode(x, y)：按住熱鍵按一下右鍵（切換選取模式；右鍵的按下、放開都攔下來，底下的程式不會跳出右鍵選單）。"""

    def __init__(self, hotkey_vks, on_start, on_move, on_end, on_cancel, on_click, on_escape, on_hotkey=None,
                 on_tap=None, on_mode=None):
        self.hotkey_vks = list(hotkey_vks)
        self.on_start, self.on_move, self.on_end = on_start, on_move, on_end
        self.on_cancel, self.on_click, self.on_escape = on_cancel, on_click, on_escape
        self.on_hotkey, self.on_tap, self.on_mode = on_hotkey, on_tap, on_mode
        self._rdown = False  # 攔下了右鍵的按下：對應的放開也要攔（放開時可能已經沒按 Alt）
        self._held: set[int] = set()  # 目前按著的熱鍵（通用鍵碼）
        self._last_down: dict[int, int] = {}  # 每個熱鍵最後一次按下事件的時間（毫秒）
        self._start = None
        self._thread = None
        self._tid = None
        self._ready = threading.Event()
        # 回呼要一直被參考著，否則會被回收而讓程式當掉
        self._mouse_proc = HOOKPROC(self._mouse)
        self._kbd_proc = HOOKPROC(self._kbd)

    def _mouse(self, code, wparam, lparam):
        if code != 0:
            return user32.CallNextHookEx(None, code, wparam, lparam)
        ms = ctypes.cast(lparam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
        x, y = ms.pt.x, ms.pt.y
        try:
            if wparam == WM_LBUTTONDOWN and all(_down(vk) for vk in self.hotkey_vks):
                self._start = (x, y)
                user32.keybd_event(VK_MASK, 0, 0, 0)
                user32.keybd_event(VK_MASK, 0, KEYEVENTF_KEYUP, 0)
                self.on_start(x, y)
                return 1
            if (wparam == WM_RBUTTONDOWN and self.on_mode and self._start is None
                    and all(_down(vk) for vk in self.hotkey_vks)):
                self._rdown = True
                user32.keybd_event(VK_MASK, 0, 0, 0)  # 同左鍵：放開 Alt 時不要跳到選單列
                user32.keybd_event(VK_MASK, 0, KEYEVENTF_KEYUP, 0)
                self.on_mode(x, y)
                return 1
            if wparam == WM_RBUTTONUP and self._rdown:
                self._rdown = False
                return 1
            if self._start is not None:
                if wparam == WM_MOUSEMOVE:
                    self.on_move(x, y)
                elif wparam == WM_LBUTTONUP:
                    (x0, y0), self._start = self._start, None
                    # 兩個方向都幾乎沒動才算點擊（沿著橫排字幕水平拖時上下只差幾像素，也是有效的拖曳）；
                    # 太細的框由 App 自己判斷
                    if abs(x - x0) < MIN_DRAG and abs(y - y0) < MIN_DRAG:
                        self.on_cancel()
                        if self.on_tap:
                            self.on_tap(x0, y0)
                    else:
                        self.on_end(x0, y0, x, y)
                    return 1
            elif wparam in (WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_MBUTTONDOWN):
                self.on_click(x, y)
        except Exception:  # 回呼出錯也不能讓掛鉤卡住滑鼠
            import traceback
            traceback.print_exc()
        return user32.CallNextHookEx(None, code, wparam, lparam)

    def _kbd(self, code, wparam, lparam):
        if code == 0:
            kb = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk = _GENERIC_VK.get(kb.vkCode)
            if vk in self.hotkey_vks and not kb.flags & LLKHF_INJECTED:
                try:
                    if wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                        # 按住時的自動重複最慢約 1 秒一次；隔更久才又收到「按下」表示漏收了放開
                        # （例如放開時在 UAC／Ctrl+Alt+Del 的安全桌面，掛鉤收不到），當成重新按下
                        last = self._last_down.get(vk)
                        if last is not None and (kb.time - last) & 0xFFFFFFFF > STUCK_MS:
                            self._held.discard(vk)
                        self._last_down[vk] = kb.time
                        before = all(k in self._held for k in self.hotkey_vks)
                        self._held.add(vk)
                        if not before and all(k in self._held for k in self.hotkey_vks) and self.on_hotkey:
                            self.on_hotkey()  # 只在剛按下時通知一次（按住時的自動重複不算）
                    elif wparam in (WM_KEYUP, WM_SYSKEYUP):
                        self._held.discard(vk)
                except Exception:
                    import traceback
                    traceback.print_exc()
        if code == 0 and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
            kb = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if kb.vkCode == VK_ESCAPE and not kb.flags & LLKHF_INJECTED:
                if self._start is not None:
                    self._start = None
                    self.on_cancel()
                self.on_escape()
        return user32.CallNextHookEx(None, code, wparam, lparam)

    def _run(self):
        self._tid = kernel32.GetCurrentThreadId()
        hm = user32.SetWindowsHookExW(WH_MOUSE_LL, self._mouse_proc, None, 0)
        hk = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._kbd_proc, None, 0)
        self._ready.set()
        if not hm or not hk:
            raise ctypes.WinError(ctypes.get_last_error())
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass
        user32.UnhookWindowsHookEx(hm)
        user32.UnhookWindowsHookEx(hk)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="hooks", daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def stop(self):
        if self._tid:
            user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
