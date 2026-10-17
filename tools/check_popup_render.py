"""結果視窗、無字幕視窗的實際畫面檢查（會真的開視窗、截螢幕，要在有螢幕的 Windows 上跑）：

    python tools/check_popup_render.py

點進原文框、改字、還鍵盤、點去別的程式、關掉再打開之後，視窗有沒有畫完整。這些動作會切換「接不接受焦點」
（ResultPopup._allow_focus，讓輸入法知道游標位置）；半透明視窗改旗標後沒重畫的話，只剩原文框、其他部分變透明（2026-10-17 的 BUG）。
判斷方式：視窗範圍裡深色像素（視窗底色）的比例，正常約 0.96；壞掉時 0.1～0.3。另外檢查 Qt 的焦點視窗、輸入法游標位置。
"""
import ctypes
import sys
from ctypes import wintypes as wt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt, QThread  # noqa: E402
from PySide6.QtGui import QGuiApplication, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit  # noqa: E402

app = QApplication([])
from jplens.config import Config  # noqa: E402
from jplens.listen_ui import ListenWindow  # noqa: E402
from jplens.ui import ResultPopup  # noqa: E402

u = ctypes.windll.user32
ok = True


def pump(n=30):
    for _ in range(n):
        QCoreApplication.processEvents()
        QThread.msleep(10)


def report(good, msg):
    global ok
    ok &= bool(good)
    print(("OK  " if good else "NG  ") + msg)


def check(w, name):
    pump()
    g = w.frameGeometry()
    img = QGuiApplication.primaryScreen().grabWindow(0, g.x() + 8, g.y() + 8, g.width() - 16, g.height() - 16).toImage()
    dark = tot = 0
    for y in range(0, img.height(), 4):
        for x in range(0, img.width(), 4):
            c = img.pixelColor(x, y)
            tot += 1
            dark += (c.red() + c.green() + c.blue()) < 240
    report(dark / tot > 0.85, f"{name}：深色比例 {dark / tot:.2f}")


def main():
    bg = QLabel("背景")  # 淺色背景：視窗變透明時看得出來
    bg.setStyleSheet("background:#88aacc;")
    bg.setGeometry(450, 250, 900, 800)
    bg.show()
    other = QLineEdit("別的程式")
    other.setGeometry(100, 100, 300, 40)
    other.show()
    cfg = Config()
    cfg.save = lambda: None
    for cls in (ResultPopup, ListenWindow):
        w = cls(cfg, type("V", (), {"listeners": []})(), "")
        if cls is ListenWindow:
            w.start("Alt＋L")
            for i, t in enumerate(["帰るの", "ほらアンブレラスカイト", "……そうか。"]):
                w.add_heard(i, t, True)
        else:
            w._show_text("……そうか。", editable=True)
            w.text.setVisible(True)
        w.move(600, 400)
        w.show()
        w.raise_()
        n = cls.__name__
        check(w, f"{n} 一開始")
        vp = w.text.viewport()
        ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(10, 10), QPointF(vp.mapToGlobal(vp.rect().topLeft())),
                         Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        w.eventFilter(vp, ev)
        check(w, f"{n} 點進原文框")
        report(QGuiApplication.focusWindow() is w.windowHandle()
               and QGuiApplication.inputMethod().cursorRectangle().width() > 0, f"{n} 點進原文框：Qt 焦點、輸入法游標位置")
        w.text.setPlainText("……そうか！")
        w._on_text_edited()
        check(w, f"{n} 改字")
        w._give_back_focus()
        check(w, f"{n} 還鍵盤")
        w.eventFilter(vp, ev)
        pump()
        u.SetForegroundWindow(wt.HWND(int(other.winId())))
        check(w, f"{n} 點去別的程式")
        w.eventFilter(vp, ev)
        pump()
        w.hide()
        pump()
        w.show()
        w.raise_()
        check(w, f"{n} 關掉再打開")
        w.hide()
        pump()
    print("全部通過" if ok else "有失敗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
