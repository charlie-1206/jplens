"""翻譯時機（按了才翻／背景預先翻好／自動翻）在結果視窗、無字幕視窗的行為檢查。用假的翻譯函式，不呼叫 API、不碰快取檔：

    python tools/check_translate_timing.py

2026-10-23 的 BUG：背景預先翻譯時同一句查兩次，第二次會跳出沒按過翻譯的譯文；預先翻譯還沒回來就再查，
會重送一次請求，無字幕視窗還會卡在「翻譯中…」。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from PySide6.QtCore import QCoreApplication, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
from jplens import ai  # noqa: E402
from jplens.config import Config  # noqa: E402
from jplens.listen_ui import ListenWindow  # noqa: E402
from jplens.ui import ResultPopup  # noqa: E402

Config.save = lambda self: None
calls = []


def fake_translate(text, cfg=None, force=False):
    hit = None if force else ai.cached_translation(text, ai.translate_lang(cfg))
    if hit is not None:
        return hit
    calls.append(text)
    time.sleep(0.4)
    zh = f"譯文{len(calls)}"
    ai._cache[(ai.translate_lang(cfg), ai.cache_key(text))] = zh
    return zh


ai.translate = fake_translate
ok = True


def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        QCoreApplication.processEvents()
        QThread.msleep(10)


def report(good, msg):
    global ok
    ok &= bool(good)
    print(("OK  " if good else "NG  ") + msg)


def make(cls, timing):
    ai._cache = {}
    calls.clear()
    cfg = Config()
    cfg.translate_timing = timing
    cfg.prefetch_min_chars = 3
    w = cls(cfg, type("V", (), {"listeners": [], "has": lambda s, x: False})(), "")
    if cls is ListenWindow:
        w.start("Alt+L")
        w.add_heard(1, TEXT, True)
    return w


def query(w):
    if isinstance(w, ListenWindow):
        w._query_line = w._sel
    w.show_result([TEXT], [], 0.1)


TEXT = "今日はいい天気ですね"
for cls in (ListenWindow, ResultPopup):
    name = cls.__name__
    # 1. 預先翻譯回來之後再查一次同一句
    w = make(cls, "prefetch")
    query(w); pump(0.8); query(w); pump(0.2)
    report(w.trans.text() == "" and w.trans_btn.isEnabled(), f"{name} 翻好後再查：不顯示譯文（譯文={w.trans.text()!r}，呼叫 {len(calls)} 次）")
    w._translate(); pump(0.1)
    report(w.trans.text() == "譯文1" and len(calls) == 1, f"{name} 再按翻譯：馬上顯示、不重送（譯文={w.trans.text()!r}，呼叫 {len(calls)} 次）")
    w.close()
    # 2. 預先翻譯還沒回來就再查一次
    w = make(cls, "prefetch")
    query(w); pump(0.1); query(w); pump(0.8)
    report(w.trans.text() == "" and w.trans_btn.isEnabled() and len(calls) == 1,
           f"{name} 翻譯中再查：不顯示、不卡「翻譯中」、不重送（譯文={w.trans.text()!r}，按鈕可按={w.trans_btn.isEnabled()}，呼叫 {len(calls)} 次）")
    w._translate(); pump(0.1)
    report(w.trans.text() == "譯文1", f"{name} 再按翻譯：顯示（譯文={w.trans.text()!r}）")
    w.close()
    # 3. 預先翻譯中按翻譯 → 結果回來要顯示，再查一次也要留著
    w = make(cls, "prefetch")
    query(w); pump(0.1); w._translate(); pump(0.8)
    report(w.trans.text() == "譯文1" and len(calls) == 1, f"{name} 翻譯中按翻譯：回來就顯示（譯文={w.trans.text()!r}，呼叫 {len(calls)} 次）")
    w.close()
    # 4. 按了才翻：翻過再查一次，照舊直接顯示
    w = make(cls, "manual")
    query(w); w._translate(); pump(0.8); query(w); pump(0.2)
    report(w.trans.text() == "譯文1" and len(calls) == 1, f"{name} 按了才翻：翻過的句子再查照舊顯示（譯文={w.trans.text()!r}）")
    w.close()
    # 5. 自動翻：查了就顯示，再查不重送
    w = make(cls, "auto")
    query(w); pump(0.1); query(w); pump(0.8)
    report(w.trans.text() == "譯文1" and len(calls) == 1, f"{name} 自動翻：翻譯中再查也顯示、不重送（譯文={w.trans.text()!r}，呼叫 {len(calls)} 次）")
    w.close()
print("全部通過" if ok else "有失敗")
