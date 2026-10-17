"""主程式：常駐在系統匣，按住 Alt 拖曳 → 擷取 → OCR → 單字清單。

    python -m jplens

三種選取模式（結果視窗的「反白／框選／漫畫」按鈕或系統匣選單切換）：
- 反白：沿著文字拖曳選字，像在文書軟體裡反白。按下 Alt 的瞬間就擷取整個螢幕並在背景偵測文字行
  （約 0.6 秒），通常在按下滑鼠前就準備好；拖曳時即時顯示選到的字。起點不在文字上時改用框選。
- 框選：拖曳出矩形，讀矩形碰到的每一行落在框裡的那段。和反白共用同一份版面（按下 Alt 時偵測），
  框裡找不到文字行才整塊辨識裁切圖。
- 漫畫：和框選一樣拖曳矩形，但按下 Alt 時另外用漫畫專用的模型找出文字區塊（bubbles.py），
  框到的區塊各自整塊讀；按住 Alt 雙擊對話框就讀那一塊（單擊不做事，免得誤觸）。框裡沒有區塊時和框選相同。

模型（RapidOCR＋manga-ocr）、字典、Sudachi 在背景執行緒預先載入；
辨識也在同一個背景執行緒依序處理，視窗不會卡住。
"""

import ctypes
import logging
import os
import sys
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCursor
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from .config import FROZEN, ROOT, Config
from .ui import BusySpinner, HighlightOverlay, ResultPopup, SelectionFrame, make_icon, native_rect

LOG_DIR = ROOT / "logs"
# 使用者保存的框選畫面（測試資料，不進版控）；打包的執行檔存在 exe 旁邊的 captures/
CAPTURE_DIR = ROOT / "captures" if FROZEN else ROOT / "tests" / "eval" / "real"
LAYOUT_MAX_AGE = 10.0  # 版面（按下 Alt 時擷取的畫面）超過幾秒就重新擷取
MIN_BOX = 6  # 框的寬或高小於這個像素數就不讀
log = logging.getLogger("jplens")


def _ask_truth(default: str) -> str | None:
    """保存畫面時問正確答案；按取消回傳 None。日文沒有空白，要設成任何地方都能換行，否則一整句不會換行。"""
    from PySide6.QtGui import QTextOption
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout
    dlg = QDialog()
    dlg.setWindowTitle("螢幕查字：保存畫面")
    dlg.setWindowFlag(Qt.WindowStaysOnTopHint)
    lay = QVBoxLayout(dlg)
    hint = QLabel("正確答案（照畫面上選到的範圍，改成正確的文字；不確定就清空）：")
    hint.setWordWrap(True)
    lay.addWidget(hint)
    edit = QPlainTextEdit(default)
    edit.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
    edit.setMinimumSize(480, 140)
    font = edit.font()
    font.setPointSizeF(font.pointSizeF() * 1.3)
    edit.setFont(font)
    lay.addWidget(edit)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    lay.addWidget(buttons)
    edit.setFocus()
    return edit.toPlainText() if dlg.exec() == QDialog.Accepted else None

class Bridge(QObject):
    """掛鉤執行緒、背景執行緒 → Qt 主執行緒（跨執行緒的 signal 會自動排入主執行緒的事件佇列）。"""
    sel_start = Signal(int, int)
    sel_move = Signal(int, int)
    sel_end = Signal(int, int, int, int)
    sel_cancel = Signal()
    click = Signal(int, int)
    tap = Signal(int, int)
    mode_key = Signal(int, int)  # 按住 Alt＋右鍵：切換選取模式
    escape = Signal()
    hotkey = Signal()
    listen_key = Signal()        # 按住熱鍵＋L：開關無字幕模式
    heard = Signal(object, object)  # 無字幕模式聽到（一句的暫定或最終文字）：(哪個 Listener, listen.Heard)
    listen_state = Signal(object, str, bool)  # 無字幕模式的狀態：(哪個 Listener, 訊息, 是否出錯)
    ready = Signal(str)
    layout_ready = Signal(int, object)
    result = Signal(int, object)
    failed = Signal(int, str)


class Worker:
    """背景執行緒：先載入模型，再依序處理工作。
    工作種類：layout（偵測整個螢幕的文字行；漫畫模式另外找文字區塊）、text（辨識反白的片段）、
    whole（整塊讀一個對話框）、bubbles（漫畫模式：整塊讀好幾個文字區塊）、box（辨識框選的矩形）。
    同一類工作只保留最新的一個（還沒輪到的舊工作直接丟掉）。"""

    def __init__(self, bridge: Bridge, dictionary, cfg: Config):
        self.bridge = bridge
        self.dictionary = dictionary
        self.cfg = cfg
        self._jobs: list[tuple] = []
        self._submitted: dict[int, float] = {}
        self._cv = threading.Condition()
        self.ready = False
        self._bubble_det = None   # 漫畫模式第一次用到（或切到漫畫模式）才載入
        self._bubble_lock = threading.Lock()
        self.bubble_error = ""    # 漫畫模型載入失敗的原因（主執行緒顯示一次）
        threading.Thread(target=self._run, name="worker", daemon=True).start()

    LAYOUT_KINDS = ("layout", "zoom")  # 版面類（zoom：小字那一帶放大重新偵測）；其餘是讀字類

    def submit(self, kind: str, job_id: int, *args):
        group = "layout" if kind in self.LAYOUT_KINDS else "read"
        if group == "read":  # 記錄排隊多久（找反白模式延遲的原因）；版面編號是另一套計數器，不記
            self._submitted[job_id] = time.perf_counter()
        with self._cv:
            if group == "read":
                for j in self._jobs:
                    if j[0] not in self.LAYOUT_KINDS:
                        self._submitted.pop(j[1], None)  # 被新工作取代、不會執行的舊工作
            self._jobs = [j for j in self._jobs if ("layout" if j[0] in self.LAYOUT_KINDS else "read") != group]
            self._jobs.append((kind, job_id, *args))
            self._cv.notify()

    def _next(self):
        with self._cv:
            while not self._jobs:
                self._cv.wait()
            # 版面優先：反白模式要先有版面才能算出選到哪些字
            i = next((k for k, j in enumerate(self._jobs) if j[0] in self.LAYOUT_KINDS), 0)
            return self._jobs.pop(i)

    def _run(self):
        try:
            t = time.perf_counter()
            from . import inkcols, layout
            from .layout import build_layout
            from .recognize import Recognizer
            from .jlearn import analysis
            from .words import extract
            # 版面的做法：環境變數 JPLENS_LAYOUT_RULES=1／JPLENS_INK=0 優先（對照測試用）
            layout.RULES = layout.RULES or self.cfg.layout_rules
            inkcols.ENABLED = inkcols.ENABLED and self.cfg.ink_columns
            self._extract, self._build_layout = extract, build_layout
            self.rec = Recognizer()
            self.rec.warmup()
            analysis.tokenize("準備できました")
            analysis.jlpt_table()
            self.ready = True
            msg = (f"就緒（載入 {time.perf_counter() - t:.0f} 秒，文字偵測用 {self.rec.det_device}、"
                   f"辨識用 {self.rec.rec_device}）")
            log.info(msg)
            self.bridge.ready.emit(msg)
        except Exception:
            log.exception("載入模型失敗")
            self.bridge.ready.emit("載入失敗，詳見 logs/jplens.log")
            return
        while True:
            kind, job_id, *args = self._next()
            try:
                getattr(self, "_do_" + kind)(job_id, *args)
            except Exception:
                log.exception("工作失敗（%s）", kind)
                if kind in ("layout", "zoom"):
                    self.bridge.layout_ready.emit(job_id, ([], None, None))
                else:
                    self.bridge.failed.emit(job_id, traceback.format_exc(limit=1))

    def _do_layout(self, job_id, img, exclude, manga=False, cursor=None, text_mode=False):
        """cursor：按下 Alt 時游標在圖裡的位置。游標附近是橫排小字就把那一帶放大重新偵測（layout.zoom_strip）；
        反白模式游標在直排正文附近就先把墨水投影的欄找好（inkcols.prewarm），拖曳第一下不用在主執行緒等。"""
        from . import inkcols, layout
        t = time.perf_counter()
        bubbles = self._find_bubbles(img, exclude) if manga else None
        t_b = time.perf_counter() - t
        lines = self._build_layout(self.rec.detect_screen(img), img, exclude)
        bands, note = [], ""
        if cursor is not None and not manga:
            x, y = cursor
            if layout.small_text(lines, (x - 60, y - 30, x + 60, y + 30)):
                t_z = time.perf_counter()
                found = layout.zoom_strip(self.rec.detect_screen, img, lines, y, exclude)
                if found:
                    lines, band = found
                    bands.append(band)
                    note = f"（小字放大重新偵測 {time.perf_counter() - t_z:.2f} 秒）"
            if text_mode and inkcols.ENABLED:
                inkcols.prewarm(img, lines, x, y)
        log.info("版面 %dx%d：%d 行、%.2f 秒%s%s", img.shape[1], img.shape[0], len(lines), time.perf_counter() - t,
                 f"（漫畫文字區塊 {len(bubbles)} 個、{t_b:.2f} 秒）" if manga else "", note)
        self.bridge.layout_ready.emit(job_id, (lines, bubbles, bands))

    def _do_zoom(self, job_id, img, lines, y, exclude):
        """反白的起點在橫排小字上、又不在按下 Alt 時放大過的那一帶：起點那一帶放大重新偵測。"""
        from . import layout
        t = time.perf_counter()
        found = layout.zoom_strip(self.rec.detect_screen, img, lines, y, exclude)
        log.info("小字放大重新偵測：%s、%.2f 秒", "找到" if found else "沒找到", time.perf_counter() - t)
        lines, band = found if found else (lines, (int(y), int(y)))
        self.bridge.layout_ready.emit(job_id, (lines, None, [band]))

    def _do_plan(self, job_id, img, lines, box):
        """框選：決定怎麼讀（layout.plan_box，橫排小字會放大重新偵測、直排小說找欄），再交給對應的讀法。
        在背景執行緒做，主執行緒不用等找欄、放大偵測。"""
        from .layout import plan_box
        how, arg = plan_box(img, lines, box, self.rec.detect_screen)
        if how == "whole":
            self._do_whole(job_id, img, *arg)
        elif how == "lines":
            self._do_text(job_id, img, lines, arg)
        elif how == "line":
            self._do_text(job_id, img, *arg)
        else:
            x0, y0, x1, y1 = box
            self._do_box(job_id, img[y0:y1, x0:x1].copy())

    def load_bubbles(self):
        """載入漫畫文字偵測模型（約 2 秒）。切到漫畫模式時在另一個執行緒先載入，第一次框選就不用等。"""
        with self._bubble_lock:
            if self._bubble_det is not None or self.bubble_error:
                return
            try:
                from .bubbles import BubbleDetector
                t = time.perf_counter()
                det = BubbleDetector()
                det.detect(np.full((64, 64, 3), 255, np.uint8))  # 暖機（GPU 第一次推論比較慢）
                self._bubble_det = det
                log.info("漫畫文字偵測模型載入完成（%s、%s，%.1f 秒）", det.path.name, det.device, time.perf_counter() - t)
            except Exception as e:
                log.exception("漫畫文字偵測模型載入失敗")
                self.bubble_error = str(e)

    def retry_bubbles(self):
        """上次載入漫畫模型失敗的話，清掉錯誤，下次用到時重新載入（切到漫畫模式時呼叫）。"""
        with self._bubble_lock:
            self.bubble_error = ""

    def _find_bubbles(self, img, exclude):
        self.load_bubbles()
        if self._bubble_det is None:
            return []
        bubbles = self._bubble_det.detect(img)
        if exclude:  # 本程式的結果視窗裡的字不算
            ex0, ey0, ex1, ey1 = exclude
            bubbles = [b for b in bubbles if not (b.x0 >= ex0 and b.y0 >= ey0 and b.x1 <= ex1 and b.y1 <= ey1)]
        return bubbles

    def _do_text(self, job_id, img, lines, segs):
        from . import ruby
        t = time.perf_counter()
        wait = t - self._submitted.pop(job_id, t)
        parts = self.rec.read_segment_parts(img, lines, segs)
        text = self.rec.join_segments(parts)
        t_ocr = time.perf_counter() - t
        # 直排小說（墨水投影找到的欄）：讀旁邊的振假名當讀音提示（人名、讀音容易選錯的字）
        hints = ruby.read(self.rec, img, lines, segs, parts) if text and any(
            lines[g.line].ruby for g in segs) else []
        t_ruby = time.perf_counter() - t - t_ocr
        texts = [text] if text else []
        words = self._extract(texts, self.dictionary, hints) if texts else []
        sec = time.perf_counter() - t
        log.info("反白 %d 段：排隊 %.2f 秒、OCR %.2f 秒、%s總計 %.2f 秒、%d 個單字：%s",
                 len(segs), wait, t_ocr, f"振假名 {len(hints)} 個 {t_ruby:.2f} 秒（{hints}）、" if hints else "",
                 sec, len(words), text)
        self.bridge.result.emit(job_id, (texts, words, sec))

    def _do_retext(self, job_id, texts):
        """使用者在結果視窗改了原文：不用辨識，直接用改好的文字查字。"""
        t = time.perf_counter()
        self._submitted.pop(job_id, None)
        words = self._extract(texts, self.dictionary)
        sec = time.perf_counter() - t
        log.info("重查（使用者修改的原文）：%.2f 秒、%d 個單字：%s", sec, len(words), " / ".join(texts))
        self.bridge.result.emit(job_id, (texts, words, sec))

    def _do_whole(self, job_id, img, crop, blank):
        """框選一個對話框大小的範圍：整塊交給 manga-ocr（它就是為「一整個對話框」設計的，振假名會自己忽略；
        漫畫的欄夾著振假名時，偵測模型常把欄切碎、漏欄、兩欄黏成一欄，依行讀會漏字或亂序）。"""
        from .jlearn import textutil
        t = time.perf_counter()
        wait = t - self._submitted.pop(job_id, t)
        text = textutil.fix_common_ocr(textutil.clean(self.rec.read_whole(img, crop, blank)))
        t_ocr = time.perf_counter() - t
        texts = [text] if text else []
        words = self._extract(texts, self.dictionary) if texts else []
        sec = time.perf_counter() - t
        log.info("框選 %dx%d（整塊讀）：排隊 %.2f 秒、OCR %.2f 秒、總計 %.2f 秒、%d 個單字：%s",
                 crop[2] - crop[0], crop[3] - crop[1], wait, t_ocr, sec, len(words), text)
        self.bridge.result.emit(job_id, (texts, words, sec))

    def _do_bubbles(self, job_id, img, crops):
        """漫畫模式：框到（或點到）的文字區塊各自整塊交給 manga-ocr，每塊一行。"""
        from .jlearn import textutil
        t = time.perf_counter()
        wait = t - self._submitted.pop(job_id, t)
        texts = [textutil.fix_common_ocr(textutil.clean(s)) for s in self.rec.read_regions(img, crops)]
        texts = [s for s in texts if s]
        t_ocr = time.perf_counter() - t
        words = self._extract(texts, self.dictionary) if texts else []
        sec = time.perf_counter() - t
        log.info("漫畫 %d 塊：排隊 %.2f 秒、OCR %.2f 秒、總計 %.2f 秒、%d 個單字：%s",
                 len(crops), wait, t_ocr, sec, len(words), " / ".join(texts))
        self.bridge.result.emit(job_id, (texts, words, sec))

    def _do_box(self, job_id, img):
        """框裡找不到文字行（版面偵測漏掉）時的備案：整塊辨識框選的裁切圖。"""
        t = time.perf_counter()
        wait = t - self._submitted.pop(job_id, t)
        texts = [b.text for b in self.rec.recognize(img)]
        t_ocr = time.perf_counter() - t
        words = self._extract(texts, self.dictionary) if texts else []
        sec = time.perf_counter() - t
        log.info("框選 %dx%d（整塊）：排隊 %.2f 秒、OCR %.2f 秒、總計 %.2f 秒、%d 個單字：%s",
                 img.shape[1], img.shape[0], wait, t_ocr, sec, len(words), " / ".join(texts)[:60])
        self.bridge.result.emit(job_id, (texts, words, sec))


_mutex = None


def _single_instance() -> bool:
    global _mutex
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # ctypes.windll 的 GetLastError 可能被中間的呼叫蓋掉
    k32.CreateMutexW.restype = ctypes.c_void_p
    _mutex = k32.CreateMutexW(None, False, "Local\\jplens_screen_lookup")  # 存著：程式結束前不關閉
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


class Screen:
    """反白模式用：按下 Alt 時擷取的某一個螢幕畫面與它的版面。"""

    def __init__(self, layout_id, img, mon):
        self.id = layout_id
        self.img = img
        self.mon = mon            # (left, top, width, height) 實體座標
        self.lines = None         # 背景偵測完成前是 None
        self.bubbles = None       # 漫畫模式的文字區塊（bubbles.Bubble）；不是漫畫模式時是 None
        self.time = time.monotonic()
        self.exclude = None       # 版面要忽略的範圍（本程式的結果視窗）
        self.bands: list[tuple[int, int]] = []  # 已經放大重新偵測過的橫帶 (上緣, 下緣)（小字）
        self.zooming = False      # 反白起點那一帶正在放大重新偵測

    def zoomed(self, y) -> bool:
        return any(a <= y <= b for a, b in self.bands)

    def contains(self, x, y) -> bool:
        l, t, w, h = self.mon
        return l <= x < l + w and t <= y < t + h

    def rel(self, x, y):
        return x - self.mon[0], y - self.mon[1]

    def abs_rect(self, r):
        return r[0] + self.mon[0], r[1] + self.mon[1], r[2] + self.mon[0], r[3] + self.mon[1]


class App:
    def __init__(self, qapp: QApplication):
        from .dictionary import Dictionary
        from .hooks import SelectionHook
        from .vocab import VocabStore

        from .vocab import SAVED_NAME
        self.qapp = qapp
        self.cfg = Config.load()
        self.dictionary = Dictionary()
        self.vocab = VocabStore(self.cfg.vocab_path / SAVED_NAME)
        self.manager = None    # 單字本視窗（開啟時才建立，關掉就釋放）
        self.settings = None   # 設定視窗
        self.bridge = Bridge()
        self.frame = SelectionFrame()
        self.highlight = HighlightOverlay()
        self.popup = ResultPopup(self.cfg, self.vocab, self.dictionary.attribution())
        self.spinner = BusySpinner()  # 游標旁的小轉圈：模型載入中、等版面、辨識中
        self.spinner.muted = set() if self.cfg.show_spinner else {"layout"}
        self.spinner.start("model")
        self.job_id = 0
        self.layout_id = 0
        self._sct = None
        self._dxcam = None     # DXGI 擷取（dxcam）：None＝還沒試、False＝不能用（多螢幕、沒裝、出錯）
        self._dx_last = None   # dxcam 上一次擷取到的畫面（畫面沒變時 grab() 回傳 None，沿用這張）
        self._last = None      # (上一次的圖, 辨識出的文字, 其他資訊)
        self._corrected = None  # 使用者在結果視窗改過、重查的原文（保存畫面時當正確答案的預設值）
        self._requery_job = None
        self._t_release = None  # 放開滑鼠的時間（記錄延遲）
        self._pending = None   # 框選放開（或漫畫模式點一下）時版面還沒好：(版面編號, 要接著做的函式, 參數)
        self._move_queued = False  # 反白：已排了一次「更新反白範圍」（滑鼠移動事件合併，算不完不會越積越多）
        self._bubble_warned = False
        self._last_tap = None  # 漫畫模式雙擊判斷：(時間, x, y)
        self.listener = None   # 無字幕模式（listen.Listener，開啟時才建立）
        self.listen_win = None  # 無字幕視窗（listen_ui.ListenWindow，第一次開啟時建立）
        self.listening = False
        self._requery_target = None  # 重查的結果要顯示在哪個視窗（查字視窗或無字幕視窗）
        self.screen = None     # 反白模式目前的 Screen
        self.drag = None       # 反白模式拖曳中：{"start", "end", "lines", "segs", "box", "pending"}

        b = self.bridge
        b.sel_start.connect(self.on_start)
        b.sel_move.connect(self.on_move)
        b.sel_end.connect(self.on_end)
        b.sel_cancel.connect(self.on_cancel)
        b.click.connect(self.on_click)
        b.tap.connect(self.on_tap)
        b.mode_key.connect(self.cycle_mode)
        # 釘選、無字幕模式時 Esc 不關
        b.escape.connect(lambda: self.popup.pin.isChecked() or self.popup.hide())
        b.listen_key.connect(self.toggle_listen)
        # 已經關掉的 Listener 晚到的訊息不要（關掉後執行緒要一下子才結束）
        b.heard.connect(lambda lst, h: lst is self.listener and self.listen_win.add_heard(h.seg, h.text, h.final, h.cont))
        b.listen_state.connect(self._on_listen_state)
        b.hotkey.connect(self.on_hotkey)
        b.ready.connect(self.on_ready)
        b.layout_ready.connect(self.on_layout)
        b.result.connect(self.on_result)
        b.failed.connect(self.on_failed)
        self.popup.quit_requested.connect(self.quit)
        self.popup.mode_changed.connect(self._sync_mode_menu)
        self.popup.vocab_requested.connect(self.open_manager)
        self.popup.requery_requested.connect(lambda texts: self._requery(texts, self.popup))

        self.tray = QSystemTrayIcon(make_icon())
        self.status_action = QAction("模型載入中…")
        self.status_action.setEnabled(False)
        menu = QMenu()
        menu.addAction(self.status_action)
        menu.addSeparator()
        group = QActionGroup(menu)
        self.mode_actions = {}
        for mode, label in (("text", "反白模式（沿著文字拖曳選字）"), ("box", "框選模式（拖曳出矩形）"),
                            ("manga", "漫畫模式（框選對話框，或雙擊對話框）")):
            act = QAction(label, group, checkable=True)
            act.setChecked(self.cfg.select_mode == mode)
            act.triggered.connect(lambda _=False, m=mode: self.set_mode(m))
            menu.addAction(act)
            self.mode_actions[mode] = act
        menu.addSeparator()
        self.listen_action = QAction(checkable=True)
        self.listen_action.triggered.connect(lambda _=False: self.toggle_listen())
        self._listen_text()
        menu.addAction(self.listen_action)
        menu.addSeparator()
        menu.addAction("單字本…", self.open_manager)
        export = menu.addMenu("匯出")
        export.addAction("單字（單字 App 備份檔）…", lambda: self.export("words"))
        export.addAction("記錄的句子（編號文字檔）…", lambda: self.export("sentences"))
        menu.addAction("設定…", self.open_settings)
        from .elevate import is_admin
        self.admin = is_admin()
        if not self.admin:  # 以系統管理員身分執行的遊戲裡要用（elevate.py）
            menu.addAction("以系統管理員身分重新啟動（遊戲裡要用）", self.restart_as_admin)
        menu.addAction("保存上一次查詢的畫面（填正確答案，當測試資料）", self.save_last_capture)
        menu.addSeparator()
        menu.addAction("結束", self.quit)
        self._menu = menu
        self.tray.setContextMenu(menu)
        # 左鍵點圖示也跳出選單（預設只有右鍵）
        self.tray.activated.connect(
            lambda reason: menu.popup(QCursor.pos()) if reason == QSystemTrayIcon.Trigger else None)
        self.tray.setToolTip("螢幕查字：模型載入中…")
        self.tray.show()

        self.worker = Worker(b, self.dictionary, self.cfg)
        self.hook = SelectionHook(self.cfg.hotkey_vks, b.sel_start.emit, b.sel_move.emit, b.sel_end.emit,
                                  b.sel_cancel.emit, b.click.emit, b.escape.emit, b.hotkey.emit, b.tap.emit,
                                  b.mode_key.emit if self.cfg.mode_key else None,
                                  b.listen_key.emit, ord(self.cfg.listen_key))
        self.hook.start()
        # 前景是比自己高權限的程式（以系統管理員身分執行的遊戲）時，掛鉤收不到按鍵：每個程式提醒一次
        self._elevated_warned: set[str] = set()
        if not self.admin:
            self._fg_timer = QTimer(interval=2000, timeout=self._check_foreground)
            self._fg_timer.start()
        if not self.dictionary.ok:
            self.tray.showMessage("螢幕查字", "找不到字典 data/dict.sqlite，請先執行 tools/build_dict.py",
                                  QSystemTrayIcon.Warning)

    # ---------------------------------------------------------------- 模式
    def set_mode(self, mode):
        self.popup.set_mode(mode)
        self._sync_mode_menu(mode)

    def cycle_mode(self, x, y):
        """按住 Alt＋右鍵：反白 → 框選 → 漫畫 輪流切換，游標旁顯示切換到哪個模式。"""
        from PySide6.QtCore import QPoint, QRect
        from PySide6.QtWidgets import QToolTip
        from .ui import MODES, phys_to_logical
        order = list(MODES)
        mode = order[(order.index(self.cfg.select_mode) + 1) % len(order)]
        self.set_mode(mode)
        QToolTip.showText(phys_to_logical(x, y) + QPoint(16, 16), f"切換到「{MODES[mode][0]}」模式", None, QRect(), 1500)

    def _sync_mode_menu(self, mode):
        self.mode_actions[mode].setChecked(True)
        if self.settings is not None:
            self.settings.sync_mode()
        self.screen = None  # 版面要重新偵測（漫畫模式另外要找文字區塊）
        self._pending = None
        self.spinner.stop("layout")  # 舊版面的結果會被丟掉，等它的轉圈不會再停
        if mode == "manga" and self.worker.bubble_error:  # 上次載入失敗：重試
            self.worker.retry_bubbles()
            self._bubble_warned = False
        self._preload_bubbles()

    def _preload_bubbles(self):
        if (self.manga_mode or self.cfg.preload_manga) and self.worker.ready:
            threading.Thread(target=self.worker.load_bubbles, name="bubbles", daemon=True).start()

    @property
    def text_mode(self) -> bool:
        return self.cfg.select_mode == "text"

    @property
    def manga_mode(self) -> bool:
        return self.cfg.select_mode == "manga"

    # ---------------------------------------------------------------- 擷取
    def _grab(self, left, top, width, height) -> np.ndarray:
        import mss
        if self._sct is None:
            self._sct = mss.mss()  # mss 物件不能跨執行緒，所以在主執行緒建立與使用
        shot = self._sct.grab({"left": left, "top": top, "width": width, "height": height})
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])  # BGRA → BGR

    def _monitor_at(self, x, y):
        import mss
        if self._sct is None:
            self._sct = mss.mss()
        for m in self._sct.monitors[1:]:
            if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]:
                return m["left"], m["top"], m["width"], m["height"]
        m = self._sct.monitors[1]
        return m["left"], m["top"], m["width"], m["height"]

    def _grab_monitor(self, mon) -> np.ndarray:
        """擷取整個螢幕（按下 Alt 時）。只有一個螢幕時用 DXGI 桌面複製（dxcam，約 2～3 ms；mss 約 35 ms），
        不能用（多螢幕、沒裝 dxcam、出錯）就用 mss。"""
        cam = self._dx_camera()
        if cam:
            try:
                f = cam.grab()
                if f is None:  # 畫面從上一次擷取後沒有變
                    f = self._dx_last
                else:
                    self._dx_last = f
                if f is not None and f.shape[:2] == (mon[3], mon[2]):
                    return np.array(f[:, :, :3], copy=True)
            except Exception:
                log.exception("DXGI 擷取失敗，改用 mss")
                self._dxcam = False
        return self._grab(*mon)

    def _dx_camera(self):
        if self._dxcam is None:
            self._dxcam = False
            try:
                import mss
                if self._sct is None:
                    self._sct = mss.mss()
                if len(self._sct.monitors) == 2:  # monitors[0] 是所有螢幕合起來；只有一個實體螢幕
                    import dxcam
                    self._dxcam = dxcam.create(output_color="BGR")
            except Exception as e:
                log.info("不用 DXGI 擷取（%s），用 mss", e)
                self._dxcam = False
        return self._dxcam

    def _prepare_screen(self, x, y):
        """擷取游標所在的螢幕，交給背景偵測文字行。"""
        mon = self._monitor_at(x, y)
        img = self._grab_monitor(mon)
        exclude = None
        if self.popup.isVisible():  # 本程式的結果視窗不算
            l, t, r, b = native_rect(self.popup)
            exclude = (l - mon[0], t - mon[1], r - mon[0], b - mon[1])
        self.layout_id += 1
        self.screen = Screen(self.layout_id, img, mon)
        self.screen.exclude = exclude
        if img.max() >= 8:
            # 偵測版面要 0.2～0.5 秒，這段時間反白畫不出來、框選放開也要等：游標旁轉圈提醒
            self.spinner.start("layout")
            self.worker.submit("layout", self.layout_id, img, exclude, self.manga_mode,
                               (x - mon[0], y - mon[1]), self.text_mode)
        else:
            self.screen.lines = []
            self.spinner.stop("layout")  # 全黑畫面不偵測，不會有版面結果來停轉圈

    def on_hotkey(self):
        """按下 Alt：反白模式先擷取畫面、在背景偵測版面，按下滑鼠時通常已經好了。"""
        if self.worker.ready and self.drag is None and not self.listening:  # 無字幕模式時不能選取，不用擷取  # 框選也用同一份版面
            self._prepare_screen(*self._cursor())

    @staticmethod
    def _cursor():
        """游標的實體座標（Qt 的 QCursor.pos() 是邏輯座標）。"""
        import ctypes.wintypes as wt
        pt = wt.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    # ---------------------------------------------------------------- 拖曳
    def on_start(self, x, y):
        if not self.text_mode:
            self.frame.begin(x, y)
            if self.screen is not None and self.screen.lines is None:
                self.spinner.start("layout")
            return
        s = self.screen
        if s is None or not s.contains(x, y) or time.monotonic() - s.time > LAYOUT_MAX_AGE:
            self._prepare_screen(x, y)
        if self.screen.lines is None:  # 版面還沒好，反白還畫不出來
            self.spinner.start("layout")
        self.drag = {"start": (x, y), "end": (x, y), "lines": [], "segs": [], "box": None, "pending": False}
        self.highlight.begin(self.screen.mon)
        self._maybe_zoom()
        self._update_highlight()

    def _maybe_zoom(self):
        """反白起點在橫排小字上、那一帶還沒放大重新偵測過：在背景放大重新偵測（偵測模型會漏掉很小又很密的字）。
        先用原本的版面畫反白，好了再換；放開時還沒好就等它。"""
        from .layout import small_text
        s, d = self.screen, self.drag
        if d is None or s is None or s.lines is None or s.zooming or not self.worker.ready:
            return
        x, y = s.rel(*d["start"])
        if s.zoomed(y) or not small_text(s.lines, (x - 60, y - 30, x + 60, y + 30)):
            return
        s.zooming = True
        self.spinner.start("layout")
        self.worker.submit("zoom", s.id, s.img, s.lines, y, s.exclude)

    def on_move(self, x, y):
        if self.drag is None:
            self.frame.update_to(x, y)
            return
        self.drag["end"] = (x, y)
        # 移動事件只記下位置，排一次更新：算反白範圍比滑鼠事件來得慢時，中間的位置直接跳過，
        # 不會一個一個排隊（之前會越積越多，反白跟不上游標、放開後也要先消化完才開始辨識）
        if not self._move_queued:
            self._move_queued = True
            QTimer.singleShot(0, self._flush_move)

    def _flush_move(self):
        self._move_queued = False
        self._update_highlight()

    def _update_highlight(self):
        from .layout import plan_drag
        d, s = self.drag, self.screen
        if d is None or s is None or s.lines is None:
            return
        # 直排小說的行來自墨水投影（不是版面的行），片段要配它自己的行
        lines, segs = plan_drag(s.img, s.lines, s.rel(*d["start"]), s.rel(*d["end"]))
        d["lines"], d["segs"] = lines, segs
        if segs:
            d["box"] = None
            self.highlight.set_rects([s.abs_rect(lines[g.line].rect(g.a, g.b)) for g in segs])
        else:  # 起點不在文字上 → 改用框選
            (x0, y0), (x1, y1) = d["start"], d["end"]
            d["box"] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.highlight.set_box(d["box"])

    def on_layout(self, layout_id, payload):
        s = self.screen
        if s is None or s.id != layout_id:
            if s is None or (s.lines is not None and not s.zooming):  # 沒有在等的版面了
                self.spinner.stop("layout")
            return
        lines, bubbles, bands = payload
        if s.lines is None:  # 按下 Alt 時的版面
            s.lines, s.bubbles = lines, bubbles
        elif bands is not None:  # 反白起點那一帶放大重新偵測的結果（失敗時 bands 是 None，照舊用原本的版面）
            s.lines = lines
        s.bands += bands or []
        s.zooming = False
        self.spinner.stop("layout")
        if self.worker.bubble_error and not self._bubble_warned:
            self._bubble_warned = True
            self.tray.showMessage("螢幕查字", "漫畫模式的模型載入失敗，先用一般框選的方式讀：" + self.worker.bubble_error,
                                  QSystemTrayIcon.Warning, 5000)
        if self._pending and self._pending[0] == layout_id:
            (_, fn, args), self._pending = self._pending, None
            fn(*args)
        if self.drag is not None:
            self._maybe_zoom()  # 拖曳開始時版面還沒好：現在才知道起點是不是小字
            self._update_highlight()
            if self.drag["pending"] and not s.zooming:
                self._finish_text()

    def on_end(self, sx, sy, ex, ey):
        self._t_release = time.perf_counter()
        if self.drag is None:
            self.frame.end()
            self._select_box(min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
            return
        self.drag["end"] = (ex, ey)
        if self.screen.lines is None or self.screen.zooming:  # 版面還沒偵測完：先顯示「辨識中」，偵測完自動接著做
            self.drag["pending"] = True
            self._show_busy(min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey))
            return
        self._update_highlight()
        self._finish_text()

    def _finish_text(self):
        d, s = self.drag, self.screen
        self.drag = None
        self.highlight.end()
        if not d["segs"]:
            if d["box"] and d["box"][2] - d["box"][0] >= MIN_BOX and d["box"][3] - d["box"][1] >= MIN_BOX:
                self._select_box(*d["box"])
            elif d["pending"]:
                self._show_error("沒有選到文字。")
            return
        rects = [s.abs_rect(d["lines"][g.line].rect(g.a, g.b)) for g in d["segs"]]
        x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
        x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
        self.job_id += 1
        self._last = (s.img, None, {"mode": "text", "start": s.rel(*d["start"]), "end": s.rel(*d["end"])})
        self._show_busy(x0, y0, x1, y1)
        self.worker.submit("text", self.job_id, s.img, d["lines"], d["segs"])

    def on_cancel(self):
        self.frame.end()
        self.spinner.stop("layout")
        if self.drag is not None:
            self.drag = None
            self.highlight.end()

    def _show_busy(self, x0, y0, x1, y1):
        if not self.popup.isVisible():  # 已開啟就留在原處（使用者可能拖到習慣的位置）
            self.popup.place_near(x0, y0, x1, y1)
        self.popup.show_busy("辨識中…" if self.worker.ready else "模型載入中，載入完成後會自動辨識…")
        self.spinner.start("query")

    def _show_error(self, msg):
        self.spinner.stop("query")
        self.popup.show_error(msg)

    def _current_screen(self, x, y):
        """(x, y) 所在螢幕的 Screen；沒有、太舊或不是那個螢幕就重新擷取（模型還沒載入完時可能是 None）。"""
        s = self.screen
        if s is None or not s.contains(x, y) or time.monotonic() - s.time > LAYOUT_MAX_AGE:
            if self.worker.ready:
                self._prepare_screen(x, y)
            s = self.screen
        return s if s is not None and s.contains(x, y) else None

    def on_tap(self, x, y):
        """漫畫模式按住 Alt 雙擊：讀游標所在的文字區塊。單擊不做事（不小心按到 Alt＋左鍵時不會跳出視窗）。"""
        if not self.manga_mode:
            return
        now, prev = time.monotonic(), self._last_tap
        self._last_tap = (now, x, y)
        interval = ctypes.windll.user32.GetDoubleClickTime() / 1000
        if prev is None or now - prev[0] > interval or abs(x - prev[1]) > 8 or abs(y - prev[2]) > 8:
            if self.worker.ready:  # 第一下就先擷取畫面、偵測版面，第二下時通常已經好了
                self._current_screen(x, y)
            return
        self._last_tap = None
        self._t_release = time.perf_counter()
        s = self._current_screen(x, y)
        if s is None:
            return
        if s.lines is None:
            self._pending = (s.id, self._tap_read, (x, y))
            self._show_busy(x, y, x + 1, y + 1)
            return
        self._tap_read(x, y)

    def _tap_read(self, x, y):
        from .bubbles import at
        s = self.screen
        hits = at(s.bubbles or [], *s.rel(x, y))
        if not hits:
            if not self.popup.isVisible():
                self.popup.place_near(x, y, x + 1, y + 1)
            self._show_error("這裡沒有找到漫畫的字。可以按住 Alt 拖曳框起來試試。")
            return
        b = hits[0]
        self._read_bubbles(s, hits, s.abs_rect((b.x0, b.y0, b.x1, b.y1)), {"point": s.rel(x, y)})

    def _read_bubbles(self, s, blocks, screen_rect, query):
        """query：怎麼選的（{"box": 框} 或 {"point": 雙擊的位置}，圖內座標），保存畫面時記下來，評估工具重跑用。"""
        H, W = s.img.shape[:2]
        crops = [b.crop(W, H) for b in blocks]
        self.job_id += 1
        self._last = (s.img, None, {"mode": "manga", **query, "crops": crops})
        self._show_busy(*screen_rect)
        self.worker.submit("bubbles", self.job_id, s.img, crops)

    def _select_box(self, x0, y0, x1, y1):
        """框選：和反白模式共用按下 Alt 時偵測的整個螢幕版面（版面還沒好就等它）。
        - 框住的是一個對話框大小的範圍（6 行以內、每行約 12 字以內）：整塊交給 manga-ocr 讀，
          裁切範圍是「框」加上框碰到的行（行可能比框寬一點，字才不會被切半；偵測漏掉的欄也會被框包進去）
        - 框得大（小說、字幕、好幾個對話框）：讀框碰到的每一行落在框裡的那段
        - 版面裡框附近完全沒有字（偵測漏掉）：框不大就整塊讀，太大才用舊的整塊辨識（先偵測再逐行）
        漫畫模式：先看框到哪些漫畫文字區塊（至少六成在框裡），有就各自整塊讀；沒有才照上面的規則。"""
        if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:  # 太細的框（沿著字拖成一條線）：當成沒框
            self.spinner.stop("layout")
            return
        s = self._current_screen(x0, y0)
        if s is not None:
            if s.lines is None:  # 版面還沒偵測完：先顯示「辨識中」，偵測完自動接著做
                self._pending = (s.id, self._select_box, (x0, y0, x1, y1))
                self._show_busy(x0, y0, x1, y1)
                return
            H, W = s.img.shape[:2]
            rx0, ry0 = s.rel(x0, y0)
            rx1, ry1 = s.rel(x1, y1)
            rx1, ry1 = min(rx1, W), min(ry1, H)  # 拖到別的螢幕時只讀這個螢幕裡的部分（左上角一定在這個螢幕）
            if rx1 - rx0 < MIN_BOX or ry1 - ry0 < MIN_BOX:
                return
            box = (rx0, ry0, rx1, ry1)
            if s.bubbles:
                from .bubbles import pick
                hits = pick(s.bubbles, box)
                if hits:
                    self._read_bubbles(s, hits, (x0, y0, x1, y1), {"box": box})
                    return
            self.job_id += 1
            self._last = (s.img, None, {"mode": "box", "box": box})
            if s.img[ry0:ry1, rx0:rx1].max() < 8:
                if not self.popup.isVisible():
                    self.popup.place_near(x0, y0, x1, y1)
                self._show_error("擷取到的是全黑畫面。如果是全螢幕遊戲，請改成「無邊框視窗」或「視窗」模式再試。")
                return
            self._show_busy(x0, y0, x1, y1)
            # 怎麼讀（layout.plan_box：找欄、小字放大重新偵測）在背景決定，主執行緒不用等
            self.worker.submit("plan", self.job_id, s.img, s.lines, box)
            return
        img = self._grab(x0, y0, x1 - x0, y1 - y0)
        self.job_id += 1
        self._last = (img, None, {"mode": "box"})
        if img.max() < 8:
            if not self.popup.isVisible():
                self.popup.place_near(x0, y0, x1, y1)
            # 獨占全螢幕的遊戲用一般方式擷取會得到全黑畫面
            self._show_error("擷取到的是全黑畫面。如果是全螢幕遊戲，請改成「無邊框視窗」或「視窗」模式再試。")
            return
        self._show_busy(x0, y0, x1, y1)
        self.worker.submit("box", self.job_id, img)

    # ---------------------------------------------------------------- 結果
    def on_ready(self, msg):
        hk = self.cfg.hotkey_name
        how = "沿著文字拖曳反白" if self.text_mode else "拖曳框選（漫畫模式也可以雙擊對話框）" if self.manga_mode else "拖曳框選"
        self.spinner.stop("model")
        self._preload_bubbles()
        if self.admin:
            msg += "（系統管理員）"
        self.status_action.setText(msg)
        self.tray.setToolTip(f"螢幕查字：{msg}\n按住 {hk} {how}")
        self.tray.showMessage("螢幕查字", f"{msg}。按住 {hk} 用滑鼠{how}日文。", QSystemTrayIcon.Information, 3000)

    def _requery(self, texts, target):
        """target：結果顯示在哪個視窗（查字視窗 popup 或無字幕視窗 listen_win）。"""
        if not self.worker.ready:
            target.status.setText("模型還在載入，等一下再按")
            return
        self.job_id += 1
        self._requery_job = self.job_id
        self._requery_target = target
        self._t_release = None
        self.worker.submit("retext", self.job_id, texts)

    def on_result(self, job_id, payload):
        if job_id == self.job_id and job_id == self._requery_job:
            # 改過的原文：當成「保存畫面」時正確答案的預設值；上一次辨識的結果（_last）不動
            texts, words, sec = payload
            target = self._requery_target or self.popup
            if target is self.popup:
                self._corrected = "\n".join(texts)
            target.show_result(texts, words, sec, "查詢" if target is not self.popup else "重查")
            return
        if job_id == self.job_id:
            self._corrected = None
            if self._last:
                self._last = (self._last[0], payload[0], self._last[2])
            self.spinner.stop("query")
            texts, words, sec = payload
            detail = f"辨識 {sec:.2f}"
            if self._t_release:  # 顯示實際等了多久（放開滑鼠到出結果），括號裡是時間花在哪
                total = time.perf_counter() - self._t_release
                if total - sec >= 0.05:  # 放開時版面還沒偵測完（或模型還在載入）
                    detail += f"、等版面 {total - sec:.2f}"
                log.info("放開滑鼠到顯示結果 %.2f 秒（%s）", total, detail)
                sec, self._t_release = total, None
            self.popup.show_result(texts, words, sec, detail)

    def on_failed(self, job_id, msg):
        if job_id == self.job_id:
            self._show_error("辨識失敗：" + msg.strip().splitlines()[-1])

    def on_click(self, x, y):
        if self.popup.isVisible() and not self.popup.pin.isChecked() and not self.popup.contains_phys(x, y):
            self.popup.hide()

    # ---------------------------------------------------------------- 無字幕模式
    def _listen_text(self):
        self.listen_action.setText(f"無字幕模式：聽電腦的聲音轉成文字（{self.cfg.listen_name}）")

    def toggle_listen(self):
        if self.listening:
            self.stop_listen()
        else:
            self.start_listen()

    def start_listen(self):
        """無字幕模式：關掉查字視窗、另外開無字幕視窗；這段時間不能框選、反白（Alt＋拖曳照常給底下的程式）。"""
        from . import listen
        from .listen_ui import ListenWindow
        if self.listen_win is None:
            w = self.listen_win = ListenWindow(self.cfg, self.vocab, self.dictionary.attribution())
            w.requery_requested.connect(lambda texts: self._requery(texts, w))
            w.stop_requested.connect(self.stop_listen)
            w.save_audio_requested.connect(self.save_listen_audio)
            w.quit_requested.connect(self.quit)
            w.vocab_requested.connect(self.open_manager)
            w.speed_changed.connect(self._set_listen_speed)
        self.listening = True
        self.listen_action.setChecked(True)
        self.hook.select_enabled = False
        self.on_cancel()
        self.popup.hide()
        self.listen_win.start(self.cfg.listen_name)
        if not listen.model_ready():
            self._on_listen_state(self.listener, f"找不到語音辨識模型（{listen.ASR_DIR}）", True)
            return
        if self.listener is None:
            lst = listen.Listener(lambda h: self.bridge.heard.emit(lst, h),
                                  lambda msg, err: self.bridge.listen_state.emit(lst, msg, err),
                                  partial=self.cfg.listen_partial, speed=self.cfg.listen_speed)
            self.listener = lst
        self.listener.start()

    def _set_listen_speed(self, speed: float):
        """無字幕視窗換了播放速度：記住，正在聽的話馬上套用。"""
        self.cfg.listen_speed = speed
        self.cfg.save()
        if self.listener is not None:
            self.listener.set_speed(speed)
        log.info("無字幕模式：播放速度 %.2f 倍", speed)

    def stop_listen(self):
        """結束無字幕模式：關掉無字幕視窗，可以繼續框選、反白。"""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None  # 停下來的執行緒結束後模型就釋放（約 300 MB），下次開啟重新載入（約 1 秒）
        self.listening = False
        self.listen_action.setChecked(False)
        self.hook.select_enabled = True
        if self.listen_win is not None:
            self.listen_win.hide()

    def _on_listen_state(self, lst, msg, error):
        if lst is not self.listener or self.listen_win is None:
            return
        self.listen_win.set_state(msg, error)
        if error:  # 錄音、辨識停了：視窗留著讓使用者看到原因，按 ✕ 或熱鍵結束
            self.listener = None

    def save_listen_audio(self):
        """無字幕視窗的「保存聲音」：最近 2 分鐘有聲音的部分（wav）和這段時間辨識出的句子（同名 txt），當測試資料。"""
        w = self.listen_win
        if self.listener is None:
            w.status.setText("沒有在聆聽，沒有聲音可以存")
            return
        folder = CAPTURE_DIR / "audio"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / time.strftime("%Y%m%d_%H%M%S.wav")
        sec = self.listener.save_recent(path)
        if not sec:
            w.status.setText("還沒聽到任何聲音，沒有東西可以存")
            return
        log.info("保存無字幕模式的聲音：%s（%.0f 秒）", path, sec)
        w.status.setText(f"已保存最近 {sec:.0f} 秒的聲音：{path.name}")
        w.status.setToolTip(str(path))

    # ---------------------------------------------------------------- 單字本、設定
    @staticmethod
    def _front(w):
        w.show()
        w.setWindowState(w.windowState() & ~Qt.WindowMinimized)
        w.raise_()
        w.activateWindow()

    def open_manager(self):
        from .manager import ManagerWindow
        if self.manager is None:
            self.manager = ManagerWindow(self.vocab, self.cfg, self.dictionary.attribution())
            self.manager.open_settings.connect(self.open_settings)
            self.manager.destroyed.connect(lambda: setattr(self, "manager", None))
        self._front(self.manager)

    def open_settings(self, tab: str = "general"):
        from .settings_ui import SettingsWindow
        if self.settings is None:
            self.settings = SettingsWindow(self.cfg, self)
            self.settings.changed.connect(self.apply_setting)
            self.settings.destroyed.connect(lambda: setattr(self, "settings", None))
        self.settings.show_tab(tab or "general")
        self._front(self.settings)

    def export(self, kind: str):
        """系統匣的「匯出」：和單字本視窗一樣先跳確認視窗。"""
        from .manager import ExportDialog
        dlg = ExportDialog(None, kind, self.vocab, self.cfg)
        if dlg.exec():
            extra = f"（重點單字 {dlg.marked} 個）" if kind == "words" else ""
            self.tray.showMessage("螢幕查字", f"已匯出 {dlg.count}{dlg.unit}{extra}到 {dlg.out.name}",
                                  QSystemTrayIcon.Information, 3000)

    def _check_foreground(self):
        from .elevate import foreground_elevated
        elevated, name = foreground_elevated()
        key = name or "?"
        if elevated and key not in self._elevated_warned:
            self._elevated_warned.add(key)
            log.info("前景程式以較高權限執行，掛鉤收不到按鍵：%s", name or "（查不到名稱）")
            self.tray.showMessage(
                "螢幕查字：這個程式裡不能用",
                f"{name or '目前的程式'} 以系統管理員身分執行，查字工具收不到按鍵。\n"
                "到系統匣選單按「以系統管理員身分重新啟動」，或在設定 → 一般打開「以系統管理員身分執行」。",
                QSystemTrayIcon.Warning, 8000)

    def restart_as_admin(self):
        from .elevate import relaunch_as_admin
        if relaunch_as_admin():
            self.quit()

    def apply_setting(self, key: str):
        """設定視窗改了設定（已寫進 config.json）：讓它馬上生效。"""
        from . import inkcols, layout
        cfg = self.cfg
        if key in ("hotkey", "listen_key"):
            self.hook.hotkey_vks = list(cfg.hotkey_vks)
            self.hook.listen_vk = ord(cfg.listen_key)
            self._listen_text()
            self.tray.setToolTip(f"螢幕查字：{self.status_action.text()}\n按住 {cfg.hotkey_name} 用滑鼠選取日文")
        elif key == "mode_key":
            self.hook.on_mode = self.bridge.mode_key.emit if cfg.mode_key else None
        elif key == "show_spinner":
            self.spinner.muted = set() if cfg.show_spinner else {"layout"}
            if not cfg.show_spinner:
                self.spinner.stop("layout")
        elif key == "select_mode":
            self.set_mode(cfg.select_mode)
        elif key in ("min_level", "show_proper", "show_text", "font_size", "popup_width"):
            self.popup.sync_from_cfg()
            if self.listen_win is not None:
                self.listen_win.sync_from_cfg()
        elif key == "listen_partial":
            from . import listen
            if self.listener is not None and self.listener.tr is not None:
                self.listener.tr.partial_every = listen.PARTIAL_EVERY if cfg.listen_partial else 0
        elif key == "preload_manga":
            self._preload_bubbles()
        elif key == "vocab_dir":
            from .vocab import SAVED_NAME
            self.vocab.set_path(cfg.vocab_path / SAVED_NAME)
        elif key in ("ink_columns", "layout_rules"):
            # 環境變數 JPLENS_LAYOUT_RULES=1／JPLENS_INK=0 優先（對照測試用，同 Worker._run）
            layout.RULES = os.environ.get("JPLENS_LAYOUT_RULES") == "1" or cfg.layout_rules
            inkcols.ENABLED = os.environ.get("JPLENS_INK", "1") != "0" and cfg.ink_columns
            self.screen = None  # 版面重新偵測

    def save_last_capture(self):
        """把上一次的圖存成 PNG（反白模式是整個螢幕），辨識結果、選取位置、正確答案存在同名 .json，
        當測試資料（tests/eval/box_eval.py 會把有正確答案的都拿來測）。
        存之前問正確答案：預設填辨識出的文字，改成正確的再按確定；不知道就留空。"""
        if not self._last:
            self.tray.showMessage("螢幕查字", "還沒有查詢過。", QSystemTrayIcon.Information, 2000)
            return
        import json
        import cv2
        img, texts, info = self._last
        truth = _ask_truth(self._corrected or "\n".join(texts or []))
        if truth is None:
            return
        truth = "".join(truth.split()) or None  # 換行、空白不算
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        path = CAPTURE_DIR / time.strftime("%Y%m%d_%H%M%S.png")
        cv2.imencode(".png", img)[1].tofile(str(path))
        path.with_suffix(".json").write_text(json.dumps({**info, "ocr": texts or [], "truth": truth},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
        self.tray.showMessage("螢幕查字", f"已保存到 {path.name}" + ("" if truth else "（沒有正確答案）"),
                              QSystemTrayIcon.Information, 2000)
        log.info("保存畫面：%s（正確答案：%s）", path, truth)

    def quit(self):
        if self.listener is not None:
            self.listener.stop()
        self.hook.stop()
        self.tray.hide()
        self.qapp.quit()


def selftest() -> int:
    """jplens.exe --selftest：不裝掛鉤、不開系統匣，載入模型、讀一張畫出來的日文、查字，
    結果寫到 logs/selftest.txt（打包後確認模型、CUDA、字典、斷詞都有帶進去）。成功回傳 0。"""
    out = []
    try:
        from PySide6.QtGui import QColor, QFont, QImage, QPainter
        from .dictionary import Dictionary
        from .recognize import Recognizer
        from .words import extract
        qapp = QApplication.instance() or QApplication(sys.argv)  # noqa: F841（畫字要用）
        t = time.perf_counter()
        rec = Recognizer()
        rec.warmup()
        out.append(f"載入 {time.perf_counter() - t:.1f} 秒；偵測 {rec.det_device}、辨識 {rec.rec_device}")
        sample = "日本語を勉強しています"
        qi = QImage(900, 160, QImage.Format_RGB888)
        qi.fill(QColor("white"))
        p = QPainter(qi)
        p.setPen(QColor("black"))
        p.setFont(QFont("Yu Gothic UI", 40))
        p.drawText(40, 105, sample)
        p.end()
        qi = qi.convertToFormat(QImage.Format_RGB888)
        arr = np.frombuffer(qi.constBits(), np.uint8).reshape(qi.height(), qi.bytesPerLine())[:, :qi.width() * 3]
        img = arr.reshape(qi.height(), qi.width(), 3)[:, :, ::-1].copy()  # RGB → BGR
        t = time.perf_counter()
        text = rec.read_whole(img)
        out.append(f"辨識：{text}（{time.perf_counter() - t:.2f} 秒）")
        words = extract([text], Dictionary())
        out.append("單字：" + "、".join(f"{w.word}【{w.reading}】{w.jlpt} {w.meaning}" for w in words[:5]))
        ok = text.replace(" ", "") == sample and bool(words)
        # 翻譯與講解（不連網）：Gemini 套件、keyring 的 Windows 認證管理員後端都有帶進去
        import keyring
        from google import genai  # noqa: F401
        backend = type(keyring.get_keyring()).__module__
        out.append(f"Gemini 套件 OK；keyring 後端：{backend}")
        ok = ok and "Windows" in backend
        # 無字幕模式（不錄音）：模型、sherpa-onnx、pyaudiowpatch 都有帶進去，辨識一段靜音不會出錯
        from . import listen
        import pyaudiowpatch  # noqa: F401
        tr = listen.Transcriber(48000)
        tr.feed(np.zeros(48000, np.float32))
        tr.flush()
        slow, r = listen.slow_down(np.zeros(48000, np.float32), 48000, 1.5)  # 倍速放慢（打包版沒有 PyAV：調整取樣率）
        ok = ok and (len(slow) > 60000 or r == 32000)
        from .listen_ui import ListenWindow
        ListenWindow(Config(), type("V", (), {"listeners": []})(), "").deleteLater()  # 視窗建得起來
        out.append(f"無字幕模式：模型 {'OK' if listen.model_ready() else '找不到'}、sherpa-onnx、pyaudiowpatch OK；"
                   f"倍速放慢用{'PyAV atempo' if listen._atempo_ok else '調整取樣率'}")
        ok = ok and listen.model_ready()
    except Exception:
        out.append(traceback.format_exc())
        ok = False
    out.append("OK" if ok else "FAILED")
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "selftest.txt").write_text("\n".join(out), encoding="utf-8")
    return 0 if ok else 1


def main():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "jplens.log", encoding="utf-8"),
                  *([logging.StreamHandler()] if sys.stdout else [])])
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    from . import elevate
    if Config.load().run_as_admin and not elevate.is_admin() and elevate.WAIT_FLAG not in sys.argv:
        if elevate.relaunch_as_admin():  # 設定了「以系統管理員身分執行」：跳 UAC，按「否」就用一般權限繼續
            return
    qapp = QApplication(sys.argv)
    qapp.setQuitOnLastWindowClosed(False)
    ok = _single_instance()
    if not ok and elevate.WAIT_FLAG in sys.argv:  # 從舊的程序重新啟動：等它結束、放掉鎖（最多 10 秒）
        for _ in range(100):
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(_mutex))
            time.sleep(0.1)
            ok = _single_instance()
            if ok:
                break
    if not ok:
        QMessageBox.warning(None, "螢幕查字", "已經在執行了（請看右下角系統匣的「字」圖示）。")
        return
    app = App(qapp)  # noqa: F841（要保持參考）
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
