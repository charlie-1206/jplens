"""設定視窗：左邊分頁、右邊內容，改了就寫進 config.json 並馬上生效（App.apply_setting）。

分頁：一般（語言〔預留〕、啟動）、查字操作、結果視窗、單字本、翻譯與 AI（整頁預留）、進階、關於。
"""

import os
import sys
import winreg
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from .config import APP_DIR, FONT_SCALES, FROZEN, HOTKEY_NAMES, POPUP_WIDTHS, ROOT
from .theme import C, JP_FONTS, QSS, badge_style, dark_title_bar

TABS = [("general", "一般", "語言與啟動"), ("lookup", "查字操作", "熱鍵與選取方式"),
        ("popup", "結果視窗", "查字結果要顯示什麼、看起來多大"), ("vocab", "單字本", "記錄存在哪裡、匯出的預設值"),
        ("ai", "翻譯與 AI", "Gemini 與本地翻譯"), ("advanced", "進階", "辨識的做法與系統狀態"),
        ("about", "關於", "版本、授權與資料來源")]
RESERVED_TABS = {"ai"}
CREDITS = [
    ("JMdict（EDRDG）", "英文意思、詞性", "CC BY-SA 4.0"),
    ("Tomoshi 開放資料（Y1Z）", "中文意思", "CC BY-SA 4.0"),
    ("中文維基詞典（kaikki.org 擷取）", "中文意思", "CC BY-SA"),
    ("Gemini 翻譯（譯自 JMdict）", "日檢詞的中文意思", "CC BY-SA 4.0"),
    ("日檢單字表（Jonathan Waller；open-anki-jlpt-decks）", "級數", "CC BY／MIT"),
    ("manga-ocr（kha-white）", "文字辨識", "Apache-2.0"),
    ("PP-OCR／RapidOCR", "文字偵測", "Apache-2.0"),
    ("comic-text-and-bubble-detector（ogkalu）", "漫畫模式", "Apache-2.0"),
    ("SudachiPy、SudachiDict", "斷詞", "Apache-2.0"),
    ("Qt（PySide6）", "介面", "LGPL-3.0"),
]

# ---------------------------------------------------------------- 開機自動啟動（目前使用者的登錄檔 Run）
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_NAME = "jplens"


def autostart_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, RUN_NAME)
        return True
    except OSError:
        return False


def autostart_command() -> str:
    """用 pythonw（沒有主控台視窗）在專案資料夾裡執行 jplens；Run 機碼沒有工作目錄，所以自己切過去。"""
    if FROZEN:  # 打包的 jplens.exe
        return f'"{sys.executable}"'
    pyw = os.environ.get("JPLENS_PYTHONW") or str(Path(sys.executable).with_name("pythonw.exe"))
    if not Path(pyw).exists():
        pyw = sys.executable
    code = f"import os,sys,runpy;os.chdir(r'{ROOT}');sys.path.insert(0,r'{ROOT}');runpy.run_module('jplens',run_name='__main__')"
    return f'"{pyw}" -c "{code}"'


def set_autostart(on: bool):
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on:
            winreg.SetValueEx(k, RUN_NAME, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(k, RUN_NAME)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------- 控制項
class Switch(QAbstractButton):
    def __init__(self, checked=False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(44, 24)

    def sizeHint(self):
        return QSize(44, 24)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        on, enabled = self.isChecked(), self.isEnabled()
        if not enabled:
            p.setOpacity(0.45)
        track = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(QColor(C["accent"] if on else C["dim"]), 1))
        p.setBrush(QColor(C["accent"]) if on else Qt.NoBrush)
        p.drawRoundedRect(track, 11.5, 11.5)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["bg"] if on else "#c8ccd4"))
        x = self.width() - 19 if on else 5
        p.drawEllipse(QRectF(x, 5, 14, 14))


def segmented(options, current, on_pick, enabled=True):
    box = QFrame()
    box.setObjectName("segBox")
    h = QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(0)
    group = QButtonGroup(box)
    for i, (val, text) in enumerate(options):
        b = QPushButton(text)
        b.setObjectName("seg")
        b.setProperty("first", "true" if i == 0 else "false")
        b.setCheckable(True)
        b.setChecked(val == current)
        b.setEnabled(enabled)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(lambda _=False, v=val: on_pick(v))
        group.addButton(b)
        h.addWidget(b)
    box.group = group
    return box


def combo(options, current, on_pick, enabled=True):
    """options：[(值, 文字, 準備中?)]；準備中的選項看得到、不能選。"""
    cb = QComboBox()
    cb.setMinimumWidth(230)
    for val, text, *soon in options:
        cb.addItem(text + ("（準備中）" if soon and soon[0] else ""), val)
        if soon and soon[0]:
            cb.model().item(cb.count() - 1).setEnabled(False)
    idx = cb.findData(current)
    cb.setCurrentIndex(max(idx, 0))
    cb.setEnabled(enabled)
    cb.currentIndexChanged.connect(lambda _: on_pick(cb.currentData()))
    return cb


def _label(text, name=None, wrap=False):
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def _button(text, slot=None):
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


class Group(QFrame):
    """一組設定（圓角的框，裡面一列一列）。"""

    def __init__(self):
        super().__init__()
        self.setObjectName("group")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(0)

    def add(self, label, desc="", control=None, note="", reserved=False, stacked=False):
        row = QFrame()
        if self.v.count():
            row.setObjectName("rowLine")
        lay = QVBoxLayout(row) if stacked else QHBoxLayout(row)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10 if stacked else 16)
        text = QVBoxLayout()
        text.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(8)
        title = _label(label)
        head.addWidget(title)
        if reserved:
            head.addWidget(_label("準備中", "reserved"))
        head.addStretch(1)
        text.addLayout(head)
        if desc:
            text.addWidget(_label(desc, "desc", wrap=True))
        if note:
            text.addWidget(_label(note, "note", wrap=True))
        lay.addLayout(text, 1)
        if control is not None:
            if stacked:
                lay.addWidget(control)
            else:
                lay.addWidget(control, 0, Qt.AlignVCenter)
        row.title = title
        self.v.addWidget(row)
        return row


class PopupPreview(QFrame):
    """結果視窗外觀的預覽：一列單字，字級、寬度跟著設定變。"""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("PopupPreview { background: #16181d; border-radius: 6px; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 10)
        v.setSpacing(8)
        self.box = QFrame()
        self.box.setObjectName("pvBox")
        v.addWidget(self.box, 0, Qt.AlignHCenter)
        self.caption = _label("", "dim")
        v.addWidget(self.caption, 0, Qt.AlignHCenter)
        bl = QVBoxLayout(self.box)
        bl.setContentsMargins(10, 8, 10, 8)
        self.row = QFrame()
        self.row.setObjectName("pvRow")
        bl.addWidget(self.row)
        rl = QVBoxLayout(self.row)
        rl.setContentsMargins(8, 6, 8, 6)
        rl.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.word, self.reading, self.badge, self.pos = QLabel("突く"), QLabel("つく"), QLabel("N3"), QLabel("五段動詞")
        for w in (self.word, self.reading, self.badge, self.pos):
            top.addWidget(w, 0, Qt.AlignVCenter)
        top.addStretch(1)
        rl.addLayout(top)
        self.meaning = QLabel("戳、刺、扎；撞、頂、捅；拄（杖）、支撐著地")
        self.meaning.setWordWrap(True)
        rl.addWidget(self.meaning)

    def apply(self, font_size: str, width_key: str):
        s, w = FONT_SCALES[font_size], POPUP_WIDTHS[width_key]
        px = lambda n: round(n * s)  # noqa: E731
        self.box.setFixedWidth(w)
        # QLabel 也是 QFrame：外框的樣式只用物件名稱指定，不然會套到每個字上
        self.box.setStyleSheet("#pvBox { background: rgb(28,30,36); border: 1px solid #555; border-radius: 8px; }")
        self.row.setStyleSheet("#pvRow { background: rgba(255,255,255,10); border-radius: 6px; }")
        self.word.setStyleSheet(f"font-size:{px(18)}px; font-weight:bold; color:white; font-family:{JP_FONTS};")
        self.reading.setStyleSheet(f"font-size:{px(13)}px; color:#9fd0ff; font-family:{JP_FONTS};")
        self.badge.setStyleSheet(badge_style("N3").replace("font-size:11px", f"font-size:{px(11)}px"))
        self.pos.setStyleSheet(f"font-size:{px(11)}px; color:#a0a0a0;")
        self.meaning.setStyleSheet(f"font-size:{px(13)}px; color:#e0e0e0;")
        self.caption.setText(f"結果視窗寬 {w} 像素")


# ---------------------------------------------------------------- 視窗
class SettingsWindow(QWidget):
    changed = Signal(str)  # 改了哪個設定（App.apply_setting 讓它馬上生效）

    def __init__(self, cfg, app=None):
        from .ui import make_icon
        super().__init__(None, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setObjectName("window")
        self.setStyleSheet(QSS)
        self.setWindowIcon(make_icon())
        self.setWindowTitle("設定 — 螢幕查字")
        self.resize(880, 640)
        self.setMinimumSize(640, 460)
        self.cfg, self.app = cfg, app

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        nav = QFrame()
        nav.setObjectName("drawer")
        nav.setFixedWidth(210)
        nv = QVBoxLayout(nav)
        nv.setContentsMargins(12, 14, 12, 12)
        nv.setSpacing(2)
        self.stack = QStackedWidget()
        self.tabs = {}
        group = QButtonGroup(self)
        for i, (key, title, sub) in enumerate(TABS):
            b = QPushButton()
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setMinimumHeight(40)
            b.setCursor(Qt.PointingHandCursor)
            row = QHBoxLayout(b)
            row.setContentsMargins(12, 0, 10, 0)
            name = QLabel(title)
            name.setAttribute(Qt.WA_TransparentForMouseEvents)
            row.addWidget(name)
            row.addStretch(1)
            if key in RESERVED_TABS:
                tag = _label("預留", "reserved")
                tag.setAttribute(Qt.WA_TransparentForMouseEvents)
                row.addWidget(tag, 0, Qt.AlignVCenter)
            b.clicked.connect(lambda _=False, k=key: self.show_tab(k))
            group.addButton(b)
            nv.addWidget(b)
            page = self._page(title, sub, key)
            self.stack.addWidget(page)
            self.tabs[key] = (b, i)
        nv.addStretch(1)
        nv.addWidget(_label("改了就自動儲存", "dim"))
        h.addWidget(nav)
        h.addWidget(self.stack, 1)
        self.show_tab("general")

    def showEvent(self, e):
        super().showEvent(e)
        dark_title_bar(self)

    def show_tab(self, key):
        b, i = self.tabs[key]
        b.setChecked(True)
        self.stack.setCurrentIndex(i)

    def _set(self, key, value):
        setattr(self.cfg, key, value)
        self.cfg.save()
        self.changed.emit(key)

    # ---------------------------------------------------------------- 分頁
    def _page(self, title, sub, key) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(24, 20, 24, 28)
        v.setSpacing(8)
        v.addWidget(_label(title, "h1"))
        v.addWidget(_label(sub, "sub"))
        v.addSpacing(8)
        self._v = v
        getattr(self, "_tab_" + key)()
        v.addStretch(1)
        scroll.setWidget(host)
        return scroll

    def _section(self, title, note=""):
        head = QHBoxLayout()
        head.setContentsMargins(4, 8, 4, 0)
        head.addWidget(_label(title, "section"))
        if note:
            head.addWidget(_label(note, "desc"))
        head.addStretch(1)
        self._v.addLayout(head)
        g = Group()
        self._v.addWidget(g)
        return g

    def _switch(self, key, slot=None):
        sw = Switch(bool(getattr(self.cfg, key)))
        sw.toggled.connect(slot or (lambda on: self._set(key, on)))
        return sw

    def _tab_general(self):
        g = self._section("語言")
        g.add("辨識的語言（要學的語言）",
              "查螢幕上哪一種語言的文字。",
              combo([("ja", "日文"), ("zh", "中文", True), ("ko", "韓文", True), ("en", "英文", True)],
                    self.cfg.src_lang, lambda v: self._set("src_lang", v)))
        g.add("解釋的語言", "單字意思、文法說明用哪一種語言顯示。",
              combo([("zh-TW", "繁體中文"), ("en", "English", True), ("zh-CN", "簡體中文", True)],
                    self.cfg.explain_lang, lambda v: self._set("explain_lang", v)))
        g.add("介面語言", "選單、按鈕上的文字。",
              combo([("zh-TW", "繁體中文"), ("en", "English", True)], self.cfg.ui_lang, lambda v: self._set("ui_lang", v)))
        g = self._section("啟動")
        sw = Switch(autostart_enabled())
        sw.toggled.connect(self._on_autostart)
        g.add("開機時自動啟動", "登入 Windows 後在系統匣待命。", sw)
        g.add("啟動時就載入漫畫模型", "切到漫畫模式時不用等，但會多用記憶體。", self._switch("preload_manga"))
        from .elevate import is_admin
        g.add("以系統管理員身分執行（遊戲裡要用）",
              "絕區零、鳴潮、星塔旅人這類遊戲以系統管理員身分執行，查字工具也要是系統管理員身分才收得到按鍵。"
              "每次啟動會跳一次 Windows 的確認視窗。",
              self._switch("run_as_admin", self._on_admin),
              note="目前已經是系統管理員身分" if is_admin() else "")

    def _on_admin(self, on):
        from .elevate import is_admin
        self._set("run_as_admin", on)
        if on and not is_admin() and self.app is not None and QMessageBox.question(
                self, "設定", "要現在以系統管理員身分重新啟動查字工具嗎？") == QMessageBox.Yes:
            self.app.restart_as_admin()

    def _on_autostart(self, on):
        try:
            set_autostart(on)
        except OSError as e:
            QMessageBox.warning(self, "設定", f"改不了開機啟動：{e}")

    def _tab_lookup(self):
        g = self._section("選取")
        g.add("熱鍵", "按住這個鍵，再用滑鼠拖曳選取。",
              segmented(list(HOTKEY_NAMES.items()), self.cfg.hotkey, self._on_hotkey))
        self.mode_seg = segmented([("text", "反白"), ("box", "框選"), ("manga", "漫畫")], self.cfg.select_mode,
                                  lambda v: self._set("select_mode", v))
        g.add("預設選取模式", "反白：沿著文字拖。框選：拖出矩形。漫畫：框住或雙擊對話框。", self.mode_seg)
        self.mode_key_row = g.add("", "依序切換 反白 → 框選 → 漫畫。", self._switch("mode_key"))
        self.spinner_row = g.add("游標旁顯示轉圈", "", self._switch("show_spinner"))
        self._hotkey_texts()

    def _hotkey_texts(self):
        hk = self.cfg.hotkey_name
        self.mode_key_row.title.setText(f"按住 {hk}＋右鍵切換模式")
        self.spinner_row.title.setText(f"按下 {hk} 時游標旁顯示轉圈（畫面分析好就停）")

    def _on_hotkey(self, v):
        self._set("hotkey", v)
        self._hotkey_texts()

    def sync_mode(self):
        """模式在別的地方（結果視窗、系統匣、Alt＋右鍵）改了：同步按鈕。"""
        for b in self.mode_seg.group.buttons():
            b.setChecked(b.text() == {"text": "反白", "box": "框選", "manga": "漫畫"}[self.cfg.select_mode])

    def _tab_popup(self):
        g = self._section("顯示內容")
        g.add("級數過濾", "只顯示這個級數以上的單字，級外的字一律顯示。結果視窗上方也能切換。",
              combo([(5, "全部"), (4, "N4 以上"), (3, "N3 以上"), (2, "N2 以上"), (1, "只顯示 N1")],
                    self.cfg.min_level, lambda v: self._set("min_level", v)))
        g.add("顯示專有名詞", "人名、地名。", self._switch("show_proper"))
        g.add("顯示辨識出的原文", "單字清單上方顯示讀到的整段文字，可以反白其中一段記錄。", self._switch("show_text"))
        g = self._section("外觀")
        self.preview = PopupPreview()
        g.add("字級", "", segmented([("s", "小"), ("m", "標準"), ("l", "大")], self.cfg.font_size,
                                     lambda v: self._look("font_size", v)))
        g.add("視窗寬度", "", segmented([("narrow", "窄"), ("std", "標準"), ("wide", "寬")], self.cfg.popup_width,
                                       lambda v: self._look("popup_width", v)))
        g.add("預覽", "", self.preview, stacked=True)
        self.preview.apply(self.cfg.font_size, self.cfg.popup_width)

    def _look(self, key, v):
        self._set(key, v)
        self.preview.apply(self.cfg.font_size, self.cfg.popup_width)

    def _tab_vocab(self):
        g = self._section("存檔")
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        self.vocab_path = QLineEdit(str(self.cfg.vocab_path))
        self.vocab_path.setObjectName("path")
        self.vocab_path.setReadOnly(True)
        h.addWidget(self.vocab_path, 1)
        h.addWidget(_button("變更…", self._change_vocab_dir))
        h.addWidget(_button("開啟資料夾", lambda: self._open_dir(self.cfg.vocab_path)))
        g.add("單字本資料夾", "記錄的單字、句子存在這裡（saved_words.jsonl、saved_sentences.jsonl）。", box, stacked=True)
        g = self._section("匯出的預設值")
        g.add("匯出單字時的範圍", "匯出確認視窗打開時預先選好的範圍。",
              combo([("all", "全部記錄"), ("marked", "只匯出重點單字")], self.cfg.export_scope,
                    lambda v: self._set("export_scope", v)))
        g.add("已學會的單字也匯出", "匯出確認視窗裡還能再改。", self._switch("export_learned"))

    def _change_vocab_dir(self):
        import shutil
        from .vocab import SAVED_NAME, SENTENCES
        old = self.cfg.vocab_path
        new = QFileDialog.getExistingDirectory(self, "選擇單字本資料夾", str(old))
        if not new or Path(new) == old:
            return
        new = Path(new)
        files = [n for n in (SAVED_NAME, SENTENCES) if (old / n).exists() and not (new / n).exists()]
        if files:
            ans = QMessageBox.question(
                self, "單字本資料夾", "要把目前的單字本複製到新的資料夾嗎？\n（選「否」會從新資料夾裡原本的內容開始，舊的檔案不會刪掉）",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ans == QMessageBox.Cancel:
                return
            if ans == QMessageBox.Yes:
                for n in files:
                    shutil.copy2(old / n, new / n)
        self._set("vocab_dir", str(new))
        self.vocab_path.setText(str(new))

    def _tab_ai(self):
        hint = QFrame()
        hint.setObjectName("hint")
        hl = QHBoxLayout(hint)
        hl.setContentsMargins(12, 8, 12, 8)
        hl.addWidget(_label("這一頁先預留位置，標「準備中」的功能還沒實作。", "note", wrap=True))
        self._v.addWidget(hint)
        g = self._section("Gemini")
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        key = QLineEdit()
        key.setEchoMode(QLineEdit.Password)
        key.setPlaceholderText("準備中：之後在這裡貼上 API Key")
        key.setEnabled(False)
        v.addWidget(key)
        has_env = bool(os.environ.get("GEMINI_API_KEY"))
        st = _label(("● 已從環境變數 GEMINI_API_KEY 讀到 Key" if has_env else "● 環境變數裡沒有 GEMINI_API_KEY"), "desc")
        st.setStyleSheet(f"color: {C['ok'] if has_env else C['sub']};")
        v.addWidget(st)
        g.add("Gemini API Key", "Key 會存在 Windows 的認證管理員，不會寫進 config.json。", box, reserved=True, stacked=True)
        g.add("模型", "品質優先，不用 Flash-Lite。", combo([("tbd", "之後列出可用的模型")], "tbd", lambda v: None,
                                                    enabled=False), reserved=True)
        g = self._section("整句翻譯")
        g.add("翻譯方式", "結果視窗多一個「翻譯」按鈕，按了才翻，同一句會記住不重翻。",
              segmented([("off", "不使用"), ("gemini", "Gemini"), ("local", "本地模型")], "off", lambda v: None,
                        enabled=False), reserved=True)

    def _tab_advanced(self):
        g = self._section("辨識")
        g.add("新版墨水投影", "框選、反白直排小說時不靠偵測模型，直接找欄與振假名。", self._switch("ink_columns"))
        g.add("舊的版本規則", "新版本有問題時才打開。", self._switch("layout_rules"))
        g = self._section("系統")
        g.add("顯示卡", "文字偵測、辨識用的裝置。", _label(self._device_text(), "sub"))
        g.add("紀錄檔", "每次查詢的時間與辨識結果，回報問題時用得到。",
              _button("開啟資料夾", lambda: self._open_dir(ROOT / "logs")))

    def _device_text(self) -> str:
        if self.app is not None and not self.app.worker.ready:
            return "模型載入中…"
        from . import gpu
        return "CUDA（NVIDIA 顯示卡）" if gpu.cuda_ready() else "CPU（沒有找到可用的 CUDA）"

    def _tab_about(self):
        g = self._section("螢幕查字")
        g.add("版本", "", _label("開發版（還沒正式發布）", "sub"))
        g.add("程式碼授權", "", _label("Apache-2.0", "sub"))
        g = self._section("資料與模型來源", "CC BY-SA 的資料要求在 App 裡標示來源")
        lst = QFrame()
        lv = QVBoxLayout(lst)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        for i, (name, use, lic) in enumerate(CREDITS):
            row = QFrame()
            if i:
                row.setObjectName("rowLine")
            rh = QHBoxLayout(row)
            rh.setContentsMargins(4, 7, 4, 7)
            col = QVBoxLayout()
            col.setSpacing(1)
            col.addWidget(_label(name))
            col.addWidget(_label(use, "dim"))
            rh.addLayout(col, 1)
            rh.addWidget(_label(lic, "sub"))
            lv.addWidget(row)
        g.add("使用的資料、模型與套件", "", lst, stacked=True)
        g.add("完整授權條文", "", _button("開啟 NOTICE.md",
                                     lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str((ROOT if (ROOT / "NOTICE.md").exists() else APP_DIR) / "NOTICE.md")))))

    def _open_dir(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
