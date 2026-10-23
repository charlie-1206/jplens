"""設定視窗：左邊分頁、右邊內容，改了就寫進 config.json 並馬上生效（App.apply_setting）。

分頁：一般（語言〔預留〕、啟動）、查字操作、結果視窗、單字本、翻譯與 AI（Gemini：API Key、翻譯、講解）、進階、關於。
"""

import os
import sys
import winreg
from pathlib import Path

import shiboken6
from PySide6.QtCore import QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout,
                               QWidget)

from .config import APP_DIR, FONT_SCALES, FROZEN, HOTKEY_NAMES, POPUP_WIDTHS, ROOT
from .i18n import t
from .theme import C, JP_FONTS, QSS, badge_style, dark_title_bar

TABS = [("general", t("一般"), t("語言與啟動")), ("lookup", t("查字操作"), t("熱鍵與選取方式")),
        ("popup", t("結果視窗"), t("查字結果要顯示什麼、看起來多大")), ("vocab", t("單字本"), t("記錄存在哪裡、匯出的預設值")),
        ("ai", t("翻譯與 AI"), t("Gemini 的 API Key、整句翻譯、句子講解")), ("advanced", t("進階"), t("辨識的做法與系統狀態")),
        ("about", t("關於"), t("版本、授權與資料來源"))]
RESERVED_TABS: set = set()  # 整頁還沒實作的分頁（側邊標「預留」）；翻譯與 AI 2026-10-17 做好了
CREDITS = [
    ("JMdict（EDRDG）", t("英文意思、詞性"), "CC BY-SA 4.0"),
    (t("Tomoshi 開放資料（Y1Z）"), t("中文意思"), "CC BY-SA 4.0"),
    (t("中文維基詞典（kaikki.org 擷取）"), t("中文意思"), "CC BY-SA"),
    (t("Gemini 翻譯（譯自 JMdict）"), t("日檢詞的中文意思"), "CC BY-SA 4.0"),
    (t("日檢單字表（Jonathan Waller；open-anki-jlpt-decks）"), t("級數"), "CC BY／MIT"),
    ("manga-ocr（kha-white）", t("文字辨識"), "Apache-2.0"),
    ("PP-OCR／RapidOCR", t("文字偵測"), "Apache-2.0"),
    ("comic-text-and-bubble-detector（ogkalu）", t("漫畫模式"), "Apache-2.0"),
    ("SudachiPy、SudachiDict", t("斷詞"), "Apache-2.0"),
    ("Qt（PySide6）", t("介面"), "LGPL-3.0"),
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
        cb.addItem(text + (t("（準備中）") if soon and soon[0] else ""), val)
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
            head.addWidget(_label(t("準備中"), "reserved"))
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
        self.word, self.reading = QLabel("突く"), QLabel("つく")  # i18n: 不翻
        self.badge, self.pos = QLabel("N3"), QLabel(t("五段動詞"))
        for w in (self.word, self.reading, self.badge, self.pos):
            top.addWidget(w, 0, Qt.AlignVCenter)
        top.addStretch(1)
        rl.addLayout(top)
        self.meaning = QLabel(t("戳、刺、扎；撞、頂、捅；拄（杖）、支撐著地"))
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
        self.caption.setText(t("結果視窗寬 {w} 像素", w=w))


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
        self.setWindowTitle(t("設定 — 螢幕查字"))
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
                tag = _label(t("預留"), "reserved")
                tag.setAttribute(Qt.WA_TransparentForMouseEvents)
                row.addWidget(tag, 0, Qt.AlignVCenter)
            b.clicked.connect(lambda _=False, k=key: self.show_tab(k))
            group.addButton(b)
            nv.addWidget(b)
            page = self._page(title, sub, key)
            self.stack.addWidget(page)
            self.tabs[key] = (b, i)
        nv.addStretch(1)
        nv.addWidget(_label(t("改了就自動儲存"), "dim"))
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
        g = self._section(t("語言"))
        g.add(t("辨識的語言（要學的語言）"),
              t("查螢幕上哪一種語言的文字。"),
              combo([("ja", t("日文")), ("zh", t("中文"), True), ("ko", t("韓文"), True), ("en", t("英文"), True)],
                    self.cfg.src_lang, lambda v: self._set("src_lang", v)))
        g.add(t("解釋的語言"), t("單字意思、文法說明、整句翻譯、講解用哪一種語言。English：單字意思用 JMdict 的英文。"
                                "改了之後新查的字才會換。"),
              combo([("zh-TW", "繁體中文"), ("en", "English"), ("zh-CN", "簡體中文", True)],  # i18n: 不翻
                    self.cfg.explain_lang, lambda v: self._set("explain_lang", v)))
        g.add(t("介面語言"), t("選單、按鈕上的文字。改了要從系統匣結束、再重新開啟查字工具才會換。"),
              combo([("zh-TW", "繁體中文"), ("en", "English")], self.cfg.ui_lang,  # i18n: 不翻
                    lambda v: self._set("ui_lang", v)))
        g = self._section(t("啟動"))
        sw = Switch(autostart_enabled())
        sw.toggled.connect(self._on_autostart)
        g.add(t("開機時自動啟動"), t("登入 Windows 後在系統匣待命。"), sw)
        g.add(t("啟動時就載入漫畫模型"), t("切到漫畫模式時不用等，但會多用記憶體。"), self._switch("preload_manga"))
        from .elevate import is_admin
        g.add(t("以系統管理員身分執行（遊戲裡要用）"),
              t("絕區零、鳴潮、星塔旅人這類遊戲以系統管理員身分執行，查字工具也要是系統管理員身分才收得到按鍵。"
                "每次啟動會跳一次 Windows 的確認視窗。"),
              self._switch("run_as_admin", self._on_admin),
              note=t("目前已經是系統管理員身分") if is_admin() else "")

    def _on_admin(self, on):
        from .elevate import is_admin
        self._set("run_as_admin", on)
        if on and not is_admin() and self.app is not None and QMessageBox.question(
                self, t("設定"), t("要現在以系統管理員身分重新啟動查字工具嗎？")) == QMessageBox.Yes:
            self.app.restart_as_admin()

    def _on_autostart(self, on):
        try:
            set_autostart(on)
        except OSError as e:
            QMessageBox.warning(self, t("設定"), t("改不了開機啟動：{err}", err=e))

    def _tab_lookup(self):
        g = self._section(t("選取"))
        g.add(t("熱鍵"), t("按住這個鍵，再用滑鼠拖曳選取。"),
              segmented(list(HOTKEY_NAMES.items()), self.cfg.hotkey, self._on_hotkey))
        self.mode_seg = segmented([("text", t("反白")), ("box", t("框選")), ("manga", t("漫畫"))], self.cfg.select_mode,
                                  lambda v: self._set("select_mode", v))
        g.add(t("預設選取模式"), t("反白：沿著文字拖。框選：拖出矩形。漫畫：框住或雙擊對話框。"), self.mode_seg)
        self.mode_key_row = g.add("", t("依序切換 反白 → 框選 → 漫畫。"), self._switch("mode_key"))
        self.spinner_row = g.add(t("游標旁顯示轉圈"), "", self._switch("show_spinner"))
        g = self._section(t("無字幕模式"))
        self.listen_row = g.add(
            "", t("把電腦正在播的聲音（直播、動畫）即時轉成文字，放進結果視窗，按「查詢」才查字。"
                  "語音辨識只用 CPU、不佔顯示卡；再按一次結束。"),
            combo([(c, c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"], self.cfg.listen_key,
                  lambda v: (self._set("listen_key", v), self._hotkey_texts())))
        g.add(t("邊講邊顯示"), t("講話中先顯示暫定的文字（灰色），講完換成最終結果。每 0.8 秒重新辨識一次，CPU 約多用單核的 10%；"
                       "關掉就講完一句才出現。"), self._switch("listen_partial"))
        self._hotkey_texts()

    def _hotkey_texts(self):
        hk = self.cfg.hotkey_name
        self.mode_key_row.title.setText(t("按住 {hk}＋右鍵切換模式", hk=hk))
        self.spinner_row.title.setText(t("按下 {hk} 時游標旁顯示轉圈（畫面分析好就停）", hk=hk))
        self.listen_row.title.setText(t("開關的按鍵：按住 {hk}＋字母（目前 {key}）", hk=hk, key=self.cfg.listen_name))

    def _on_hotkey(self, v):
        self._set("hotkey", v)
        self._hotkey_texts()

    def sync_mode(self):
        """模式在別的地方（結果視窗、系統匣、Alt＋右鍵）改了：同步按鈕。"""
        for b in self.mode_seg.group.buttons():
            b.setChecked(b.text() == {"text": t("反白"), "box": t("框選"), "manga": t("漫畫")}[self.cfg.select_mode])

    def _tab_popup(self):
        g = self._section(t("顯示內容"))
        g.add(t("級數過濾"), t("只顯示這個級數以上的單字，級外的字一律顯示。結果視窗上方也能切換。"),
              combo([(5, t("全部")), (4, t("N4 以上")), (3, t("N3 以上")), (2, t("N2 以上")), (1, t("只顯示 N1"))],
                    self.cfg.min_level, lambda v: self._set("min_level", v)))
        g.add(t("顯示專有名詞"), t("人名、地名。"), self._switch("show_proper"))
        g.add(t("顯示辨識出的原文"), t("單字清單上方顯示讀到的整段文字，可以反白其中一段記錄。"), self._switch("show_text"))
        g = self._section(t("外觀"))
        self.preview = PopupPreview()
        g.add(t("字級"), "", segmented([("s", t("小")), ("m", t("標準")), ("l", t("大"))], self.cfg.font_size,
                                       lambda v: self._look("font_size", v)))
        g.add(t("視窗寬度"), "", segmented([("narrow", t("窄")), ("std", t("標準")), ("wide", t("寬"))],
                                         self.cfg.popup_width, lambda v: self._look("popup_width", v)))
        g.add(t("預覽"), "", self.preview, stacked=True)
        self.preview.apply(self.cfg.font_size, self.cfg.popup_width)

    def _look(self, key, v):
        self._set(key, v)
        self.preview.apply(self.cfg.font_size, self.cfg.popup_width)

    def _tab_vocab(self):
        g = self._section(t("存檔"))
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        self.vocab_path = QLineEdit(str(self.cfg.vocab_path))
        self.vocab_path.setObjectName("path")
        self.vocab_path.setReadOnly(True)
        h.addWidget(self.vocab_path, 1)
        h.addWidget(_button(t("變更…"), self._change_vocab_dir))
        h.addWidget(_button(t("開啟資料夾"), lambda: self._open_dir(self.cfg.vocab_path)))
        g.add(t("單字本資料夾"), t("記錄的單字、句子存在這裡（saved_words.jsonl、saved_sentences.jsonl）。"), box, stacked=True)
        g = self._section(t("匯出的預設值"))
        g.add(t("匯出單字時的範圍"), t("匯出確認視窗打開時預先選好的範圍。"),
              combo([("all", t("全部記錄")), ("marked", t("只匯出重點單字"))], self.cfg.export_scope,
                    lambda v: self._set("export_scope", v)))
        g.add(t("已學會的單字也匯出"), t("匯出確認視窗裡還能再改。"), self._switch("export_learned"))

    def _change_vocab_dir(self):
        import shutil
        from .vocab import SAVED_NAME, SENTENCES
        old = self.cfg.vocab_path
        new = QFileDialog.getExistingDirectory(self, t("選擇單字本資料夾"), str(old))
        if not new or Path(new) == old:
            return
        new = Path(new)
        files = [n for n in (SAVED_NAME, SENTENCES) if (old / n).exists() and not (new / n).exists()]
        if files:
            ans = QMessageBox.question(
                self, t("單字本資料夾"),
                t("要把目前的單字本複製到新的資料夾嗎？\n（選「否」會從新資料夾裡原本的內容開始，舊的檔案不會刪掉）"),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ans == QMessageBox.Cancel:
                return
            if ans == QMessageBox.Yes:
                for n in files:
                    shutil.copy2(old / n, new / n)
        self._set("vocab_dir", str(new))
        self.vocab_path.setText(str(new))

    def _tab_ai(self):
        from . import ai
        g = self._section("Gemini")
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        row = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText(t("貼上 API Key（Google AI Studio 取得）"))
        row.addWidget(self.key_edit, 1)
        row.addWidget(_button(t("儲存"), self._save_key))
        self.key_clear = _button(t("刪除"), self._clear_key)
        row.addWidget(self.key_clear)
        v.addLayout(row)
        st_row = QHBoxLayout()
        self.key_status = _label("", "desc", wrap=True)
        st_row.addWidget(self.key_status, 1)
        self.key_test = _button(t("測試"), self._test_key)
        self.key_test.setToolTip(t("翻譯一句「こんにちは」確認 Key 能用（會用掉一次免費額度）"))
        st_row.addWidget(self.key_test)
        v.addLayout(st_row)
        g.add("API Key", t("存在 Windows 的認證管理員（加密，只有你的 Windows 帳號讀得到），不會寫進 config.json 或紀錄檔。"
                         "沒有存的話用環境變數 GEMINI_API_KEY。"), box, stacked=True)
        self._show_key_status()

        g = self._section(t("整句翻譯"), t("結果視窗"))
        g.add(t("翻譯來源"), t("預設用 Google Gemini（{model}，上面的 Key）。"
                       "也可以改用任何 OpenAI 相容的服務（Groq、OpenRouter、DeepSeek、OpenAI，或本機的 Ollama、LM Studio）。",
                       model=ai.TRANSLATE_MODEL),
              segmented([("gemini", "Google Gemini"), ("openai", t("自訂服務"))], self.cfg.translate_provider,
                        self._on_provider), stacked=True)
        self.custom_box = QWidget()
        cv = QVBoxLayout(self.custom_box)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(6)
        self.url_edit = QLineEdit(self.cfg.openai_base_url)
        self.url_edit.setPlaceholderText(t("網址，例如 https://api.groq.com/openai/v1"))
        self.url_edit.textChanged.connect(lambda text: self._set_custom("openai_base_url", text))
        cv.addWidget(self.url_edit)
        self.model_edit = QLineEdit(self.cfg.openai_model)
        self.model_edit.setPlaceholderText(t("模型名稱，例如 llama-3.3-70b-versatile"))
        self.model_edit.textChanged.connect(lambda text: self._set_custom("openai_model", text))
        cv.addWidget(self.model_edit)
        row = QHBoxLayout()
        self.okey_edit = QLineEdit()
        self.okey_edit.setEchoMode(QLineEdit.Password)
        self.okey_edit.setPlaceholderText(t("API Key（本機服務可以不填）"))
        row.addWidget(self.okey_edit, 1)
        row.addWidget(_button(t("儲存"), self._save_okey))
        self.okey_clear = _button(t("刪除"), self._clear_okey)
        row.addWidget(self.okey_clear)
        cv.addLayout(row)
        st = QHBoxLayout()
        self.okey_status = _label("", "desc", wrap=True)
        st.addWidget(self.okey_status, 1)
        self.okey_test = _button(t("測試"), self._test_custom)
        self.okey_test.setToolTip(t("翻譯一句「こんにちは」確認網址、模型、Key 都對（會用掉一次額度）"))
        st.addWidget(self.okey_test)
        cv.addLayout(st)
        self.custom_row = g.add(t("自訂服務"), t("網址、模型名稱存在 config.json；Key 存在 Windows 認證管理員"
                                       "（沒存的話用環境變數 OPENAI_API_KEY）。"), self.custom_box, stacked=True)
        self._show_okey_status()
        self._on_provider(self.cfg.translate_provider, save=False)
        g.add(t("翻譯時機"), t("背景預先翻好：查字結果一出來就先翻，按「翻譯」時多半已經好了（每次查字都會用一次額度，翻過的不重翻）。"),
              segmented([("manual", t("按了才翻")), ("prefetch", t("背景預先翻好")), ("auto", t("自動翻並顯示"))],
                        self.cfg.translate_timing, lambda v: self._set("translate_timing", v)), stacked=True)
        spin = QSpinBox()
        spin.setRange(1, 50)
        spin.setSuffix(t(" 字以上"))
        spin.setValue(self.cfg.prefetch_min_chars)
        spin.valueChanged.connect(lambda v: self._set("prefetch_min_chars", v))
        g.add(t("預先翻譯的長度"), t("原文（不算標點）至少幾個字才在背景先翻；太短的（「えっ」「うん」）按了才翻。"), spin)

        g = self._section(t("句子講解"), t("單字本 → 句子"))
        g.add(t("講解模型"), t("翻譯、拆解詞與文法、語氣。講解會存起來，之後打開不用再呼叫。"),
              combo(list(ai.EXPLAIN_MODELS.items()), self.cfg.explain_model, lambda m: self._set("explain_model", m)))

    # ---------------------------------------------------------------- 自訂翻譯服務
    def _on_provider(self, v, save=True):
        if save:
            self._set("translate_provider", v)
        self.custom_row.setVisible(v == "openai")

    def _set_custom(self, key: str, text: str) -> bool:
        """網址、模型名稱打字就存（config.json 是明文）：看起來像 API Key 的不存（貼錯欄位），網址要 http:// 或 https:// 開頭才存。"""
        from . import ai
        val = text.strip()
        if ai.looks_like_key(val):
            self._show_okey_status(t("● 這看起來是 API Key，請貼到下面的 Key 欄位（這一欄會存成明文，所以沒有存）"), False)
            return False
        if key == "openai_base_url" and val and not val.lower().startswith(("http://", "https://")):
            self._show_okey_status(t("● 網址要以 http:// 或 https:// 開頭"), False)
            return False
        self._set(key, val)
        self._show_okey_status()
        return True

    def _show_okey_status(self, msg: str = "", ok: bool | None = None):
        from . import ai
        key, src = ai.get_key("openai")
        if not msg:
            msg = {"keyring": t("● Key 已存在 Windows 認證管理員（{key}）", key=ai.masked(key)),
                   "env": t("● Key 用環境變數 OPENAI_API_KEY（{key}）", key=ai.masked(key)),
                   "": t("● 沒有 Key（本機服務不需要）")}[src]
            ok = True if src else None
        self.okey_status.setText(msg)
        self.okey_status.setStyleSheet(f"color: {C['ok'] if ok else C['sub'] if ok is None else '#e0a080'};")
        self.okey_clear.setEnabled(src == "keyring")

    def _save_okey(self):
        from . import ai
        k = self.okey_edit.text().strip()
        if not k:
            return
        try:
            ai.set_key(k, "openai")
        except Exception as e:  # noqa: BLE001
            self._show_okey_status(t("● 存不進去：{err}", err=e), False)
            return
        self.okey_edit.clear()
        self._show_okey_status()

    def _clear_okey(self):
        from . import ai
        ai.clear_key("openai")
        self._show_okey_status()

    def _test_custom(self):
        from . import ai
        if not (self._set_custom("openai_base_url", self.url_edit.text())
                and self._set_custom("openai_model", self.model_edit.text())):
            return
        self.okey_test.setEnabled(False)
        self._show_okey_status(t("● 測試中…"), None)
        url, model = self.cfg.openai_base_url, self.cfg.openai_model

        def done(zh, err):
            if not shiboken6.isValid(self):  # 測試中把設定視窗關掉了（WA_DeleteOnClose）
                return
            self.okey_test.setEnabled(True)
            self._show_okey_status(t("● 可以用：こんにちは → {zh}", zh=zh) if not err else f"● {err}", not err)
        ai.run_async(lambda: ai.openai_chat(url, model, ai.TRANSLATE_PROMPT.format(text="こんにちは")), done)  # i18n: 不翻

    def _show_key_status(self, msg: str = "", ok: bool | None = None):
        from . import ai
        key, src = ai.get_key()
        if not msg:
            msg = {"keyring": t("● 已存在 Windows 認證管理員（{key}）", key=ai.masked(key)),
                   "env": t("● 用環境變數 GEMINI_API_KEY（{key}）", key=ai.masked(key)),
                   "": t("● 還沒有設定 Key：翻譯、講解不能用")}[src]
            ok = bool(src)
        self.key_status.setText(msg)
        self.key_status.setStyleSheet(f"color: {C['ok'] if ok else C['sub'] if ok is None else '#e0a080'};")
        self.key_clear.setEnabled(src == "keyring")
        self.key_test.setEnabled(bool(src))

    def _save_key(self):
        from . import ai
        k = self.key_edit.text().strip()
        if not k:
            return
        try:
            ai.set_key(k)
        except Exception as e:  # noqa: BLE001
            self._show_key_status(t("● 存不進去：{err}", err=e), False)
            return
        self.key_edit.clear()
        self._show_key_status()

    def _clear_key(self):
        from . import ai
        ai.clear_key()
        self._show_key_status()

    def _test_key(self):
        from . import ai
        self.key_test.setEnabled(False)
        self._show_key_status(t("● 測試中…"), None)

        def done(zh, err):
            if not shiboken6.isValid(self):
                return
            self._show_key_status()
            self._show_key_status((t("● 可以用：こんにちは → {zh}", zh=zh) if not err else f"● {err}"), not err)
        ai.run_async(lambda: ai._generate(ai.TRANSLATE_MODEL, ai.TRANSLATE_PROMPT.format(text="こんにちは")), done)  # i18n: 不翻

    def _tab_advanced(self):
        g = self._section(t("辨識"))
        g.add(t("新版墨水投影"), t("框選、反白直排小說時不靠偵測模型，直接找欄與振假名。"), self._switch("ink_columns"))
        g.add(t("舊的版本規則"), t("新版本有問題時才打開。"), self._switch("layout_rules"))
        g = self._section(t("系統"))
        g.add(t("顯示卡"), t("文字偵測、辨識用的裝置。"), _label(self._device_text(), "sub"))
        g.add(t("紀錄檔"), t("每次查詢的時間與辨識結果，回報問題時用得到。"),
              _button(t("開啟資料夾"), lambda: self._open_dir(ROOT / "logs")))

    def _device_text(self) -> str:
        if self.app is not None and not self.app.worker.ready:
            return t("模型載入中…")
        from . import gpu
        b = gpu.backend()
        return gpu.name() if b != "cpu" else t("CPU（沒有找到可用的 CUDA／DirectML，或顯示卡比 CPU 慢）")

    def _tab_about(self):
        g = self._section(t("螢幕查字"))
        import onnxruntime as ort

        from . import __version__
        flavor = t("DirectML 版") if "DmlExecutionProvider" in ort.get_available_providers() else t("CUDA 版")
        g.add(t("版本"), "", _label(t("v{ver}（{flavor}）", ver=__version__, flavor=flavor), "sub"))
        g.add(t("程式碼授權"), "", _label("Apache-2.0", "sub"))
        g = self._section(t("資料與模型來源"), t("CC BY-SA 的資料要求在 App 裡標示來源"))
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
        g.add(t("使用的資料、模型與套件"), "", lst, stacked=True)
        g.add(t("完整授權條文"), "", _button(t("開啟 NOTICE.md"), lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str((ROOT if (ROOT / "NOTICE.md").exists() else APP_DIR) / "NOTICE.md")))))

    def _open_dir(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
