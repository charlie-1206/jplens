"""Qt 視窗：框選外框、結果視窗。

座標：掛鉤與擷取用實體像素；Qt 用邏輯座標（縮放 125%／150% 時兩者不同）。
Qt 6 在 Windows 上每個螢幕的左上角邏輯座標＝實體座標，螢幕內再除以縮放比例，見 phys_to_logical。
"""

import ctypes
import ctypes.wintypes as wt
import re

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from .words import WordInfo

LEVEL_CHOICES = [("全部", 5), ("N4 以上", 4), ("N3 以上", 3), ("N2 以上", 2), ("N1", 1)]
LEVEL_COLORS = {"N5": "#6b8e6b", "N4": "#4f8a8b", "N3": "#4a6fa5", "N2": "#8a5fa8", "N1": "#b5524a",
                "級外": "#7a7a7a", "專有名詞": "#9a7b3c"}
BORDER = 3
LIST_MAX_H = 420  # 單字清單最高多少（邏輯像素），超過就捲動
MODES = {  # 選取模式：按鈕文字、說明
    "text": ("反白", "目前是反白模式：按住 Alt 沿著文字拖曳選字。"),
    "box": ("框選", "目前是框選模式：按住 Alt 拖曳出矩形。"),
    "manga": ("漫畫", "目前是漫畫模式：按住 Alt 框住對話框，或按住 Alt 雙擊對話框。\n"
                      "用漫畫專用的模型找出對話框裡的字、整塊讀，不會漏欄；只適合漫畫（字幕、小說請用另外兩種）。"),
}
LIST_SENSES = 3  # 清單裡每個字最多顯示幾個意思（英文少一個，因為比較長）；完整內容在滑鼠提示


def phys_to_logical(x: int, y: int) -> QPoint:
    """實體座標 → Qt 邏輯座標。剛好在螢幕右／下邊界上（框選拖到螢幕邊緣）或不在任何螢幕裡時，
    用最近的那個螢幕的縮放比例換算（不然 150% 縮放的螢幕會差 1.5 倍）。"""
    best, best_d = None, None
    for s in QGuiApplication.screens():
        g, r = s.geometry(), s.devicePixelRatio()
        x1, y1 = g.x() + g.width() * r, g.y() + g.height() * r
        d = max(g.x() - x, 0, x - x1) + max(g.y() - y, 0, y - y1)  # 0 = 在螢幕裡或邊界上
        if g.x() <= x < x1 and g.y() <= y < y1:
            d = -1  # 在螢幕裡面優先（兩個螢幕相鄰時，邊界上的點也在右邊那個螢幕裡）
        if best_d is None or d < best_d:
            best, best_d = (g, r), d
    if best is None:
        return QPoint(x, y)
    g, r = best
    return QPoint(round(g.x() + (x - g.x()) / r), round(g.y() + (y - g.y()) / r))


def native_rect(w: QWidget) -> tuple[int, int, int, int]:
    rc = wt.RECT()
    ctypes.windll.user32.GetWindowRect(wt.HWND(int(w.winId())), ctypes.byref(rc))
    return rc.left, rc.top, rc.right, rc.bottom


class SelectionFrame(QWidget):
    """框選時顯示的外框。畫在選取範圍外側，所以擷取畫面時不會拍進去；滑鼠事件直接穿透。"""

    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._origin = None

    def begin(self, x, y):
        self._origin = (x, y)
        self.update_to(x, y)
        self.show()

    def update_to(self, x, y):
        if self._origin is None:
            return
        x0, y0 = self._origin
        a = phys_to_logical(min(x0, x), min(y0, y))
        b = phys_to_logical(max(x0, x), max(y0, y))
        self.setGeometry(QRect(a, b).adjusted(-BORDER, -BORDER, BORDER, BORDER))
        self.update()

    def end(self):
        self._origin = None
        self.hide()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(QPen(QColor(255, 80, 60), BORDER, Qt.DashLine))
        p.drawRect(self.rect().adjusted(BORDER // 2, BORDER // 2, -(BORDER - BORDER // 2), -(BORDER - BORDER // 2)))


class BusySpinner(QWidget):
    """跟著滑鼠游標的小轉圈：模型載入中、等版面偵測、辨識中時顯示。
    不改 Windows 的系統游標（工具當掉時游標會一直停在忙碌狀態）。滑鼠事件直接穿透。
    多個等待可能同時發生（模型還在載入時就查詢），全部結束才隱藏。"""

    SIZE = 22    # 邏輯像素
    OFFSET = 16  # 離游標尖端多遠（右下方）

    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(self.SIZE, self.SIZE)
        self._reasons: set[str] = set()
        self.muted: set[str] = set()  # 不顯示轉圈的原因（設定視窗關掉「按下熱鍵時轉圈」＝{"layout"}）
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self, reason: str):
        if reason in self.muted:
            return
        self._reasons.add(reason)
        if not self._timer.isActive():
            self._tick()
            self.show()
            self._timer.start(30)

    def stop(self, reason: str):
        self._reasons.discard(reason)
        if not self._reasons and self._timer.isActive():
            self._timer.stop()
            self.hide()

    def _tick(self):
        pt = wt.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        p = phys_to_logical(pt.x, pt.y)
        self.move(p.x() + self.OFFSET, p.y() + self.OFFSET)
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(3, 3, -3, -3)
        p.setPen(QPen(QColor(0, 0, 0, 90), 4))            # 深色底圈：亮的背景上也看得到
        p.drawEllipse(r)
        pen = QPen(QColor(255, 210, 122), 3)              # 和結果視窗標題同色
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(r, -self._angle * 16, 100 * 16)


STYLE = """
QFrame#popup { background: rgb(28, 30, 36); border: 1px solid #555; border-radius: 8px; }
QLabel { color: #e8e8e8; }
QLabel#title { font-size: 13px; font-weight: bold; color: #ffd27a; }
QLabel#text { color: #cfd6e6; font-size: 14px; background: rgba(255,255,255,18); border-radius: 4px; padding: 6px; }
QLabel#word { font-size: 18px; font-weight: bold; color: #ffffff; }
QLabel#grammar { font-size: 15px; font-weight: bold; color: #ffffff; }
QLabel#section { font-size: 12px; font-weight: bold; color: #ffd27a; padding: 4px 2px 0px 2px; }
QLabel#reading { font-size: 13px; color: #9fd0ff; }
QLabel#pos { font-size: 11px; color: #a0a0a0; }
QLabel#meaning { font-size: 13px; color: #e0e0e0; }
QLabel#meaning_en { font-size: 12px; color: #a8a8a8; }
QLabel#also { font-size: 12px; color: #b8c4d4; }
QLabel#status, QLabel#attr { font-size: 11px; color: #8a8a8a; }
QFrame#row { background: rgba(255,255,255,10); border-radius: 6px; }
QPushButton { color: #e8e8e8; background: #3a3f4b; border: 1px solid #555; border-radius: 4px; padding: 3px 8px; }
QPushButton:hover { background: #4a5160; }
QPushButton#saved { color: #8fd18f; background: transparent; border: 1px solid #4a6a4a; }
QPushButton#flat { background: transparent; border: none; font-size: 14px; padding: 2px 6px; }
QPushButton#flat:checked { color: #ffd27a; background: #5a4c26; border: 1px solid #ffd27a; border-radius: 4px; }
QPushButton#mode { font-size: 12px; padding: 1px 8px; }
QComboBox, QCheckBox { color: #e8e8e8; font-size: 12px; }
QComboBox { background: #3a3f4b; border: 1px solid #555; border-radius: 4px; padding: 1px 6px; }
QComboBox QAbstractItemView { background: #2b2e36; color: #e8e8e8; selection-background-color: #4a5160; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #555b66; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
"""


def scaled_style(scale: float) -> str:
    """結果視窗的樣式表，字級乘上設定的倍數（config.FONT_SCALES）。"""
    return re.sub(r"font-size: (\d+)px", lambda m: f"font-size: {round(int(m.group(1)) * scale)}px", STYLE)


class WordRow(QFrame):
    def __init__(self, w: WordInfo, saved: bool, on_save, scale: float = 1.0):
        super().__init__()
        self.setObjectName("row")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        top = QHBoxLayout()
        word = QLabel(w.word)
        word.setObjectName("grammar" if w.is_grammar else "word")
        top.addWidget(word)
        if w.reading and w.reading != w.word:
            rd = QLabel(w.reading)
            rd.setObjectName("reading")
            top.addWidget(rd)
        badge = QLabel(w.jlpt)
        badge.setStyleSheet(f"background:{LEVEL_COLORS.get(w.jlpt, '#777')}; color:white; border-radius:3px;"
                            f"padding:0px 5px; font-size:{round(11 * scale)}px;")
        top.addWidget(badge)
        pos = QLabel(w.pos)
        pos.setObjectName("pos")
        top.addWidget(pos)
        top.addStretch(1)
        self.btn = QPushButton()
        self.btn.setFixedWidth(82)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.clicked.connect(lambda: on_save(w, self))
        self.set_saved(saved)
        top.addWidget(self.btn)
        lay.addLayout(top)

        # 冷僻的詞義（排在最後）只有在沒有其他意思時才列出來
        zh = w.zh[:len(w.zh) - w.rare_zh] or w.zh
        en_list = w.en[:len(w.en) - w.rare_en] or w.en
        if zh:
            m = QLabel("；".join(zh[:LIST_SENSES]))
            m.setObjectName("meaning")
        else:
            en = "; ".join(en_list[:LIST_SENSES - 1]) if en_list else w.jlpt_en
            m = QLabel(f"〔英〕{en}" if en else "（字典查不到）")
            m.setObjectName("meaning_en")
        m.setWordWrap(True)
        m.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(m)
        if w.also:  # 同寫法的其他常用讀音（「要」よう／かなめ）：斷詞選的讀音不一定是文中的那個
            also = QLabel("也讀：" + "　".join(f"{rd}（{mean}）" for rd, mean in w.also))
            also.setObjectName("also")
            also.setWordWrap(True)
            also.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lay.addWidget(also)
        tip = [f"{w.word} {w.jlpt}" if w.is_grammar else f"{w.word}【{w.reading}】{w.jlpt}"]
        if w.formation:
            tip.append("接續：" + w.formation)
        if w.zh:
            main, rare = w.zh[:len(w.zh) - w.rare_zh], w.zh[len(w.zh) - w.rare_zh:]
            tip.append("中：" + "；".join(main or rare) + (f"　〔少用〕{'；'.join(rare)}" if main and rare else ""))
        if w.en or w.jlpt_en:
            tip.append("英：" + (" / ".join(w.en) if w.en else w.jlpt_en))
        for rd, mean in w.also:
            tip.append(f"也讀【{rd}】：{mean}")
        for n in w.notes:  # 文法：JLPT_grammer 的說明（同一條規則對到好幾條時各自標名稱）
            head = f"〔{n['pattern']}〕" if len(w.notes) > 1 else ""
            if n.get("meaning"):
                tip.append(f"說明{head}：{n['meaning']}")
            if n.get("formation") and n["formation"] != w.formation:
                tip.append(f"接續{head}：{n['formation']}")
            if n.get("example"):
                tip.append(f"例句{head}：{n['example']}")
        tip.append("句子：" + w.sentence)
        self.setToolTip("\n".join(tip))

    def set_saved(self, saved: bool):
        self.btn.setText("✓ 已記錄" if saved else "＋ 記錄")
        self.btn.setObjectName("saved" if saved else "")
        self.btn.setEnabled(not saved)
        self.btn.style().unpolish(self.btn)
        self.btn.style().polish(self.btn)


class HighlightOverlay(QWidget):
    """反白模式：拖曳時在選到的字上畫半透明色塊（滑鼠事件直接穿透）。
    起點不在文字上時改畫虛線矩形（放開後會用框選模式辨識）。座標用實體像素傳進來。"""

    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._rects: list[tuple[int, int, int, int]] = []
        self._box = None

    def begin(self, mon: tuple[int, int, int, int]):
        """mon：螢幕的實體座標 (left, top, width, height)。"""
        l, t, w, h = mon
        a, b = phys_to_logical(l, t), phys_to_logical(l + w - 1, t + h - 1)
        self.setGeometry(QRect(a, b))
        self._rects, self._box = [], None
        self.show()
        self.update()

    def set_rects(self, rects):
        self._rects, self._box = list(rects), None
        self.update()

    def set_box(self, box):
        self._rects, self._box = [], box
        self.update()

    def end(self):
        self._rects, self._box = [], None
        self.hide()

    def _local(self, x0, y0, x1, y1) -> QRect:
        o = self.geometry().topLeft()
        return QRect(phys_to_logical(x0, y0) - o, phys_to_logical(x1, y1) - o)

    def paintEvent(self, _):
        p = QPainter(self)
        if self._box:
            p.setPen(QPen(QColor(255, 80, 60), BORDER, Qt.DashLine))
            p.drawRect(self._local(*self._box))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(60, 150, 255, 90))
        for r in self._rects:
            p.drawRect(self._local(*r))


class ResultPopup(QWidget):
    quit_requested = Signal()
    mode_changed = Signal(str)
    vocab_requested = Signal()  # 右鍵選單「開啟單字本」

    def __init__(self, cfg, vocab, attribution: str):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.cfg, self.vocab = cfg, vocab
        self.words: list[WordInfo] = []
        self._drag = None

        frame = QFrame(self)
        frame.setObjectName("popup")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        lay = self.frame_lay = QVBoxLayout(frame)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        head = QHBoxLayout()
        self.title = QLabel("查字")
        self.title.setObjectName("title")
        head.addWidget(self.title)
        head.addStretch(1)
        self.level = QComboBox()
        for label, lv in LEVEL_CHOICES:
            self.level.addItem(label, lv)
        self.level.setCurrentIndex(next(i for i, (_, lv) in enumerate(LEVEL_CHOICES) if lv == cfg.min_level)
                                   if cfg.min_level in dict(LEVEL_CHOICES).values() else 2)
        self.level.currentIndexChanged.connect(self._on_filter)
        self.level.setToolTip("只顯示這個級數以上的單字（級外的字一律顯示）")
        head.addWidget(self.level)
        self.proper = QCheckBox("專有名詞")
        self.proper.setChecked(cfg.show_proper)
        self.proper.toggled.connect(self._on_filter)
        head.addWidget(self.proper)
        self.text_btn = self._flat("文", "顯示／隱藏辨識出的原文", checkable=True, checked=cfg.show_text)
        self.text_btn.toggled.connect(self._on_toggle_text)
        head.addWidget(self.text_btn)
        self.mode_btn = QPushButton()
        self.mode_btn.setObjectName("mode")
        self.mode_btn.setCursor(Qt.PointingHandCursor)
        self.mode_btn.clicked.connect(self._toggle_mode)
        self._show_mode()
        head.addWidget(self.mode_btn)
        self.pin = self._flat("📌", "釘選：點別的地方時不要自動關閉", checkable=True)
        self.pin.toggled.connect(lambda on: self.pin.setToolTip(
            "已釘選：點別的地方時不會關閉（再按一次取消）" if on else "釘選：點別的地方時不要自動關閉"))
        head.addWidget(self.pin)
        close = self._flat("✕", "關閉（Esc）")
        close.clicked.connect(self.hide)
        head.addWidget(close)
        lay.addLayout(head)

        self.text = QLabel()
        self.text.setObjectName("text")
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.text)

        # 記錄原句：查完單字還是看不懂整句時，先存起來之後再查
        self.sent_row = QWidget()
        row = QHBoxLayout(self.sent_row)
        row.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("反白原文可只記錄其中一段")
        hint.setObjectName("status")
        row.addWidget(hint)
        row.addStretch(1)
        self.sent_btn = QPushButton("＋ 記錄原句")
        self.sent_btn.setFocusPolicy(Qt.NoFocus)  # 按按鈕時不要搶走焦點，否則原文的反白會被取消
        self.sent_btn.setCursor(Qt.PointingHandCursor)
        self.sent_btn.clicked.connect(self._save_sentence)
        row.addWidget(self.sent_btn)
        lay.addWidget(self.sent_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_host = QWidget()
        self.list_lay = QVBoxLayout(self.list_host)
        self.list_lay.setContentsMargins(0, 0, 0, 0)
        self.list_lay.setSpacing(4)
        self.scroll.setWidget(self.list_host)
        lay.addWidget(self.scroll, 1)

        foot = QHBoxLayout()
        self.status = QLabel()
        self.status.setObjectName("status")
        foot.addWidget(self.status)
        foot.addStretch(1)
        attr = QLabel("字典：Tomoshi、維基詞典、JMdict")
        attr.setObjectName("attr")
        attr.setToolTip(attribution)
        foot.addWidget(attr)
        lay.addLayout(foot)
        self.texts: list[str] = []
        self._apply_look()

    def _apply_look(self):
        """字級、寬度（設定視窗的「結果視窗 → 外觀」）。"""
        from .config import FONT_SCALES, POPUP_WIDTHS
        self.scale = FONT_SCALES[self.cfg.font_size]
        self.setStyleSheet(scaled_style(self.scale))
        self.setFixedWidth(POPUP_WIDTHS[self.cfg.popup_width])

    def sync_from_cfg(self):
        """設定視窗改了級數、專有名詞、原文、字級、寬度：同步標題列的控制項並重畫。"""
        for w in (self.level, self.proper, self.text_btn):
            w.blockSignals(True)
        idx = self.level.findData(self.cfg.min_level)
        self.level.setCurrentIndex(idx if idx >= 0 else 0)
        self.proper.setChecked(self.cfg.show_proper)
        self.text_btn.setChecked(self.cfg.show_text)
        for w in (self.level, self.proper, self.text_btn):
            w.blockSignals(False)
        self._apply_look()
        self.text.setVisible(self.cfg.show_text or not self.texts)
        self.sent_row.setVisible(self.cfg.show_text and bool(self.texts))
        if self.isVisible():
            self._render()

    def _show_mode(self):
        label, tip = MODES[self.cfg.select_mode]
        self.mode_btn.setText(label)
        self.mode_btn.setToolTip(tip + "\n按一下切換（反白 → 框選 → 漫畫）。")

    def set_mode(self, mode: str):
        self.cfg.select_mode = mode
        self.cfg.save()
        self._show_mode()

    def _toggle_mode(self):
        order = list(MODES)
        self.set_mode(order[(order.index(self.cfg.select_mode) + 1) % len(order)])
        self.mode_changed.emit(self.cfg.select_mode)

    def _flat(self, text, tip, checkable=False, checked=False):
        b = QPushButton(text)
        b.setObjectName("flat")
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        if checkable:
            b.setCheckable(True)
            b.setChecked(checked)
        return b

    # ---------------------------------------------------------------- 顯示
    def place_near(self, x0, y0, x1, y1):
        """放在框選範圍旁邊（右邊放不下就放左邊，再不行就疊在上面），不超出螢幕。
        只在視窗還沒開啟時呼叫；已開啟時維持原位（可能是使用者拖過的位置）。"""
        a, b = phys_to_logical(x0, y0), phys_to_logical(x1, y1)
        scr = QGuiApplication.screenAt(a) or QGuiApplication.primaryScreen()
        area = scr.availableGeometry()
        w, h = self.width(), self.height()
        gap = 12
        if b.x() + gap + w <= area.right():
            x = b.x() + gap
        elif a.x() - gap - w >= area.left():
            x = a.x() - gap - w
        else:
            x = min(max(area.left(), b.x() - w), area.right() - w)
        y = min(max(area.top(), a.y()), area.bottom() - h)
        self.move(x, y)

    def show_busy(self, msg="辨識中…"):
        self.title.setText("查字")
        self.text.setText(msg)
        self.text.setVisible(True)
        self.texts = []
        self.sent_row.setVisible(False)
        self._clear_list()
        self.status.setText("")
        self._fit()
        self.show()
        self.raise_()

    def show_error(self, msg):
        self.text.setText(msg)
        self.text.setVisible(True)
        self.sent_row.setVisible(False)
        self._fit()
        self.show()

    def _fit(self):
        """高度跟著內容：單字少時視窗變矮，多時最高 LIST_MAX_H 再捲動。

        清單高度直接把每一列的高度加起來，不用 list_host.sizeHint()：剛加進「已顯示」容器的列
        要等下一輪事件迴圈才會顯示，在那之前版面會把它們當成隱藏、高度算成 0。"""
        items = [self.list_lay.itemAt(i) for i in range(self.list_lay.count())]
        widgets = [it.widget() for it in items if it.widget() is not None]
        inner = self.width() - 2 * 10 - 10  # 外框左右邊距、捲軸
        want = sum(w.heightForWidth(inner) if w.hasHeightForWidth() else w.sizeHint().height() for w in widgets)
        want += self.list_lay.spacing() * max(0, len(widgets) - 1)
        self.scroll.setFixedHeight(min(want + 2, LIST_MAX_H))
        self.scroll.setVisible(bool(widgets))
        self.frame_lay.activate()  # 內層（外框裡）的版面要先更新，外層算出的高度才不會慢一步
        self.layout().activate()
        h = self.heightForWidth(self.width()) if self.hasHeightForWidth() else self.sizeHint().height()
        self.resize(self.width(), max(h, self.minimumSizeHint().height()))
        self._keep_on_screen()

    def _keep_on_screen(self):
        """變高之後如果超出螢幕下緣就往上移；其他時候不動位置。"""
        area = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        x = min(max(area.left(), self.x()), area.right() - self.width())
        y = min(max(area.top(), self.y()), area.bottom() - self.height())
        if (x, y) != (self.x(), self.y()):
            self.move(x, y)

    def show_result(self, texts: list[str], words: list[WordInfo], seconds: float, detail: str = ""):
        """seconds：放開滑鼠到出結果實際等了多久；detail：時間花在哪裡（顯示在括號裡）。"""
        self._detail = detail
        self.words = words
        self.texts = texts
        self.text.setText("\n".join(texts) if texts else "（沒有辨識到日文）")
        self.text.setVisible(self.cfg.show_text or not texts)
        self.sent_row.setVisible(self.cfg.show_text and bool(texts))
        self.sent_btn.setText("＋ 記錄原句")
        self._seconds = seconds
        self._render()
        self.show()
        self.raise_()

    def _clear_list(self):
        while self.list_lay.count():
            item = self.list_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _render(self):
        from .words import visible
        self._clear_list()
        shown = visible(self.words, self.cfg.min_level, self.cfg.show_proper)
        words = [w for w in shown if not w.is_grammar]
        points = [w for w in shown if w.is_grammar]
        for w in words:
            self.list_lay.addWidget(WordRow(w, self.vocab.has(w), self._save, self.scale))
        if points:
            # 屬於文法句型的詞（と思った 的「思う」）不另外列成單字，在這裡看得到它在句子裡的作用
            head = QLabel("文法")
            head.setObjectName("section")
            head.setToolTip("句子裡的文法句型；組成句型的詞（例如「と思う」的「思う」）不另外列成單字")
            self.list_lay.addWidget(head)
            for w in points:
                self.list_lay.addWidget(WordRow(w, self.vocab.has(w), self._save, self.scale))
        if not shown:
            empty = QLabel("（沒有符合級數條件的單字）" if self.words else "")
            empty.setObjectName("status")
            self.list_lay.addWidget(empty)
        self.list_lay.addStretch(1)
        hidden = len(self.words) - len(shown)
        self.title.setText(f"{len(words)} 個單字" + (f"・{len(points)} 文法" if points else "")
                           + (f"（隱藏 {hidden}）" if hidden else ""))
        self.title.setMinimumWidth(self.title.sizeHint().width())  # 標題列擠的時候不要截掉字
        detail = getattr(self, "_detail", "")
        self.status.setText(f"{getattr(self, '_seconds', 0):.2f} 秒" + (f"（{detail}）" if detail else ""))
        self.status.setToolTip("放開滑鼠到出結果實際等了多久。\n"
                               "等版面：按下 Alt 後偵測整個畫面的文字位置還沒做完；辨識：讀字＋查字典")
        self._fit()

    def _save(self, w: WordInfo, row: WordRow):
        self.vocab.add(w)
        row.set_saved(True)
        self.status.setText(f"已記錄「{w.word}」")

    def _save_sentence(self):
        sel = self.text.selectedText().replace("\u2029", "\n").strip()
        text = sel or "\n".join(self.texts)
        if not text:
            return
        if self.vocab.add_sentence(text, [w.word for w in self.words]):
            self.sent_btn.setText("✓ 已記錄原句")
            self.status.setText("已記錄" + ("反白的部分" if sel else "原句"))
        else:
            self.status.setText("這句已經記錄過了")

    def _on_filter(self):
        self.cfg.min_level = self.level.currentData()
        self.cfg.show_proper = self.proper.isChecked()
        self.cfg.save()
        self._render()

    def _on_toggle_text(self, on):
        self.cfg.show_text = on
        self.cfg.save()
        self.text.setVisible(on)
        self.sent_row.setVisible(on and bool(self.texts))
        self._fit()

    def contains_phys(self, x, y) -> bool:
        if not self.isVisible():
            return False
        l, t, r, b = native_rect(self)
        return l <= x < r and t <= y < b

    def contextMenuEvent(self, e):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("開啟單字本", self.vocab_requested.emit)
        menu.addAction("關閉視窗", self.hide)
        menu.addAction("結束螢幕查字", self.quit_requested.emit)
        menu.exec(e.globalPos())

    # ---------------------------------------------------------------- 拖曳移動
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None


def make_icon():
    from PySide6.QtGui import QFont, QIcon, QPixmap
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#d9534f"))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 12, 12)
    p.setPen(QColor("white"))
    f = QFont("Yu Gothic UI", 34)
    f.setBold(True)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignCenter, "字")
    p.end()
    return QIcon(pm)

