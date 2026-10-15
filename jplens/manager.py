"""單字本視窗：瀏覽、搜尋、篩選記錄的單字、文法、句子；標重點單字、已學會；編輯、刪除（可復原）；
匯出前先跳確認視窗（範圍、已學會的要不要匯出、存到哪裡）。

一般的系統視窗（可縮放、有工作列圖示），和無邊框置頂的結果視窗分開；開啟時才建立，關掉就釋放。
"""

import html
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QPainter, QPalette, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QScrollArea,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from .theme import C, QSS, badge_style, dark_title_bar, icon
from .vocab import key_of, one_line

VIEWS = {"words": "單字", "grammar": "文法", "sentences": "句子"}
LEVELS = ["N5", "N4", "N3", "N2", "N1", "級外"]
LEVEL_RANK = {"N1": 0, "N2": 1, "N3": 2, "N4": 3, "N5": 4}
POS_CATS = ["名詞", "動詞", "形容詞", "副詞", "慣用語"]
DATE_OPTS = [("all", "全部"), ("today", "今天"), ("7", "7 天內"), ("30", "30 天內")]
STATUS_OPTS = [("all", "全部"), ("learning", "學習中"), ("learned", "已學會")]
MARK_OPTS = [("all", "全部"), ("marked", "只看重點單字")]
SORTS = [("new", "最新記錄"), ("old", "最早記錄"), ("level", "級數（難到易）"), ("kana", "五十音")]
MAX_TAGS = 6  # 句子卡片最多列幾個單字標籤


# ---------------------------------------------------------------- 小工具
def is_grammar(rec: dict) -> bool:
    return rec.get("pos") == "文法"


def level_of(rec: dict) -> str:
    lv = rec.get("jlpt", "")
    return lv if lv in LEVEL_RANK else "級外"  # 專有名詞也算級外


def pos_cat(pos: str) -> str:
    if pos.startswith("名詞"):
        return "名詞"
    if "動詞" in pos:
        return "動詞"
    if "形容" in pos:
        return "形容詞"
    if pos == "副詞":
        return "副詞"
    if pos in ("慣用語", "複合詞"):
        return "慣用語"
    return pos


def _parse(saved_at: str):
    try:
        return datetime.fromisoformat(saved_at)
    except (TypeError, ValueError):
        return None


def days_ago(saved_at: str) -> int:
    t = _parse(saved_at)
    return (date.today() - t.date()).days if t else 10 ** 6


def date_text(saved_at: str) -> str:
    t = _parse(saved_at)
    if not t:
        return ""
    d = (date.today() - t.date()).days
    if d == 0:
        return "今天 " + t.strftime("%H:%M")
    if d == 1:
        return "昨天 " + t.strftime("%H:%M")
    return t.strftime("%m/%d" if t.year == date.today().year else "%Y/%m/%d")


def kana_key(s: str) -> str:
    """五十音排序用：片假名換成平假名。"""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def highlight(sentence: str, word: str) -> str:
    """原句裡標出這個字（HTML）。活用過的動詞找不到整個字時，去掉詞尾的假名再找（突く → 突いて 的「突」）。"""
    text = one_line(sentence)
    w = word.lstrip("〜～")
    for k in range(len(w), 0, -1):
        part = w[:k]
        if k < len(w) and not any("一" <= c <= "鿿" for c in part) and k < 2:
            break  # 只剩一個假名，太容易誤標
        i = text.find(part)
        if i >= 0:
            return (html.escape(text[:i]) + f'<span style="color:{C["accent"]}; font-weight:bold;">'
                    + html.escape(part) + "</span>" + html.escape(text[i + k:]))
    return html.escape(text)


class ElidedLabel(QLabel):
    """一行放不下時尾巴變成「…」。"""

    def __init__(self, text=""):
        super().__init__()
        self._full = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setToolTip(text)

    def sizeHint(self):
        return QSize(80, self.fontMetrics().height() + 2)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(self.palette().color(QPalette.WindowText))
        p.drawText(self.rect(), Qt.AlignLeft | Qt.AlignVCenter,
                   self.fontMetrics().elidedText(self._full, Qt.ElideRight, self.width()))


class ClickArea(QWidget):
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
        else:
            super().keyPressEvent(e)


def _label(text, name=None, wrap=False, select=False):
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    if select:
        lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lb


def _button(text, name=None, slot=None, tip=None):
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


def _sep():
    f = QFrame()
    f.setObjectName("sep")
    return f


def _clear(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().hide()  # 不然要等到真的刪掉之前，舊的元件還會疊在新的上面畫一次
            item.widget().deleteLater()
        elif item.layout():
            _clear(item.layout())


# ---------------------------------------------------------------- 卡片
class WordCard(QFrame):
    """一個單字（或文法）：收合時一行半，點一下展開看全部意思、原句、按鈕。"""

    def __init__(self, rec: dict, win: "ManagerWindow", expanded: bool):
        super().__init__()
        self.setObjectName("card")
        self.win = win
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(4, 0, 0, 0)
        self._lay.setSpacing(0)
        self.build(rec, expanded)

    def build(self, rec: dict, expanded: bool):
        _clear(self._lay)
        self.rec, self.expanded = rec, expanded
        self.setProperty("open", "true" if expanded else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        marked, learned = bool(rec.get("marked")), bool(rec.get("learned"))

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(0)
        star = QToolButton()
        star.setIcon(icon("star", C["accent"] if marked else C["dim"], C["accent"] if marked else "none"))
        star.setIconSize(QSize(20, 20))
        star.setFixedSize(40, 44)
        star.setCursor(Qt.PointingHandCursor)
        star.setToolTip("取消重點單字" if marked else "設為重點單字（匯出時標成 App 的重點）")
        star.clicked.connect(lambda: self.win.set_flag(self, "marked", not marked))
        head.addWidget(star, 0, Qt.AlignTop)

        area = ClickArea()
        area.clicked.connect(lambda: self.win.toggle_card(self))
        row = QHBoxLayout(area)
        row.setContentsMargins(4, 9, 14, 9)
        row.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(_label(rec.get("word", ""), "word"))
        reading = rec.get("reading", "")
        if reading and reading != rec.get("word"):
            top.addWidget(_label(reading, "reading"), 0, Qt.AlignBottom)
        lv = rec.get("jlpt", "")
        if lv:
            badge = QLabel(lv)
            badge.setStyleSheet(badge_style(lv))
            top.addWidget(badge, 0, Qt.AlignVCenter)
        top.addWidget(_label(rec.get("pos", ""), "dim"), 0, Qt.AlignVCenter)
        if learned:
            top.addWidget(_label("✓ 已學會", "learned"), 0, Qt.AlignVCenter)
        top.addStretch(1)
        col.addLayout(top)
        zh = rec.get("meaning_zh") or []
        summary = ElidedLabel("；".join(zh) if zh else "〔英〕" + "; ".join(rec.get("meaning_en") or []))
        summary.setObjectName("summary")
        col.addWidget(summary)
        row.addLayout(col, 1)
        row.addWidget(_label(date_text(rec.get("saved_at", "")), "dim"), 0, Qt.AlignVCenter)
        chev = QLabel()
        chev.setPixmap(icon("chev_up" if expanded else "chev_down", C["dim"]).pixmap(16, 16))
        row.addWidget(chev, 0, Qt.AlignVCenter)
        head.addWidget(area, 1)
        self._lay.addLayout(head)
        if expanded:
            self._lay.addWidget(self._detail(rec, learned))

    def _detail(self, rec: dict, learned: bool) -> QFrame:
        d = QFrame()
        d.setObjectName("cardDetail")
        v = QVBoxLayout(d)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(10)
        zh, en = rec.get("meaning_zh") or [], rec.get("meaning_en") or []
        senses = QVBoxLayout()
        senses.setSpacing(4)
        for i, m in enumerate(zh or en, 1):
            line = QHBoxLayout()
            num = _label(str(i), "dim")
            num.setFixedWidth(16)
            num.setAlignment(Qt.AlignRight | Qt.AlignTop)
            line.addWidget(num, 0, Qt.AlignTop)
            line.addWidget(_label(m, "meaning", wrap=True, select=True), 1)
            senses.addLayout(line)
        if zh and en:
            e = _label("英：" + " / ".join(en), "en", wrap=True, select=True)
            e.setContentsMargins(22, 0, 0, 0)
            senses.addWidget(e)
        v.addLayout(senses)
        if rec.get("sentence", "").strip():
            q = QFrame()
            q.setObjectName("quote")
            qv = QVBoxLayout(q)
            qv.setContentsMargins(10, 8, 10, 8)
            qv.setSpacing(4)
            qv.addWidget(_label("原句", "note"))
            s = _label(highlight(rec["sentence"], rec.get("word", "")), "jp", wrap=True, select=True)
            s.setTextFormat(Qt.RichText)
            qv.addWidget(s)
            v.addWidget(q)
        acts = QHBoxLayout()
        acts.setSpacing(8)
        t = _parse(rec.get("saved_at", ""))
        acts.addWidget(_label("記錄於 " + t.strftime("%Y-%m-%d %H:%M") if t else "", "dim"))
        acts.addStretch(1)
        acts.addWidget(_button("取消已學會" if learned else "標記已學會", "learnedBtn" if learned else None,
                               lambda: self.win.set_flag(self, "learned", not learned)))
        acts.addWidget(_button("編輯", slot=lambda: self.win.edit(self.rec)))
        acts.addWidget(_button("複製", slot=lambda: self.win.copy_word(self.rec)))
        acts.addWidget(_button("刪除", "danger", lambda: self.win.delete(self.rec)))
        v.addLayout(acts)
        return d


class SentenceCard(QFrame):
    def __init__(self, rec: dict, win: "ManagerWindow"):
        super().__init__()
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        v.addWidget(_label(one_line(rec.get("text", "")), "jp", wrap=True, select=True))
        row = QHBoxLayout()
        row.setSpacing(6)
        words = rec.get("words") or []
        for w in words[:MAX_TAGS]:
            row.addWidget(_label(w, "tag"), 0, Qt.AlignVCenter)
        if len(words) > MAX_TAGS:
            more = _label(f"＋{len(words) - MAX_TAGS}", "dim")
            more.setToolTip("、".join(words[MAX_TAGS:]))
            row.addWidget(more)
        row.addStretch(1)
        row.addWidget(_label(date_text(rec.get("saved_at", "")), "dim"))
        row.addWidget(_button("複製", slot=lambda: win.copy_sentence(rec)))
        row.addWidget(_button("刪除", "danger", lambda: win.delete_sentence(rec)))
        v.addLayout(row)


# ---------------------------------------------------------------- 篩選面板
class FilterPanel(QWidget):
    """按漏斗跳出的面板（Qt.Popup：點外面就關）。"""

    def __init__(self, win: "ManagerWindow"):
        super().__init__(win, Qt.Popup)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(QSS)
        self.win = win
        box = QFrame(self)
        box.setObjectName("popupPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)
        self.lay = QVBoxLayout(box)
        self.lay.setContentsMargins(14, 14, 14, 12)
        self.lay.setSpacing(12)
        self.setFixedWidth(360)

    def build(self):
        _clear(self.lay)
        f, cards = self.win.f, self.win.view != "sentences"
        if cards:
            self._section("級數（可複選）", [(lv, lv) for lv in LEVELS], f["levels"], multi=True)
            if self.win.view == "words":
                self._section("詞性（可複選）", [(p, p) for p in POS_CATS], f["pos"], multi=True)
        self._section("記錄時間", DATE_OPTS, f, key="date")
        if cards:
            self._section("狀態", STATUS_OPTS, f, key="status")
            self._section("重點單字", MARK_OPTS, f, key="marked")
        foot = QHBoxLayout()
        foot.addWidget(_button("清除條件", "plain", self._clear_all))
        foot.addStretch(1)
        foot.addWidget(_button("完成", "primary", self.close))
        self.lay.addWidget(_sep())
        self.lay.addLayout(foot)
        self.adjustSize()

    def _section(self, title, opts, target, multi=False, key=None):
        self.lay.addWidget(_label(title, "desc"))
        row = QHBoxLayout()
        row.setSpacing(6)
        group = None if multi else QButtonGroup(self)
        for val, text in opts:
            b = _button(text, "chip")
            b.setCheckable(True)
            b.setChecked(val in target if multi else target[key] == val)
            if multi:
                b.toggled.connect(lambda on, v=val: self._toggle(target, v, on))
            else:
                group.addButton(b)
                b.clicked.connect(lambda _=False, v=val: self._pick(key, v))
            row.addWidget(b)
        row.addStretch(1)
        self.lay.addLayout(row)

    def _toggle(self, target: set, val, on):
        (target.add if on else target.discard)(val)
        self.win.refresh()

    def _pick(self, key, val):
        self.win.f[key] = val
        self.win.refresh()

    def _clear_all(self):
        self.win.clear_filters()
        self.build()


# ---------------------------------------------------------------- 對話框
class ExportDialog(QDialog):
    """匯出確認視窗：範圍（全部／目前的篩選結果／只匯出重點單字）、已學會的要不要匯出、存到哪裡。"""

    def __init__(self, parent, kind: str, store, cfg, filtered=None):
        super().__init__(parent)
        self.kind, self.store, self.cfg = kind, store, cfg
        words = kind == "words"
        self.setWindowTitle("匯出單字 App 備份檔" if words else "匯出句子")
        self.setStyleSheet(QSS)
        self.setMinimumWidth(480)
        recs = store.records() if words else store.sentences()
        self.scopes = [("all", "全部記錄（單字＋文法）" if words else "全部句子", recs)]
        if filtered is not None:
            self.scopes.append(("filtered", "目前畫面的搜尋、篩選結果", filtered))
        if words:
            self.scopes.append(("marked", "只匯出重點單字", [r for r in recs if r.get("marked")]))
        self.unit = " 個" if words else " 句"

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(12)
        v.addWidget(_label("匯出範圍", "desc"))
        self.group = QButtonGroup(self)
        self.radios = []
        default = cfg.export_scope if words else "all"
        for i, (key, text, _) in enumerate(self.scopes):
            rb = QRadioButton()
            rb.setChecked(key == default)
            rb.toggled.connect(self._update)
            self.group.addButton(rb, i)
            self.radios.append((rb, text))
            v.addWidget(rb)
        if self.group.checkedId() < 0:
            self.radios[0][0].setChecked(True)
        self.learned = QCheckBox("已學會的單字也匯出")
        self.learned.setChecked(cfg.export_learned)
        self.learned.toggled.connect(self._update)
        self.hint = _label("", "summary", wrap=True)
        hint_box = QFrame()
        hint_box.setObjectName("hint")
        hb = QHBoxLayout(hint_box)
        hb.setContentsMargins(10, 8, 10, 8)
        star = QLabel()
        star.setPixmap(icon("star", C["accent"], C["accent"]).pixmap(16, 16))
        hb.addWidget(star, 0, Qt.AlignTop)
        hb.addWidget(self.hint, 1)
        if words:
            v.addWidget(self.learned)
            v.addWidget(hint_box)
        else:
            v.addWidget(_label("每行一句，依記錄順序編號（1. 2. 3.）。", "desc"))
        v.addWidget(_label("儲存位置", "desc"))
        prow = QHBoxLayout()
        stamp = datetime.now().strftime("%Y%m%d")
        name = f"vocabulary_{stamp}.txt" if words else f"sentences_{stamp}.txt"
        self.path = QLineEdit(str(store.path.parent / name))
        self.path.setObjectName("path")
        prow.addWidget(self.path, 1)
        prow.addWidget(_button("變更…", slot=self._browse))
        v.addLayout(prow)
        v.addSpacing(4)
        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(_button("取消", slot=self.reject))
        self.ok = _button("匯出", "primary", self._export)
        self.ok.setDefault(True)
        foot.addWidget(self.ok)
        v.addLayout(foot)
        self.count = 0
        self._update()

    def _filter(self, recs):
        if self.kind == "words" and not self.learned.isChecked():
            return [r for r in recs if not r.get("learned")]
        return recs

    def _selected(self):
        return self._filter(self.scopes[max(self.group.checkedId(), 0)][2])

    def _update(self):
        for (rb, text), (_, _, recs) in zip(self.radios, self.scopes):
            rb.setText(f"{text}　{len(self._filter(recs))}{self.unit}")
        sel = self._selected()
        marked = sum(1 for r in sel if r.get("marked"))
        self.hint.setText(f"其中重點單字 {marked} 個，匯出時寫成 isMarked: 1，在單字 App 裡會是標記的單字。")
        self.ok.setText(f"匯出 {len(sel)}{self.unit}")
        self.ok.setEnabled(bool(sel))

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(self, self.windowTitle(), self.path.text(), "文字檔 (*.txt)")
        if path:
            self.path.setText(path)

    def _export(self):
        path = Path(self.path.text().strip())
        if not path.name:
            return
        if path.exists() and QMessageBox.question(self, self.windowTitle(), f"{path.name} 已經存在，要覆蓋嗎？") \
                != QMessageBox.Yes:
            return
        sel = self._selected()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.kind == "words":
                self.count = self.store.export_app_backup(path, sel)
                self.marked = sum(1 for r in sel if r.get("marked"))
            else:
                self.count = self.store.export_sentences(path, sel)
        except OSError as e:
            QMessageBox.warning(self, self.windowTitle(), f"存檔失敗：{e}")
            return
        self.out = path
        self.accept()


class EditDialog(QDialog):
    def __init__(self, parent, rec: dict):
        super().__init__(parent)
        self.setWindowTitle(f"編輯「{rec.get('word', '')}」")
        self.setStyleSheet(QSS)
        self.setMinimumWidth(480)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(6)
        self.word = QLineEdit(rec.get("word", ""))
        self.reading = QLineEdit(rec.get("reading", ""))
        self.zh = QPlainTextEdit("\n".join(rec.get("meaning_zh") or []))
        self.zh.setFixedHeight(96)
        self.sentence = QPlainTextEdit(rec.get("sentence", ""))
        self.sentence.setFixedHeight(72)
        for title, w, tip in (("單字", self.word, ""), ("讀音", self.reading, ""),
                              ("中文意思", self.zh, "一行一個意思"), ("原句", self.sentence, "")):
            row = QHBoxLayout()
            row.addWidget(_label(title, "desc"))
            if tip:
                row.addWidget(_label(tip, "dim"))
            row.addStretch(1)
            v.addLayout(row)
            v.addWidget(w)
            v.addSpacing(4)
        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(_button("取消", slot=self.reject))
        ok = _button("儲存", "primary", self.accept)
        ok.setDefault(True)
        foot.addWidget(ok)
        v.addLayout(foot)

    def values(self) -> dict:
        return {"word": self.word.text().strip(), "reading": self.reading.text().strip(),
                "meaning_zh": [ln.strip() for ln in self.zh.toPlainText().splitlines() if ln.strip()],
                "sentence": self.sentence.toPlainText().strip()}


# ---------------------------------------------------------------- 視窗
class ManagerWindow(QWidget):
    open_settings = Signal(str)  # 要打開設定視窗的哪一頁

    def __init__(self, store, cfg, attribution: str = ""):
        from .ui import make_icon
        super().__init__(None, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setObjectName("window")
        self.setStyleSheet(QSS)
        self.setWindowIcon(make_icon())
        self.setWindowTitle("單字本 — 螢幕查字")
        self.resize(960, 700)
        self.setMinimumSize(560, 420)
        self.store, self.cfg = store, cfg
        self.view = "words"
        self.f = {"levels": set(), "pos": set(), "date": "all", "status": "all", "marked": "all"}
        self.expanded: str | None = None
        self.shown: list[dict] = []
        self._dirty = True
        self._undo = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(12, 10, 12, 8)
        bar.setSpacing(8)
        self.menu_btn = self._tool("menu", "選單：單字／文法／句子、匯出、設定", self.toggle_drawer)
        bar.addWidget(self.menu_btn)
        self.search = QLineEdit()
        self.search.setObjectName("search")
        self.search.addAction(icon("search", C["sub"]), QLineEdit.LeadingPosition)
        self.search.setClearButtonEnabled(True)
        self._search_timer = QTimer(self, singleShot=True, interval=150, timeout=self.refresh)
        self.search.textChanged.connect(lambda _: self._search_timer.start())
        bar.addWidget(self.search, 1)
        self.filter_btn = self._tool("filter", "篩選", self.open_filter)
        self.filter_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.addWidget(self.filter_btn)
        root.addLayout(bar)
        QShortcut(QKeySequence.Find, self, activated=lambda: (self.search.setFocus(), self.search.selectAll()))

        ctx = QHBoxLayout()
        ctx.setContentsMargins(16, 0, 16, 6)
        ctx.setSpacing(6)
        self.count = _label("", "sub")
        ctx.addWidget(self.count)
        self.chips = QHBoxLayout()
        self.chips.setSpacing(6)
        ctx.addLayout(self.chips)
        ctx.addStretch(1)
        ctx.addWidget(_label("排序", "sub"))
        self.sort = QComboBox()
        for key, text in SORTS:
            self.sort.addItem(text, key)
        self.sort.currentIndexChanged.connect(lambda _: self.refresh())
        ctx.addWidget(self.sort)
        root.addLayout(ctx)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        self.list = QVBoxLayout(host)
        self.list.setContentsMargins(16, 2, 12, 16)
        self.list.setSpacing(8)
        self.scroll.setWidget(host)
        root.addWidget(self.scroll, 1)

        self._build_drawer(attribution)
        self._build_toast()
        self.filter_panel = None
        store.listeners.append(self._on_store_changed)
        self.refresh()

    # ---------------------------------------------------------------- 元件
    def _tool(self, name, tip, slot):
        b = QToolButton()
        b.setIcon(icon(name, C["text"]))
        b.setIconSize(QSize(20, 20))
        b.setMinimumSize(44, 40)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(lambda: slot())  # clicked 會帶 checked＝False 進來，toggle_drawer 會當成「關閉」
        return b

    def _build_drawer(self, attribution):
        self.scrim = QWidget(self)
        self.scrim.setObjectName("scrim")
        self.scrim.setAttribute(Qt.WA_StyledBackground)
        self.scrim.mousePressEvent = lambda e: self.toggle_drawer(False)
        self.drawer = QFrame(self)
        self.drawer.setObjectName("drawer")
        self.drawer.setFixedWidth(250)
        v = QVBoxLayout(self.drawer)
        v.setContentsMargins(12, 14, 12, 12)
        v.setSpacing(2)
        v.addWidget(_label("記錄的內容", "dim"))
        self.nav = {}
        group = QButtonGroup(self)
        for key, text in VIEWS.items():
            b = _button("", "nav")
            b.setCheckable(True)
            b.setChecked(key == self.view)
            row = QHBoxLayout(b)
            row.setContentsMargins(10, 0, 10, 0)
            name, num = QLabel(text), _label("", "dim")
            for lb in (name, num):
                lb.setAttribute(Qt.WA_TransparentForMouseEvents)
            row.addWidget(name)
            row.addStretch(1)
            row.addWidget(num)
            b.setMinimumHeight(40)
            b.clicked.connect(lambda _=False, k=key: self.set_view(k))
            group.addButton(b)
            self.nav[key] = (b, name, num)
            v.addWidget(b)
        v.addSpacing(6)
        v.addWidget(_sep())
        v.addSpacing(6)
        v.addWidget(_label("匯出", "dim"))
        v.addWidget(_button("單字 → 單字 App 備份檔", "nav", lambda: self.export("words")))
        v.addWidget(_button("句子 → 文字檔", "nav", lambda: self.export("sentences")))
        v.addSpacing(6)
        v.addWidget(_sep())
        v.addSpacing(6)
        v.addWidget(_button("設定", "nav", lambda: self._settings("general")))
        v.addWidget(_button("關於與資料來源", "nav", lambda: self._settings("about")))
        v.addStretch(1)
        note = _label("字典資料：JMdict、Tomoshi、維基詞典（CC BY-SA 4.0）", "dim", wrap=True)
        if attribution:
            note.setToolTip(attribution)
        v.addWidget(note)
        self.scrim.hide()
        self.drawer.hide()

    def _build_toast(self):
        self.toast = QFrame(self)
        self.toast.setObjectName("toastBox")
        h = QHBoxLayout(self.toast)
        h.setContentsMargins(14, 6, 6, 6)
        h.setSpacing(10)
        self.toast_text = _label("", "toast")
        h.addWidget(self.toast_text)
        self.undo_btn = _button("復原", "link", self._do_undo)
        h.addWidget(self.undo_btn)
        self.toast.hide()
        self._toast_timer = QTimer(self, singleShot=True, interval=4000, timeout=self.toast.hide)

    def show_toast(self, text, undo=None):
        self._undo = undo
        self.toast_text.setText(text)
        self.undo_btn.setVisible(undo is not None)
        self.toast.adjustSize()
        self._place_toast()
        self.toast.show()
        self.toast.raise_()
        self._toast_timer.start()

    def _place_toast(self):
        self.toast.move((self.width() - self.toast.width()) // 2, self.height() - self.toast.height() - 18)

    def _do_undo(self):
        undo, self._undo = self._undo, None
        self.toast.hide()
        if undo:
            undo()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.scrim.setGeometry(self.rect())
        self.drawer.setGeometry(0, 0, self.drawer.width(), self.height())
        if self.toast.isVisible():
            self._place_toast()

    def showEvent(self, e):
        super().showEvent(e)
        dark_title_bar(self)
        if self._dirty:
            self.refresh()

    def closeEvent(self, e):
        if self._on_store_changed in self.store.listeners:
            self.store.listeners.remove(self._on_store_changed)
        super().closeEvent(e)

    # ---------------------------------------------------------------- 狀態
    def toggle_drawer(self, show=None):
        show = not self.drawer.isVisible() if show is None else show
        self.scrim.setVisible(show)
        self.drawer.setVisible(show)
        if show:
            self.scrim.raise_()
            self.drawer.raise_()

    def set_view(self, view):
        self.view = view
        self.expanded = None
        self.toggle_drawer(False)
        self.search.setPlaceholderText({"words": "搜尋單字、讀音、意思、原句", "grammar": "搜尋文法、意思、例句",
                                        "sentences": "搜尋句子、單字"}[view])
        self.refresh()
        self.scroll.verticalScrollBar().setValue(0)

    def open_filter(self):
        self.filter_panel = FilterPanel(self)
        self.filter_panel.build()
        b = self.filter_btn
        self.filter_panel.move(b.mapToGlobal(QPoint(b.width() - self.filter_panel.width(), b.height() + 4)))
        self.filter_panel.show()

    def clear_filters(self):
        self.f = {"levels": set(), "pos": set(), "date": "all", "status": "all", "marked": "all"}
        self.refresh()

    def _settings(self, tab):
        self.toggle_drawer(False)
        self.open_settings.emit(tab)

    def _on_store_changed(self):
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    # ---------------------------------------------------------------- 清單
    def _matches(self, r: dict, q: str) -> bool:
        f, cards = self.f, self.view != "sentences"
        d = days_ago(r.get("saved_at", ""))
        if (f["date"] == "today" and d != 0) or (f["date"] == "7" and d >= 7) or (f["date"] == "30" and d >= 30):
            return False
        if cards:
            if f["levels"] and level_of(r) not in f["levels"]:
                return False
            if self.view == "words" and f["pos"] and pos_cat(r.get("pos", "")) not in f["pos"]:
                return False
            if (f["status"] == "learned") != bool(r.get("learned")) and f["status"] != "all":
                return False
            if f["marked"] == "marked" and not r.get("marked"):
                return False
        if not q:
            return True
        if cards:
            hay = " ".join([r.get("word", ""), r.get("reading", ""), " ".join(r.get("meaning_zh") or []),
                            " ".join(r.get("meaning_en") or []), r.get("sentence", "")])
        else:
            hay = r.get("text", "") + " " + " ".join(r.get("words") or [])
        return q in hay.lower()

    def _active(self):
        """目前的篩選條件：[(標籤, 拿掉這個條件的函式)]。"""
        f, out = self.f, []
        cards = self.view != "sentences"
        if cards:
            for lv in sorted(f["levels"], key=LEVELS.index):
                out.append((lv, lambda v=lv: f["levels"].discard(v)))
            if self.view == "words":
                for p in sorted(f["pos"], key=POS_CATS.index):
                    out.append((p, lambda v=p: f["pos"].discard(v)))
        if f["date"] != "all":
            out.append((dict(DATE_OPTS)[f["date"]], lambda: f.__setitem__("date", "all")))
        if cards and f["status"] != "all":
            out.append((dict(STATUS_OPTS)[f["status"]], lambda: f.__setitem__("status", "all")))
        if cards and f["marked"] != "all":
            out.append(("重點單字", lambda: f.__setitem__("marked", "all")))
        return out

    def refresh(self):
        self._dirty = False
        recs = self.store.records()
        groups = {"words": [r for r in recs if not is_grammar(r)], "grammar": [r for r in recs if is_grammar(r)],
                  "sentences": self.store.sentences()}
        for key, (b, _, num) in self.nav.items():
            num.setText(str(len(groups[key])))
            b.setChecked(key == self.view)
        source = groups[self.view]
        q = self.search.text().strip().lower()
        shown = [r for r in source if self._matches(r, q)]
        sort = self.sort.currentData()
        if sort == "old":
            shown.sort(key=lambda r: r.get("saved_at", ""))
        elif sort == "level" and self.view != "sentences":
            shown.sort(key=lambda r: r.get("saved_at", ""), reverse=True)
            shown.sort(key=lambda r: LEVEL_RANK.get(r.get("jlpt", ""), 9))
        elif sort == "kana":
            shown.sort(key=lambda r: kana_key(r.get("reading") or r.get("word", "")) if self.view != "sentences"
                       else r.get("text", ""))
        else:
            shown.sort(key=lambda r: r.get("saved_at", ""), reverse=True)
        self.shown = shown

        active = self._active()
        _clear(self.chips)
        for text, remove in active:
            b = _button(f"{text}  ✕", "activeChip", tip=f"拿掉條件「{text}」")
            b.clicked.connect(lambda _=False, rm=remove: (rm(), self.refresh()))
            self.chips.addWidget(b)
        n = len(active)
        self.filter_btn.setText(str(n) if n else "")
        self.filter_btn.setObjectName("active" if n else "")
        self.filter_btn.setIcon(icon("filter", C["accent"] if n else C["text"]))
        self.filter_btn.style().unpolish(self.filter_btn)
        self.filter_btn.style().polish(self.filter_btn)
        self.count.setText(f"符合 {len(shown)} / {len(source)} 個" if (n or q) else f"{VIEWS[self.view]} {len(source)} 個")

        pos = self.scroll.verticalScrollBar().value()
        _clear(self.list)
        if self.view == "sentences":
            for r in shown:
                self.list.addWidget(SentenceCard(r, self))
        else:
            for r in shown:
                self.list.addWidget(WordCard(r, self, key_of(r) == self.expanded))
        if not shown:
            empty = QVBoxLayout()
            empty.setContentsMargins(0, 60, 0, 0)
            msg = _label(f"還沒有記錄的{VIEWS[self.view]}" if not source else f"沒有符合的{VIEWS[self.view]}", "sub")
            msg.setAlignment(Qt.AlignCenter)
            empty.addWidget(msg)
            if source:
                b = _button("清除搜尋與篩選", slot=lambda: (self.search.clear(), self.clear_filters()))
                empty.addWidget(b, 0, Qt.AlignHCenter)
            self.list.addLayout(empty)
        self.list.addStretch(1)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(pos))

    # ---------------------------------------------------------------- 動作
    def toggle_card(self, card: WordCard):
        key = key_of(card.rec)
        old = self.expanded
        self.expanded = None if old == key else key
        for i in range(self.list.count()):  # 只重畫展開／收合的兩張，不整個重建
            w = self.list.itemAt(i).widget()
            if isinstance(w, WordCard) and key_of(w.rec) in (old, key):
                w.build(w.rec, key_of(w.rec) == self.expanded)

    def _update(self, card: WordCard, **fields):
        """改一個單字並只重畫那張卡片（清單其他部分不動，篩選結果下次重新整理才更新）。"""
        self.store.listeners.remove(self._on_store_changed)
        try:
            new = self.store.update(key_of(card.rec), **fields)
        finally:
            self.store.listeners.append(self._on_store_changed)
        card.build(new, card.expanded)
        return new

    def set_flag(self, card: WordCard, field: str, value: bool):
        new = self._update(card, **{field: value})
        word = new.get("word", "")
        texts = {("marked", True): f"「{word}」設為重點單字", ("marked", False): f"「{word}」取消重點",
                 ("learned", True): f"「{word}」標記為已學會", ("learned", False): f"「{word}」改回學習中"}
        self.show_toast(texts[field, value])

    def edit(self, rec: dict):
        dlg = EditDialog(self, rec)
        if dlg.exec() != QDialog.Accepted:
            return
        vals = dlg.values()
        if not vals["word"]:
            QMessageBox.warning(self, "編輯", "單字不能是空的。")
            return
        was_open = self.expanded == key_of(rec)
        if was_open:  # 改了單字或讀音時，存檔觸發的重新整理要認得新的鍵
            self.expanded = key_of({**rec, **vals})
        try:
            new = self.store.update(key_of(rec), **vals)
        except ValueError as e:
            if was_open:
                self.expanded = key_of(rec)
            QMessageBox.warning(self, "編輯", str(e))
            return
        self.show_toast(f"已儲存「{new['word']}」")

    def _confirm_delete(self, what: str) -> bool:
        box = QMessageBox(QMessageBox.Warning, "刪除", f"確定要刪除{what}嗎？\n刪除後幾秒內還可以按「復原」。",
                          QMessageBox.Cancel, self)
        box.setStyleSheet(QSS)
        ok = box.addButton("刪除", QMessageBox.DestructiveRole)
        ok.setObjectName("danger")
        box.setDefaultButton(QMessageBox.Cancel)
        box.exec()
        return box.clickedButton() is ok

    def delete(self, rec: dict):
        if not self._confirm_delete(f"「{rec.get('word', '')}」"):
            return
        res = self.store.delete(key_of(rec))
        if res:
            idx, old = res
            self.show_toast(f"已刪除「{rec.get('word', '')}」", lambda: self.store.restore(idx, old))

    def delete_sentence(self, rec: dict):
        if not self._confirm_delete("這個句子"):
            return
        res =self.store.delete_sentence(rec.get("text", ""), rec.get("saved_at", ""))
        if res:
            idx, old = res
            self.show_toast("已刪除句子", lambda: self.store.restore_sentence(idx, old))

    def copy_word(self, rec: dict):
        rd = rec.get("reading", "")
        text = rec.get("word", "") + (f"【{rd}】" if rd and rd != rec.get("word") else "")
        zh = rec.get("meaning_zh") or rec.get("meaning_en") or []
        QGuiApplication.clipboard().setText(text + ("　" + "；".join(zh) if zh else ""))
        self.show_toast(f"已複製「{rec.get('word', '')}」")

    def copy_sentence(self, rec: dict):
        QGuiApplication.clipboard().setText(one_line(rec.get("text", "")))
        self.show_toast("已複製句子")

    def export(self, kind: str):
        self.toggle_drawer(False)
        same = (kind == "sentences") == (self.view == "sentences")
        dlg = ExportDialog(self, kind, self.store, self.cfg, list(self.shown) if same else None)
        if dlg.exec() == QDialog.Accepted:
            extra = f"（重點單字 {dlg.marked} 個）" if kind == "words" else ""
            self.show_toast(f"已匯出 {dlg.count}{dlg.unit}{extra}到 {dlg.out.name}")
