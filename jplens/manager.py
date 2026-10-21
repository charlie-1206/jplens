"""單字本視窗：瀏覽、搜尋、篩選記錄的單字、文法、句子；標重點單字、已學會；編輯、刪除（可復原）；
匯出前先跳確認視窗（範圍、已學會的要不要匯出、存到哪裡）。

一般的系統視窗（可縮放、有工作列圖示），和無邊框置頂的結果視窗分開；開啟時才建立，關掉就釋放。
"""

import html
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QEvent, QModelIndex, QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QGuiApplication, QKeySequence, QPainter, QPalette, QPen,
                           QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QListView, QMessageBox,
                               QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QStackedWidget,
                               QStyle, QStyledItemDelegate, QToolButton, QToolTip, QVBoxLayout, QWidget)

from . import speech
from .theme import C, LEVEL_BG, QSS, badge_style, dark_title_bar, icon
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
    """一行放不下時尾巴變成「…」。natural：放得下時照文字本身的寬度（展開卡片的單字、讀音、日期），
    放不下才縮（最窄 MIN_ELIDED）；不然這些標籤撐大卡片的最小寬度，窄視窗時版面被硬壓、算高度也用錯的寬度。"""
    MIN_ELIDED = 40

    def __init__(self, text="", name=None, natural=False):
        super().__init__(text if natural else "")  # natural 的大小照 QLabel 自己算（和一般標籤同高，收合、展開才對得齊）
        self._full, self._natural = text, natural
        if name:
            self.setObjectName(name)
        self.setSizePolicy(QSizePolicy.Preferred if natural else QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setToolTip(text)

    def sizeHint(self):
        return super().sizeHint() if self._natural else QSize(80, self.fontMetrics().height() + 2)

    def minimumSizeHint(self):
        h = self.sizeHint()
        return QSize(min(self.MIN_ELIDED, h.width()), h.height())

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


def _speak_button(tip: str, slot) -> QToolButton:
    """朗讀按鈕（喇叭）：展開的單字卡片上，唸單字、唸原句。"""
    b = QToolButton()
    b.setIcon(icon("speaker", C["dim"], size=18))
    b.setIconSize(QSize(18, 18))
    b.setFixedSize(30, 28)
    b.setCursor(Qt.PointingHandCursor)
    b.setToolTip(tip)
    b.clicked.connect(slot)
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
        top.addWidget(ElidedLabel(rec.get("word", ""), "word", natural=True))
        reading = rec.get("reading", "")
        if reading and reading != rec.get("word"):
            # 也讀（記錄時結果視窗的「也讀」）直接接在讀音後面：よう；かなめ
            top.addWidget(ElidedLabel(self.win.store.reading_with_also(rec), "reading", natural=True), 0, Qt.AlignBottom)
        if expanded and rec.get("pos") != "文法":  # 文法句型（〜であれ／〜であろうと）不是一個能唸的詞
            top.addWidget(_speak_button(f"朗讀「{speech.word_text(rec)}」",
                                        lambda: self.win.speak(speech.word_text(self.rec))), 0, Qt.AlignVCenter)
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
        summary = ElidedLabel("；".join((["；".join(zh) if zh else "〔英〕" + "; ".join(rec.get("meaning_en") or [])])
                                       + self.win.store.also_meanings(rec)))
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
        for m in self.win.store.also_meanings(rec):
            a = _label(m, "meaning", wrap=True, select=True)
            a.setContentsMargins(22, 0, 0, 0)
            senses.addWidget(a)
        v.addLayout(senses)
        if rec.get("sentence", "").strip():
            q = QFrame()
            q.setObjectName("quote")
            qv = QVBoxLayout(q)
            qv.setContentsMargins(10, 8, 10, 8)
            qv.setSpacing(4)
            head = QHBoxLayout()
            head.setSpacing(4)
            head.addWidget(_label("原句", "note"))
            head.addWidget(_speak_button("朗讀原句", lambda: self.win.speak(self.rec.get("sentence", ""))))
            head.addStretch(1)
            qv.addLayout(head)
            s = _label(highlight(rec["sentence"], rec.get("word", "")), "jp", wrap=True, select=True)
            s.setTextFormat(Qt.RichText)
            qv.addWidget(s)
            v.addWidget(q)
        acts = QHBoxLayout()
        acts.setSpacing(8)
        t = _parse(rec.get("saved_at", ""))
        acts.addWidget(ElidedLabel("記錄於 " + t.strftime("%Y-%m-%d %H:%M") if t else "", "dim", natural=True), 1)
        acts.addStretch(1)
        acts.addWidget(_button("取消已學會" if learned else "標記已學會", "learnedBtn" if learned else None,
                               lambda: self.win.set_flag(self, "learned", not learned)))
        acts.addWidget(_button("編輯", slot=lambda: self.win.edit(self.rec)))
        acts.addWidget(_button("複製", slot=lambda: self.win.copy_word(self.rec)))
        acts.addWidget(_button("刪除", "danger", lambda: self.win.delete(self.rec)))
        v.addLayout(acts)
        return d


# ---------------------------------------------------------------- 單字清單（虛擬化）
# 收合的卡片用畫的（CardDelegate.paint），只畫看得到的那幾張；展開的那一張才放真的 WordCard（setIndexWidget），
# 按鈕、選取文字都沿用元件。單字再多，打開、搜尋、捲動的時間都一樣（以前每張卡片都是元件，1,000 個字重新整理要 2 秒）。
# 尺寸、位置照 WordCard 收合時的版面（QSS 的字級、邊框 1、星號 40×44、內距 4／9／14），改樣式時兩邊一起改。
LIST_MARGIN = (16, 2, 12, 16)  # 清單左、上、右、下的留白（和句子頁的 self.list 一樣）
CARD_GAP = 8


class CardModel(QAbstractListModel):
    """目前畫面上的單字紀錄（搜尋、篩選、排序後）。"""

    def __init__(self):
        super().__init__()
        self.recs: list[dict] = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.recs)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.UserRole and index.isValid():
            return self.recs[index.row()]
        return None

    def set_records(self, recs: list[dict]):
        self.beginResetModel()
        self.recs = recs
        self.endResetModel()

    def set_record(self, row: int, rec: dict):
        self.recs[row] = rec
        i = self.index(row)
        self.dataChanged.emit(i, i)

    def row_of(self, key: str) -> int:
        return next((i for i, r in enumerate(self.recs) if key_of(r) == key), -1)


class CardDelegate(QStyledItemDelegate):
    """畫收合的單字卡片；點星號切換重點、點其他地方展開。展開的那一列高度問 ManagerWindow（真的 WordCard 有多高）。"""

    def __init__(self, win: "ManagerWindow"):
        super().__init__(win)
        self.win = win
        # 字型、標籤的內距直接量套了樣式表（QSS）的 QLabel，和元件版的卡片一模一樣，改 QSS 也會跟著變
        probe = QWidget()
        probe.setStyleSheet(QSS)

        def label(name=None, style=None, text="N1"):
            lb = QLabel(text, probe)
            if name:
                lb.setObjectName(name)
            if style:
                lb.setStyleSheet(style)
            lb.ensurePolished()
            s = lb.sizeHint()
            return QFont(lb.font()), s.width() - QFontMetrics(lb.font()).horizontalAdvance(text), s.height()
        self.f_word, _, self.top_h = label("word", text="字")
        self.f_reading, _, self.reading_h = label("reading", text="よ")
        self.f_summary = label("summary")[0]
        self.f_dim, _, self.dim_h = label("dim")
        self.f_badge, self.badge_pad, self.badge_h = label(style=badge_style("N1"))
        self.f_learned, self.learned_pad, self.learned_h = label("learned", text="✓ 已學會")
        probe.deleteLater()
        self.sum_h = QFontMetrics(self.f_summary).height() + 2  # ElidedLabel.sizeHint
        self.card_h = 2 + max(44, 9 + self.top_h + 2 + self.sum_h + 9)  # 上下邊框各 1
        self._icons = {}

    def _icon(self, name, color, fill="none"):
        key = (name, color, fill)
        if key not in self._icons:
            self._icons[key] = icon(name, color, fill)
        return self._icons[key]

    # ---- 版面
    def _card_rect(self, rect: QRect, row: int) -> QRect:
        top = LIST_MARGIN[1] if row == 0 else 0
        return QRect(rect.left() + LIST_MARGIN[0], rect.top() + top,
                     rect.width() - LIST_MARGIN[0] - LIST_MARGIN[2], self.card_h)

    def row_extra(self, row: int) -> int:
        """這一列除了卡片以外的高度：卡片間距、第一列上面、最後一列下面的留白。"""
        return (LIST_MARGIN[1] if row == 0 else 0) + (LIST_MARGIN[3] if row == self.win.model.rowCount() - 1 else CARD_GAP)

    def _star_rect(self, card: QRect) -> QRect:
        return QRect(card.left() + 1 + 4, card.top() + 1, 40, 44)

    def sizeHint(self, option, index):
        row = index.row()
        h = self.win.open_height(row)
        return QSize(option.rect.width(), (h if h is not None else self.card_h) + self.row_extra(row))

    def summary_text(self, rec: dict) -> str:
        zh = rec.get("meaning_zh") or []
        return "；".join((["；".join(zh) if zh else "〔英〕" + "; ".join(rec.get("meaning_en") or [])])
                        + self.win.store.also_meanings(rec))

    def _layout(self, rec: dict, card: QRect):
        """收合卡片裡各部分的位置與內容：[(種類, 矩形, 文字)]，paint 和 tooltip 共用。"""
        items = []
        x0 = card.left() + 1 + 4 + 40 + 4
        right = card.right() - 1 - 14
        top = card.top() + 1 + 9
        chev = QRect(right - 16 + 1, card.top() + (card.height() - 16) // 2, 16, 16)
        items.append(("chev", chev, ""))
        date = date_text(rec.get("saved_at", ""))
        dw, dh = QFontMetrics(self.f_dim).horizontalAdvance(date), self.dim_h
        drect = QRect(chev.left() - 12 - dw, card.top() + (card.height() - dh) // 2, dw, dh)
        items.append(("date", drect, date))
        col_right = drect.left() - 12 if date else chev.left() - 12
        # 第一行：單字、讀音、級數、詞性、已學會（間距 8，放不下的就不畫，和元件版一樣靠右邊被切掉）
        x = x0
        word = rec.get("word", "")
        ww = QFontMetrics(self.f_word).horizontalAdvance(word)
        items.append(("word", QRect(x, top, ww, self.top_h), word))
        x += ww + 8
        reading = rec.get("reading", "")
        if reading and reading != word:
            text = self.win.store.reading_with_also(rec)
            w, h = QFontMetrics(self.f_reading).horizontalAdvance(text), self.reading_h
            items.append(("reading", QRect(x, top + self.top_h - h, w, h), text))  # 元件版是 AlignBottom
            x += w + 8
        lv = rec.get("jlpt", "")
        if lv:
            w, h = QFontMetrics(self.f_badge).horizontalAdvance(lv) + self.badge_pad, self.badge_h
            items.append(("badge", QRect(x, top + (self.top_h - h) // 2, w, h), lv))
            x += w + 8
        pos = rec.get("pos", "")
        if pos:
            w, h = QFontMetrics(self.f_dim).horizontalAdvance(pos), self.dim_h
            items.append(("pos", QRect(x, top + (self.top_h - h) // 2, w, h), pos))
            x += w + 8
        if rec.get("learned"):
            text = "✓ 已學會"
            w, h = QFontMetrics(self.f_learned).horizontalAdvance(text) + self.learned_pad, self.learned_h
            items.append(("learned", QRect(x, top + (self.top_h - h) // 2, w, h), text))
        items.append(("summary", QRect(x0, top + self.top_h + 2, max(col_right - x0, 0), self.sum_h), self.summary_text(rec)))
        return items, col_right

    # ---- 畫
    def paint(self, p: QPainter, option, index):
        rec = index.data(Qt.UserRole)
        if rec is None or self.win.open_height(index.row()) is not None:
            return  # 展開的那一列由真的 WordCard 蓋住
        card = self._card_rect(option.rect, index.row())
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        focus = bool(option.state & QStyle.State_HasFocus)
        p.setPen(QPen(QColor(C["card_open"] if focus else C["card_bd"]), 1))
        p.setBrush(QColor(C["card"]))
        p.drawRoundedRect(QRectF(card).adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)
        marked = bool(rec.get("marked"))
        star = self._star_rect(card)
        self._icon("star", C["accent"] if marked else C["dim"], C["accent"] if marked else "none").paint(
            p, QRect(star.center().x() - 9, star.center().y() - 9, 20, 20))
        items, col_right = self._layout(rec, card)
        p.setClipRect(QRect(card.left(), card.top(), col_right - card.left(), card.height()))
        for kind, r, text in items:
            if kind in ("chev", "date"):
                continue
            if kind == "word":
                p.setFont(self.f_word)
                p.setPen(QColor("#ffffff"))
                p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, text)
            elif kind == "reading":
                p.setFont(self.f_reading)
                p.setPen(QColor(C["reading"]))
                p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, text)
            elif kind == "badge":
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(LEVEL_BG.get(text, "#61656e")))
                p.drawRoundedRect(QRectF(r), 3, 3)
                p.setFont(self.f_badge)
                p.setPen(QColor("white"))
                p.drawText(r, Qt.AlignCenter, text)
            elif kind == "pos":
                p.setFont(self.f_dim)
                p.setPen(QColor(C["dim"]))
                p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, text)
            elif kind == "learned":
                p.setPen(QPen(QColor(C["ok_bd"]), 1))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(QRectF(r).adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)
                p.setFont(self.f_learned)
                p.setPen(QColor(C["ok"]))
                p.drawText(r, Qt.AlignCenter, text)
            elif kind == "summary":
                p.setFont(self.f_summary)
                p.setPen(QColor("#d6d9de"))
                p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter,
                           QFontMetrics(self.f_summary).elidedText(text, Qt.ElideRight, r.width()))
        p.setClipping(False)
        for kind, r, text in items:
            if kind == "date":
                p.setFont(self.f_dim)
                p.setPen(QColor(C["dim"]))
                p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, text)
            elif kind == "chev":
                self._icon("chev_down", C["dim"]).paint(p, r)
        p.restore()

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)  # 預設會照「文字欄」往右縮幾個像素，展開的卡片就和其他卡片對不齊

    # ---- 滑鼠
    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            return True  # 不要選取、拖曳
        if event.type() != QEvent.MouseButtonRelease or event.button() != Qt.LeftButton:
            return False
        card = self._card_rect(option.rect, index.row())
        pos = event.position().toPoint()
        if card.contains(pos):
            # 按下時 return True 擋掉了 Qt 自己設目前列：這裡補上，之後按 Enter／空白鍵才是動這一張（不是之前方向鍵停的那張）
            self.win.cards.setCurrentIndex(index)
        if self._star_rect(card).contains(pos):
            rec = index.data(Qt.UserRole)
            self.win.set_flag_row(index.row(), "marked", not rec.get("marked"))
        elif card.contains(pos):
            self.win.toggle_row(index.row())
        return True

    def helpEvent(self, event, view, option, index):
        rec = index.data(Qt.UserRole)
        if rec is None or event.type() != QEvent.ToolTip:
            return super().helpEvent(event, view, option, index)
        card = self._card_rect(option.rect, index.row())
        pos = event.pos()
        tip = ""
        if self._star_rect(card).contains(pos):
            tip = "取消重點單字" if rec.get("marked") else "設為重點單字（匯出時標成 App 的重點）"
        else:
            for kind, r, text in self._layout(rec, card)[0]:
                if kind == "summary" and r.contains(pos):
                    tip = text
        if tip:
            QToolTip.showText(event.globalPos(), tip, view)
        else:
            QToolTip.hideText()
        return True


class CardView(QListView):
    """單字清單：一次只畫看得到的卡片。Enter／空白鍵展開目前那張。"""
    toggled = Signal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName("cards")
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setFrameShape(QFrame.NoFrame)
        self.setMouseTracking(True)
        self.viewport().setAutoFillBackground(False)
        self.resized = None  # ManagerWindow 設：寬度變了要重算展開那張的高度

    def updateGeometries(self):
        super().updateGeometries()
        self.verticalScrollBar().setSingleStep(24)  # ScrollPerPixel 預設一格只捲幾個像素

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.resized and e.size().width() != e.oldSize().width():
            self.resized()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        self.viewport().setCursor(Qt.PointingHandCursor if self.indexAt(e.position().toPoint()).isValid()
                                  else Qt.ArrowCursor)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space) and self.currentIndex().isValid():
            self.toggled.emit(self.currentIndex().row())
        else:
            super().keyPressEvent(e)


class SentenceCard(QFrame):
    def __init__(self, rec: dict, win: "ManagerWindow"):
        super().__init__()
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        v.addWidget(_label(one_line(rec.get("text", "")), "jp", wrap=True, select=True))
        if rec.get("translation"):
            v.addWidget(_label(rec["translation"], "summary", wrap=True, select=True))
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
        # 講解（Gemini，ai.py）：第一次按才呼叫，結果存進句子紀錄，之後只是展開／收合
        key = (rec.get("text", ""), rec.get("saved_at", ""))
        has, busy, opened = bool(rec.get("explanation")), key in win.explaining, key in win.explain_open
        label = "講解中…" if busy else ("收合講解" if has and opened else "講解" if not has else "展開講解")
        btn = _button(label, slot=lambda: win.explain_sentence(rec),
                      tip="用 Gemini 講解這句：翻譯、拆解詞與文法、語氣。模型在設定 → 翻譯與 AI 選。")
        btn.setEnabled(not busy)
        row.addWidget(btn)
        row.addWidget(_button("複製", slot=lambda: win.copy_sentence(rec)))
        row.addWidget(_button("刪除", "danger", lambda: win.delete_sentence(rec)))
        v.addLayout(row)
        err = win.explain_errors.get(key)
        if err:
            v.addWidget(_label(err, "note", wrap=True))
        if has and opened:
            box = QFrame()
            box.setObjectName("quote")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(12, 10, 12, 10)
            md = QLabel()
            md.setTextFormat(Qt.MarkdownText)
            md.setText(rec["explanation"])
            md.setWordWrap(True)
            md.setTextInteractionFlags(Qt.TextSelectableByMouse)
            bl.addWidget(md)
            foot = QHBoxLayout()
            from .ai import EXPLAIN_MODELS
            model = rec.get("explain_model", "")
            foot.addWidget(_label(EXPLAIN_MODELS.get(model, model).split("（")[0], "dim"))
            foot.addStretch(1)
            foot.addWidget(_button("重新講解", slot=lambda: win.explain_sentence(rec, again=True),
                                   tip="用目前設定的模型再講解一次（會蓋掉這份講解）"))
            bl.addLayout(foot)
            v.addWidget(box)


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
        self.hint.setText(f"其中重點單字 {marked} 個。")
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


class AddWordDialog(QDialog):
    """手動新增單字：打單字後按「查字典」（或 Enter）自動填讀音、詞性、級數、意思，都可以再改。"""

    def __init__(self, parent, dictionary):
        super().__init__(parent)
        self.dictionary = dictionary
        self.setWindowTitle("新增單字")
        self.setStyleSheet(QSS)
        self.setMinimumWidth(480)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(6)
        v.addWidget(_label("單字", "desc"))
        row = QHBoxLayout()
        self.word = QLineEdit()
        self.word.setPlaceholderText("例：乗っかる（動詞打辭書形或活用形都可以）")
        self.word.returnPressed.connect(self.lookup)
        row.addWidget(self.word, 1)
        row.addWidget(_button("查字典", slot=self.lookup))
        v.addLayout(row)
        self.status = _label("", "dim", wrap=True)
        v.addWidget(self.status)
        grid = QHBoxLayout()
        self.reading, self.pos = QLineEdit(), QLineEdit()
        self.level = QComboBox()
        for lv in ["級外", "N5", "N4", "N3", "N2", "N1"]:
            self.level.addItem(lv)
        for title, w in (("讀音", self.reading), ("詞性", self.pos), ("級數", self.level)):
            col = QVBoxLayout()
            col.addWidget(_label(title, "desc"))
            col.addWidget(w)
            grid.addLayout(col, 2 if w is self.reading else 1)
        v.addLayout(grid)
        row = QHBoxLayout()
        row.addWidget(_label("中文意思", "desc"))
        row.addWidget(_label("一行一個意思", "dim"))
        row.addStretch(1)
        v.addLayout(row)
        self.zh = QPlainTextEdit()
        self.zh.setFixedHeight(96)
        v.addWidget(self.zh)
        v.addWidget(_label("原句（可以空白）", "desc"))
        self.sentence = QPlainTextEdit()
        self.sentence.setFixedHeight(60)
        v.addWidget(self.sentence)
        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(_button("取消", slot=self.reject))
        self.ok = _button("新增", "primary", self.accept)
        foot.addWidget(self.ok)
        v.addLayout(foot)
        self._en: list[str] = []
        self._also: list = []

    def lookup(self):
        text = self.word.text().strip()
        if not text or self.dictionary is None:
            return
        from .words import extract
        infos = [w for w in extract([text], self.dictionary) if not w.is_grammar]
        info = next((w for w in infos if w.word == text), infos[0] if infos else None)
        if info is None:
            self.status.setText("字典裡找不到，請自己填讀音和意思。")
            return
        if info.word != text:
            self.word.setText(info.word)  # 活用形換成辭書形（乗っかった → 乗っかる）
        self.reading.setText(info.reading)
        self.pos.setText(info.pos)
        self.level.setCurrentText(info.jlpt if info.jlpt in ("N5", "N4", "N3", "N2", "N1") else "級外")
        self.zh.setPlainText("\n".join(info.zh))
        self._en = info.en or ([info.jlpt_en] if info.jlpt_en else [])
        self._also = [[rd, mean] for rd, mean in info.also]
        more = f"（同一段文字還有：{'、'.join(w.word for w in infos if w is not info)}）" if len(infos) > 1 else ""
        self.status.setText(f"已從字典填入「{info.word}」{more}")

    def record(self) -> dict:
        lv = self.level.currentText()
        return {"word": self.word.text().strip(), "reading": self.reading.text().strip() or self.word.text().strip(),
                "pos": self.pos.text().strip(), "jlpt": "" if lv == "級外" else lv,
                "meaning_zh": [ln.strip() for ln in self.zh.toPlainText().splitlines() if ln.strip()],
                "meaning_en": self._en, "sentence": self.sentence.toPlainText().strip(), "also": self._also}


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
        self.explaining: set = set()     # 正在講解的句子 (text, saved_at)
        self.explain_open: set = set()   # 展開講解的句子
        self.explain_errors: dict = {}   # 講解失敗的訊息
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
        self.menu_btn = self._tool("menu", "選單：單字／文法／句子、單字測驗、匯入匯出、設定", self.toggle_drawer)
        bar.addWidget(self.menu_btn)
        self.page_title = _label("單字測驗", "pageTitle")  # 測驗頁沒有搜尋、篩選，改顯示標題
        self.page_title.hide()
        bar.addWidget(self.page_title)
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

        self.ctx_box = QWidget()
        ctx = QHBoxLayout(self.ctx_box)
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
        root.addWidget(self.ctx_box)

        # 單字、文法：虛擬化清單；句子（每張都有按鈕、講解，數量也少）和「沒有符合的」：一般的捲動區
        self.model = CardModel()
        self.cards = CardView()
        self.cards.setModel(self.model)
        self.delegate = CardDelegate(self)
        self.cards.setItemDelegate(self.delegate)
        self.cards.toggled.connect(self.toggle_row)
        self.cards.resized = self._open_resized
        self._open_box: QWidget | None = None  # 展開的那張：外框（留白）＋ WordCard
        self._open_row = -1  # 它在第幾列（排版時每一列都會問高度，不能每次從頭找：展開第 1,500 列時重新整理要 2 秒）
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        self.list = QVBoxLayout(host)
        self.list.setContentsMargins(*LIST_MARGIN)
        self.list.setSpacing(CARD_GAP)
        self.scroll.setWidget(host)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.cards)
        self.stack.addWidget(self.scroll)
        from .quiz_ui import QuizPage
        self.quiz_page = QuizPage(self)
        self.stack.addWidget(self.quiz_page)
        root.addWidget(self.stack, 1)

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
        outer = QVBoxLayout(self.drawer)
        outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()  # 項目多，視窗矮的時候要能捲
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget()
        sc.setWidget(host)
        outer.addWidget(sc)
        v = QVBoxLayout(host)
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
        v.addWidget(_label("練習", "dim"))
        self.quiz_btn = _button("單字測驗", "nav", lambda: self.set_view("quiz"))
        self.quiz_btn.setCheckable(True)
        self.quiz_btn.setMinimumHeight(40)
        group.addButton(self.quiz_btn)
        v.addWidget(self.quiz_btn)
        v.addSpacing(6)
        v.addWidget(_sep())
        v.addSpacing(6)
        v.addWidget(_label("單字本", "dim"))
        v.addWidget(_button("新增單字…", "nav", self.add_word))
        v.addWidget(_button("匯入單字…", "nav", self.import_words))
        v.addWidget(_button("匯出：單字 App 備份檔", "nav", lambda: self.export("words")))
        v.addWidget(_button("匯出：完整備份（可匯回）", "nav", self.export_full))
        v.addWidget(_button("匯出：句子 → 文字檔", "nav", lambda: self.export("sentences")))
        clear = _button("清空單字本…", "nav", self.clear_words)
        clear.setStyleSheet(f"color: {C['danger']};")
        v.addWidget(clear)
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
        speech.prepare()  # 語音引擎要等事件迴圈跑過才就緒：先建好，第一次按朗讀才不用等
        if self._dirty:
            self.refresh()

    def closeEvent(self, e):
        self.quiz_page.leave()
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
        if self.view == "quiz" and view != "quiz":
            self.quiz_page.leave()
        self.view = view
        self.expanded = None
        self.toggle_drawer(False)
        quiz = view == "quiz"
        for w in (self.search, self.filter_btn, self.ctx_box):
            w.setVisible(not quiz)
        self.page_title.setVisible(quiz)
        if quiz:
            self.refresh()
            self.stack.setCurrentWidget(self.quiz_page)
            self.quiz_page.enter()
            self.quiz_page.setFocus()
            return
        self.search.setPlaceholderText({"words": "搜尋單字、讀音、意思、原句", "grammar": "搜尋文法、意思、例句",
                                        "sentences": "搜尋句子、單字"}[view])
        self.refresh()
        self.scroll.verticalScrollBar().setValue(0)
        self.cards.verticalScrollBar().setValue(0)

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
                            " ".join(rd for rd, _ in self.store.also_of(r)),
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
        if self.view == "quiz":
            self.quiz_btn.setChecked(True)
            self.quiz_page.store_changed()
            return
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

        if self.view != "sentences" and shown:
            bar = self.cards.verticalScrollBar()
            pos = bar.value()
            self._open_box, self._open_row = None, -1  # 重設模型時 Qt 會刪掉放在清單裡的元件
            self.model.set_records(shown)
            self.stack.setCurrentWidget(self.cards)
            row = self.model.row_of(self.expanded) if self.expanded else -1
            if row >= 0:
                self._open(row)
            self.cards.doItemsLayout()
            bar.setValue(pos)
            _clear(self.list)
            return
        self.model.set_records([])
        self._open_box, self._open_row = None, -1
        self.stack.setCurrentWidget(self.scroll)
        pos = self.scroll.verticalScrollBar().value()
        _clear(self.list)
        for r in shown:
            self.list.addWidget(SentenceCard(r, self))
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
    def open_height(self, row: int) -> int | None:
        """展開的那一列卡片有多高（不含間距）；其他列回傳 None（收合，用畫的）。"""
        if self._open_box is None or row != self._open_row:
            return None
        card = self._open_box.layout().itemAt(0).widget()
        w = self.cards.viewport().width() - LIST_MARGIN[0] - LIST_MARGIN[2]
        return card.heightForWidth(w) if card.hasHeightForWidth() else card.sizeHint().height()

    def _open(self, row: int):
        """把第 row 列換成真的 WordCard（展開）。"""
        rec = self.model.recs[row]
        box = QWidget()
        box.setAttribute(Qt.WA_StyledBackground, False)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(LIST_MARGIN[0], LIST_MARGIN[1] if row == 0 else 0, LIST_MARGIN[2],
                               self.delegate.row_extra(row) - (LIST_MARGIN[1] if row == 0 else 0))
        lay.addWidget(WordCard(rec, self, True))
        lay.addStretch(1)
        self._open_box, self._open_row = box, row
        self.cards.setIndexWidget(self.model.index(row), box)
        self.delegate.sizeHintChanged.emit(self.model.index(row))

    def _close(self):
        """收合展開的那張（換回用畫的）。一定要用 setIndexWidget(…, None) 請 Qt 自己移除：
        Qt 的清單內部用指標記著放進去的元件，自己 deleteLater 的話它不知道，之後重新排版會碰到已經刪掉的元件而閃退
        （10-21 打包版：點星號後點開另一張就當掉；測試沒抓到是因為那塊記憶體剛好還沒被重用）。"""
        row, box = self._open_row, self._open_box
        self._open_box, self._open_row = None, -1
        if row >= 0:
            # 鍵盤焦點在展開的卡片裡（標題上按 Enter 收合）：卡片拿掉後焦點會不見，交回清單、停在這一列
            focused = box is not None and box.isAncestorOf(QApplication.focusWidget())
            self.cards.setIndexWidget(self.model.index(row), None)
            self.delegate.sizeHintChanged.emit(self.model.index(row))
            if focused:
                self.cards.setCurrentIndex(self.model.index(row))
                self.cards.setFocus(Qt.OtherFocusReason)

    def _open_resized(self):
        if self._open_row >= 0:
            self.delegate.sizeHintChanged.emit(self.model.index(self._open_row))

    def toggle_row(self, row: int):
        key = key_of(self.model.recs[row])
        was = self.expanded
        self._close()
        self.expanded = None if was == key else key
        if self.expanded:
            self._open(row)

    def toggle_card(self, card: WordCard):
        """展開那張的標題被點：收合。"""
        row = self.model.row_of(key_of(card.rec))
        if row >= 0:
            self.toggle_row(row)

    def _update(self, rec: dict, **fields) -> dict:
        """改一個單字，只更新那一列（清單其他部分不動，篩選結果下次重新整理才更新）。"""
        self.store.listeners.remove(self._on_store_changed)
        try:
            new = self.store.update(key_of(rec), **fields)
        finally:
            self.store.listeners.append(self._on_store_changed)
        row = self.model.row_of(key_of(rec))
        if row >= 0:
            self.model.set_record(row, new)
            if self._open_box is not None and self.expanded == key_of(rec):
                card = self._open_box.layout().itemAt(0).widget()
                card.build(new, True)
                self.delegate.sizeHintChanged.emit(self.model.index(row))
        return new

    def set_flag_row(self, row: int, field: str, value: bool):
        self._flag(self.model.recs[row], field, value)

    def set_flag(self, card: WordCard, field: str, value: bool):
        self._flag(card.rec, field, value)

    def _flag(self, rec: dict, field: str, value: bool):
        new = self._update(rec, **{field: value})
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

    def explain_sentence(self, rec: dict, again: bool = False):
        """句子頁的「講解」：已經有講解就展開／收合；沒有（或按「重新講解」）才呼叫 Gemini。"""
        from . import ai
        key = (rec.get("text", ""), rec.get("saved_at", ""))
        if key in self.explaining:
            return
        self.explain_errors.pop(key, None)
        if rec.get("explanation") and not again:
            self.explain_open.symmetric_difference_update({key})
            self.refresh()
            return
        self.explaining.add(key)
        model = self.cfg.explain_model

        def done(text, err):
            from shiboken6 import isValid
            if not isValid(self):  # 等講解的時候視窗關掉了：講解照樣存起來，只是不用重畫
                if not err:
                    self.store.update_sentence(key[0], key[1], explanation=text, explain_model=model)
                return
            self.explaining.discard(key)
            if err:
                self.explain_errors[key] = err
            else:
                self.explain_open.add(key)
                if self.store.update_sentence(key[0], key[1], explanation=text, explain_model=model):
                    return  # 存檔會觸發重新整理
            self.refresh()
        ai.run_async(lambda: ai.explain(key[0], model), done)
        self.refresh()

    def delete_sentence(self, rec: dict):
        if not self._confirm_delete("這個句子"):
            return
        res =self.store.delete_sentence(rec.get("text", ""), rec.get("saved_at", ""))
        if res:
            idx, old = res
            self.show_toast("已刪除句子", lambda: self.store.restore_sentence(idx, old))

    def speak(self, text: str):
        """朗讀（展開卡片上的喇叭）；沒有日文語音時提示怎麼裝。"""
        speech.speak(text, self.show_toast)

    def copy_word(self, rec: dict):
        rd = rec.get("reading", "")
        text = rec.get("word", "") + (f"【{self.store.reading_with_also(rec)}】" if rd and rd != rec.get("word") else "")
        zh = (rec.get("meaning_zh") or rec.get("meaning_en") or []) + self.store.also_meanings(rec)
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

    # ---------------------------------------------------------------- 單字本管理
    def add_word(self):
        self.toggle_drawer(False)
        dlg = AddWordDialog(self, self.store.dictionary)
        while dlg.exec() == QDialog.Accepted:
            rec = dlg.record()
            if not rec["word"] or not (rec["meaning_zh"] or rec["meaning_en"]):
                QMessageBox.warning(self, "新增單字", "要有單字和至少一個意思。")
                continue
            if not self.store.add_record(rec):
                QMessageBox.warning(self, "新增單字", f"單字本裡已經有「{rec['word']}【{rec['reading']}】」。")
                continue
            self.show_toast(f"已新增「{rec['word']}」")
            return

    def import_words(self):
        self.toggle_drawer(False)
        path, _ = QFileDialog.getOpenFileName(self, "匯入單字", str(self.store.path.parent),
                                              "單字 App 備份檔、完整備份 (*.txt *.json *.jsonl);;所有檔案 (*)")
        if not path:
            return
        try:
            n = self.store.import_file(Path(path))
        except (OSError, ValueError, UnicodeDecodeError) as e:
            QMessageBox.warning(self, "匯入單字", f"沒辦法匯入：{e}")
            return
        QMessageBox.information(self, "匯入單字", f"新增 {n['added']} 個、補上資料 {n['updated']} 個、"
                                                f"略過 {n['skipped']} 個（單字本裡已經有了，或檔案裡重複）。")

    def export_full(self):
        self.toggle_drawer(False)
        name = f"單字本備份_{datetime.now():%Y%m%d}.jsonl"
        path, _ = QFileDialog.getSaveFileName(self, "完整備份", str(self.store.path.parent / name), "單字本備份 (*.jsonl)")
        if not path:
            return
        try:
            n = self.store.export_jsonl(Path(path))
        except OSError as e:
            QMessageBox.warning(self, "完整備份", f"存檔失敗：{e}")
            return
        self.show_toast(f"已備份 {n} 個（含文法）到 {Path(path).name}，可以用「匯入單字」匯回來")

    def clear_words(self):
        self.toggle_drawer(False)
        n = len(self.store.records())
        if not n:
            self.show_toast("單字本已經是空的")
            return
        box = QMessageBox(QMessageBox.Warning, "清空單字本",
                          f"確定要清空單字本嗎？會刪掉 {n} 個單字和文法（記錄的句子不受影響）。\n"
                          "清空前會自動另存一份備份在單字本資料夾，幾秒內也可以按「復原」。",
                          QMessageBox.Cancel, self)
        box.setStyleSheet(QSS)
        ok = box.addButton("清空", QMessageBox.DestructiveRole)
        ok.setObjectName("danger")
        box.setDefaultButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is not ok:
            return
        old, backup = self.store.clear()
        self.show_toast(f"已清空（備份：{backup.name}）", lambda: self.store.restore_all(old))
