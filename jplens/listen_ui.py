"""無字幕視窗：無字幕模式（listen.py）聽到的句子一句一句往下排（像直播的字幕紀錄），點一句放進下面的原文框，按「查詢」才查字。

和一般查字的結果視窗（ui.ResultPopup）分開（使用者 10-17：兩個模式共用原文框會互相蓋掉），
但查字、翻譯、記錄原句、單字清單沿用 ResultPopup（這裡是它的子類別）。

- 句子清單：最新的在最下面，進來時平順地捲上去；講話中先顯示暫定的文字（灰色、結尾「…」，邊講邊長出來），講完換成最終結果。
  **已經出現的句子不會被取代**（使用者 10-17：看上一句時目前這句被蓋掉不行）。
- 往上捲去看舊句子時清單停止自動跟到最新，下面出現「↓ 有 n 句新的」，按了才回到最新。
- 一直講被強制切開的片段（listen.Heard.cont）接在同一行（使用者 10-17：一句話先顯示後半、切開後前半才「插進」兩行中間）。
- 原文框是「選取的句子」，**跟著正在講的那一行**（講話中的暫定文字也是，使用者 10-17 晚：原文框應該是現在聽到的話，不是上一句）；
  點過舊句子、查詢過也照樣跟上新的一行——但點了舊句子之後，要等**點的時候還沒出現的新一行**才換過去
  （否則正在講的那行每 0.8 秒更新一次，點了舊句子還來不及按查詢就被換掉；2026-10-18 審查）。
  不換的情況：原文框正在改字、輸入法組字中、鍵盤在原文框裡（點進去了）、清單往上捲在看舊句子（按「↓ 回到最新」再跟上）、
  按了查詢／翻譯但那行還在講（等它講完才查、才翻，這段時間原文框停在那行）。
  查詢結果（單字清單）不會跟著換，「收起查詢結果」那列寫著是哪一句的結果。
- 譯文存在那一行、顯示在清單裡那一行下面（使用者 10-17 晚：按了翻譯、下一句進來後譯文就不見了）；點回那行，原文框下面也會顯示。
- 播放速度（倍速）選單：影片開幾倍速就選幾倍，listen.py 辨識前放慢回原速（config.listen_speed）。
- 點別的地方、按 Esc 不會關；✕ 關閉並結束無字幕模式。位置記在 config.listen_pos（第一次在主螢幕右下角）。
"""

from html import escape

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from .ui import ResultPopup

MAX_LINES = 300   # 清單最多留幾句
LIST_H = 168      # 句子清單的高度（邏輯像素，乘上字級）
SPEEDS = (1.0, 1.25, 1.5, 1.75, 2.0)  # 同 listen.SPEEDS（開視窗時還不需要載入 listen）

STYLE = """
QLabel#listen { font-size: 12px; color: #9fd0ff; }
QLabel#listen_err { font-size: 12px; color: #e0a080; }
QScrollArea#lines { background: rgba(255,255,255,8); border: 1px solid #3a3f4b; border-radius: 6px; }
QLabel#line { font-size: 14px; color: #cfd6e6; padding: 4px 8px; border-radius: 4px; border-left: 3px solid transparent; }
QLabel#line:hover { background: rgba(255,255,255,14); }
QLabel#line[partial="true"] { color: #a4adbd; }
QLabel#line[sel="true"] { background: rgba(255,210,122,28); border-left: 3px solid #ffd27a; }
QPushButton#newer { font-size: 12px; color: #ffd27a; background: #3a3424; border: 1px solid #6a5a30; border-radius: 4px; padding: 3px 10px; }
QPushButton#newer:hover { background: #4a4230; }
QPushButton#fold { font-size: 12px; color: #a8b0c0; background: rgba(255,255,255,8); border: 1px solid #3a3f4b; border-radius: 4px; padding: 4px 10px; text-align: left; }
QPushButton#fold:hover { color: #ffffff; background: rgba(255,255,255,16); }
QPushButton#tiny { font-size: 12px; background: transparent; border: none; padding: 2px 6px; color: #a8b0c0; }
QPushButton#tiny:hover { color: #ffffff; }
"""


class Line(QLabel):
    """清單裡的一行：一句話，可能由幾個片段組成（一直講被強制切開的接在一起）。最後一個片段還在講時是暫定的（灰一點、結尾「…」）。
    translation：按「翻譯」得到的譯文，顯示在這行下面。"""
    clicked = Signal(object)

    def __init__(self):
        super().__init__()
        self.setObjectName("line")
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setCursor(Qt.PointingHandCursor)
        self.parts: list[list] = []  # [片段編號 seg, 文字, 是不是最終]
        self.translation = ""
        self.set_selected(False)

    @property
    def full(self) -> str:
        return "".join(p[1] for p in self.parts)

    @property
    def partial(self) -> bool:
        return any(not p[2] for p in self.parts)

    @property
    def last_seg(self) -> int:
        return self.parts[-1][0] if self.parts else -1

    def has_seg(self, seg: int) -> bool:
        return any(p[0] == seg for p in self.parts)

    def set_part(self, seg: int, text: str, final: bool):
        """更新（沒有就加上）片段 seg；最終文字是空的就拿掉那個片段。"""
        part = next((p for p in self.parts if p[0] == seg), None)
        if final and not text:
            if part is not None:
                self.parts.remove(part)
        elif part is None:
            self.parts.append([seg, text, final])
        else:
            part[1], part[2] = text, final
        self._render()

    def set_translation(self, zh: str):
        self.translation = zh
        self._render()

    def _render(self):
        html = escape(self.full) + (" …" if self.partial else "")
        if self.translation:
            html += f'<div style="color:#c9e3b5; font-size:12px; margin-top:2px;">{escape(self.translation)}</div>'
        self.setText(html)
        self.setProperty("partial", "true" if self.partial else "false")
        self.setToolTip("點一下放到下面的原文框" + ("（還在講）" if self.partial else ""))
        self._polish()

    def set_selected(self, on: bool):
        self.setProperty("sel", "true" if on else "false")
        self._polish()

    def _polish(self):
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self)
            e.accept()
            return
        super().mousePressEvent(e)


class ListenWindow(ResultPopup):
    extra_style = STYLE
    stop_requested = Signal()   # 按了 ✕：結束無字幕模式
    save_audio_requested = Signal()
    speed_changed = Signal(float)  # 選了播放速度（倍速）

    def __init__(self, cfg, vocab, attribution: str):
        super().__init__(cfg, vocab, attribution)
        self.query_always = True
        self.empty_hint = "（點上面的句子放到這裡；也可以自己輸入，按 Enter 查詢）"
        self.mode_btn.hide()  # 選取模式、釘選和無字幕無關（這個視窗本來就不會自動關）
        self.pin.hide()
        self.text_btn.hide()  # 無字幕模式一定要看得到原文框
        self.edit_hint.hide()  # 按鈕列只放按鈕，窄的視窗才不會擠在一起
        self.requery_btn.setText("🔍 查詢")
        self.requery_btn.setToolTip("查原文框裡這句的單字（Enter 也可以）")
        self.title.setText("無字幕模式")

        lay = self.frame_lay
        at = lay.indexOf(self.text)
        # 狀態列：聆聽中／載入中／出錯，右邊是「保存最近 60 秒的聲音」
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 2, 0, 2)
        bar.setSpacing(8)
        self.state = QLabel()
        self.state.setObjectName("listen")
        self.state.setWordWrap(True)
        bar.addWidget(self.state, 1)
        self.speed = QComboBox()
        for s in SPEEDS:
            self.speed.addItem("原速" if s == 1 else f"{s:g}×", s)
        self.speed.setCurrentIndex(SPEEDS.index(cfg.listen_speed) if cfg.listen_speed in SPEEDS else 0)
        self.speed.setFocusPolicy(Qt.NoFocus)
        self.speed.setToolTip("影片的播放速度：開倍速看就選一樣的倍數（辨識前先放慢回原速，比較準）")
        self.speed.currentIndexChanged.connect(lambda i: self.speed_changed.emit(self.speed.itemData(i)))
        bar.addWidget(self.speed, 0, Qt.AlignTop)
        self.save_btn = QPushButton("💾 保存聲音")
        self.save_btn.setObjectName("tiny")
        self.save_btn.setFocusPolicy(Qt.NoFocus)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setToolTip("把最近 2 分鐘有聲音的部分（影片停著的時間不算）和辨識結果存下來（漏句、切錯時按，當測試資料）")
        self.save_btn.clicked.connect(self.save_audio_requested.emit)
        bar.addWidget(self.save_btn, 0, Qt.AlignTop)
        lay.insertLayout(at, bar)

        # 句子清單
        self.area = QScrollArea()
        self.area.setObjectName("lines")
        self.area.setWidgetResizable(True)
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.area.setFocusPolicy(Qt.NoFocus)
        host = QWidget()
        self.lines_lay = QVBoxLayout(host)
        self.lines_lay.setContentsMargins(6, 6, 6, 6)
        self.lines_lay.setSpacing(2)
        self.lines_lay.addStretch(1)  # 句子少的時候貼在下面（最新的在最下面）
        self.waiting = QLabel("（等待聲音…）")
        self.waiting.setObjectName("status")
        self.waiting.setAlignment(Qt.AlignCenter)
        self.lines_lay.addWidget(self.waiting)
        self.area.setWidget(host)
        lay.insertWidget(at + 1, self.area)
        self.newer = QPushButton()
        self.newer.setObjectName("newer")
        self.newer.setFocusPolicy(Qt.NoFocus)
        self.newer.setCursor(Qt.PointingHandCursor)
        self.newer.clicked.connect(self._jump_latest)
        self.newer.hide()
        lay.insertWidget(at + 2, self.newer, 0, Qt.AlignHCenter)
        lay.insertSpacing(at + 3, 2)

        # 收起／展開查詢結果（單字清單）：查完字視窗會變很高，擋住影片
        self.fold_btn = QPushButton()
        self.fold_btn.setObjectName("fold")
        self.fold_btn.setFocusPolicy(Qt.NoFocus)
        self.fold_btn.setCursor(Qt.PointingHandCursor)
        self.fold_btn.clicked.connect(self._toggle_fold)
        self.fold_btn.hide()
        lay.insertWidget(lay.indexOf(self.scroll), self.fold_btn)

        self.lines: list[Line] = []
        self._sel: Line | None = None
        self._follow = True    # 清單自動捲到最新（原文框也跟著換成最新一句）
        self._queried = ""     # 查詢結果是哪一句的
        self._unseen = 0       # 不跟著捲時，新進來幾句
        self._pending = None   # 按了查詢／翻譯但那行還在講：("query" 或 "translate", 那一行)，講完才做
        self._hold_seg = None  # 點了舊句子：片段編號到這裡為止的行都不自動換過去（等新的一行）
        self._query_line = None  # 「查詢」是查哪一行的（結果回來時原文框已經換到別行，就換回目前選的）
        self._want_y = None    # 使用者放的高度：查詢結果展開時被往上推，收起後回到這裡
        sb = self.area.verticalScrollBar()
        self._anim = QPropertyAnimation(sb, b"value", self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        # 只有使用者自己捲（滾輪、拖曳、點捲軸）才決定要不要繼續跟到最新；動畫和內容變長造成的捲動不算
        sb.actionTriggered.connect(self._user_scrolled)
        sb.sliderMoved.connect(self._user_scrolled)
        sb.rangeChanged.connect(lambda lo, hi: self._follow and self._scroll_to(hi))
        self._apply_look()

    # ---------------------------------------------------------------- 外觀、位置
    def _apply_look(self):
        super()._apply_look()
        if hasattr(self, "area"):
            self.area.setFixedHeight(round(LIST_H * self.scale))

    def _set_title(self, text: str):
        super()._set_title(text if text == "無字幕模式" else f"無字幕｜{text}")

    # ---------------------------------------------------------------- 查詢結果
    def show_result(self, texts, words, seconds, detail=""):
        """查詢結果回來。ResultPopup.show_result 會把原文框換成查的字、清掉譯文和還在跑的翻譯；
        這裡把它們還原（2026-10-18 審查：先按翻譯再按查詢，「翻譯中…」不見；查詢中換了行，原文框變回舊的那句）。"""
        self.list_folded = False  # 查了新的一句：展開看結果
        self._queried = "".join(texts)
        job, sel, asked = self._job, self._sel, self._query_line
        super().show_result(texts, words, seconds, detail)
        if sel is None or sel not in self.lines:
            return
        if asked is not sel:  # 查詢的時候是別行：原文框留在現在選的這行
            self.texts = [sel.full]
            self._show_text(sel.full, editable=True)
            if sel.translation:
                self._set_translation(sel.translation)
        elif job is not None and not job["done"] and job.get("line") is sel:
            self._job = job  # 翻譯還在跑：結果回來照樣顯示在這裡
            self._set_translation("翻譯中…", busy=True)
        elif sel.translation and not self._zh:
            self._set_translation(sel.translation)
        self._fit()

    def _render(self):
        super()._render()
        self._fold_text()

    def _fold_text(self):
        from .words import visible
        shown = visible(self.words, self.cfg.min_level, self.cfg.show_proper)
        n_words = sum(not w.is_grammar for w in shown)
        n_gram = len(shown) - n_words
        q = self._queried if len(self._queried) <= 12 else self._queried[:11] + "…"
        what = f"「{q}」{n_words} 個單字" + (f"・{n_gram} 文法" if n_gram else "")
        self.fold_btn.setText(f"▸ 展開查詢結果（{what}）" if self.list_folded else f"▾ 收起查詢結果（{what}）")
        self.fold_btn.setVisible(bool(self.words))

    def _toggle_fold(self):
        self.list_folded = not self.list_folded
        self._fold_text()
        self._fit()

    def sync_from_cfg(self):
        super().sync_from_cfg()
        self.text.setVisible(True)
        self.sent_row.setVisible(True)
        self._fit()

    def place(self):
        """上次拖過去的地方；沒有（或那個位置已經不在任何螢幕上）就放主螢幕右下角。"""
        self._want_y = None
        self._fit()
        pos = self.cfg.listen_pos
        if pos and QGuiApplication.screenAt(QPoint(pos[0] + 20, pos[1] + 20)):
            self.move(pos[0], pos[1])
            self._want_y = pos[1]
            self._keep_on_screen()
            return
        area = QGuiApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.width() - 16, area.bottom() - self.height() - 16)
        self._want_y = self.y()

    def _keep_on_screen(self):
        """變高時超出螢幕下緣就往上推；變矮時（收起查詢結果）回到使用者放的高度（2026-10-18 審查：收起後懸在螢幕中間）。"""
        if self._want_y is None:
            return super()._keep_on_screen()
        area = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        x = min(max(area.left(), self.x()), area.right() - self.width())
        y = min(max(area.top(), self._want_y), area.bottom() - self.height())
        if (x, y) != (self.x(), self.y()):
            self.move(x, y)

    def mouseReleaseEvent(self, e):
        if self._drag is not None:  # 記住拖到哪裡，下次從這裡開始
            self.cfg.listen_pos = [self.x(), self.y()]
            self._want_y = self.y()
            self.cfg.save()
        super().mouseReleaseEvent(e)

    def _close(self):
        self.stop_requested.emit()
        self.hide()

    # ---------------------------------------------------------------- 開始、狀態
    def start(self, hotkey_name: str):
        for ln in self.lines:
            ln.deleteLater()
        self.lines, self._sel = [], None
        self._follow, self._unseen = True, 0
        self._pending = self._hold_seg = self._query_line = None
        self._queried = ""
        self.newer.hide()
        self.waiting.show()
        self.words, self.texts = [], []
        self._clear_list()
        self.fold_btn.hide()
        self._set_title("無字幕模式")
        self._show_text("", editable=True)
        self.text.setVisible(True)
        self.sent_row.setVisible(True)  # 沒有句子時也可以自己輸入再按查詢
        self.status.setText(f"{hotkey_name} 或 ✕ 結束")
        self.set_state("載入語音辨識…", False)
        if not self.isVisible():
            self.place()
        self.show()
        self.raise_()

    def set_state(self, msg: str, error: bool):
        self.state.setText(("⚠ " if error else "🎙 ") + msg)
        self.state.setObjectName("listen_err" if error else "listen")
        self.state.style().unpolish(self.state)
        self.state.style().polish(self.state)
        self._fit()

    # ---------------------------------------------------------------- 聽到的句子
    def add_heard(self, seg: int, text: str, final: bool, cont: bool = False):
        """listen.Heard：同一個片段（seg）先來暫定文字、最後來最終文字（空的＝那個片段不算，拿掉）。
        cont：接在上一個片段後面（一直講被強制切開），放在同一行。"""
        line = next((ln for ln in reversed(self.lines[-5:]) if ln.has_seg(seg)), None)
        if line is None:
            if not text:
                return
            if cont and self.lines and self.lines[-1].last_seg == seg - 1:
                line = self.lines[-1]
            else:
                line = Line()
                line.clicked.connect(self._pick)
                self.waiting.hide()
                self.lines_lay.addWidget(line)
                self.lines.append(line)
                if not self._follow:
                    self._unseen += 1
                    self._show_newer()
                while len(self.lines) > MAX_LINES:
                    old = self.lines.pop(0)
                    if old is self._sel:
                        self._sel = None
                    old.deleteLater()
        line.set_part(seg, text, final)
        if not line.parts:  # 唯一的片段最後是空的：整行拿掉
            self.lines.remove(line)
            line.deleteLater()
            if self._pending and self._pending[1] is line:
                self._pending = None
                self._set_translation("")
                self.status.setText("那句沒有辨識出文字")
            if line is self._sel:
                self._sel = None
            return
        if line is self._sel:
            if not self._editing():
                self._refresh_sel()
        elif (self._follow and not self._editing() and self._pending is None
              and (self._hold_seg is None or line.parts[0][0] > self._hold_seg)):
            self._hold_seg = None
            self._select(line)
        if self._pending and self._pending[1] is line and not line.partial:
            what = self._pending[0]
            self._pending = None
            if what == "translate":
                self._translate()
            else:
                self._requery()

    def _select(self, line: Line):
        if self._sel is not None and self._sel in self.lines:
            self._sel.set_selected(False)
        self._sel = line
        self._pending = None
        line.set_selected(True)
        self._job = None  # 上一句還在翻譯的話，結果不要顯示在這句底下（會存在那一行）
        if self._pin_job is not None:
            self._release_pin()
        self.texts = [line.full]
        self._show_text(line.full, editable=True)
        self.text.setVisible(True)  # 「文」按鈕關掉時原文框也要看得到（無字幕模式就是看原文）
        self.sent_row.setVisible(True)
        self.sent_btn.setText("＋ 記錄原句")
        if line.translation:
            self._set_translation(line.translation)
        self._fit()

    def _refresh_sel(self):
        """選取的那一行文字變了（還在講、接上了下一個片段）：原文框跟著換，譯文、查詢結果不動。"""
        line = self._sel
        if self.texts == [line.full]:
            return
        self.texts = [line.full]
        self.text.blockSignals(True)
        self.text.setPlainText(line.full)
        self.text.blockSignals(False)
        h = self.text.height()
        self._fit_text()
        if self.text.height() != h:
            self._fit()

    def _pick(self, line: Line):
        """點了一行：放進原文框。點的是舊句子時，正在講的那行不會把它換掉，等下一個新的一行出現才換過去。"""
        self._select(line)
        last = self._last_line()
        self._hold_seg = last.last_seg if last is not None and last is not line else None

    def _editing(self) -> bool:
        """原文框不要被換掉：改過字還沒查、輸入法組字中、鍵盤在原文框裡（點進去要打字）。"""
        return self._editing_text() or self.text.hasFocus()

    def _last_line(self):
        return self.lines[-1] if self.lines else None

    def _waiting_line(self, what: str) -> bool:
        """選取的那行還在講：等講完再查詢／翻譯（原文框改過字就直接用改好的字）。"""
        line = self._sel
        if line is None or not line.partial or self._editing_text():
            return False
        self._pending = (what, line)
        if what == "translate":
            self._set_translation("這句講完就翻譯…", busy=True)
        else:
            self.status.setText("這句講完就查詢…")
        self._fit()
        return True

    def _requery(self):
        if self._waiting_line("query"):
            return
        self._query_line = self._sel
        super()._requery()

    def _request(self, src: str, show: bool, force: bool = False):
        super()._request(src, show, force)
        self._job["line"] = self._sel  # 譯文存到這一行（之後原文框換到別行也留著）

    def _job_done(self, job):
        line = job.get("line")
        if line is not None and job["show"] and not job["err"] and job["zh"] and line in self.lines:
            line.set_translation(job["zh"])
        super()._job_done(job)

    def _set_translation(self, zh: str, error: bool = False, busy: bool = False):
        super()._set_translation(zh, error, busy)
        if zh and not error and not busy and getattr(self, "_sel", None) is not None and self._sel in self.lines:
            self._sel.set_translation(zh)  # 快取裡現成的譯文、背景先翻好的也記在那一行

    def _translate(self, pin: bool = False):
        if self._waiting_line("translate"):
            return
        super()._translate(pin=False)  # 這個視窗本來就不會自動關，不用釘選

    # ---------------------------------------------------------------- 捲動
    def _scroll_to(self, value: int):
        self._anim.stop()
        sb = self.area.verticalScrollBar()
        if sb.value() == value:
            return
        self._anim.setStartValue(sb.value())
        self._anim.setEndValue(value)
        self._anim.start()

    def _user_scrolled(self, *_):
        self._anim.stop()
        QTimer.singleShot(0, self._check_follow)  # actionTriggered 時捲軸還沒移動

    def _check_follow(self):
        sb = self.area.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        if at_bottom != self._follow:
            self._follow = at_bottom
            if at_bottom:
                self._unseen = 0
            self._show_newer()

    def _show_newer(self):
        self.newer.setText(f"↓ 有 {self._unseen} 句新的，回到最新" if self._unseen else "↓ 回到最新")
        if self.newer.isVisible() != (not self._follow):
            self.newer.setVisible(not self._follow)
            self._fit()

    def _jump_latest(self):
        self._follow, self._unseen = True, 0
        self._hold_seg = None
        self._show_newer()
        self._scroll_to(self.area.verticalScrollBar().maximum())
        last = self._last_line()
        if last is not None:
            self._select(last)
