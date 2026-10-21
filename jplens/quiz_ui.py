"""單字本視窗的「單字測驗」頁：設定（題型、題數、範圍、級數、時間）→ 作答 → 結果（重測答錯的）。

出題在 quiz.py（答案唯一、些微混淆）。第一次出題要算每個意思的向量（約 0.7 毫秒一個、300 個字約 1 秒），
在背景執行緒做；之後從快取讀。答錯時可以按「這個也對」：算答對，並記住這兩個字以後不放在同一題。
"""

import random
import threading
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea,
                               QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from . import quiz
from .vocab import key_of

MODES = [("meaning", "看單字選意思"), ("word", "看意思選單字"), ("mixed", "混合")]
COUNTS = [(10, "10 題"), (20, "20 題"), (30, "30 題")]
SCOPES = [("all", "全部"), ("marked", "重點單字"), ("recent", "最近記錄的")]
LEVELS = ["N5", "N4", "N3", "N2", "N1", "級外"]
LEARNED = [(False, "不出"), (True, "也出")]
TIMES = [(5, "5 秒"), (10, "10 秒"), (15, "15 秒"), (0, "不限")]
AUTO_NEXT_MS = 900  # 答對後自動下一題
# 這次執行期間記住設定（關掉單字本視窗再打開還在）
SETTINGS = {"mode": "meaning", "count": 10, "scope": "all", "recent": 50, "levels": set(), "learned": False, "time": 10}


def _label(text, name=None, wrap=False):
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def _button(text, name=None, slot=None):
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(lambda _=False: slot())
    return b


def level_of(rec: dict) -> str:
    lv = rec.get("jlpt", "")
    return lv if lv in LEVELS[:5] else "級外"


class _Relay(QObject):
    done = Signal(object, object)  # (回呼, 結果)


class OptionCard(QFrame):
    """一個選項：大塊、可以換行；答完後下面多一行對照（選意思時列出是哪個字，選單字時列出讀音和意思）。"""
    clicked = Signal()

    def __init__(self, num: int):
        super().__init__()
        self.setObjectName("quizOpt")
        self.setAttribute(Qt.WA_Hover)
        self.setCursor(Qt.PointingHandCursor)
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(12)
        self.num = _label(str(num), "quizNum")
        h.addWidget(self.num, 0, Qt.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(2)
        self.text = _label("", "quizOptText", wrap=True)
        self.sub = _label("", "quizOptSub", wrap=True)
        v.addWidget(self.text)
        v.addWidget(self.sub)
        h.addLayout(v, 1)
        for lb in (self.num, self.text, self.sub):
            lb.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.set_state("")

    def set_state(self, state: str):
        """""：可以選；"correct"／"wrong"：答完後的正確答案／選錯的；"done"：答完後的其他選項。"""
        self.setProperty("state", state)
        self.setCursor(Qt.PointingHandCursor if not state else Qt.ArrowCursor)
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()


class QuizPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setFocusPolicy(Qt.StrongFocus)  # 數字鍵作答、Enter 下一題
        self.sem: quiz.Semantic | None = None
        self.maker: quiz.QuizMaker | None = None
        self.questions: list[quiz.Question] = []
        self.results: list[dict] = []  # {"q", "chosen"（None＝超時）, "ok", "same"（按了這個也對）}
        self.i = 0
        self.answered = False
        self._job = 0  # 第幾次出題：離開測驗頁就加一，背景出好的舊題目不用
        self._relay = _Relay()
        self._relay.done.connect(lambda cb, res: cb(res))
        self._timer = QTimer(self, interval=50, timeout=self._tick)
        self._next_timer = QTimer(self, singleShot=True, interval=AUTO_NEXT_MS, timeout=self.next_question)
        self.pages = QStackedWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.pages)
        self._build_setup()
        self._build_loading()
        self._build_question()
        self._build_result()

    # ---------------------------------------------------------------- 設定
    def _section(self, v, title, opts, key, multi=False):
        v.addWidget(_label(title, "desc"))
        row = QHBoxLayout()
        row.setSpacing(6)
        group = None if multi else QButtonGroup(self)
        for val, text in opts:
            b = _button(text, "chip")
            b.setCheckable(True)
            b.setChecked(val in SETTINGS[key] if multi else SETTINGS[key] == val)
            if multi:
                b.toggled.connect(lambda on, x=val: ((SETTINGS[key].add if on else SETTINGS[key].discard)(x),
                                                    self.update_count()))
            else:
                group.addButton(b)
                b.clicked.connect(lambda _=False, x=val: (SETTINGS.__setitem__(key, x), self.update_count()))
            row.addWidget(b)
        return row

    def _build_setup(self):
        page = QScrollArea()
        page.setWidgetResizable(True)
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(24, 12, 24, 20)
        v.setSpacing(10)
        v.addWidget(_label("從單字本出題。選項裡不會有意思一樣的字（答錯時如果你覺得也對，可以按「這個也對」）。", "sub", wrap=True))
        v.addSpacing(4)
        for title, opts, key in (("題型", MODES, "mode"), ("題數", COUNTS, "count")):
            row = self._section(v, title, opts, key)
            row.addStretch(1)
            v.addLayout(row)
        row = self._section(v, "範圍", SCOPES, "scope")
        self.recent = QSpinBox()
        self.recent.setRange(5, 10000)
        self.recent.setSingleStep(10)
        self.recent.setValue(SETTINGS["recent"])
        self.recent.setSuffix(" 個")
        self.recent.valueChanged.connect(lambda n: (SETTINGS.__setitem__("recent", n), self.update_count()))
        row.addWidget(self.recent)
        row.addStretch(1)
        v.addLayout(row)
        row = self._section(v, "級數（可複選，不選＝全部）", [(lv, lv) for lv in LEVELS], "levels", multi=True)
        row.addStretch(1)
        v.addLayout(row)
        for title, opts, key in (("已學會的單字", LEARNED, "learned"), ("每題時間", TIMES, "time")):
            row = self._section(v, title, opts, key)
            row.addStretch(1)
            v.addLayout(row)
        v.addSpacing(10)
        foot = QHBoxLayout()
        self.count_label = _label("", "sub")
        foot.addWidget(self.count_label)
        foot.addStretch(1)
        self.start_btn = _button("開始測驗", "primary", self.start)
        self.start_btn.setMinimumWidth(140)
        foot.addWidget(self.start_btn)
        v.addLayout(foot)
        v.addStretch(1)
        page.setWidget(host)
        self.setup_page = page
        self.pages.addWidget(page)

    def candidates(self) -> list[dict]:
        """符合設定、可以出題的單字（有意思、不是文法）。"""
        recs = [r for r in self.win.store.records() if r.get("pos") != "文法" and quiz.meanings(r) and r.get("word")]
        if SETTINGS["scope"] == "marked":
            recs = [r for r in recs if r.get("marked")]
        elif SETTINGS["scope"] == "recent":
            recs = sorted(recs, key=lambda r: r.get("saved_at", ""), reverse=True)[:SETTINGS["recent"]]
        if not SETTINGS["learned"]:
            recs = [r for r in recs if not r.get("learned")]
        if SETTINGS["levels"]:
            recs = [r for r in recs if level_of(r) in SETTINGS["levels"]]
        return recs

    def update_count(self):
        n = len(self.candidates())
        self.recent.setEnabled(SETTINGS["scope"] == "recent")
        self.count_label.setText(f"符合條件的單字 {n} 個" + ("" if n >= SETTINGS["count"] or not n
                                                       else f"（不到 {SETTINGS['count']} 題，就出 {n} 題）"))
        self.start_btn.setEnabled(n > 0)

    def enter(self):
        """切到測驗頁。"""
        if self.pages.currentWidget() is self.setup_page or not self.questions:
            self.pages.setCurrentWidget(self.setup_page)
            self.update_count()

    def store_changed(self):
        if self.pages.currentWidget() is self.setup_page:
            self.update_count()

    # ---------------------------------------------------------------- 準備題目（背景）
    def _build_loading(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.addStretch(1)
        self.loading = _label("準備題目中…", "sub")
        self.loading.setAlignment(Qt.AlignCenter)
        v.addWidget(self.loading)
        v.addStretch(2)
        self.loading_page = page
        self.pages.addWidget(page)

    def start(self, targets: list[dict] | None = None):
        targets = self.candidates() if targets is None else targets
        if not targets:
            return
        n, mode = min(SETTINGS["count"], len(targets)), SETTINGS["mode"]
        first = self.sem is None or not self.sem._vec
        self.loading.setText("準備題目中…" + ("（第一次要分析每個意思，單字多的話要幾秒）" if first else ""))
        self.pages.setCurrentWidget(self.loading_page)
        pool = [r for r in self.win.store.records() if r.get("pos") != "文法" and quiz.meanings(r)]
        store = self.win.store
        self._job += 1
        job = self._job

        def work():
            try:
                if self.sem is None:
                    self.sem = quiz.Semantic(cache=quiz.VEC_CACHE)
                same = quiz.SamePairs(store.path.parent / quiz.SAME_NAME)
                maker = quiz.QuizMaker(pool, key_of, self.sem, same, quiz.jlpt_fill(store.dictionary))
                by_key = {key_of(r): r for r in maker.pool}
                ts = [by_key.get(key_of(t), t) for t in targets]  # 用 pool 裡同一個物件（選項比對靠 is）
                res = (maker, maker.make(ts, mode, n), "")
            except Exception as e:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).exception("出題失敗")
                res = (None, [], str(e))
            self._relay.done.emit(lambda r: self._ready(r, job), res)

        threading.Thread(target=work, name="quiz", daemon=True).start()

    def _ready(self, res, job: int):
        if job != self._job:  # 準備題目時離開了測驗頁或關掉視窗：不然會在背景開始作答、倒數計時
            return
        maker, questions, err = res
        if err or not questions:
            self.pages.setCurrentWidget(self.setup_page)
            self.update_count()
            self.win.show_toast("出題失敗：" + err if err else "湊不出選項（單字太少），請先記錄更多單字")
            return
        self.maker, self.questions, self.results, self.i = maker, questions, [], 0
        self.pages.setCurrentWidget(self.question_page)
        self.show_question()

    # ---------------------------------------------------------------- 作答
    def _build_question(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 8, 24, 16)
        v.setSpacing(10)
        top = QHBoxLayout()
        self.progress = _label("", "sub")
        top.addWidget(self.progress)
        top.addStretch(1)
        self.score = _label("", "sub")
        top.addWidget(self.score)
        top.addWidget(_button("結束", None, self.finish))
        v.addLayout(top)
        self.timebar = QProgressBar()
        self.timebar.setObjectName("quizTime")
        self.timebar.setTextVisible(False)
        self.timebar.setRange(0, 1000)
        self.timebar.setFixedHeight(6)
        v.addWidget(self.timebar)
        self.prompt = _label("", "quizPrompt", wrap=True)
        self.prompt.setAlignment(Qt.AlignCenter)
        self.prompt.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addSpacing(6)
        v.addWidget(self.prompt)
        self.prompt_sub = _label("", "quizPromptSub", wrap=True)
        self.prompt_sub.setAlignment(Qt.AlignCenter)
        v.addWidget(self.prompt_sub)
        v.addSpacing(6)
        self.opts = []
        for k in range(quiz.N_OPTIONS):
            o = OptionCard(k + 1)
            o.clicked.connect(lambda k=k: self.answer(k))
            self.opts.append(o)
            v.addWidget(o)
        feed = QHBoxLayout()
        self.feedback = _label("", "quizFeedback", wrap=True)
        feed.addWidget(self.feedback, 1)
        self.same_btn = _button("這個也對", None, self.mark_same)
        self.same_btn.setToolTip("你選的也是正確的意思：這題算答對，以後這兩個字不會出現在同一題")
        feed.addWidget(self.same_btn)
        self.next_btn = _button("下一題", "primary", self.next_question)
        self.next_btn.setMinimumWidth(110)
        feed.addWidget(self.next_btn)
        v.addLayout(feed)
        v.addStretch(1)
        self.question_page = page
        self.pages.addWidget(page)

    @property
    def q(self) -> quiz.Question:
        return self.questions[self.i]

    def show_question(self):
        q = self.q
        self.answered = False
        self._next_timer.stop()
        self.progress.setText(f"第 {self.i + 1} / {len(self.questions)} 題")
        self.score.setText(f"答對 {sum(r['ok'] for r in self.results)}")
        self.prompt.setProperty("kind", q.mode)
        self.prompt.style().unpolish(self.prompt)
        self.prompt.style().polish(self.prompt)
        self.prompt.setText(q.prompt)
        self.prompt_sub.setText("選出意思" if q.mode == "meaning" else "選出單字")
        for k, o in enumerate(self.opts):
            o.text.setText(q.label(k))
            o.text.setProperty("kind", q.mode)
            o.text.style().unpolish(o.text)
            o.text.style().polish(o.text)
            o.sub.hide()
            o.set_state("")
        self.feedback.setText("")
        self.same_btn.hide()
        self.next_btn.hide()
        self._limit = SETTINGS["time"]
        self.timebar.setVisible(bool(self._limit))
        self.timebar.setValue(1000)
        self._t0 = time.monotonic()
        if self._limit:
            self._timer.start()
        self.setFocus()

    def _tick(self):
        left = self._limit - (time.monotonic() - self._t0)
        self.timebar.setValue(max(0, int(left / self._limit * 1000)))
        if left <= 0:
            self.answer(None)

    def answer(self, k: int | None):
        if self.answered:
            return
        self.answered = True
        self._timer.stop()
        q = self.q
        ok = k == q.answer
        self.results.append({"q": q, "chosen": k, "ok": ok, "same": False})
        for i, o in enumerate(self.opts):
            o.set_state("correct" if i == q.answer else "wrong" if i == k else "done")
            rec = q.options[i]
            rd = rec.get("reading", "")
            word = rec.get("word", "") + (f"【{rd}】" if rd and rd != rec.get("word") else "")
            o.sub.setText(word if q.mode == "meaning" else (f"【{rd}】　" if rd and rd != rec.get("word") else "")
                          + quiz.shown(rec))
            o.sub.show()
        t = q.target
        rd = t.get("reading", "")
        self.prompt_sub.setText((f"【{rd}】" if rd and rd != t.get("word") else "") if q.mode == "meaning"
                                else t.get("word", "") + (f"【{rd}】" if rd and rd != t.get("word") else ""))
        self.score.setText(f"答對 {sum(r['ok'] for r in self.results)}")
        if ok:
            self.feedback.setText("✓ 答對了")
            self._next_timer.start()
        else:
            self.feedback.setText("✗ 時間到" if k is None else "✗ 答錯了")
            self.same_btn.setVisible(k is not None)
        self.next_btn.setText("看結果" if self.i == len(self.questions) - 1 else "下一題")
        self.next_btn.show()
        self.next_btn.setFocus()

    def mark_same(self):
        """答錯但使用者覺得也對：算答對，記住這兩個字以後不放同一題。"""
        r = self.results[-1]
        chosen = self.q.options[r["chosen"]]
        if self.maker and self.maker.same is not None:
            self.maker.same.add(key_of(self.q.target), key_of(chosen))
        r["ok"] = r["same"] = True
        self.same_btn.hide()
        self.feedback.setText(f"✓ 算答對：以後「{self.q.target.get('word')}」和「{chosen.get('word')}」不會出現在同一題")
        self.score.setText(f"答對 {sum(x['ok'] for x in self.results)}")
        self.next_btn.setFocus()

    def next_question(self):
        if not self.answered:
            return
        self._next_timer.stop()
        if self.i + 1 >= len(self.questions):
            self.finish()
            return
        self.i += 1
        self.show_question()

    def keyPressEvent(self, e):
        if self.pages.currentWidget() is self.question_page:
            if not self.answered and Qt.Key_1 <= e.key() < Qt.Key_1 + quiz.N_OPTIONS:
                self.answer(e.key() - Qt.Key_1)
                return
            if self.answered and e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
                self.next_question()
                return
        super().keyPressEvent(e)

    # ---------------------------------------------------------------- 結果
    def _build_result(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 8, 24, 16)
        v.setSpacing(10)
        self.result_title = _label("", "quizScore")
        self.result_title.setAlignment(Qt.AlignCenter)
        v.addWidget(self.result_title)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.retry_btn = _button("", "primary", self.retry_wrong)
        btns.addWidget(self.retry_btn)
        btns.addWidget(_button("再來一次", None, lambda: self.start()))
        btns.addWidget(_button("回到設定", None, self.back_to_setup))
        btns.addStretch(1)
        v.addLayout(btns)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        self.result_list = QVBoxLayout(host)
        self.result_list.setContentsMargins(0, 4, 8, 4)
        self.result_list.setSpacing(6)
        scroll.setWidget(host)
        v.addWidget(scroll, 1)
        self.result_page = page
        self.pages.addWidget(page)

    def finish(self):
        self._timer.stop()
        self._next_timer.stop()
        n, ok = len(self.results), sum(r["ok"] for r in self.results)  # 中途按「結束」：還沒作答的題目不算
        self.result_title.setText(f"{ok} / {n}" + (f"　{round(ok * 100 / n)}%" if n else ""))
        wrong = [r for r in self.results if not r["ok"]]
        self.retry_btn.setText(f"重測答錯的 {len(wrong)} 題")
        self.retry_btn.setVisible(bool(wrong))
        while self.result_list.count():
            item = self.result_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for r in sorted(self.results, key=lambda r: r["ok"]):
            self.result_list.addWidget(self._result_row(r))
        self.result_list.addStretch(1)
        self.pages.setCurrentWidget(self.result_page)
        if self.sem is not None:
            self.sem.save()

    def _result_row(self, r) -> QFrame:
        q, t = r["q"], r["q"].target
        row = QFrame()
        row.setObjectName("card")
        h = QHBoxLayout(row)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(10)
        mark = _label("✓" if r["ok"] else "✗", "quizOk" if r["ok"] else "quizNg")
        h.addWidget(mark, 0, Qt.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(2)
        rd = t.get("reading", "")
        v.addWidget(_label(t.get("word", "") + (f"【{rd}】" if rd and rd != t.get("word") else ""), "quizResWord"))
        v.addWidget(_label("；".join(quiz.meanings(t)), "summary", wrap=True))
        if not r["ok"]:
            your = "時間到，沒有作答" if r["chosen"] is None else "你選：" + q.label(r["chosen"]) + (
                f"（{q.options[r['chosen']].get('word', '')}）" if q.mode == "meaning" else "")
            v.addWidget(_label(your, "dim", wrap=True))
        elif r["same"]:
            v.addWidget(_label("你按了「這個也對」", "dim"))
        h.addLayout(v, 1)
        return row

    def retry_wrong(self):
        wrong = [r["q"].target for r in self.results if not r["ok"]]
        if not wrong or self.maker is None:
            return
        qs = [q for q in (self.maker.question(t, r["q"].mode) for t, r in
                          zip(wrong, [r for r in self.results if not r["ok"]])) if q]
        random.shuffle(qs)
        if qs:
            self.questions, self.results, self.i = qs, [], 0
            self.pages.setCurrentWidget(self.question_page)
            self.show_question()

    def back_to_setup(self):
        self.questions = []
        self.pages.setCurrentWidget(self.setup_page)
        self.update_count()

    def leave(self):
        """離開測驗頁或關掉視窗：停計時、放掉模型。"""
        self._timer.stop()
        self._next_timer.stop()
        self._job += 1
        if self.pages.currentWidget() is self.loading_page:
            self.pages.setCurrentWidget(self.setup_page)
        if self.sem is not None:
            self.sem.unload()
