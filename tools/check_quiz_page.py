"""單字本「單字測驗」頁的自動檢查（會開視窗；用暫存的假單字本和暫存的向量快取，不碰真的資料）：

    python tools/check_quiz_page.py

切到測驗頁（搜尋列藏起來）、範圍和級數的篩選、出題（選項唯一）、數字鍵作答、答對自動下一題、
答錯按「這個也對」（記住配對）、時間到、中途結束、重測答錯的、單字本改變時更新數量、離開測驗頁放掉模型。
"""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from PySide6.QtCore import QCoreApplication, Qt, QThread  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
from jplens import quiz, quiz_ui  # noqa: E402
from jplens.config import Config  # noqa: E402
from jplens.manager import ManagerWindow  # noqa: E402
from jplens.vocab import VocabStore, key_of  # noqa: E402

WORDS = [  # (單字, 讀音, 級數, 意思)
    ("毛布", "もうふ", "N3", "毛毯、毯子"), ("羽毛", "うもう", "N1", "羽毛、羽絨"), ("布団", "ふとん", "N4", "被褥、棉被"),
    ("机", "つくえ", "N5", "桌子、書桌"), ("椅子", "いす", "N5", "椅子"), ("窓", "まど", "N5", "窗戶"),
    ("即座", "そくざ", "N1", "即刻、當場"), ("早急", "さっきゅう", "N1", "即時、迅速"), ("緒", "いとぐち", "", "開端、起點"),
    ("端", "はし", "N3", "開頭、起端；邊緣"), ("重んじる", "おもんじる", "N1", "重視、尊重、看重"),
    ("尊顔", "そんがん", "", "尊顏、尊容"), ("加重", "かじゅう", "N1", "加重、增加重量"), ("荷", "に", "N2", "行李、貨物；負擔、責任"),
    ("夜通し", "よどおし", "N1", "整夜、通宵"), ("見通し", "みとおし", "N2", "展望、前景；視野"),
    ("撤去", "てっきょ", "N1", "撤除、拆除"), ("甘んじる", "あまんじる", "", "甘心、甘願、安於"),
    ("甘く見る", "あまくみる", "", "輕視、小看"), ("部類", "ぶるい", "N1", "類別、種類"), ("部下", "ぶか", "N3", "部下、下屬"),
    ("交わる", "まじわる", "N1", "相交、交叉；交往"), ("割り切る", "わりきる", "N1", "徹底解決、乾脆"),
    ("仕舞い込む", "しまいこむ", "", "收起來、藏起來"), ("受け持つ", "うけもつ", "N2", "負責、掌管"),
    ("先程", "さきほど", "N3", "剛才、方才"), ("手立て", "てだて", "N1", "手段、方法"), ("用例", "ようれい", "N1", "用例、例子"),
    ("思い返す", "おもいかえす", "N1", "回想、重新考慮"), ("名案", "めいあん", "N1", "好主意、妙計"),
    ("容易い", "たやすい", "N1", "容易、簡單"), ("瞬く", "またたく", "N1", "眨眼；閃爍"), ("呆気ない", "あっけない", "N1", "沒意思、不過癮"),
    ("遺憾", "いかん", "N1", "遺憾、可惜"), ("決行", "けっこう", "N1", "斷然實行"), ("乗り切る", "のりきる", "N1", "渡過、克服"),
    ("跳ね上がる", "はねあがる", "N1", "跳起；暴漲"), ("行き届く", "いきとどく", "N1", "周到、周密"), ("毛皮", "けがわ", "N2", "毛皮"),
    ("布地", "ぬのじ", "N1", "布料"),
]
ok = True


def pump(n=20):
    for _ in range(n):
        QCoreApplication.processEvents()
        QThread.msleep(10)


def report(good, msg):
    global ok
    ok &= bool(good)
    print(("OK  " if good else "NG  ") + msg)


d = Path(tempfile.mkdtemp())
quiz.VEC_CACHE = d / "vectors.npz"
with open(d / "saved_words.jsonl", "w", encoding="utf-8") as f:
    for i, (w, rd, lv, zh) in enumerate(WORDS):
        f.write(json.dumps({"word": w, "reading": rd, "pos": "名詞", "jlpt": lv, "meaning_zh": zh.split("；"),
                            "saved_at": f"2026-10-{1 + i % 28:02d}T10:00:00", "marked": i % 4 == 0,
                            "learned": i % 9 == 8}, ensure_ascii=False) + "\n")
store = VocabStore(d / "saved_words.jsonl")
win = ManagerWindow(store, Config.load())
win.resize(820, 760)
win.show()
pump()
qp = win.quiz_page
S = quiz_ui.SETTINGS
S.update(mode="meaning", count=10, scope="all", recent=50, levels=set(), learned=False, time=10)

# ---- 切到測驗頁
win.set_view("quiz")
pump()
report(not win.search.isVisible() and win.page_title.isVisible() and win.stack.currentWidget() is qp,
       "切到測驗頁：搜尋列藏起來、顯示標題")
n_all = len([w for i, w in enumerate(WORDS) if i % 9 != 8])
report(f"{n_all} 個" in qp.count_label.text(), f"數量（不含已學會）：{qp.count_label.text()}")
S["scope"] = "marked"
qp.update_count()
report(f"{sum(1 for i in range(len(WORDS)) if i % 4 == 0 and i % 9 != 8)} 個" in qp.count_label.text(),
       f"只出重點單字：{qp.count_label.text()}")
S["scope"], S["levels"] = "all", {"N5"}
qp.update_count()
report(f"{sum(1 for i, w in enumerate(WORDS) if w[2] == 'N5' and i % 9 != 8)} 個" in qp.count_label.text(),
       f"只出 N5：{qp.count_label.text()}")
S["levels"] = set()
qp.update_count()


def start():
    t = time.perf_counter()
    QTest.mouseClick(qp.start_btn, Qt.LeftButton)
    while qp.pages.currentWidget() is not qp.question_page and time.perf_counter() - t < 30:
        pump(5)
    return time.perf_counter() - t


t = start()
report(len(qp.questions) == 10, f"出題：{len(qp.questions)} 題，{t:.2f} 秒（第一次要算向量）")
good = all(len({q.label(i) for i in range(4)}) == 4 and not any(
    qp.maker.same_meaning(q.target, o) for o in q.options if o is not q.target) for q in qp.questions)
report(good, "每題 4 個選項都不同、沒有和答案意思重疊的")

# ---- 數字鍵答對 → 自動下一題（鍵盤事件送給目前有焦點的元件，和實際使用一樣）
win.activateWindow()
pump()
report(QApplication.focusWidget() is qp, f"作答時鍵盤在測驗頁上（{type(QApplication.focusWidget()).__name__}）")
QTest.keyClick(QApplication.focusWidget() or qp, Qt.Key_1 + qp.q.answer)
pump(5)
report(qp.answered and qp.results[-1]["ok"] and qp.next_btn.isVisible(), "數字鍵答對")
QTest.qWait(quiz_ui.AUTO_NEXT_MS + 200)
report(qp.i == 1 and not qp.answered, "答對後自動下一題")

# ---- 答錯 → 這個也對
wrong = next(i for i in range(4) if i != qp.q.answer)
target, chosen = qp.q.target, qp.q.options[wrong]
QTest.mouseClick(qp.opts[wrong], Qt.LeftButton)
pump(5)
report(not qp.results[-1]["ok"] and qp.same_btn.isVisible(), "點錯的選項：答錯、出現「這個也對」")
QTest.mouseClick(qp.same_btn, Qt.LeftButton)
pump(5)
saved = quiz.SamePairs(d / quiz.SAME_NAME)
report(qp.results[-1]["ok"] and saved.has(key_of(target), key_of(chosen)) and qp.maker.same_meaning(target, chosen),
       f"按「這個也對」：算答對、記住「{target['word']}」和「{chosen['word']}」")
QTest.keyClick(QApplication.focusWidget() or qp, Qt.Key_Return)
pump(5)
report(qp.i == 2, "Enter：下一題")

# ---- 時間到
qp._t0 -= qp._limit + 1
pump(10)
report(qp.answered and qp.results[-1]["chosen"] is None and not qp.results[-1]["ok"] and not qp.same_btn.isVisible(),
       "時間到：算答錯、沒有「這個也對」")
QTest.keyClick(QApplication.focusWidget() or qp, Qt.Key_Return)
pump(5)
wrong = next(i for i in range(4) if i != qp.q.answer)
QTest.keyClick(QApplication.focusWidget() or qp, Qt.Key_1 + wrong)
pump(5)

# ---- 中途結束 → 結果 → 重測答錯的
QTest.mouseClick(next(b for b in qp.question_page.findChildren(quiz_ui.QPushButton) if b.text() == "結束"), Qt.LeftButton)
pump()
n_wrong = sum(not r["ok"] for r in qp.results)
report(qp.pages.currentWidget() is qp.result_page and len(qp.results) == 4 and n_wrong == 2,
       f"中途結束：結果 {qp.result_title.text()}，答錯 {n_wrong} 題")
QTest.mouseClick(qp.retry_btn, Qt.LeftButton)
pump()
report(qp.pages.currentWidget() is qp.question_page and len(qp.questions) == 2, "重測答錯的：2 題")

# ---- 看意思選單字
qp.back_to_setup()
S["mode"] = "word"
start()
report(all(q.mode == "word" and q.prompt == quiz.shown(q.target) for q in qp.questions), "看意思選單字：題目是意思")
S["mode"] = "mixed"
qp.back_to_setup()
start()
modes = {q.mode for q in qp.questions}
report(modes == {"meaning", "word"}, f"混合：{modes}")

# ---- 單字本改變時更新數量；離開測驗頁放掉模型
qp.back_to_setup()
store.add_record({"word": "窓口", "reading": "まどぐち", "meaning_zh": ["窗口、櫃台"]})
pump()
report(f"{n_all + 1} 個" in qp.count_label.text(), f"新增單字後數量更新：{qp.count_label.text()}")
win.set_view("words")
pump()
report(win.search.isVisible() and not win.page_title.isVisible() and qp.sem._sess is None
       and quiz.VEC_CACHE.exists(), "切回單字頁：搜尋列回來、模型放掉、向量存檔")

# ---- 準備題目時就離開：背景出好的題目不能自己開始作答、倒數（以前會在背景一題一題超時判錯）
win.set_view("quiz")
pump()
qp.back_to_setup()
QTest.mouseClick(qp.start_btn, Qt.LeftButton)
loading = qp.pages.currentWidget() is qp.loading_page
win.set_view("words")
for _ in range(400):
    pump(5)
    if not any(t.name == "quiz" for t in threading.enumerate()):
        break
pump()
report(loading and qp.pages.currentWidget() is qp.setup_page and not qp._timer.isActive() and not qp.results,
       "準備題目時切回單字頁：題目出好後不會在背景開始作答")

win.close()
print("全部通過" if ok else "有失敗的項目")
sys.exit(0 if ok else 1)
