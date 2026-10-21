"""單字本清單（虛擬化：收合的卡片用畫的、展開的那張是真的 WordCard）的自動檢查，會開視窗：

    python tools/check_manager_list.py          # 3,000 個假單字
    python tools/check_manager_list.py 10000    # 指定數量

用暫存資料夾的假單字本，不碰真的單字本。用滑鼠、鍵盤事件實際去點，檢查：
速度（打開、重新整理、搜尋、標星號）、點星號／點卡片、展開的那張和那一列對齊、展開時按「標記已學會」、
視窗變窄、捲到中間後單字本被改（別處記了新單字）、搜尋把展開的那張濾掉、刪掉展開的那張再復原、最後一列展開、
Enter 鍵、在展開的卡片上捲滾輪、切換單字／文法／句子頁、每一列的位置是不是連續不重疊。
"""
import ctypes
import ctypes.wintypes
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from PySide6.QtCore import QCoreApplication, QPoint, Qt, QThread  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QToolButton  # noqa: E402

app = QApplication([])
from jplens.config import Config  # noqa: E402
from jplens.manager import ClickArea, ManagerWindow  # noqa: E402
from jplens.vocab import VocabStore, key_of  # noqa: E402
from jplens.words import WordInfo  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
ok = True


def pump(n=20):
    for _ in range(n):
        QCoreApplication.processEvents()
        QThread.msleep(5)


def report(good, msg):
    global ok
    ok &= bool(good)
    print(("OK  " if good else "NG  ") + msg)


def timed_nowait(f):
    t = time.perf_counter()
    f()
    QCoreApplication.processEvents()
    return time.perf_counter() - t


def timed(f):
    t = time.perf_counter()
    f()
    pump(3)
    return time.perf_counter() - t


d = Path(tempfile.mkdtemp())
with open(d / "saved_words.jsonl", "w", encoding="utf-8") as f:
    for i in range(N):
        f.write(json.dumps({
            "word": f"単語{i}", "reading": f"たんご{i}", "pos": "文法" if i % 50 == 49 else "名詞",
            "jlpt": ["N5", "N4", "N3", "N2", "N1", ""][i % 6], "meaning_zh": [f"意思{i}", "第二個意思" * (i % 3 + 1)],
            "meaning_en": ["meaning"], "sentence": f"これは単語{i}の例文です。" * (i % 4 + 1), "also": [],
            "saved_at": f"2024-01-01T00:00:{i % 60:02d}".replace("00:00", f"{(i // 3600) % 24:02d}:{(i // 60) % 60:02d}"),
            "marked": i % 7 == 0, "learned": i % 5 == 0}, ensure_ascii=False) + "\n")
store = VocabStore(d / "saved_words.jsonl")
cfg = Config.load()

win = ManagerWindow(store, cfg)
win.setWindowFlag(Qt.WindowStaysOnTopHint)  # 送真的滾輪時要在最上面
win.resize(900, 700)
t_open = timed(win.show)
pump()
view, model, bar = win.cards, win.model, win.cards.verticalScrollBar()
words = len([r for r in store.records() if r.get("pos") != "文法"])
report(model.rowCount() == words, f"單字頁 {model.rowCount()} 列（應為 {words}）")
report(t_open < 1.0, f"打開 {t_open:.2f} 秒")
report((t := timed(win.refresh)) < 0.5, f"重新整理 {t:.2f} 秒")


def rect(row):
    return view.visualRect(model.index(row))


def click(pt, button=Qt.LeftButton):
    QTest.mouseClick(view.viewport(), button, Qt.NoModifier, pt)
    pump()


def star_point(row):
    r = rect(row)
    return QPoint(r.left() + 16 + 1 + 4 + 20, r.top() + (2 if row == 0 else 0) + 22)


def body_point(row):
    r = rect(row)
    return QPoint(r.left() + 300, r.top() + (2 if row == 0 else 0) + 30)


def check_rows(tag):
    """看得到的每一列：位置連續、不重疊；展開那張的外框和它那一列一樣大。"""
    first = view.indexAt(QPoint(10, 1)).row()
    last = view.indexAt(QPoint(10, view.viewport().height() - 2)).row()
    if first < 0:
        report(False, f"{tag}：找不到看得到的列")
        return
    last = model.rowCount() - 1 if last < 0 else last
    good = all(rect(i).bottom() + 1 == rect(i + 1).top() for i in range(first, last))
    box = win._open_box
    # Qt 內部記著的元件：只能有展開的那一列（收合時自己刪元件、沒告訴 Qt 的話會留一筆，之後重新排版就閃退，10-21 的 BUG）
    QCoreApplication.processEvents()
    stray = [i for i in range(first, last + 1) if view.indexWidget(model.index(i)) is not None
             and view.indexWidget(model.index(i)) is not box]
    if stray:
        report(False, f"{tag}：第 {stray} 列還掛著已經收合的卡片")
    good &= not stray
    if box is not None:
        row = model.row_of(win.expanded)
        good &= box.geometry() == rect(row) and view.indexWidget(model.index(row)) is box
        card = box.layout().itemAt(0).widget()
        good &= abs(card.height() - win.open_height(row)) <= 1
    report(good, f"{tag}：第 {first}～{last} 列位置連續、展開的卡片對齊")


# ---- 點星號（收合的卡片）
rec = model.recs[3]
was = bool(rec.get("marked"))
t = timed(lambda: click(star_point(3)))
now = next(r for r in store.records() if key_of(r) == key_of(rec))
report(now.get("marked") == (not was) and model.recs[3].get("marked") == (not was) and win.expanded is None,
       f"點第 3 列的星號：重點 {was} → {now.get('marked')}，沒有展開（{t:.2f} 秒）")

# ---- 點卡片展開、再點標題收合
click(body_point(2))
report(win.expanded == key_of(model.recs[2]) and win._open_box is not None, "點第 2 列：展開")
check_rows("展開第 2 列")
card = win._open_box.layout().itemAt(0).widget()
head = card.findChild(ClickArea)
QTest.mouseClick(head, Qt.LeftButton, Qt.NoModifier, QPoint(200, 10))
pump()
report(win.expanded is None and win._open_box is None, "點展開那張的標題：收合")
check_rows("收合後")

# ---- 展開 A 再點 B
click(body_point(1))
click(body_point(4) if win.open_height(1) is None else QPoint(300, rect(4).top() + 30))
report(win.expanded == key_of(model.recs[4]) and model.row_of(win.expanded) == 4, "展開第 1 列後點第 4 列：只有第 4 列展開")
check_rows("展開第 4 列")

# ---- 展開的卡片按「標記已學會」
card = win._open_box.layout().itemAt(0).widget()
btn = next(b for b in card.findChildren(QPushButton) if "已學會" in b.text())
before = bool(model.recs[4].get("learned"))
QTest.mouseClick(btn, Qt.LeftButton)
pump()
after = next(r for r in store.records() if key_of(r) == win.expanded).get("learned")
card = win._open_box.layout().itemAt(0).widget()
report(after == (not before) and win._open_box is not None and any(
    ("取消已學會" if after else "標記已學會") == b.text() for b in card.findChildren(QPushButton)),
    f"展開的卡片按已學會：{before} → {after}，卡片還開著、按鈕文字更新")
check_rows("按已學會後")

# ---- 視窗變窄（展開那張變高）
h1 = win.open_height(4)
win.resize(560, 700)
pump()
h2 = win.open_height(4)
report(h2 >= h1, f"視窗變窄：展開的卡片高度 {h1} → {h2}")
check_rows("變窄後")
win.resize(900, 700)
pump()
check_rows("變回原本寬度")

# ---- 捲到中間後，別的地方記了新單字（單字本改變 → 重新整理）
bar.setValue(bar.maximum() // 2)
pump()
mid = bar.value()
row = view.indexAt(QPoint(300, 50)).row()
click(QPoint(300, rect(row).top() + 30))
key = win.expanded
t = timed_nowait(lambda: store.add(WordInfo(word="新しい", reading="あたらしい", pos="い形容詞", level=5, proper=False,
                                      sentence="新しい単語", zh=["新的"])))
pump()
report(win.expanded == key and win._open_box is not None and abs(bar.value() - mid) <= rect(0).height() + 50,
       f"捲到中間（{mid}）展開一張後記了新單字：展開的還在、捲動位置 {bar.value()}（{t:.2f} 秒）")
check_rows("記新單字後")

# ---- 搜尋把展開的那張濾掉、再清掉
for ch in "単語12":  # 一個字一個字打（QTest.keyClick 送漢字會變亂碼並讓程式當掉，所以直接插進輸入框）
    win.search.insert(ch)
    QTest.qWait(30)
QTest.qWait(300)
pump()
n = model.rowCount()
want = sum(1 for r in store.records() if "単語12" in r["word"] and r.get("pos") != "文法")
report(n == want and all("単語12" in r["word"] for r in model.recs), f"搜尋「単語12」：{n} 列（應為 {want}）")
check_rows("搜尋後")
t = timed(lambda: (win.search.clear(), QTest.qWait(250)))
report(model.rowCount() == words + 1, f"清掉搜尋：{model.rowCount()} 列（{t:.2f} 秒，含等打字的 0.15 秒）")

# ---- 刪掉展開的那張、復原
bar.setValue(0)
pump()
click(body_point(0))
rec0 = model.recs[0]
res = store.delete(key_of(rec0))
pump()
report(win._open_box is None and model.row_of(key_of(rec0)) < 0, "刪掉展開的第 0 列：清單更新、沒有殘留的卡片")
check_rows("刪除後")
store.restore(*res)
pump()
report(model.row_of(key_of(rec0)) == 0 and win.expanded == key_of(rec0) and win._open_box is not None,
       "復原：回到第 0 列、照樣展開")
check_rows("復原後（第 0 列展開）")

# ---- 最後一列展開
bar.setValue(bar.maximum())
pump()
last = model.rowCount() - 1
click(QPoint(300, rect(last).top() + 30))
bar.setValue(bar.maximum())
pump()
report(win.expanded == key_of(model.recs[last]), "最後一列展開")
check_rows("最後一列展開")
box = win._open_box
gap = view.viewport().height() - (box.geometry().bottom() + 1) if box else -1
report(box is not None and abs(rect(last).bottom() + 1 - view.viewport().height()) <= 1,
       f"捲到底：最後一列貼齊底部（剩 {gap} 像素，含 16 像素留白在列裡）")

# ---- Enter 鍵
bar.setValue(0)
pump()
view.setFocus()
view.setCurrentIndex(model.index(5))
QTest.keyClick(view, Qt.Key_Return)
pump()
report(win.expanded == key_of(model.recs[5]), "Enter：展開目前那張")
QTest.keyClick(view, Qt.Key_Return)
pump()
report(win.expanded is None, "再按 Enter：收合")
# 方向鍵停在第 5 列、滑鼠點開第 2 列，按 Enter 要收合第 2 列（以前動到的是第 5 列）
QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, rect(2).center())
pump()
QTest.keyClick(view, Qt.Key_Return)
pump()
report(win.expanded is None and view.currentIndex().row() == 2, "滑鼠點開後按 Enter：收合點開的那張")
# 展開的卡片有朗讀按鈕：單字（唸讀音）、原句
win.toggle_row(2)
pump()
tips = [b.toolTip() for b in win._open_box.findChildren(QToolButton) if b.toolTip().startswith("朗讀")]
report(tips == [f"朗讀「{model.recs[2]['reading']}」", "朗讀原句"], f"展開的卡片有朗讀按鈕：{tips}")
win.toggle_row(2)
pump()
# 鍵盤焦點在展開卡片的標題上按 Enter 收合：焦點回到清單（以前卡片拿掉後焦點就不見了）
win.toggle_row(2)
pump()
area = win._open_box.findChild(ClickArea)
area.setFocus()
QTest.keyClick(area, Qt.Key_Return)
pump()
report(win.expanded is None and QApplication.focusWidget() is view and view.currentIndex().row() == 2,
       f"標題上按 Enter 收合：焦點回到清單（{type(QApplication.focusWidget()).__name__}）")
# 最窄的視窗、長的單字：展開的卡片最小寬度不能比實際寬，不然版面被硬壓、原句高度算錯被切掉
w0 = win.size()
model.recs[3] = {**model.recs[3], "word": "〜であれ／〜であろうと", "reading": "であれ・であろうと",
                 "sentence": "雨であれ雪であれ、明日の試合は必ず予定どおりに行うと監督は皆の前で言った。"}
win.resize(win.minimumWidth(), w0.height())
pump()
win.toggle_row(3)
pump()
card = win._open_box.layout().itemAt(0).widget()
jp = next(lb for lb in card.findChildren(QLabel) if lb.objectName() == "jp")
report(card.minimumSizeHint().width() <= card.width() and jp.height() >= jp.heightForWidth(jp.width()),
       f"最窄視窗展開長單字：卡片最小寬 {card.minimumSizeHint().width()} ≤ {card.width()}、原句 {jp.height()} ≥ {jp.heightForWidth(jp.width())}")
win.toggle_row(3)
win.refresh()
win.resize(w0)
pump()

# ---- 在展開的卡片上捲滾輪
# Qt 6 不讓程式自己送的滾輪事件往上層傳（QApplication::notify），所以用 Windows 送真的滾輪：
# 視窗暫時置頂、滑鼠移到展開的卡片上捲一格，再把滑鼠移回原位
click(body_point(1))
card = win._open_box.layout().itemAt(0).widget()
u = ctypes.windll.user32
old = ctypes.wintypes.POINT()
u.GetCursorPos(ctypes.byref(old))
results = []
for name, pt in (("單字", QPoint(100, 25)), ("意思", QPoint(120, 80)), ("原句", QPoint(300, card.height() - 80))):
    v0 = bar.value()
    g = card.mapToGlobal(pt)
    ratio = card.devicePixelRatioF()
    u.SetCursorPos(int(g.x() * ratio), int(g.y() * ratio))
    pump(5)
    u.mouse_event(0x0800, 0, 0, ctypes.c_uint32(-120 & 0xFFFFFFFF).value, 0)  # MOUSEEVENTF_WHEEL，往下一格
    pump(15)
    results.append(f"{name} {v0}→{bar.value()}")
    good = bar.value() > v0
    bar.setValue(0)
    pump()
    card = win._open_box.layout().itemAt(0).widget()
    if not good:
        break
u.SetCursorPos(old.x, old.y)
report(good, "在展開的卡片上捲滾輪，清單會捲動：" + "、".join(results))

# ---- 展開的卡片上的按鈕：編輯（改了單字本身，鍵也變了）、複製、刪除後復原
import jplens.manager as M  # noqa: E402


def press(text):
    card = win._open_box.layout().itemAt(0).widget()
    QTest.mouseClick(next(b for b in card.findChildren(QPushButton) if b.text() == text), Qt.LeftButton)
    pump()


bar.setValue(0)
pump()
if win.expanded is None:
    click(body_point(1))
row = model.row_of(win.expanded)


class FakeEdit:
    def __init__(self, parent, rec):
        self.rec = rec

    def exec(self):
        return M.QDialog.Accepted

    def values(self):
        return {"word": self.rec["word"] + "改", "reading": self.rec["reading"], "meaning_zh": ["改過的意思"],
                "sentence": self.rec.get("sentence", "")}


real_edit, M.EditDialog = M.EditDialog, FakeEdit
old_word = model.recs[row]["word"]
press("編輯")
M.EditDialog = real_edit
new_row = model.row_of(win.expanded)
report(win.expanded and win.expanded.startswith(old_word + "改") and win._open_box is not None
       and model.recs[new_row]["meaning_zh"] == ["改過的意思"], f"編輯「{old_word}」→「{old_word}改」：照樣展開、意思更新")
check_rows("編輯後")
press("複製")
report(QApplication.clipboard().text().startswith(old_word + "改"), f"複製：{QApplication.clipboard().text()[:30]}")
real_confirm, win._confirm_delete = win._confirm_delete, lambda what: True
key = win.expanded
press("刪除")
win._confirm_delete = real_confirm
report(model.row_of(key) < 0 and win._open_box is None and win.undo_btn.isVisible(), "刪除：從清單消失、出現「復原」")
QTest.mouseClick(win.undo_btn, Qt.LeftButton)
pump()
report(model.row_of(key) >= 0 and win.expanded == key and win._open_box is not None, "按復原：回來、照樣展開")
check_rows("復原後")

# ---- 排序、篩選
for i in range(win.sort.count()):
    t = timed_nowait(lambda: win.sort.setCurrentIndex(i))
    report(model.rowCount() == words + 1, f"排序「{win.sort.currentText()}」：{t:.2f} 秒")
    check_rows(f"排序「{win.sort.currentText()}」")
win.sort.setCurrentIndex(0)
win.f["levels"] = {"N1"}
win.f["marked"] = "marked"
win.refresh()
pump()
want = sum(1 for r in store.records() if r.get("jlpt") == "N1" and r.get("marked") and r.get("pos") != "文法")
report(model.rowCount() == want, f"篩選 N1＋重點單字：{model.rowCount()} 列（應為 {want}）")
check_rows("篩選後")
win.clear_filters()
pump()

# ---- 切換頁面
for v in ("grammar", "sentences", "words"):
    t = timed(lambda: win.set_view(v))
    shown = win.stack.currentWidget() is (win.scroll if v == "sentences" else win.cards)
    report(shown, f"切到 {v}：{t:.2f} 秒，顯示對的清單（{model.rowCount()} 列）")
check_rows("切回單字頁")

# ---- 速度：標星號、整頁捲動
t = time.perf_counter()
QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, star_point(2))
QCoreApplication.processEvents()
t = time.perf_counter() - t
report(t < 0.2, f"標星號 {t:.2f} 秒（含寫回單字本檔案）")
t0 = time.perf_counter()
for i in range(20):
    bar.setValue(bar.maximum() * i // 20)
    view.viewport().repaint()
dt = (time.perf_counter() - t0) / 20
report(dt < 0.05, f"捲動重畫一頁 {dt * 1000:.1f} 毫秒")

win.close()
print("全部通過" if ok else "有失敗的項目")
sys.exit(0 if ok else 1)
