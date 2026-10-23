"""介面截圖（檢查英文介面的排版：文字有沒有被切掉、有沒有重疊），會開視窗：

    python tools/shot_ui.py en 輸出資料夾        # 英文介面（JPLENS_UI_LANG=en；解釋的語言也用英文）
    python tools/shot_ui.py zh-TW 輸出資料夾     # 中文介面（對照）

截：結果視窗、無字幕視窗、設定的每個分頁、單字本（單字、展開的卡片、篩選、文法、句子、單字測驗、匯出、手動新增）。
用暫存資料夾的假單字本，不碰真的單字本；不呼叫 Gemini。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "i18n_work" / f"shots_{LANG}")
os.environ["JPLENS_UI_LANG"] = LANG  # 要在 import jplens 的介面模組之前

from PySide6.QtCore import QCoreApplication, QEvent, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
from jplens import lang  # noqa: E402
from jplens.config import Config  # noqa: E402

Config.save = lambda self: None  # 截圖時切分頁、改設定都不寫回 config.json
lang.set_explain("en" if LANG == "en" else "zh-TW")
from jplens.dictionary import Dictionary  # noqa: E402
from jplens.listen_ui import ListenWindow  # noqa: E402
from jplens.manager import AddWordDialog, ExportDialog, ManagerWindow  # noqa: E402
from jplens.settings_ui import TABS, SettingsWindow  # noqa: E402
from jplens.ui import ResultPopup  # noqa: E402
from jplens.vocab import VocabStore  # noqa: E402
from jplens.words import extract  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
cfg = Config.load()
cfg.explain_lang = "en" if LANG == "en" else "zh-TW"


def pump(n=30):
    for _ in range(n):
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)  # 收合的卡片真的刪掉，不然會留在下一張截圖上
        QThread.msleep(10)


def shot(w, name):
    pump()
    w.grab().save(str(OUT / f"{name}.png"))
    print("  ", name)


d = Path(tempfile.mkdtemp())
SAMPLE = "要するに、彼は辛いカレーが好きじゃないってことだ"
words = extract([SAMPLE], Dictionary())
with open(d / "saved_words.jsonl", "w", encoding="utf-8") as f:
    for i, w in enumerate(words * 4):
        f.write(json.dumps({"word": w.word, "reading": w.reading, "pos": w.pos, "jlpt": w.jlpt if w.level else "",
                            "meaning_zh": w.zh, "meaning_en": w.en or [w.jlpt_en], "sentence": SAMPLE, "also": w.also,
                            "saved_at": f"2026-10-{20 + i % 3}T12:00:00", "marked": i % 3 == 0, "learned": i % 4 == 0},
                           ensure_ascii=False) + "\n")
with open(d / "saved_sentences.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps({"text": SAMPLE, "words": [w.word for w in words], "zh": "", "saved_at": "2026-10-22T12:00:00"},
                       ensure_ascii=False) + "\n")
store = VocabStore(d / "saved_words.jsonl")

print("結果視窗")
popup = ResultPopup(cfg, store, "")
popup.show_result([SAMPLE], words, 0.21, "")
popup.move(40, 40)
shot(popup, "popup")
popup._show_text(SAMPLE, editable=True)
popup._set_translation("Basically, he doesn't like spicy curry.")
shot(popup, "popup_translated")
popup.hide()

print("無字幕視窗")
lw = ListenWindow(cfg, store, "")
lw.start(cfg.listen_name)
lw.set_state("Speakers (Realtek Audio)", False)
for i, text in enumerate(["今日はいい天気ですね", "要するに、彼は辛いカレーが好きじゃないってことだ"]):
    lw.add_heard(i, text, True)
lw._query_line = lw._sel
lw.show_result([SAMPLE], words, 0.12)
shot(lw, "listen")
lw.hide()

print("設定")
sw = SettingsWindow(cfg)
sw.resize(980, 720)
sw.show()
for key, *_ in TABS:
    sw.show_tab(key)
    shot(sw, f"settings_{key}")
sw.hide()

print("單字本")
mw = ManagerWindow(store, cfg)
mw.resize(960, 720)
mw.show()
shot(mw, "manager_words")
if mw.model.rowCount():
    mw.toggle_row(0)
    shot(mw, "manager_expanded")
    mw.toggle_row(0)
mw.open_filter()
shot(mw, "manager_filter")
mw.toggle_drawer(False)
for view in ("grammar", "sentences", "quiz"):
    mw.set_view(view)
    shot(mw, f"manager_{view}")
mw.set_view("words")
for cls, name in ((ExportDialog, "export"), (AddWordDialog, "add_word")):
    try:
        dlg = cls(mw, "words", store, cfg) if cls is ExportDialog else cls(mw, Dictionary())
        dlg.show()
        shot(dlg, f"dialog_{name}")
        dlg.close()
    except Exception as e:  # noqa: BLE001（對話框的參數不同：截不到就跳過）
        print("   跳過", name, e)
print("存在", OUT)
