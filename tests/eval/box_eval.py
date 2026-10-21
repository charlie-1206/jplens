"""框選測試：和 app 相同的 plan_box → 整塊讀／依行讀，對照正確答案。

    python tests/eval/box_eval.py

圖在 tests/eval/real/（使用者保存的真實畫面，有著作權、不進版控）。
每個例子都是實際用的時候讀錯過的框：漫畫對話框漏欄、兩欄黏成一欄、振假名、框邊切到半個字。
同一個框也用漫畫模式讀一次（bubbles.pick 框到文字區塊就各自整塊讀，否則和框選相同）。
2026-10-15 起漫畫只用漫畫模式讀（版面補丁規則預設關閉，框選讀漫畫會變差）：漫畫的例子（cases.json 標 manga）
以漫畫模式計分，框選的結果只列出來參考；其他例子以框選計分。
"""
import json
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jplens.bubbles import BubbleDetector, pick  # noqa: E402
from jplens.layout import build_layout, plan_box  # noqa: E402
from jplens.recognize import Recognizer  # noqa: E402
from jplens.jlearn import textutil  # noqa: E402

R = ROOT / "tests" / "eval" / "real"
# 例子（框、正確答案）和圖放在一起，不進版控（正解是漫畫、小說的台詞，有著作權）：
# cases.json＝[{"file": 圖檔名, "box": [x0, y0, x1, y1], "truth": 正確答案, "manga": 漫畫（以漫畫模式計分）}]
# 2026-10-15 的 23 個：漫畫對話框漏欄、兩欄黏成一欄、振假名、框邊切到半個字（ちょくちょく）、
# BUG A（左邊那欄連振假名被框成寬框，這欄被當成振假名丟掉）、記事本小字（偵測整行漏掉，讀成「だ」）
CASES_PATH = R / "cases.json"
if not CASES_PATH.exists():
    sys.exit(f"找不到 {CASES_PATH}（使用者保存的真實畫面與正確答案，不進版控）")
CASES = [(str(R / c["file"]), tuple(c["box"]), c["truth"], c.get("manga", False))
         for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))]


def norm(t):
    t = unicodedata.normalize("NFKC", textutil.clean(t))
    return "".join(c for c in t if unicodedata.category(c)[0] in "LN")


rec = Recognizer()
rec.warmup()
det = BubbleDetector()
cache = {}


def run(f, box, manga=False):
    if f not in cache:
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        cache[f] = (img, build_layout(rec.detect_screen(img), img), det.detect(img))
    img, lines, bubbles = cache[f]
    if manga:
        hits = pick(bubbles, box)
        if hits:
            H, W = img.shape[:2]
            texts = [textutil.fix_common_ocr(textutil.clean(t)) for t in rec.read_regions(img, [b.crop(W, H) for b in hits])]
            return f"漫畫×{len(hits)}", "".join(texts)
    how, arg = plan_box(img, lines, box, rec.detect_screen, rec.read_segment_parts)  # 和 app 相同：橫排小字放大重新偵測
    if how == "whole":
        text = textutil.fix_common_ocr(textutil.clean(rec.read_whole(img, *arg)))
    elif how == "lines":
        text = rec.read_segments(img, lines, arg)
    elif how == "line":
        text = rec.read_segments(img, *arg)
    else:
        x0, y0, x1, y1 = box
        text = " / ".join(b.text for b in rec.recognize(img[y0:y1, x0:x1]))
    return how, text


ok = n = ok_m = n_m = ok_mb = 0
for f, box, truth, manga in CASES:
    how, text = run(f, box)
    good = norm(text) == norm(truth)
    if manga:
        ok_mb += good
    else:
        ok, n = ok + good, n + 1
    print(f"{'✓' if good else '✗'} {how:5s} {box[2] - box[0]}x{box[3] - box[1]}：{text}" + ("" if good else f"　（正解 {truth}）")
          + ("　〔漫畫用框選，參考〕" if manga else ""))
    if manga:
        how, text = run(f, box, manga=True)
        good = norm(text) == norm(truth)
        ok_m, n_m = ok_m + good, n_m + 1
        print(f"   {'✓' if good else '✗'} 漫畫模式 {how}：{text}")
print(f"框選（漫畫以外）{ok}/{n}；漫畫（漫畫模式）{ok_m}/{n_m}；參考：漫畫用框選 {ok_mb}/{n_m}")



# ---------------------------------------------------------------- 使用者保存畫面時填了正確答案的（app 的「保存上一次查詢的畫面」）
# 照當時的方式重跑：框選（box）、反白（text：起點終點，和 app 一樣起點是橫排小字就放大重新偵測）、
# 漫畫模式（manga：框或雙擊的位置）。反白的起點、終點和 app 一樣用 layout.plan_drag（直排小說走墨水投影）。
def run_saved(f, d):
    from jplens.bubbles import at
    from jplens.layout import plan_drag, small_text, zoom_strip
    if d["mode"] == "box" and "box" in d:
        return run(f, tuple(d["box"]))
    run(f, (0, 0, 1, 1))  # 確保版面、漫畫區塊已算好
    img, lines, bubbles = cache[f]
    H, W = img.shape[:2]
    if d["mode"] == "text":
        (sx, sy), end = d["start"], tuple(d["end"])
        if small_text(lines, (sx - 60, sy - 30, sx + 60, sy + 30)):
            found = zoom_strip(rec.detect_screen, img, lines, sy)
            lines = found[0] if found else lines
        tl, segs = plan_drag(img, lines, (sx, sy), end)
        return "反白", rec.read_segments(img, tl, segs) if segs else ""
    if d["mode"] == "manga":
        hits = pick(bubbles, tuple(d["box"])) if "box" in d else at(bubbles, *d["point"]) if "point" in d else []
        texts = [textutil.fix_common_ocr(textutil.clean(t)) for t in rec.read_regions(img, [b.crop(W, H) for b in hits])]
        return f"漫畫×{len(hits)}", "".join(texts)
    return "略過", ""


saved = []
for j in sorted(R.glob("2*.json")):
    d = json.loads(j.read_text(encoding="utf-8"))
    if isinstance(d, dict) and d.get("truth") and j.with_suffix(".png").exists():
        saved.append((str(j.with_suffix(".png")), d))
if saved:
    ok_s = 0
    for f, d in saved:
        how, text = run_saved(f, d)
        good = norm(text) == norm(d["truth"])
        ok_s += good
        print(f"{'✓' if good else '✗'} 保存 {Path(f).stem} {how}：{text}" + ("" if good else f"　（正解 {d['truth']}）"))
    print(f"保存的畫面（有正確答案）{ok_s}/{len(saved)}")
