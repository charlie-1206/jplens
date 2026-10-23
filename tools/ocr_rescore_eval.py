"""manga-ocr 候選重新評分（recognize.pick_candidate）的權重評估：把真的日文句子畫成文字圖，比較各權重的辨識結果。

    python tools/ocr_rescore_eval.py [句數]          # 預設 1500 句 × 6 種字型／樣式

句子：維基詞典例句（data/raw/zhwikt_ja.jsonl，和 reading_eval 同一份），長度 6～30 字，打亂後取前 N 句。
偶數句是「調參」組、奇數句是「驗證」組：權重只看調參組決定，驗證組用來確認沒有看著答案調。
每一行照 app 的方式讀（_gray_for_rec → read_long：長行切段、接縫重讀）；beam search 的候選有快取，換權重不用重跑模型。
印出每個權重：整句全對的句數、字錯誤率（編輯距離 ÷ 字數）、和權重 0（現在的做法）比變好／變差的句數，
以及驗證組變差的例子。字型用 Windows 內建的日文字型（遊ゴシック、MS ゴシック、HG 系列）。
"""
import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from jplens import recognize  # noqa: E402
from jplens.jlearn import textutil  # noqa: E402

FONTS = Path("C:/Windows/Fonts")
# (名稱, 字型檔, 字的大小, 樣式)：一般網頁／介面、小字、明朝、教科書體、字幕（白字黑邊、彩色背景）、深色背景
STYLES = [("ゴシック", "YuGothM.ttc", 30, "plain"), ("小字", "msgothic.ttc", 17, "plain"),
          ("明朝", "HGRMB.TTC", 34, "plain"), ("教科書", "HGRKK.TTC", 38, "dark"),
          ("字幕", "HGRSGU.TTC", 50, "subtitle"), ("ポップ", "HGRPP1.TTC", 40, "plain")]
WEIGHTS = [0, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4]


def sentences(n: int) -> list[str]:
    seen = []
    for line in open(ROOT / "data" / "raw" / "zhwikt_ja.jsonl", encoding="utf-8"):
        for s in json.loads(line).get("senses", []):
            for ex in s.get("examples", []):
                t = "".join((ex.get("text") or "").split())
                if 6 <= len(t) <= 30 and _japanese(t):
                    seen.append(t)
    seen = list(dict.fromkeys(seen))
    random.Random(0).shuffle(seen)
    return seen[:n]


def _japanese(t: str) -> bool:
    """日文句子：有假名，而且每個字都在日文字元集（cp932）裡（例句偶爾混著中文翻譯：我们、莫名其妙，离奇）。"""
    if "，" in t or not any("぀" <= c <= "ヿ" for c in t):
        return False
    try:
        t.encode("cp932")
    except UnicodeEncodeError:
        return False
    return True


def render(text: str, font: str, size: int, style: str, rnd: random.Random) -> np.ndarray:
    f = ImageFont.truetype(str(FONTS / font), size)
    stroke = max(2, size // 12) if style == "subtitle" else 0
    x0, y0, x1, y1 = f.getbbox(text, stroke_width=stroke)
    pad_y = max(3, size // 6)
    w, h = x1 - x0 + 4, y1 - y0 + 2 * pad_y
    if style == "subtitle":  # 動畫畫面：柔和的彩色漸層
        a, b = (np.array([rnd.randint(120, 255) for _ in range(3)], np.float32) for _ in range(2))
        grad = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        im = Image.fromarray(np.repeat((a * (1 - grad) + b * grad).astype(np.uint8), h, 0))
        fill, stroke_fill = (255, 255, 255), (20, 20, 20)
    elif style == "dark":
        im, fill, stroke_fill = Image.new("RGB", (w, h), (34, 36, 40)), (225, 225, 225), None
    else:
        im, fill, stroke_fill = Image.new("RGB", (w, h), (255, 255, 255)), (20, 20, 20), None
    ImageDraw.Draw(im).text((2 - x0, pad_y - y0), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    if style == "subtitle":
        im = im.filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(im)[:, :, ::-1].copy()  # BGR，和螢幕截圖相同


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", textutil.clean(t))
    return "".join(c for c in t if unicodedata.category(c)[0] in "LN")


def edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    rec = recognize.Recognizer()
    rec.warmup()
    mo = rec._mo
    memo: dict[bytes, list] = {}
    raw = mo.candidates

    def cached(grays):
        keys = [hashlib.blake2b(g.tobytes() + bytes(str(g.shape), "ascii"), digest_size=16).digest() for g in grays]
        todo = [i for i, k in enumerate(keys) if k not in memo]
        for i, c in zip(todo, raw([grays[i] for i in todo]) if todo else []):
            memo[keys[i]] = c
        return [memo[k] for k in keys]
    mo.candidates = cached

    sents = sentences(n)
    rnd = random.Random(1)
    items = []  # (句子編號, 樣式名, 灰階圖)
    for k, s in enumerate(sents):
        for name, font, size, style in STYLES:
            items.append((k, name, recognize._gray_for_rec(render(s, font, size, style, rnd))))
    print(f"{len(sents)} 句 × {len(STYLES)} 種樣式 = {len(items)} 行")

    results = {}
    for w in WEIGHTS:
        mo.rescore = lambda c, w=w: recognize.pick_candidate(c, w)
        out = []
        for i in range(0, len(items), 64):
            part = items[i:i + 64]
            out += rec.read_long([g for _, _, g in part], [False] * len(part))
        results[w] = out
        print(f"  權重 {w:g} 完成（快取 {len(memo)} 張）", flush=True)

    truth = [norm(sents[k]) for k, _, _ in items]
    base = [norm(t) for t in results[0]]
    for split, keep in (("調參（偶數句）", 0), ("驗證（奇數句）", 1)):
        idx = [i for i, (k, _, _) in enumerate(items) if k % 2 == keep]
        chars = sum(len(truth[i]) for i in idx)
        print(f"\n{split}：{len(idx)} 行、{chars} 字")
        for w in WEIGHTS:
            got = [norm(results[w][i]) for i in idx]
            errs = [edit(g, truth[i]) for g, i in zip(got, idx)]
            berrs = [edit(base[i], truth[i]) for i in idx]
            better = sum(e < b for e, b in zip(errs, berrs))
            worse = sum(e > b for e, b in zip(errs, berrs))
            exact = sum(e == 0 for e in errs)
            print(f"  權重 {w:<7g} 全對 {exact:5d}（{exact / len(idx):.2%}）  字錯誤率 {sum(errs) / chars:.3%}"
                  f"  比現在好 {better:3d}、差 {worse:3d}")
        by_style = {}
        for i in idx:
            by_style.setdefault(items[i][1], []).append(i)
        print("  各樣式的字錯誤率（權重 0 → 各權重）：")
        for name, ii in by_style.items():
            c = sum(len(truth[i]) for i in ii)
            print(f"    {name:6s} " + "  ".join(f"{sum(edit(norm(results[w][i]), truth[i]) for i in ii) / c:.2%}"
                                             for w in WEIGHTS))
    for w in WEIGHTS[1:]:
        idx = [i for i, (k, _, _) in enumerate(items) if k % 2 == 1]
        diff = [i for i in idx if norm(results[w][i]) != base[i]]
        worse = [i for i in diff if edit(norm(results[w][i]), truth[i]) > edit(base[i], truth[i])]
        better = [i for i in diff if edit(norm(results[w][i]), truth[i]) < edit(base[i], truth[i])]
        print(f"\n權重 {w:g} 驗證組：改了 {len(diff)} 行（好 {len(better)}、差 {len(worse)}）")
        for tag, lst in (("差", worse), ("好", better)):
            for i in lst[:8]:
                print(f"  {tag} [{items[i][1]}] {results[0][i]} → {results[w][i]}　（正解 {sents[items[i][0]]}）")


if __name__ == "__main__":
    main()
