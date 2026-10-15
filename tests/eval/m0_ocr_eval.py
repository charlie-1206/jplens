"""M0：比較各種辨識策略在合成樣本上的準確率與速度。

    python tests/eval/m0_ocr_eval.py            # 合成樣本
    python tests/eval/m0_ocr_eval.py 圖檔 ...    # 另外印出真實截圖的辨識結果（沒有標準答案）
    python tests/eval/m0_ocr_eval.py --fp32      # manga-ocr 改用全精度（預設是半精度）

時間取「第一次」辨識該圖的時間：實際使用時每次框選的尺寸都不同，不會有重複尺寸的快取效果。

準確率：去掉標點與空白後的字元準確率（1 - 編輯距離 / 標準答案長度）。
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from jplens.recognize import Recognizer  # noqa: E402
from jplens.jlearn import textutil  # noqa: E402
from samples import load_all  # noqa: E402

MODES = ["lines", "whole", "blocks"]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run(rec, img, mode):
    t = time.perf_counter()
    blocks = rec.recognize(img, mode)
    return "".join(b.text for b in blocks), time.perf_counter() - t


def main():
    t = time.perf_counter()
    rec = Recognizer()
    if "--fp32" in sys.argv:
        sys.argv.remove("--fp32")
        rec._mo.model.float()
    print(f"載入模型 {time.perf_counter() - t:.1f} 秒；manga-ocr：{rec.rec_device}")
    rec.warmup()

    real = sys.argv[1:]
    if real:
        for path in real:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            print(f"\n== {Path(path).name} {img.shape[1]}x{img.shape[0]}")
            for mode in MODES:
                text, sec = run(rec, img, mode)
                print(f"  {mode:<6} {sec * 1000:6.0f} ms  {text}")
        return

    total = {m: [0, 0, 0.0] for m in MODES}
    for name, img, gt in load_all():
        ref = textutil.norm(gt)
        print(f"\n== {name} {img.shape[1]}x{img.shape[0]}")
        for mode in MODES:
            text, sec = run(rec, img, mode)
            e = min(lev(textutil.norm(text), ref), len(ref))
            total[mode][0] += e
            total[mode][1] += len(ref)
            total[mode][2] += sec
            print(f"  {mode:<6} {100 - 100 * e / len(ref):5.1f}%  {sec * 1000:6.0f} ms  {text}")
    print("\n== 合計")
    for mode, (e, n, sec) in total.items():
        print(f"  {mode:<6} {100 - 100 * e / n:5.1f}%  平均 {sec / len(load_all()) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
