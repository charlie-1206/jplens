"""從螢幕錄影（有燒錄字幕）做無字幕模式的評估資料：聲音存成 wav、字幕用查字工具的 OCR 讀成正確台詞。

    python tools/sub_truth.py 錄影.mp4 名稱 [--fps 2] [--band 0.755 0.895] [--x 0.05 0.95]

輸出 tests/eval/real/audio/名稱.wav（48 kHz 單聲道）與 名稱_truth.txt（格式同 listen_compare：[mm:ss] 台詞）。
--band：字幕所在的高度範圍（畫面高度的比例；YouTube 全螢幕的蓮ノ空影片是 0.755～0.895，不含下面的［角色名］）。
字幕 OCR 不保證全對（會有幾個錯字），只看相對變化；要更準可以手動修 _truth.txt。
另外輸出 名稱_subs.tsv（每一則字幕的開始、結束秒數），找「哪一句表現不好」時用。
"""
import argparse
import difflib
import sys
import unicodedata
import wave
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "eval" / "real" / "audio"


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    return "".join(c for c in t if unicodedata.category(c)[0] in "LN")


def main():
    import av
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("name")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--band", type=float, nargs=2, default=(0.755, 0.895))
    ap.add_argument("--x", type=float, nargs=2, default=(0.05, 0.95))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 聲音
    chunks = []
    with av.open(a.video) as f:
        res = av.AudioResampler(format="s16", layout="mono", rate=48000)
        for fr in f.decode(audio=0):
            for r in res.resample(fr):
                chunks.append(r.to_ndarray().reshape(-1))
    pcm = np.concatenate(chunks).astype(np.int16)
    with wave.open(str(OUT / f"{a.name}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(pcm.tobytes())
    print(f"聲音 {len(pcm) / 48000:.1f} 秒")

    # 字幕
    from jplens.recognize import Recognizer
    rec = Recognizer()
    reads = []  # (秒, 文字)
    nxt = 0.0
    with av.open(a.video) as f:
        for fr in f.decode(video=0):
            t = float(fr.pts * fr.time_base)
            if t < nxt:
                continue
            nxt = t + 1 / a.fps
            img = fr.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            crop = np.ascontiguousarray(img[int(h * a.band[0]):int(h * a.band[1]), int(w * a.x[0]):int(w * a.x[1])])
            text = "".join(b.text for b in rec.recognize(crop))
            reads.append((t, text))
    # 連續相似的讀數併成一則字幕（每則取出現最多次的讀法）
    subs = []  # [開始, 結束, {讀法: 次數}]
    for t, text in reads:
        n = norm(text)
        if len(n) < 2:
            continue
        if subs and t - subs[-1][1] <= 1.5 / a.fps:
            best = max(subs[-1][2], key=subs[-1][2].get)
            if difflib.SequenceMatcher(None, norm(best), n).ratio() >= 0.6:
                subs[-1][1] = t
                subs[-1][2][text] = subs[-1][2].get(text, 0) + 1
                continue
        subs.append([t, t, {text: 1}])
    lines, tsv = [], []
    for s, e, c in subs:
        best = max(c, key=lambda k: (c[k], len(k)))
        if e - s < 0.4 and c[best] < 2:  # 只閃一下（換場、OCR 雜訊）
            continue
        lines.append(f"[{int(s) // 60:02d}:{int(s) % 60:02d}] {best}")
        tsv.append(f"{s:.1f}\t{e + 1 / a.fps:.1f}\t{best}")
    head = f"# 由 tools/sub_truth.py 從 {Path(a.video).name} 的燒錄字幕 OCR 產生（可能有錯字）\n"
    (OUT / f"{a.name}_truth.txt").write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"{a.name}_subs.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    print(f"字幕 {len(lines)} 則（讀了 {len(reads)} 張）")


if __name__ == "__main__":
    main()
