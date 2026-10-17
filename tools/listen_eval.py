"""無字幕模式（jplens/listen.py 的 Transcriber）離線測試：影片聲音照即時的方式一小段一小段餵進去，和逐字稿比對。

    python tools/listen_eval.py 影片.mp4 逐字稿.txt [--rate 48000] [--chunk 0.05] [--partial 0]
    python tools/listen_eval.py 聲音.wav - --rate 48000      # 沒有逐字稿（只看切句、CPU），例如「保存最近 60 秒的聲音」存的檔

和 tools/asr_eval.py 的差別：走的是查字工具實際用的程式（原始取樣率進來、語音偵測用線性內插降取樣、
沒聲音時補靜音），所以可以確認即時版的準確率和評估時一樣。計分用 asr_eval 的 seg_cer。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def load_audio(path: str, rate: int) -> np.ndarray:
    import av
    out = []
    with av.open(path) as f:
        stream = f.streams.audio[0]
        res = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in f.decode(stream):
            for fr in res.resample(frame):
                out.append(fr.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    import wave
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main():
    import asr_eval
    from jplens.listen import Transcriber
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("transcript")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--chunk", type=float, default=0.05)  # 錄音一次送來幾秒（listen.py 是 1/20 秒）
    ap.add_argument("--partial", type=float, default=None)  # 暫定文字的間隔（0＝不送；預設照 listen.py）
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    audio = load_audio(a.video, a.rate) if not a.video.endswith(".wav") else load_wav(a.video)
    t = time.perf_counter()
    tr = Transcriber(a.rate) if a.partial is None else Transcriber(a.rate, partial_every=a.partial)
    print(f"載入 {time.perf_counter() - t:.1f} 秒")
    rows, times, partials = [], [], 0
    n = int(a.chunk * a.rate)
    c0 = time.process_time()
    for i in range(0, len(audio) + n, n):
        got = tr.feed(audio[i:i + n]) if i < len(audio) else tr.flush()
        for h in got:
            if not h.final:
                partials += 1
                continue
            times.append(h.sec)
            rows.append({"start": round(h.start, 1), "dur": 0, "sec": round(h.sec, 3), "text": h.text})
            if not a.quiet:
                print(f"[{h.start:6.1f}s 輸出於 {(i + n) / a.rate:6.1f}s] {h.text}")
    cpu = time.process_time() - c0
    total = len(audio) / a.rate
    if a.transcript == "-":
        print(f"聲音 {total:.0f} 秒、{len(rows)} 句、暫定 {partials} 次；CPU 每秒聲音 {cpu / total:.3f} 秒")
        return
    seg = asr_eval.seg_scores(rows, asr_eval.ref_lines(a.transcript))
    print(f"聲音 {total:.0f} 秒、{len(rows)} 句、暫定 {partials} 次；字元錯 {seg[0]:.1%}、讀音錯 {seg[1]:.1%}；"
          f"每句辨識中位數 {np.median(times):.2f} 秒；CPU 每秒聲音 {cpu / total:.3f} 秒")


if __name__ == "__main__":
    main()
