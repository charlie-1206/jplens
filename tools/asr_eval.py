"""無字幕模式的評估：影片聲音 → 語音偵測切句 → 語音辨識，和逐字稿比對。

    python tools/asr_eval.py 影片.mp4 逐字稿.txt --engine reazon|whisper [--start 秒] [--dur 秒] [--out 結果.json]

- 語音偵測：Silero VAD（sherpa-onnx），照即時的方式一小段一小段餵進去，一句講完才送去辨識。
- reazon ：ReazonSpeech k2 v2（sherpa-onnx 的 zipformer transducer，CPU）。data/models/asr/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/
- whisper：kotoba-whisper v2.0（faster-whisper／CTranslate2 int8，GPU）。data/models/asr/kotoba-whisper-v2.0-ct2/
  （由 HF 的 kotoba-tech/kotoba-whisper-v2.0 用 ct2-transformers-converter --quantization int8_float16 轉成）
- 逐字稿：video_translate 的格式「[mm:ss] 說話者｜台詞」（字幕 OCR 來的，偶爾有雜訊行）。取時間範圍內的台詞接起來當正確答案。
- 計分：逐字稿有很多沒配音的台詞（主角、部分配角），所以主要看 seg_cer：每句辨識結果和同一段時間的台詞做局部比對
  （字元錯誤率，去掉標點、全半形統一；分母是辨識出的字數）。seg_cer_reading 兩邊都先用 Sudachi 轉成片假名讀音再比，
  寫法不同（わたし／私）不算錯。漏掉、沒辨識到的句子量不到。cer_all 是和整份逐字稿比，只當參考。
- 每個引擎開一個程序跑（記憶體才量得準）：記錄載入時間、每句辨識時間、RAM（USS）、整張顯示卡已用的記憶體變化。
"""
import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASR = ROOT / "data" / "models" / "asr"
SR = 16000
PREROLL = 0.3  # 每句往前多帶幾秒
PAD = 0.2      # 每句頭尾補幾秒靜音


def load_audio(path: str, start: float, dur: float | None) -> np.ndarray:
    import av
    out = []
    with av.open(path) as f:
        stream = f.streams.audio[0]
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        if start:
            f.seek(int(start / stream.time_base), stream=stream)
        for frame in f.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0
            if dur is not None and t > start + dur:
                break
            for fr in res.resample(frame):
                out.append(fr.to_ndarray().reshape(-1))
    a = np.concatenate(out).astype(np.float32)
    # seek 會跳到前一個關鍵幀：裁掉 start 之前的部分（約略）
    return a[: int(dur * SR)] if dur is not None else a


def vad_segments(audio: np.ndarray) -> tuple[list[tuple[float, np.ndarray]], float]:
    """照即時的方式每 512 個樣本（32 ms）餵一次，回傳 [(開始秒數, 這句的聲音)] 和 VAD 花的 CPU 時間。"""
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(ASR / "silero_vad.onnx")
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.4   # 停頓 0.4 秒就算一句結束
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = 15     # 一直講超過 15 秒也切一次
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    spans = []
    t = time.process_time()
    win = cfg.silero_vad.window_size
    for i in range(0, len(audio), win):
        vad.accept_waveform(audio[i:i + win])
        while not vad.empty():
            spans.append((vad.front.start, vad.front.start + len(vad.front.samples)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        spans.append((vad.front.start, vad.front.start + len(vad.front.samples)))
        vad.pop()
    # 語音偵測判定「開始講話」時第一個音節常已經過了一點：往前多帶 PREROLL 秒（即時的時候就是保留最近的一小段緩衝），
    # 頭尾再補 PAD 秒的靜音（ReazonSpeech 沒有前後靜音時常漏掉第一個詞）
    pre, pad = int(PREROLL * SR), np.zeros(int(PAD * SR), np.float32)
    segs = [(max(0, a - pre) / SR, np.concatenate([pad, audio[max(0, a - pre):b], pad])) for a, b in spans]
    return segs, time.process_time() - t


class Reazon:
    def __init__(self, threads=2):
        import sherpa_onnx
        d = ASR / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
        pick = lambda stem: str(next(iter(sorted(d.glob(f"{stem}*int8*.onnx"))), None) or next(d.glob(f"{stem}*.onnx")))
        self.files = {k: pick(k) for k in ("encoder", "decoder", "joiner")}
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self.files["encoder"], decoder=self.files["decoder"], joiner=self.files["joiner"],
            tokens=str(d / "tokens.txt"), num_threads=threads, sample_rate=SR, feature_dim=80,
            decoding_method="greedy_search")
        self.name = f"ReazonSpeech k2 v2（int8、CPU {threads} 執行緒）"

    def __call__(self, a: np.ndarray) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(SR, a)
        self.rec.decode_stream(s)
        return s.result.text


class Whisper:
    def __init__(self, beam=1):
        from jplens import gpu
        gpu.cuda_ready()  # 先載入 CUDA／cuDNN 的 DLL（CTranslate2 也要 cuBLAS 12、cuDNN 9）
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(ASR / "kotoba-whisper-v2.0-ct2"), device="cuda", compute_type="int8_float16")
        self.beam = beam
        self.name = f"kotoba-whisper v2.0（int8、GPU、beam {beam}）"

    def __call__(self, a: np.ndarray) -> str:
        segs, _ = self.model.transcribe(a, language="ja", task="transcribe", beam_size=self.beam,
                                        condition_on_previous_text=False, vad_filter=False,
                                        without_timestamps=True)
        return "".join(s.text for s in segs)


# ---------------------------------------------------------------- 計分
def reference(path: str, start: float, end: float) -> str:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[(\d+):(\d+)\]\s*(.*)", line)
        if not m:
            continue
        t = int(m.group(1)) * 60 + int(m.group(2))
        if start - 2 <= t <= end + 2:
            out.append(m.group(3).split("｜")[-1])
    return "".join(out)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    return "".join(c for c in t if unicodedata.category(c)[0] in "LN")


_tok = None


def reading(t: str) -> str:
    global _tok
    if _tok is None:
        from sudachipy import dictionary
        _tok = dictionary.Dictionary().create()
    t = norm(t)
    return "".join(m.reading_form() for m in _tok.tokenize(t)) if t else ""


def cer(hyp: str, ref: str) -> float:
    if not ref:
        return 0.0
    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        cur = [i] + [0] * len(ref)
        for j, r in enumerate(ref, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (h != r))
        prev = cur
    return prev[-1] / len(ref)


def ref_lines(path: str) -> list[tuple[int, str]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[(\d+):(\d+)\]\s*(.*)", line)
        if m:
            out.append((int(m.group(1)) * 60 + int(m.group(2)), m.group(3).split("｜")[-1]))
    return out


def local_errors(hyp: str, ref: str) -> int:
    """hyp 對 ref 裡最像的一段（ref 頭尾可以略過）的編輯距離。逐字稿有很多沒配音的台詞（主角、部分配角），
    只比「唸出來的那句」辨識得準不準；漏掉沒辨識到的句子這裡量不到。"""
    if not hyp:
        return 0
    prev = [0] * (len(ref) + 1)  # ref 開頭可以略過
    for i, h in enumerate(hyp, 1):
        cur = [i] + [0] * len(ref)
        for j, r in enumerate(ref, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (h != r))
        prev = cur
    return min(prev)  # ref 結尾可以略過


def seg_scores(rows, lines) -> tuple[float, float]:
    """每句和逐字稿同一段時間（前後 15 秒）的台詞做局部比對；回傳（字元錯誤率、讀音錯誤率），分母是辨識出的字數。"""
    e = n = er = nr = 0
    for r in rows:
        ref = "".join(t for s, t in lines if r["start"] - 15 <= s <= r["start"] + r["dur"] + 15)
        h, rf = norm(r["text"]), norm(ref)
        e, n = e + local_errors(h, rf), n + len(h)
        hr, rr = reading(r["text"]), reading(ref)
        er, nr = er + local_errors(hr, rr), nr + len(hr)
    return (e / n if n else 0), (er / nr if nr else 0)


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("transcript")
    ap.add_argument("--engine", choices=["reazon", "whisper"], required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    import psutil
    proc = psutil.Process()

    audio = load_audio(a.video, a.start, a.dur)
    segs, vad_cpu = vad_segments(audio)
    total = len(audio) / SR
    print(f"聲音 {total:.0f} 秒；語音偵測切出 {len(segs)} 句，VAD CPU {vad_cpu:.2f} 秒（即時的 {vad_cpu / total:.1%}）")

    g0, m0 = gpu_used_mib(), proc.memory_full_info().uss
    t = time.perf_counter()
    eng = Reazon(a.threads) if a.engine == "reazon" else Whisper(a.beam)
    eng(np.zeros(SR, np.float32))  # 暖機
    load = time.perf_counter() - t
    rows, times, cpu = [], [], 0.0
    for st, s in segs:
        t, c = time.perf_counter(), time.process_time()
        text = eng(s)
        times.append(time.perf_counter() - t)
        cpu += time.process_time() - c
        rows.append({"start": round(a.start + st, 1), "dur": round(len(s) / SR, 1), "sec": round(times[-1], 3), "text": text})
        print(f"[{a.start + st:6.1f}s {len(s) / SR:4.1f}s → {times[-1]:.2f}s] {text}")
    g1, m1 = gpu_used_mib(), proc.memory_full_info().uss
    hyp = "".join(r["text"] for r in rows)
    ref = reference(a.transcript, a.start, a.start + total)
    seg = seg_scores(rows, ref_lines(a.transcript))
    res = {
        "engine": eng.name, "audio_sec": round(total), "segments": len(segs), "load_sec": round(load, 1),
        "sec_median": round(float(np.median(times)), 3) if times else 0,
        "sec_p90": round(float(np.percentile(times, 90)), 3) if times else 0,
        "cpu_per_audio_sec": round(cpu / total, 3),
        "ram_mb": round((m1 - m0) / 2**20), "vram_mib": g1 - g0 if g0 >= 0 else None,
        "cer_all": round(cer(norm(hyp), norm(ref)), 3),  # 和整份逐字稿比（含沒配音的台詞，偏高，只當參考）
        "seg_cer": round(seg[0], 3), "seg_cer_reading": round(seg[1], 3),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
