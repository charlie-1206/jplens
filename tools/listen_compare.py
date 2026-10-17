"""無字幕模式的切句／語音偵測設定比較（第六個工作階段寫的）：每組參數跑一次，印出含漏字的讀音錯誤率。

    python tools/listen_compare.py DEF "VAD_KIND=silero4" "SPLIT_CHECK=0" "ON=0.4,OFF=0.25" ...

資料（都不進版控）：
- tests/eval/real/audio/*.wav ＋ 同名 *_truth.txt：有正確台詞的聲音（20261016_hasu_anime：使用者的蓮ノ空 3D 動畫，台詞照燒錄字幕寫的；
  hasunosora1：蓮ノ空劇情，照燒錄字幕 OCR 整理，沒拿來調過參數）
- ../video_translate 的 vivian_story1～3（逐字稿有沒配音的台詞，只看相對變化）
- C:/jplens/captures/audio/ 裡使用者保存的聲音（沒有正確台詞：只印字數；環境變數 SHOW=1 印出全文）
參數名稱是 jplens/listen.py 的模組常數（VAD_KIND、SOFT_VAD 是字串，其他是數字）。
"""
import sys, os, wave, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import jplens.listen as L
import asr_eval
from listen_eval import load_audio
VT = ROOT.parent / "video_translate"


def wav(p):
    w = wave.open(str(p))
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


DATA = []
for f in sorted((ROOT / "tests/eval/real/audio").glob("*_truth.txt")):
    DATA.append((f.stem.replace("_truth", ""), wav(f.with_name(f.stem.replace("_truth", "") + ".wav")), str(f)))
for i in (1, 2, 3):
    v = VT / "video_source" / f"vivian_story{i}.mp4"
    if v.exists():
        DATA.append((f"story{i}", load_audio(str(v), 48000), str(VT / "transcripts" / f"vivian_story{i}.txt")))
CLIPS = [(Path(f).stem, wav(f)) for f in sorted(glob.glob("C:/jplens/captures/audio/*.wav"))]
NAMES = ("ON", "OFF", "MIN_SPEECH", "MIN_SILENCE", "SOFT_LEN", "SHORT_SILENCE", "HARD_LEN", "SOFT_ON", "SOFT_MIN",
         "VAD_KIND", "SOFT_VAD", "SPLIT_CHECK", "SPLIT_GAIN", "PAD", "PREROLL", "TAIL")
DEFAULT = {k: getattr(L, k) for k in NAMES}


class _H:
    def __init__(self, text): self.text = text


def run(a):
    """同一個 seg 後來的最終文字蓋掉前面的（合併重新辨識），空的拿掉。"""
    tr = L.Transcriber(48000, partial_every=0)
    segs = {}
    for i in range(0, len(a) + 2400, 2400):
        for h in (tr.feed(a[i:i + 2400]) if i < len(a) else tr.flush()):
            if h.final:
                segs[h.seg] = h.text
    return [_H(t) for t in segs.values() if t]


for combo in sys.argv[1:]:
    for k, v in DEFAULT.items():
        setattr(L, k, v)
    for kv in filter(lambda x: x and x != "DEF", combo.split(",")):
        k, v = kv.split("=")
        setattr(L, k, v if k in ("VAD_KIND", "SOFT_VAD") else float(v))
    res = []
    for name, a, truth in DATA:
        out = run(a)
        hyp = "".join(h.text for h in out)
        ref = "".join(t for _, t in asr_eval.ref_lines(truth))
        rd = asr_eval.cer(asr_eval.reading(hyp), asr_eval.reading(ref))
        res.append(f"{name} {rd:.1%}（{len(out)} 句、{len(asr_eval.norm(hyp))}/{len(asr_eval.norm(ref))} 字）")
    for name, a in CLIPS:
        out = run(a)
        res.append(f"{name} {len(out)} 句 {len(asr_eval.norm(''.join(h.text for h in out)))} 字")
        if os.environ.get("SHOW"):
            print("   ", name, "／".join(h.text for h in out))
    print(combo, "｜", "；".join(res), flush=True)
