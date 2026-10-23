"""無字幕模式：即時把電腦播放的聲音轉成文字，交給無字幕視窗（listen_ui.py），使用者按「查詢」才查字。

流程：WASAPI loopback 錄「預設播放裝置」的聲音（pyaudiowpatch）→ 轉單聲道 →
Silero VAD 每 32 ms 算一次「有人在講話」的機率 → 自己切句（Segmenter）→ ReazonSpeech k2 v2（sherpa-onnx，int8，只用 CPU）。

切句（2026-10-17 第二版，原本用 sherpa-onnx 的 VoiceActivityDetector）：
- 停頓 MIN_SILENCE 秒算一句結束；一句已經超過 SOFT_LEN 秒時，停頓 SHORT_SILENCE 秒就切（直播、動畫常一口氣講很久，不要積成一大句）。
- 超過 HARD_LEN 秒還沒停頓：在最後幾秒裡「講話機率最低」的地方切，下一句**從切點接著開始**（不留空隙）。
  sherpa-onnx 的最長限制是硬切，切點附近的字會不見（使用者 10-17 的影片：「もちろん私も含めてだけれど」整句消失）。
- 講話中每 PARTIAL_EVERY 秒把目前聽到的部分重新辨識一次，先送出「暫定」的文字（邊講邊長出來），講完再送最終結果。
- 倍速播放（使用者選的播放速度，set_speed）：每句辨識前先放慢回原速（PyAV 的 atempo，不變調），停頓、長度的門檻照倍速縮短。
  ReazonSpeech 聽不懂加速的講話：1.5 倍時讀音錯誤率 7.5% → 27%，只縮短門檻 22%，放慢回原速 10.6%（tools/listen_compare.py 的評估資料加速後量的）。
  自動判斷倍速不可靠（每秒假名數本來就因人因句差很多），所以讓使用者選。

- 評估見 tools/asr_eval.py：每句往前多帶 PREROLL 秒、頭尾補 PAD 秒靜音，
  否則 ReazonSpeech 常漏掉句首。
- 語音偵測只要 16 kHz：先 7 kHz 低通再線性內插降取樣；送去辨識的保留原始取樣率，交給 sherpa-onnx 自己重新取樣。
- loopback 在沒有聲音播放時通常完全不送資料：等不到資料就自己補靜音，句子才會結束。
- 換了播放裝置（插上耳機、藍牙）：PortAudio 的裝置清單只在初始化時讀一次，所以「一段時間完全沒聲音」時重新初始化，
  改錄新的預設裝置。
- sherpa-onnx 自帶一份 onnxruntime.dll（CPU 版），和查字用的 onnxruntime-gpu 同名：先 import onnxruntime，
  程序裡用的就是 GPU 版那份（API 向下相容，實測兩邊都正常、OCR 仍在 GPU）。
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import APP_DIR
from .i18n import t

log = logging.getLogger("jplens")
ASR_DIR = APP_DIR / "data" / "models" / "asr"
REAZON_DIR = ASR_DIR / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
SR = 16000          # 語音偵測用的取樣率
FRAME = 512         # 語音偵測一次看幾個樣本（32 ms）
FRAME_SEC = FRAME / SR
ON, OFF = 0.5, 0.35  # 講話機率超過 ON 開始一句；低於 OFF 算停頓（Silero 建議的兩段門檻）
PREROLL = 0.3       # 每句往前多帶幾秒（判定開始講話時，第一個音節常已經過了一點）
TAIL = 0.2          # 每句講完後多帶幾秒（尾音）
PAD = 0.5           # 每句頭尾補幾秒靜音（0.2 時 ReazonSpeech 有時把後半句吃掉：「大丈夫。ボクは、頑張れる」只剩「大丈夫」）
MIN_SILENCE = 0.4   # 停頓幾秒算一句結束
SOFT_LEN = 2.0      # 一句超過這麼長以後……
SHORT_SILENCE = 0.15  # ……停頓這麼久就切（不同人接著講常只停 0.15 秒，併成一段時 ReazonSpeech 會吃掉小聲的那個人）
HARD_LEN = 6.0      # 一直講超過這麼長：在最後 CUT_WINDOW 秒裡講話機率最低的地方切
CUT_WINDOW = 3.0
MIN_SPEECH = 0.2    # 講話不到這麼久的當雜音
# 小聲、氣音的台詞（「心の声」這類）講話機率常只有 0.2～0.35，到不了 ON：兩句之間的空檔另外用低門檻找，
# 機率 ≥ SOFT_ON 的部分累積超過 SOFT_MIN 秒就也送去辨識（直接把 ON／OFF 降低的話，句子之間有背景音樂時停頓判斷不出來，句子會併在一起）
SOFT_ON = 0.2
SOFT_MIN = 1.0
# 一段裡有兩三個人接著講時，ReazonSpeech 常把其中一個人整段吃掉（使用者 10-17 的 MyGO 短片：4.6 秒只讀出「バンドです」）。
# 超過 SPLIT_CHECK 秒的句子另外從中間講話機率最低的地方切成兩半各辨識一次，兩半的字數和 ≥ SPLIT_GAIN × 整段的字數就用兩半的（整段吃掉了內容）
SPLIT_CHECK = 2.5
SPLIT_GAIN = 1.2
PARTIAL_EVERY = 0.8  # 講話中每隔多久送一次暫定的文字（0＝不送）：每次都重新辨識整句到目前為止的聲音
IDLE_FILL = 0.2     # 等不到聲音資料多久就補一段靜音
REOPEN_AFTER = 5.0  # 完全沒聲音多久就重新找預設播放裝置
KEEP_SEC = 60       # 原始聲音保留多久（切句往前拿 PREROLL）
SAVE_SEC = 120      # 「保存聲音」存最近多少秒「有聲音」的部分（沒在播的時間不算：使用者常是影片停了一陣子才按）
SPEEDS = (1.0, 1.25, 1.5, 1.75, 2.0)  # 可以選的播放速度


# 語音偵測模型：silero4（sherpa-onnx 附的 Silero v4）、silero6（官方 v6.2.3）；比較用：silero5（v5.1.2）、ten（TEN VAD）。
# 10-17 比較（含漏字的讀音錯誤率，使用者的動畫影片）：v4 9.0%、**v6 6.1%**、TEN 6.4～7.2%（句子較長）、v5 16.4%。
# v6 沒人講話時機率很低、切得乾淨，但小聲的氣音台詞（「あっという間に当日だ」）幾乎給 0；v4 給 0.2～0.35。
# 所以主要切句用 v6、兩句之間的空檔補漏用 v4（SOFT_VAD）：6.4%，氣音台詞也讀得到。
VAD_KIND = "silero6"
SOFT_VAD = "silero4"  # 空檔補漏用的語音偵測模型（None＝和主要的同一個）
VAD_TEST = APP_DIR / "data" / "models" / "vad_test"  # 比較用的模型（tools/listen_eval.py --vad）


class _SileroV4:
    def __init__(self, ort, so, path):
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.h = self.c = np.zeros((2, 1, 64), np.float32)

    def __call__(self, frame):
        p, self.h, self.c = self.s.run(None, {"x": frame[None, :], "h": self.h, "c": self.c})
        return float(p[0, 0])


class _SileroV5:
    """v5 以後：狀態合成一個 state、每次要帶前一段的最後 64 個樣本當上下文。"""

    def __init__(self, ort, so, path):
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros(64, np.float32)
        self.sr = np.array(SR, np.int64)

    def __call__(self, frame):
        x = np.concatenate([self.ctx, frame])[None, :]
        p, self.state = self.s.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.ctx = frame[-64:]
        return float(p[0, 0])


class _TenVad:
    """TEN VAD（Apache-2.0）：官方的 ten_vad.dll，每 256 個樣本（16 ms）一個機率，這裡兩個取大的湊成 32 ms。"""

    def __init__(self, path):
        import ctypes as C
        self.C = C
        self.lib = C.CDLL(str(path))
        self.lib.ten_vad_create.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_float]
        self.lib.ten_vad_process.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_float), C.POINTER(C.c_int32)]
        self.h = C.c_void_p(0)
        assert self.lib.ten_vad_create(C.byref(self.h), 256, C.c_float(0.5)) == 0
        self.p, self.flag = C.c_float(), C.c_int32()

    def __call__(self, frame):
        x = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        out = 0.0
        for i in (0, 256):
            part = np.ascontiguousarray(x[i:i + 256])
            self.lib.ten_vad_process(self.h, part.ctypes.data, 256, self.C.byref(self.p), self.C.byref(self.flag))
            out = max(out, self.p.value)
        return out


def make_vad(kind: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if kind == "silero4":
        return _SileroV4(ort, so, ASR_DIR / "silero_vad.onnx")
    if kind == "silero6":
        return _SileroV5(ort, so, ASR_DIR / "silero_vad_v6.onnx")
    if kind == "silero5":
        return _SileroV5(ort, so, VAD_TEST / "silero_v5.onnx")
    if kind == "ten":
        return _TenVad(VAD_TEST / "ten_vad.dll")
    raise ValueError(kind)


def model_ready() -> bool:
    return ((ASR_DIR / "silero_vad.onnx").exists() and (ASR_DIR / "silero_vad_v6.onnx").exists()
            and any(REAZON_DIR.glob("encoder*.onnx")))


_atempo_ok = None  # PyAV 能不能用（None＝還沒試過）


def slow_down(audio: np.ndarray, rate: int, speed: float) -> tuple[np.ndarray, int]:
    """倍速播放的聲音放慢回原速，回傳（聲音, 取樣率）。用 PyAV（ffmpeg）的 atempo：不變調，和瀏覽器的倍速播放做法類似。
    沒有 PyAV 時退而求其次：取樣率當成 rate / speed 交給辨識（等於放慢＋音調變低；1.5 倍時讀音錯誤率比 atempo 多約 2 個百分點）。
    **打包版不帶 PyAV**：它附的 FFmpeg 含 x264／x265（GPL），公開發布的打包版要維持寬鬆授權，所以打包版一律用調整取樣率。
    比較過自己用 numpy 寫的 WSOLA：比 atempo 差、比較慢，不用。"""
    global _atempo_ok
    if _atempo_ok is not False:
        try:
            import av
            g = av.filter.Graph()
            src = g.add_abuffer(format="flt", sample_rate=rate, layout="mono")
            at = g.add("atempo", f"{1 / speed:.6f}")
            sink = g.add("abuffersink")
            src.link_to(at)
            at.link_to(sink)
            g.configure()
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio, np.float32).reshape(1, -1),
                                            format="flt", layout="mono")
            fr.sample_rate, fr.pts = rate, 0
            g.push(fr)
            g.push(None)
            out = []
            while True:
                try:
                    out.append(g.pull().to_ndarray().reshape(-1))
                except (av.BlockingIOError, av.EOFError):
                    break
            _atempo_ok = True
            return (np.concatenate(out).astype(np.float32) if out else audio), rate
        except Exception:  # noqa: BLE001（沒有 PyAV、ffmpeg 出錯）
            if _atempo_ok is None:
                log.exception("無字幕模式：PyAV 的 atempo 不能用，倍速改用調整取樣率")
            _atempo_ok = False
    return audio, max(8000, int(rate / speed))


def ending(text: str) -> str:
    """一段話最後一個詞看起來是不是講完了（Sudachi 詞性）：
    "end"＝講完（終助詞、終止形、命令形、意志形、感動詞）；"cont"＝句子不會停在這裡（格助詞、係助詞、連用形、接續詞…）；
    "?"＝分不出來（名詞結尾；接續助詞「けど」「から」「て」口語常拿來收尾：「わかってるけど」「ちょっと待って」）。"""
    from .jlearn.analysis import tokenize
    toks = [t for t in tokenize(text) if not t.is_punct()]
    if not toks:
        return "?"
    p1, p2, cform = toks[-1].pos[0], toks[-1].pos[1], toks[-1].pos[5]
    if p1 == "助詞":  # i18n: 不翻
        return "end" if p2 == "終助詞" else "?" if p2 == "接続助詞" else "cont"  # i18n: 不翻
    if p1 in ("動詞", "形容詞", "助動詞"):  # i18n: 不翻
        if cform.startswith(("終止形", "命令形", "意志推量形")):  # i18n: 不翻
            return "end"
        return "cont" if cform.startswith(("連用形", "仮定形", "未然形")) else "?"  # i18n: 不翻
    if p1 == "感動詞":  # i18n: 不翻
        return "end"
    return "cont" if p1 in ("接続詞", "連体詞", "接頭辞") else "?"  # i18n: 不翻


def starts_turn(text: str) -> bool:
    """一段話以感嘆詞開頭（えっ、ねえ、そう、はい…）：多半是換人講話或新的一句，不接在上一段後面。"""
    from .jlearn.analysis import tokenize
    toks = [t for t in tokenize(text) if not t.is_punct()]
    return bool(toks) and toks[0].pos[0] == "感動詞"  # i18n: 不翻


@dataclass
class Heard:
    seg: int        # 第幾句（暫定與最終用同一個編號）
    start: float    # 這句從聲音的第幾秒開始
    text: str
    final: bool     # False＝講到一半的暫定文字
    sec: float      # 辨識花了幾秒
    cont: bool = False  # 接在上一句後面（上一句一直講、被強制切開）：畫面上接在同一行


class Transcriber:
    """餵原始取樣率的單聲道聲音（feed），回傳 [Heard]（講話中的暫定文字、一句講完的最終文字）。
    不碰錄音裝置，離線測試（tools/listen_eval.py）可以直接拿影片的聲音餵。"""

    def __init__(self, rate: int, threads: int = 1, partial_every: float = PARTIAL_EVERY, vad: str = None,
                 speed: float = 1.0):
        # threads：2 個執行緒每次辨識只快約 20%，CPU 卻多用 3 倍（實測），所以用 1 個
        import onnxruntime as ort  # 先載 GPU 版的 onnxruntime.dll（見檔案開頭）
        import sherpa_onnx
        self.vad_kind = vad or VAD_KIND
        pick = lambda stem: str(next(iter(sorted(REAZON_DIR.glob(f"{stem}*int8*.onnx"))), None)
                                or next(REAZON_DIR.glob(f"{stem}*.onnx")))
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"),
            tokens=str(REAZON_DIR / "tokens.txt"), num_threads=threads, sample_rate=SR, feature_dim=80,
            decoding_method="greedy_search")
        self.partial_every = partial_every
        self._seg = 0
        self.set_speed(speed)
        self.set_rate(rate)

    def set_speed(self, speed: float):
        """播放速度（倍速）：停頓、長度的門檻照比例縮短，辨識前把聲音放慢回原速。講到一半換也可以。"""
        s = self.speed = max(0.5, min(2.0, float(speed)))
        self.min_silence, self.short_silence = MIN_SILENCE / s, SHORT_SILENCE / s
        self.soft_len, self.hard_len, self.cut_window = SOFT_LEN / s, HARD_LEN / s, CUT_WINDOW / s
        self.split_check, self.soft_min = SPLIT_CHECK / s, SOFT_MIN / s

    def set_rate(self, rate: int):
        """換了錄音裝置（取樣率可能不同）：從頭開始。"""
        self.rate = int(rate)
        self.step = self.rate / SR  # 每個 16 kHz 樣本對應幾個原始樣本
        # 降取樣前先低通（7 kHz）：只用線性內插的話高頻折回來，語音偵測變差（實測切出的句子少一成、字元錯多 1 個百分點）
        self._fir = None
        if self.rate > SR:
            n = int(self.rate / SR * 40) | 1
            k = np.arange(n) - (n - 1) / 2
            h = np.sinc(2 * 7000 / self.rate * k) * np.hamming(n)
            self._fir = (h / h.sum()).astype(np.float32)
        self._hist = np.zeros(0 if self._fir is None else len(self._fir) - 1, np.float32)
        self._t = 0.0                            # 下一個 16 kHz 樣本落在 _tail 的哪個位置（線性內插）
        self._tail = np.zeros(0, np.float32)     # 還沒用完的原始樣本
        self._pending = np.zeros(0, np.float32)  # 湊不滿一個 FRAME 的 16 kHz 樣本
        self._keep = np.zeros(int(KEEP_SEC * self.rate), np.float32)  # 最近 KEEP_SEC 秒的原始聲音（環狀，第 i 個樣本在 i % 長度）
        self._total = 0                          # 到目前為止收到幾個原始樣本
        self._vad = make_vad(self.vad_kind)
        self._soft_vad = make_vad(SOFT_VAD) if SOFT_VAD and SOFT_VAD != self.vad_kind else None
        self._p2 = 0.0  # 補漏用的機率（_soft_vad 算的；沒有就是主要的機率）
        self._frame = 0          # 已經算過機率的 FRAME 數
        self._probs: list[float] = []  # 目前這句每個 FRAME 的講話機率（從 _start 開始）
        self._start = None       # 目前這句從第幾個 FRAME 開始（None＝沒在講話）
        self._last_speech = 0    # 目前這句最後一個講話的 FRAME
        self._speech_frames = 0  # 目前這句有幾個 FRAME 在講話
        self._partial_at = 0     # 上次送暫定文字時在第幾個 FRAME
        self._partial_text = ""
        self._cont = False       # 目前這句是接在強制切開的上一句後面（開頭不往前帶）
        self._prev_hi = 0        # 上一句送去辨識的聲音到第幾個原始樣本（往前多帶時不要和上一句重疊，否則字會重複）
        self._soft = None        # 空檔裡的小聲講話從第幾個 FRAME 開始（SOFT_ON）
        self._soft_last = 0
        self._soft_voiced = 0

    # ---------------------------------------------------------------- 降取樣
    def _to16k(self, x: np.ndarray) -> np.ndarray:
        if self._fir is not None:
            h = np.concatenate([self._hist, x])
            self._hist = h[len(h) - len(self._hist):]
            x = np.convolve(h, self._fir, mode="valid").astype(np.float32)
        buf = np.concatenate([self._tail, x])
        if len(buf) < 2 or self._t > len(buf) - 1:
            self._tail = buf
            return np.zeros(0, np.float32)
        n = int((len(buf) - 1 - self._t) // self.step) + 1
        idx = self._t + np.arange(n) * self.step
        out = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        nxt = idx[-1] + self.step
        drop = min(int(nxt), len(buf) - 1)
        self._tail, self._t = buf[drop:], nxt - drop
        return out

    def _prob(self, frame: np.ndarray) -> float:
        return self._vad(frame)

    # ---------------------------------------------------------------- 切句
    def feed(self, x: np.ndarray) -> list[Heard]:
        x = np.asarray(x, np.float32)[-len(self._keep):]
        i = self._total % len(self._keep)
        n = min(len(x), len(self._keep) - i)
        self._keep[i:i + n] = x[:n]
        self._keep[:len(x) - n] = x[n:]
        self._total += len(x)
        y = np.concatenate([self._pending, self._to16k(x)])
        k = len(y) // FRAME * FRAME
        self._pending = y[k:]
        out = []
        for j in range(0, k, FRAME):
            fr = y[j:j + FRAME]
            p = self._prob(fr)
            self._p2 = self._soft_vad(fr) if self._soft_vad is not None else p
            out += self._step(p)
        if self._start is not None and self.partial_every:
            out += self._partial()
        return out

    def flush(self) -> list[Heard]:
        """聲音結束（離線測試）：講到一半的那句直接結束。"""
        if self._start is None:
            return self._close_soft(self._soft_last + 1) if self._soft is not None else []
        return self._finish(self._last_speech + 1)

    def _step(self, p: float) -> list[Heard]:
        f = self._frame
        self._frame += 1
        if self._start is None:
            if p >= ON:
                out = self._close_soft(f) if self._soft is not None else []
                self._start, self._last_speech, self._speech_frames = f, f, 1
                self._probs = [p]
                self._partial_at, self._partial_text = f, ""
                return out
            if self._p2 >= SOFT_ON:
                if self._soft is None:
                    self._soft, self._soft_voiced = f, 0
                self._soft_last = f
                self._soft_voiced += 1
            elif self._soft is not None and (f - self._soft_last) * FRAME_SEC >= self.min_silence:
                return self._close_soft(self._soft_last + 1)
            if self._soft is not None and (f + 1 - self._soft) * FRAME_SEC >= self.hard_len:
                return self._close_soft(f + 1)
            return []
        self._probs.append(p)
        if p >= OFF:
            if p >= ON or f - self._last_speech <= 1:
                self._last_speech = f
            self._speech_frames += 1
        length = (f + 1 - self._start) * FRAME_SEC
        silence = (f - self._last_speech) * FRAME_SEC
        if silence >= self.min_silence or (length >= self.soft_len and silence >= self.short_silence):
            return self._finish(self._last_speech + 1)
        if length >= self.hard_len:  # 一直講：在最後 CUT_WINDOW 秒裡講話機率最低的地方切，下一句從那裡接著開始
            w = int(self.cut_window / FRAME_SEC)
            tail = np.convolve(self._probs[-w:], np.ones(3) / 3, mode="same")
            cut = f + 1 - w + int(np.argmin(tail[1:-1])) + 1
            rest = self._probs[cut - self._start:]
            out = self._finish(cut, contiguous=True)
            self._start, self._last_speech = cut, f
            self._probs = list(rest)
            self._speech_frames = sum(q >= OFF for q in rest)
            self._partial_at, self._partial_text = cut, ""
            return out
        return []

    def _audio(self, a: int, b: int, pre: float, tail: float) -> tuple[int, np.ndarray]:
        """第 a～b 個 FRAME 的原始聲音（往前多帶 pre 秒、往後 tail 秒），頭尾補靜音。"""
        size = len(self._keep)
        hi = min(int((b * FRAME + tail * SR) * self.step), self._total)
        lo = max(int((a * FRAME - pre * SR) * self.step), self._total - size, self._prev_hi)
        lo = min(lo, int(a * FRAME * self.step))  # 上一句的尾巴已經多帶到這句裡面：至少從這句開始講話的地方開始
        pad = np.zeros(int(PAD * self.rate), np.float32)
        return lo, np.concatenate([pad, self._keep[np.arange(lo, hi) % size], pad])

    def _decode(self, audio: np.ndarray) -> tuple[str, float]:
        t = time.perf_counter()
        rate = self.rate
        if self.speed != 1:
            audio, rate = slow_down(audio, rate, self.speed)
        s = self.rec.create_stream()
        s.accept_waveform(rate, audio)
        self.rec.decode_stream(s)
        return s.result.text.strip(), time.perf_counter() - t

    def _finish(self, end: int, contiguous: bool = False) -> list[Heard]:
        """目前這句到第 end 個 FRAME 結束。contiguous：強制切開（後面接著講），尾巴不多帶，下一句的開頭也不往前帶。"""
        start, seg, cont = self._start, self._seg, self._cont
        pre = 0.0 if cont else PREROLL
        self._cont = contiguous
        speech = self._speech_frames * FRAME_SEC
        probs = self._probs
        self._start, self._probs, self._speech_frames = None, [], 0
        self._soft = None
        self._seg += 1
        if speech < MIN_SPEECH and not self._partial_text:
            return []
        lo, audio = self._audio(start, end, pre, 0.0 if contiguous else TAIL)
        self._prev_hi = lo + len(audio) - 2 * int(PAD * self.rate)
        text, sec = self._decode(audio)
        if SPLIT_CHECK and (end - start) * FRAME_SEC >= self.split_check and len(probs) >= 10:
            text, sec = self._split_check(audio, start, end, probs, text, sec)
        if not text and not self._partial_text:
            return []
        return [Heard(seg, lo / self.rate, text, True, sec, cont)]

    def _split_check(self, audio, start, end, probs, text, sec):
        """整段的結果 vs 從中間（30%～70% 之間講話機率最低的地方）切開兩半各自辨識：兩半明顯比較多字就用兩半的。"""
        n = len(probs)
        sm = np.convolve(probs, np.ones(3) / 3, mode="same")
        i = int(n * 0.3) + int(np.argmin(sm[int(n * 0.3):int(n * 0.7)]))
        pad = int(PAD * self.rate)
        body = audio[pad:len(audio) - pad]
        cut = len(body) - int((end - start - i) * FRAME * self.step)  # 從結尾往回算（開頭有 PREROLL）
        if not 0 < cut < len(body):
            return text, sec
        z = np.zeros(pad, np.float32)
        a, s1 = self._decode(np.concatenate([z, body[:cut], z]))
        b, s2 = self._decode(np.concatenate([z, body[cut:], z]))
        cnt = lambda t: sum(1 for ch in t if ch.isalnum())
        if cnt(a) + cnt(b) >= SPLIT_GAIN * cnt(text) + 2:
            return a + b, sec + s1 + s2
        return text, sec + s1 + s2

    def _close_soft(self, end: int) -> list[Heard]:
        """空檔裡的小聲講話到第 end 個 FRAME 結束：夠長就送去辨識（當成獨立的一句）。"""
        start, voiced = self._soft, self._soft_voiced * FRAME_SEC
        self._soft = None
        if voiced < self.soft_min:
            return []
        self._cont = False
        lo, audio = self._audio(start, end, PREROLL, TAIL)
        self._prev_hi = lo + len(audio) - 2 * int(PAD * self.rate)
        text, sec = self._decode(audio)
        seg = self._seg
        self._seg += 1
        return [Heard(seg, lo / self.rate, text, True, sec)] if text else []

    def _partial(self) -> list[Heard]:
        f = self._frame
        if (f - self._partial_at) * FRAME_SEC < self.partial_every or (f - self._start) * FRAME_SEC < 0.5:
            return []
        self._partial_at = f
        pre = 0.0 if self._cont else PREROLL
        lo, audio = self._audio(self._start, f, pre, 0.0)
        text, sec = self._decode(audio)
        if not text or text == self._partial_text:
            return []
        self._partial_text = text
        return [Heard(self._seg, lo / self.rate, text, False, sec, self._cont)]

    def recent(self, seconds: float = KEEP_SEC) -> np.ndarray:
        """最近 seconds 秒的原始聲音（「保存最近 60 秒的聲音」用）。"""
        n = min(int(seconds * self.rate), self._total, len(self._keep))
        return self._keep[np.arange(self._total - n, self._total) % len(self._keep)].copy()


class Listener:
    """錄喇叭聲音 → Transcriber。on_heard(Heard) 在背景執行緒呼叫（請轉到 Qt 主執行緒）；
    on_state(訊息, 是否錯誤)：開始聆聽（裝置名稱）、換裝置、出錯。"""

    def __init__(self, on_heard, on_state, partial: bool = True, speed: float = 1.0):
        self.on_heard, self.on_state = on_heard, on_state
        self.partial = partial  # 邊講邊顯示（設定可關）
        self.speed = speed      # 播放速度（倍速，無字幕視窗的選單）
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self.tr: Transcriber | None = None
        self.lines: list[tuple[float, str]] = []  # (時間, 最終文字)：保存聲音時一起存
        self._sound: list[tuple[float, np.ndarray]] = []  # (時間, 聲音)：最近 SAVE_SEC 秒有聲音的部分
        self._sound_n = 0
        self._sound_rate = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="listen", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def set_speed(self, speed: float):
        self.speed = speed
        if self.tr is not None:
            self.tr.set_speed(speed)

    def _keep_sound(self, x: np.ndarray, rate: int):
        if rate != self._sound_rate:
            self._sound, self._sound_n, self._sound_rate = [], 0, rate
        self._sound.append((time.time(), x))
        self._sound_n += len(x)
        while self._sound_n - len(self._sound[0][1]) >= SAVE_SEC * rate:
            self._sound_n -= len(self._sound.pop(0)[1])

    def save_recent(self, path) -> float:
        """最近 SAVE_SEC 秒有聲音的部分存成 wav（16 位元單聲道；中間沒在播的時間跳過），回傳秒數；
        同名 .txt 是這段時間辨識出的句子（前面是時刻）。"""
        import wave
        chunks, rate = list(self._sound), self._sound_rate
        if not chunks:
            return 0.0
        a = np.concatenate([c for _, c in chunks])
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
        t0 = chunks[0][0] - 1
        lines = [time.strftime("%H:%M:%S ", time.localtime(t)) + text for t, text in self.lines if t >= t0]
        head = time.strftime("聲音從 %H:%M:%S", time.localtime(chunks[0][0])) + time.strftime(" 到 %H:%M:%S（沒在播的時間跳過）", time.localtime(chunks[-1][0]))  # i18n: 不翻
        path.with_suffix(".txt").write_text("\n".join([head] + lines), encoding="utf-8")
        return len(a) / rate

    def _open(self):
        import pyaudiowpatch as pa
        p = pa.PyAudio()
        try:
            dev = p.get_default_wasapi_loopback()
        except Exception:  # noqa: BLE001（沒有播放裝置、WASAPI 不能用）
            p.terminate()
            raise
        ch = max(1, int(dev["maxInputChannels"]))
        rate = int(dev["defaultSampleRate"])

        def cb(data, frames, info, status):
            self._q.put(np.frombuffer(data, np.float32).reshape(-1, ch).mean(axis=1))
            return None, pa.paContinue

        stream = p.open(format=pa.paFloat32, channels=ch, rate=rate, input=True, frames_per_buffer=rate // 20,
                        input_device_index=dev["index"], stream_callback=cb)
        name = dev["name"].replace(" [Loopback]", "")
        return p, stream, rate, name

    @staticmethod
    def _close(p, stream):
        try:
            stream.stop_stream()
            stream.close()
        except Exception:  # noqa: BLE001
            pass
        p.terminate()

    def _run(self):
        import ctypes
        # 這條執行緒自己初始化 COM：PortAudio 只在第一次初始化時（在那條執行緒）做，程序裡已經有別的 PyAudio
        # （例如上一次聆聽還沒完全結束）時，這裡開 WASAPI 會回「Unanticipated host error」（實測）
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        p = stream = None
        try:
            t_open = time.perf_counter()
            p, stream, rate, name = self._open()
            tr = self.tr = Transcriber(rate, partial_every=PARTIAL_EVERY if self.partial else 0, speed=self.speed)
            log.info("無字幕模式：開始聆聽「%s」（%d Hz）；載入 %.1f 秒", name, rate, time.perf_counter() - t_open)
            self.on_state(t("聆聽中：{name}", name=name), False)
            last_sound = time.perf_counter()
            while not self._stop.is_set():
                try:
                    x = self._q.get(timeout=IDLE_FILL)
                except queue.Empty:
                    x = np.zeros(int(IDLE_FILL * rate), np.float32)  # 沒有聲音在播：補靜音，講到一半的句子才會結束
                now = time.perf_counter()
                if np.abs(x).max(initial=0) > 1e-4:
                    last_sound = now
                    self._keep_sound(x, rate)
                for h in tr.feed(x):
                    if h.final:
                        log.info("聽到（辨識 %.2f 秒）：%s", h.sec, h.text)
                        self.lines = (self.lines + [(time.time(), h.text)])[-200:]
                    self.on_heard(h)
                if now - last_sound > REOPEN_AFTER:  # 很久沒聲音：可能換了播放裝置，重新找預設的
                    last_sound = now
                    self._close(p, stream)
                    p = stream = None
                    p, stream, new_rate, new_name = self._open()
                    if new_name != name:
                        log.info("無字幕模式：播放裝置換成「%s」（%d Hz）", new_name, new_rate)
                        self.on_state(t("聆聽中：{name}", name=new_name), False)
                    if new_rate != rate:
                        tr.set_rate(new_rate)
                    name, rate = new_name, new_rate
                    while not self._q.empty():  # 丟掉舊裝置剩下的
                        self._q.get_nowait()
        except Exception as e:  # noqa: BLE001
            log.exception("無字幕模式出錯")
            self.on_state(t("無字幕模式出錯：{err}", err=e), True)
        finally:
            if p is not None:
                self._close(p, stream)
            while not self._q.empty():
                self._q.get_nowait()
            log.info("無字幕模式：停止聆聽")
