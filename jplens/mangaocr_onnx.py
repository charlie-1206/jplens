"""manga-ocr 的 ONNX 版：不需要 PyTorch／transformers（常駐記憶體約少一半、啟動快很多）。

模型由 tools/export_manga_ocr_onnx.py 從 kha-white/manga-ocr-base 匯出到 data/models/manga-ocr-onnx/：
- encoder.onnx：圖 (B,3,224,224) → 兩層交叉注意力的 K/V（已為每個 beam 複製好，B×4 列）
- decoder.onnx：一次解一個字，自注意力的 K/V 快取由呼叫端帶進帶出（beam_idx 指定每一列接續哪一列的快取）
- vocab.txt、meta.json（精度、beam 數）

解碼和 transformers 的 generate() 相同（manga-ocr 的 config.json：beam search 4 束、不重複 3-gram、
length_penalty 2.0、early_stopping）；_beam_search 照 transformers 5.9 generation/utils.py 的 _beam_search 寫。
2026-10-15 實測：box_eval＋select_eval 實際送進 manga-ocr 的 291 張圖，fp32 和原本 PyTorch 版逐字相同。
唯一刻意的不同：不重複 n-gram 從 3 放寬成 6（NGRAM，環境變數 JPLENS_NGRAM 可改回 3 對照），否則「ちょくちょく」讀不出來。

用 ONNX Runtime 的 IOBinding：K/V 快取一直留在顯示卡上，每一步只有 logits 傳回 CPU。
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models" / "manga-ocr-onnx"
START, EOS = 2, 3
N_SPECIAL = 5  # [PAD] [UNK] [CLS] [SEP] [MASK]：解碼時略過（同 skip_special_tokens）
LAYERS = 2
NGRAM = 6  # 不重複 n-gram：manga-ocr 原本是 3，會讀不出「ちょくちょく」「ハハハハ」這類重複（第二個「ちょく」被禁止，只好改讀成「ぢょく」）


def available(model_dir: Path = MODEL_DIR) -> bool:
    return all((model_dir / f).exists() for f in ("encoder.onnx", "decoder.onnx", "vocab.txt", "meta.json"))


def post_process(text: str) -> str:
    """同 manga_ocr.ocr.post_process（那個模組一 import 就會載入 torch，所以抄過來）。"""
    import jaconv
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    return jaconv.h2z(text, ascii=True, digit=True)


def preprocess(gray: np.ndarray) -> np.ndarray:
    """灰階圖 → (3,224,224)：PIL 雙線性縮放、(x/255-0.5)/0.5。原本 transformers＋torchvision 的縮放最多差 2 個灰階，
    實測 291 張圖辨識結果完全相同。"""
    im = Image.fromarray(gray).convert("RGB").resize((224, 224), Image.BILINEAR)
    return ((np.asarray(im, np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)


def _log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _topk(a, k):
    """每列最大的 k 個（由大到小）的索引，同 torch.topk。"""
    part = np.argpartition(-a, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(a, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def _take(a, idx):
    """a: (B, N, ...)，idx: (B, M) → (B, M, ...)。"""
    return np.take_along_axis(a, idx.reshape(idx.shape + (1,) * (a.ndim - 2)), 1)


def beam_search(step, batch: int, num_beams=4, max_length=300, length_penalty=2.0, ngram=3) -> np.ndarray:
    """step(tokens (N,), src (N,), cur) → 下一個字的 logits (N, V)。N = batch×num_beams；
    src[i]＝第 i 列要接續上一步第幾列的 K/V 快取；cur＝這一步餵進去的字的位置（從 0 開始）。
    回傳每張圖最好的序列 (batch, max_length)，未用到的位置是 EOS。"""
    B, K = batch, num_beams
    keep = 2 * K  # transformers：max(2, 1 + eos 數) × num_beams
    cur, prompt = 1, 1
    run_seq = np.full((B, K, max_length), EOS, np.int64)  # 未填的位置：pad_token_id=0 為假值，transformers 改填 eos
    run_seq[:, :, 0] = START
    seqs = run_seq.copy()
    run_scores = np.zeros((B, K), np.float32)
    run_scores[:, 1:] = -1e9
    beam_scores = np.full((B, K), -1e9, np.float32)
    finished = np.zeros((B, K), bool)
    unsat = np.ones((B, 1), bool)
    top_mask = np.arange(keep) < K
    src = np.arange(B * K)
    flat_ofs = (np.arange(B) * K)[:, None]
    while True:
        flat = run_seq[:, :, :cur].reshape(B * K, cur)
        lp = _log_softmax(step(flat[:, -1], src, cur - 1).astype(np.float32))
        V = lp.shape[1]
        if ngram and cur + 1 >= ngram:  # NoRepeatNGramLogitsProcessor：前 ngram-1 個字和以前出現過的一樣時，禁止接同一個字
            for i in range(B * K):
                toks = flat[i].tolist()
                prefix = tuple(toks[cur + 1 - ngram:cur])
                banned = [toks[j + ngram - 1] for j in range(len(toks) - ngram + 1)
                          if tuple(toks[j:j + ngram - 1]) == prefix]
                if banned:
                    lp[i, banned] = -np.inf
        acc = (lp.reshape(B, K, V) + run_scores[:, :, None]).reshape(B, K * V)
        idx = _topk(acc, keep)
        tlp = np.take_along_axis(acc, idx, 1)
        beam, tok = idx // V, idx % V
        tseq = _take(run_seq, beam).copy()
        tseq[:, :, cur] = tok
        hits = (tok == EOS) | (cur + 1 >= max_length)
        # 下一輪繼續跑的束
        trl = tlp + hits.astype(np.float32) * -1e9
        n = _topk(trl, K)
        run_seq, run_scores = _take(tseq, n), np.take_along_axis(trl, n, 1)
        src = (np.take_along_axis(beam, n, 1) + flat_ofs).reshape(-1)
        # 更新已完成的序列
        just = hits & top_mask[None, :]
        fl = tlp / ((cur + 1 - prompt) ** length_penalty)
        fl = fl + finished.all(1, keepdims=True).astype(np.float32) * -1e9
        fl = fl + (~unsat).astype(np.float32) * -1e9
        fl = (fl + (~just).astype(np.float32) * -1e9).astype(np.float32)
        ms = np.concatenate([seqs, tseq], 1)
        msc = np.concatenate([beam_scores, fl], 1)
        mf = np.concatenate([finished, just], 1)
        t = _topk(msc, K)
        seqs, beam_scores, finished = _take(ms, t), np.take_along_axis(msc, t, 1), np.take_along_axis(mf, t, 1)
        cur += 1
        best_run = run_scores[:, :1] / ((cur - prompt) ** length_penalty)
        worst_fin = np.where(finished, beam_scores.min(1, keepdims=True), -1e9)
        unsat = unsat & np.any(best_run > worst_fin, axis=-1, keepdims=True)
        if not (unsat.any() and not finished.all() and not hits.all()):
            break
    return seqs[:, 0]


class MangaOcrOnnx:
    def __init__(self, providers: list, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.num_beams = meta["num_beams"]
        self.ngram = int(os.environ.get("JPLENS_NGRAM", NGRAM))
        self.dtype = np.float16 if meta["dtype"] == "fp16" else np.float32
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # 不和其他 session 搶 CPU
        self.enc = ort.InferenceSession(str(model_dir / "encoder.onnx"), so, providers=providers)
        self.dec = ort.InferenceSession(str(model_dir / "decoder.onnx"), so, providers=providers)
        p = self.dec.get_providers()[0]
        self.device = {"CUDAExecutionProvider": "cuda", "DmlExecutionProvider": "dml"}.get(p, "cpu")
        self.vocab = (model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
        self._ort = ort

    def _decode(self, ids) -> str:
        toks = [self.vocab[i] for i in ids if i >= N_SPECIAL]
        return post_process(" ".join(toks).replace(" ##", "").strip())

    def __call__(self, grays: list[np.ndarray]) -> list[str]:
        if not grays:
            return []
        ort, dev, dt = self._ort, self.device, self.dtype
        x = np.stack([preprocess(g) for g in grays]).astype(dt)
        io = self.enc.io_binding()
        io.bind_cpu_input("pixel_values", x)
        cross_names = [o.name for o in self.enc.get_outputs()]
        for name in cross_names:
            io.bind_output(name, dev)
        self.enc.run_with_iobinding(io)
        cross = dict(zip(cross_names, io.get_outputs()))
        n = len(grays) * self.num_beams
        empty = np.zeros((n, 12, 0, 64), dt)
        past = {f"{kv}{i}": ort.OrtValue.ortvalue_from_numpy(empty, dev, 0) for i in range(LAYERS) for kv in "kv"}

        def step(tokens, src, pos):
            nonlocal past
            io = self.dec.io_binding()
            io.bind_cpu_input("input_ids", tokens.reshape(-1, 1).astype(np.int64))
            io.bind_cpu_input("position", np.array([pos], np.int64))
            io.bind_cpu_input("beam_idx", src.astype(np.int64))
            for name, v in (*past.items(), *cross.items()):
                io.bind_ortvalue_input(name, v)
            io.bind_output("logits", "cpu")
            out_names = [f"new_{k}" for k in past]
            for name in out_names:
                io.bind_output(name, dev)
            self.dec.run_with_iobinding(io)
            outs = io.get_outputs()
            past = dict(zip(past, outs[1:]))
            return outs[0].numpy()

        seqs = beam_search(step, len(grays), self.num_beams, ngram=self.ngram)
        return [self._decode(s.tolist()) for s in seqs]
