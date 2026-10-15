"""把 manga-ocr（kha-white/manga-ocr-base，safetensors 版）匯出成 ONNX，給 jplens/mangaocr_onnx.py 用。

    python tools/export_manga_ocr_onnx.py              # fp32（預設；和 PyTorch 版逐字相同）
    python tools/export_manga_ocr_onnx.py --fp16       # fp16（檔案小一半，291 張有 3 張雜訊圖結果不同）
    python tools/export_manga_ocr_onnx.py --check imgs.pkl   # 匯出後拿一批灰階圖和 PyTorch generate() 逐字比對

只有匯出時需要 torch、transformers、manga-ocr；查字工具執行時不需要。

為了 KV 快取，解碼器不直接匯出 transformers 的 BertLMHeadModel，而是用下面的 KVDecoder 重組
（共用同一份權重；先在 PyTorch 裡驗證和 generate() 結果相同才匯出）：
- 編碼器順便算好兩層交叉注意力的 K/V，並為每個 beam 複製（解碼時每一步都重用）
- 解碼器一次餵一個字；自注意力的 K/V 快取由呼叫端帶進帶出，beam_idx 指定每一列接續哪一列的快取（beam 重新排序）
沒裝 onnx 套件也能匯出：torch 只在有自訂 onnxscript 函式時才需要它，這裡原樣略過。
"""

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jplens import mangaocr_onnx as mo  # noqa: E402
from jplens.jlearn import ocr  # noqa: E402

HEADS, HEAD_DIM = 12, 64


def _snapshot() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(ocr.MANGA_OCR_REPO, revision=ocr.MANGA_OCR_REVISION,
                             allow_patterns=["*.json", "*.txt", "*.safetensors"])


def _split(x):  # (N, T, 768) → (N, 12, T, 64)
    return x.view(x.shape[0], x.shape[1], HEADS, HEAD_DIM).transpose(1, 2)


def _attend(q, k, v):
    w = torch.softmax(q @ k.transpose(-1, -2) / HEAD_DIM ** 0.5, dim=-1)
    a = w @ v  # (N, 12, T, 64)
    return a.transpose(1, 2).reshape(a.shape[0], a.shape[2], HEADS * HEAD_DIM)


class EncoderKV(nn.Module):
    def __init__(self, model, num_beams):
        super().__init__()
        self.encoder, self.layers, self.num_beams = model.encoder, model.decoder.bert.encoder.layer, num_beams

    def forward(self, pixel_values):
        h = self.encoder(pixel_values=pixel_values).last_hidden_state
        h = torch.repeat_interleave(h, self.num_beams, dim=0)
        out = []
        for layer in self.layers:
            ca = layer.crossattention.self
            out += [_split(ca.key(h)), _split(ca.value(h))]
        return tuple(out)  # ck0, cv0, ck1, cv1


class KVDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        bert = model.decoder.bert
        self.emb, self.layers, self.cls = bert.embeddings, bert.encoder.layer, model.decoder.cls

    def forward(self, input_ids, position, beam_idx, k0, v0, k1, v1, ck0, cv0, ck1, cv1):
        e = self.emb
        x = e.word_embeddings(input_ids) + e.position_embeddings(position).unsqueeze(0) \
            + e.token_type_embeddings.weight[0]
        x = e.LayerNorm(x)
        presents = []
        for layer, (pk, pv), (ck, cv) in zip(self.layers, ((k0, v0), (k1, v1)), ((ck0, cv0), (ck1, cv1))):
            sa = layer.attention.self
            k = torch.cat([pk.index_select(0, beam_idx), _split(sa.key(x))], dim=2)
            v = torch.cat([pv.index_select(0, beam_idx), _split(sa.value(x))], dim=2)
            presents += [k, v]
            a = _attend(_split(sa.query(x)), k, v)
            x = layer.attention.output.LayerNorm(layer.attention.output.dense(a) + x)
            a = _attend(_split(layer.crossattention.self.query(x)), ck, cv)
            x = layer.crossattention.output.LayerNorm(layer.crossattention.output.dense(a) + x)
            h = layer.intermediate(x)
            x = layer.output.LayerNorm(layer.output.dense(h) + x)
        return (self.cls(x)[:, 0].float(), *presents)


def torch_runner(model, dtype, device, num_beams):
    """KVDecoder＋mo.beam_search 在 PyTorch 裡跑（驗證重組的解碼器和 generate() 相同）。"""
    enc, dec = EncoderKV(model, num_beams), KVDecoder(model)

    @torch.inference_mode()
    def run(grays):
        x = torch.from_numpy(np.stack([mo.preprocess(g) for g in grays])).to(device, dtype)
        cross = enc(x)
        n = len(grays) * num_beams
        past = [torch.zeros((n, HEADS, 0, HEAD_DIM), dtype=dtype, device=device)] * 4

        def step(tokens, src, pos):
            nonlocal past
            out = dec(torch.from_numpy(tokens).to(device).view(-1, 1), torch.tensor([pos], device=device),
                      torch.from_numpy(src).to(device), *past, *cross)
            past = list(out[1:])
            return out[0].cpu().numpy()

        return mo.beam_search(step, len(grays), num_beams)

    return run


def export(model, dtype, device, num_beams, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    from torch.onnx._internal import onnx_proto_utils
    onnx_proto_utils._add_onnxscript_fn = lambda proto, custom_opsets: proto  # 沒裝 onnx 套件時略過（見上）
    enc, dec = EncoderKV(model, num_beams).eval(), KVDecoder(model).eval()
    x = torch.randn(2, 3, 224, 224, dtype=dtype, device=device)
    with torch.inference_mode():
        cross = enc(x)
    names_c = ["ck0", "cv0", "ck1", "cv1"]
    torch.onnx.export(enc, (x,), str(out_dir / "encoder.onnx"), input_names=["pixel_values"], output_names=names_c,
                      dynamic_axes={"pixel_values": {0: "b"}, **{c: {0: "n"} for c in names_c}},
                      opset_version=16, do_constant_folding=True)
    n = 2 * num_beams
    past = [torch.randn((n, HEADS, 3, HEAD_DIM), dtype=dtype, device=device) for _ in range(4)]
    ids = torch.full((n, 1), 100, dtype=torch.int64, device=device)
    pos = torch.tensor([3], device=device)
    beam_idx = torch.arange(n, device=device)
    names_p = ["k0", "v0", "k1", "v1"]
    torch.onnx.export(dec, (ids, pos, beam_idx, *past, *cross), str(out_dir / "decoder.onnx"),
                      input_names=["input_ids", "position", "beam_idx", *names_p, *names_c],
                      output_names=["logits", *[f"new_{p}" for p in names_p]],
                      dynamic_axes={"input_ids": {0: "n"}, "beam_idx": {0: "n"}, "logits": {0: "n"},
                                    **{p: {0: "n", 2: "t"} for p in names_p},
                                    **{f"new_{p}": {0: "n", 2: "t1"} for p in names_p},
                                    **{c: {0: "n"} for c in names_c}},
                      opset_version=16, do_constant_folding=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", type=Path, default=mo.MODEL_DIR)
    ap.add_argument("--check", type=Path, help="pickle：灰階圖的 list（或 (圖, 文字) 的 list），和 PyTorch generate() 比對")
    args = ap.parse_args()

    from manga_ocr.ocr import MangaOcrModel, post_process
    from transformers import AutoTokenizer
    path = _snapshot()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32
    if dtype == torch.float16 and device == "cpu":
        sys.exit("fp16 匯出需要 CUDA")
    # eager 注意力：用 PyTorch 的 SDPA 時，匯出工具會把 ViT 升成 fp32（fp16 版的檔案大一倍、輸出變 fp32）
    model = MangaOcrModel.from_pretrained(path, attn_implementation="eager").to(dtype).to(device).eval()
    gc = model.generation_config
    num_beams = gc.num_beams
    assert (gc.no_repeat_ngram_size, gc.length_penalty, gc.early_stopping, gc.max_length) == (3, 2.0, True, 300), \
        "manga-ocr 的生成設定變了，mangaocr_onnx.beam_search 要跟著改"
    tok = AutoTokenizer.from_pretrained(path, tokenizer_type="bert-japanese")

    grays = None
    if args.check:
        data = pickle.load(open(args.check, "rb"))
        grays = [d[0] if isinstance(d, tuple) else d for d in data]
        app = [d[1] for d in data] if data and isinstance(data[0], tuple) else None  # 查字工具當時的輸出
        # 1. 重組的解碼器（PyTorch）＋numpy beam search vs. generate()
        ref, mine = [], []
        run = torch_runner(model, dtype, device, num_beams)
        with torch.inference_mode():
            for i in range(0, len(grays), 16):
                x = torch.from_numpy(np.stack([mo.preprocess(g) for g in grays[i:i + 16]])).to(device, dtype)
                ref += [post_process(tok.decode(s, skip_special_tokens=True))
                        for s in model.generate(x, max_length=300).cpu()]
                mine += [post_process(tok.decode(s, skip_special_tokens=True)) for s in run(grays[i:i + 16])]
        same = sum(a == b for a, b in zip(ref, mine))
        print(f"KV 快取解碼器（PyTorch）vs. generate()：{same}/{len(ref)} 張逐字相同")

    t = time.perf_counter()
    export(model, dtype, device, num_beams, args.out)
    for f in ("vocab.txt",):
        shutil.copy(Path(path) / f, args.out / f)
    (args.out / "meta.json").write_text(json.dumps(
        {"dtype": "fp16" if args.fp16 else "fp32", "num_beams": num_beams, "source": ocr.MANGA_OCR_REPO,
         "revision": ocr.MANGA_OCR_REVISION}, indent=2), encoding="utf-8")
    sizes = {p.name: f"{p.stat().st_size / 1e6:.0f} MB" for p in args.out.glob("*.onnx")}
    print(f"匯出完成（{time.perf_counter() - t:.0f} 秒）→ {args.out}：{sizes}")

    if grays is not None:
        import onnxruntime as ort
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        onnx = mo.MangaOcrOnnx(prov, args.out)
        out = []
        for i in range(0, len(grays), 16):
            out += onnx(grays[i:i + 16])
        same = sum(a == b for a, b in zip(ref, out))
        print(f"ONNX（{onnx.device}）vs. generate()：{same}/{len(ref)} 張逐字相同")
        for a, b in zip(ref, out):
            if a != b:
                print(f"  generate：{a[:50]}\n  ONNX    ：{b[:50]}")
        if app:
            same = sum(a == b for a, b in zip(app, out))
            print(f"ONNX vs. 查字工具原本的輸出（PyTorch fp16）：{same}/{len(app)} 張逐字相同")
            for a, b in zip(app, out):
                if a != b:
                    print(f"  原本：{a[:50]}\n  ONNX：{b[:50]}")
        del ort


if __name__ == "__main__":
    main()
