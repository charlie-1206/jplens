"""無字幕模式的斷句評估：切點切在哪裡、一則字幕被拆成幾行、一行混到幾則字幕、切點附近錯多少字。

    python tools/listen_seg_eval.py 輸出.json [參數=值,…]     # 跑 tests/eval/real/audio/ 有正確台詞的聲音，結果存成 json
    python tools/listen_seg_eval.py 比較 a.json b.json …       # 把幾次的結果排成表
    python tools/listen_seg_eval.py 重算 a.json 輸出.json [顯示規則]  # 不重新辨識，只換「顯示時怎麼接成一行」重新計分
        顯示規則：now（目前）、pause:0.6（停頓短於 0.6 秒就接）、join:0.8:0.3（最後一個詞是「沒講完」停頓 < 0.8 秒就接、「分不出來」< 0.3 秒）

參數名稱是 jplens/listen.py 的模組常數（同 tools/listen_compare.py）。
listen_compare 只量整段的讀音錯誤率，分不出錯是斷句造成的還是模型讀錯；這裡用正確台詞（_truth.txt，一行是一則字幕）當句子邊界：
- 辨識結果和正確台詞都轉成讀音（Sudachi），用編輯距離對齊（rapidfuzz，只有這個工具用），把每個切點對應到正確台詞的位置。
- **切在詞中間**：切點對應的位置不是正確台詞 Sudachi 斷詞的邊界（「メロメロ」被切開這類）。對齊在錯字附近會偏一兩個字，只看相對變化。
- **切點附近錯字率**：正確台詞在切點前後各 4 個讀音裡，沒對上的字（漏、錯）＋多出來的字，比整體錯字率高多少。
- **漏掉的字幕**：一則字幕的讀音對上不到三成（整句沒聽到、或讀得面目全非）。
- **拆開／混到**（都以字幕計）：一則字幕對到 ≥ 2 個片段（或顯示的行）＝拆開；和別的字幕擠在同一個片段（行）裡＝混到。
  字幕卡片不一定等於「一個人講完一句」（同一個人的長句常分兩張卡），所以「混到」偏高，只看相對變化。
片段＝送去辨識的一段（Heard）；行＝無字幕視窗顯示的一行（目前只有強制切開的下一段接在同一行，listen_ui.Line）。
"""
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")
AUDIO = ROOT / "tests" / "eval" / "real" / "audio"
NEAR = 4  # 切點前後幾個讀音算「切點附近」
SHOW = os.environ.get("SHOW")  # SHOW=1：列出切在詞中間的切點
NAMES = ("ON", "OFF", "MIN_SPEECH", "MIN_SILENCE", "SOFT_LEN", "SHORT_SILENCE", "HARD_LEN", "CUT_WINDOW", "SOFT_ON",
         "SOFT_MIN", "VAD_KIND", "SOFT_VAD", "SPLIT_CHECK", "SPLIT_GAIN", "PAD", "PREROLL", "TAIL")


def wav(p) -> tuple[np.ndarray, int]:
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(1), w.getframerate()


# ---------------------------------------------------------------- 辨識
def transcribe(audio: np.ndarray, rate: int) -> list[dict]:
    """照即時的方式餵（每次 0.05 秒），回傳每個片段：開始、結束（秒，講話的範圍）、怎麼結束的、是不是接在上一段、文字。"""
    import jplens.listen as L

    class Rec(L.Transcriber):
        def _finish(self, end, contiguous=False):
            start = self._start
            silence = (self._frame - 1 - self._last_speech) * L.FRAME_SEC
            how = "強制" if contiguous else "停頓" if silence >= self.min_silence else "短停頓"
            out = super()._finish(end, contiguous)
            for h in out:
                meta[h.seg] = (start * L.FRAME_SEC, end * L.FRAME_SEC, how)
            return out

        def _close_soft(self, end):
            start = self._soft
            out = super()._close_soft(end)
            for h in out:
                meta[h.seg] = (start * L.FRAME_SEC, end * L.FRAME_SEC, "補漏")
            return out

    meta, segs = {}, {}
    tr = Rec(rate, partial_every=0)
    n = rate // 20
    for i in range(0, len(audio) + n, n):
        for h in (tr.feed(audio[i:i + n]) if i < len(audio) else tr.flush()):
            if h.final:
                segs[h.seg] = h
    return [{"seg": s, "start": round(meta[s][0], 3), "end": round(meta[s][1], 3), "how": meta[s][2],
             "cont": h.cont, "text": h.text} for s, h in sorted(segs.items()) if h.text]


# ---------------------------------------------------------------- 顯示時怎麼接成一行
def lines_now(segs: list[dict]) -> list[list[int]]:
    """目前的無字幕視窗：強制切開的下一段（cont）接在上一段的同一行。回傳每一行由哪幾個片段（索引）組成。"""
    out = []
    for i, s in enumerate(segs):
        if out and s["cont"] and segs[out[-1][-1]]["seg"] == s["seg"] - 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def lines_join(cont_gap: float, maybe_gap: float, end_gap: float = 0.0, turn: bool = False):
    """強制切開的照舊接；另外上一段結束到這段開始的停頓（秒）短於門檻就接在同一行。門檻看目前那行最後一個詞（listen.ending）：
    句子不會停在那裡（cont）、分不出來（?）、講完了（end）各一個。只看停頓的話三個設一樣。"""
    from jplens.listen import ending, starts_turn

    def rule(segs):
        out = []
        for i, s in enumerate(segs):
            if out:
                prev = segs[out[-1][-1]]
                gap = s["start"] - prev["end"]
                text = "".join(segs[j]["text"] for j in out[-1])
                limit = {"cont": cont_gap, "?": maybe_gap, "end": end_gap}[ending(text)] if (cont_gap, maybe_gap) != (end_gap,) * 2 else end_gap
                if (s["cont"] and prev["seg"] == s["seg"] - 1) or (gap < limit and not (turn and starts_turn(s["text"]))):
                    out[-1].append(i)
                    continue
            out.append([i])
        return out
    return rule


def get_rule(name: str):
    """now；pause:秒（只看停頓）；join:cont秒:?秒[:end秒]（看停頓＋最後一個詞）；後面加 +turn：下一段以感嘆詞開頭就不接。"""
    if name == "now":
        return lines_now
    turn = name.endswith("+turn")
    kind, *v = name.removesuffix("+turn").split(":")
    v = [float(x) for x in v]
    return lines_join(v[0], v[0], v[0], turn) if kind == "pause" else lines_join(*v, turn=turn)


# ---------------------------------------------------------------- 計分
_tok = None


def words(text: str) -> list[str]:
    """一段文字每個詞的讀音（同 asr_eval.reading，但留著詞的邊界）。"""
    global _tok
    import asr_eval
    if _tok is None:
        from sudachipy import dictionary
        _tok = dictionary.Dictionary().tokenizer()
    t = asr_eval.norm(text)
    return [m.reading_form() for m in _tok.tokenize(t)] if t else []


def score(segs: list[dict], truth: Path, rule) -> dict:
    import asr_eval
    from rapidfuzz.distance import Levenshtein
    # 正確台詞：讀音串、詞的邊界、字幕的邊界
    ref, wb, subs = "", {0}, []
    for _, t in asr_eval.ref_lines(str(truth)):
        a = len(ref)
        for w in words(t):
            ref += w
            wb.add(len(ref))
        if len(ref) > a:
            subs.append((a, len(ref)))
    # 辨識結果：每個片段的讀音範圍
    hyp, spans = "", []
    for s in segs:
        a = len(hyp)
        hyp += "".join(words(s["text"]))
        spans.append((a, len(hyp)))
    ops = Levenshtein.opcodes(hyp, ref)
    bad = np.zeros(len(ref) + 1)  # 正確台詞每個位置：沒對上（漏、錯）＋這裡多出來的字
    for op in ops:
        if op.tag in ("replace", "delete"):
            bad[op.dest_start:op.dest_end] += 1
        if op.tag == "insert":
            bad[op.dest_start] += op.src_end - op.src_start
        if op.tag == "replace" and op.src_end - op.src_start > op.dest_end - op.dest_start:
            bad[op.dest_start] += (op.src_end - op.src_start) - (op.dest_end - op.dest_start)

    def h2r(p: int) -> int:
        """辨識結果第 p 個讀音的位置對應到正確台詞的哪裡。"""
        for op in ops:
            if op.src_start <= p < op.src_end or (p == op.src_end and op is ops[-1]):
                k = (p - op.src_start) / max(1, op.src_end - op.src_start)
                return round(op.dest_start + k * (op.dest_end - op.dest_start))
            if op.src_start == op.src_end == p:  # 這裡漏了一段：切點當成在漏掉那段的開頭
                return op.dest_start
        return len(ref)

    lines = rule(segs)
    line_spans = [(spans[g[0]][0], spans[g[-1]][1]) for g in lines]

    def overlap_counts(units):
        """units：辨識的範圍（片段或行）。回傳（拆開的字幕數, 和別的字幕擠在同一個單位裡的字幕數）。"""
        rs = [(h2r(a), h2r(b)) for a, b in units]
        per_sub = [0] * len(subs)
        mixed = set()
        for a, b in rs:
            hit = []
            for i, (sa, sb) in enumerate(subs):
                need = min(3, max(1, (sb - sa + 1) // 2))
                if min(b, sb) - max(a, sa) >= need:
                    per_sub[i] += 1
                    hit.append(i)
            if len(hit) >= 2:
                mixed.update(hit)
        return sum(c >= 2 for c in per_sub), len(mixed)

    wbs = np.array(sorted(wb))
    cuts = []  # 片段之間的切點：怎麼切的、在不在詞的邊界、切點附近錯幾個字
    for i in range(len(segs) - 1):
        r = h2r(spans[i][1])
        lo, hi = max(0, r - NEAR), min(len(ref), r + NEAR)
        mid = int(np.min(np.abs(wbs - r)) > 0)
        cuts.append({"how": segs[i]["how"], "mid": mid, "near_bad": float(bad[lo:hi].sum()), "near_n": hi - lo})
        if SHOW and mid:
            print(f"    切在詞中間（{segs[i]['how']}）：…{segs[i]['text'][-8:]}｜{segs[i + 1]['text'][:8]}…  正確：…{ref[max(0, r - 8):r]}｜{ref[r:r + 8]}…")
    dist = Levenshtein.distance(hyp, ref)
    hit = np.zeros(len(ref))  # 正確台詞每個字有沒有對上
    for op in ops:
        if op.tag == "equal":
            hit[op.dest_start:op.dest_end] = 1
    missed = [i for i, (a, b) in enumerate(subs) if hit[a:b].mean() < 0.3]
    if SHOW:
        for i in missed:
            print(f"    漏掉：{ref[subs[i][0]:subs[i][1]]}")
    return {"ref": len(ref), "dist": dist, "subs": len(subs), "segs": len(segs), "lines": len(lines), "missed": len(missed),
            "split_seg": overlap_counts(spans)[0], "mixed_seg": overlap_counts(spans)[1],
            "split_line": overlap_counts(line_spans)[0], "mixed_line": overlap_counts(line_spans)[1], "cuts": cuts}


def summary(res: dict) -> dict:
    """幾個檔案加總。"""
    tot = {k: sum(r.get(k, 0) for r in res.values()) for k in ("ref", "dist", "subs", "segs", "lines", "missed", "split_seg",
                                                        "mixed_seg", "split_line", "mixed_line")}
    cuts = [c for r in res.values() for c in r["cuts"]]
    tot["cuts"] = len(cuts)
    tot["mid"] = sum(c["mid"] for c in cuts)
    tot["near"] = sum(c["near_bad"] for c in cuts) / max(1, sum(c["near_n"] for c in cuts))
    forced = [c for c in cuts if c["how"] == "強制"]
    tot["forced"] = len(forced)
    tot["forced_mid"] = sum(c["mid"] for c in forced)
    tot["forced_near"] = sum(c["near_bad"] for c in forced) / max(1, sum(c["near_n"] for c in forced))
    tot["how"] = {h: [sum(c["how"] == h for c in cuts), sum(c["mid"] for c in cuts if c["how"] == h)]
                  for h in ("停頓", "短停頓", "強制", "補漏")}
    return tot


def row(name: str, t: dict) -> str:
    return (f"{name}｜讀音錯 {t['dist'] / t['ref']:.1%}｜片段 {t['segs']}、行 {t['lines']}／字幕 {t['subs']}｜"
            f"拆開 片段 {t['split_seg']}・行 {t['split_line']}｜混到 片段 {t['mixed_seg']}・行 {t['mixed_line']}｜"
            f"漏掉的字幕 {t.get('missed', 0)}｜切在詞中間 {t['mid']}/{t['cuts']}（強制 {t['forced_mid']}/{t['forced']}）｜"
            f"切點附近錯 {t['near']:.1%}（強制 {t['forced_near']:.1%}）｜"
            + "、".join(f"{h} {n}（詞中間 {m}）" for h, (n, m) in t.get("how", {}).items() if n))


def report(data: dict, rule) -> dict:
    res = {}
    for name, d in data["files"].items():
        res[name] = score(d, AUDIO / f"{name}_truth.txt", rule)
        print(row(name, summary({name: res[name]})))
    print(row("合計", summary(res)))
    return res


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "比較":
        for p in a[1:]:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            print(row(f"{Path(p).stem}（{d['config'] or '預設'}，顯示 {d.get('rule', 'now')}）", d["summary"]))
        return
    if a[0] == "重算":
        d = json.loads(Path(a[1]).read_text(encoding="utf-8"))
        d["rule"] = a[3] if len(a) > 3 else "now"
        res = report(d, get_rule(d["rule"]))
        d["summary"] = summary(res)
        Path(a[2]).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    import jplens.listen as L
    config = a[1] if len(a) > 1 else ""
    for kv in filter(None, config.split(",")):
        k, v = kv.split("=")
        assert k in NAMES, k
        setattr(L, k, v if k in ("VAD_KIND", "SOFT_VAD") else float(v))
    data = {"config": config, "rule": "now", "files": {}}
    for truth in sorted(AUDIO.glob("*_truth.txt")):
        name = truth.stem.removesuffix("_truth")
        audio, rate = wav(AUDIO / f"{name}.wav")
        data["files"][name] = transcribe(audio, rate)
    res = report(data, lines_now)
    data["summary"] = summary(res)
    Path(a[0]).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
