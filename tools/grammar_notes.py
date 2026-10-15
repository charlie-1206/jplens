"""用 JLPT_grammer 的說明補充文法規則：產生 jplens/jlearn/grammar_notes.json（結果視窗的滑鼠提示用）。

    python tools/grammar_notes.py

JLPT_grammer（使用者請 Gemini 整理的 750 條）每條有接續、語意、例句（附中譯），比規則本身的說明詳細
（語感、使用場合、な形＋な／名詞＋の 這類細節）。規則的 meaning 很短，留在清單上；這裡的內容只放在滑鼠提示。
- 現有規則（grammar_rules.py）：名稱相近、又比對得到 JLPT_grammer 例句的條目（同 tools/grammar_batches.covered）
- Gemini 寫的規則（grammar_rules_extra.jsonl）：本來就是從某一條來的（key）
同一個句型在不同級數各列一次的（〜がたい N3／N2／N1）只取級數最接近規則的那條；一條規則最多 MAX 條。
級數不改（規則和 JLPT_grammer 不同的列在最後，教材本來就會差一級）。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import grammar_batches as gb  # noqa: E402
from jplens.jlearn import grammar  # noqa: E402
from jplens.jlearn.analysis import tokenize  # noqa: E402
from jplens.jlearn.grammar import EXTRA_PATH  # noqa: E402
from jplens.jlearn.grammar_rules import RULES  # noqa: E402

OUT = ROOT / "jplens" / "jlearn" / "grammar_notes.json"
MAX = 3


def build() -> tuple[dict, list]:
    entries = [e for e in gb.load_entries() if e.get("example_ja")]
    by_key = {e["key"]: e for e in entries}
    cands = {}
    for e in entries:
        hit = grammar.match_line(tokenize(e["example_ja"]), RULES)
        for r in gb.related_rules(e):
            if r in hit:
                cands.setdefault(r.id, []).append(e)
    levels = {r.id: r.level for r in RULES}
    for line in EXTRA_PATH.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        levels[d["id"]] = d["level"]
        if d.get("key") in by_key:
            cands.setdefault(d["id"], []).append(by_key[d["key"]])
    notes, mismatch = {}, []
    for rid, es in cands.items():
        lv = levels[rid]
        kept, names = [], set()
        for e in sorted(es, key=lambda e: (abs(e["level"] - (lv or 3)), -e["level"])):
            n = gb._names(e["pattern"])
            if n & names or len(kept) >= MAX:
                continue
            names |= n
            kept.append(e)
        notes[rid] = [{k: e.get(k, "") for k in ("key", "level", "pattern", "meaning", "formation", "example")}
                      for e in kept]
        if lv is not None and all(e["level"] != lv for e in es):
            mismatch.append((rid, lv, sorted({e["level"] for e in es}, reverse=True)))
    return notes, mismatch


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    notes, mismatch = build()
    OUT.write_text(json.dumps(dict(sorted(notes.items())), ensure_ascii=False, indent=0), encoding="utf-8")
    n_old = sum(rid in {r.id for r in RULES} for rid in notes)
    print(f"寫入 {OUT.relative_to(ROOT)}：{len(notes)} 條規則（現有規則 {n_old}、Gemini 寫的 {len(notes) - n_old}）")
    rules = {r.id: r for r in grammar.all_rules()}
    print(f"級數和 JLPT_grammer 不同的現有規則 {sum(rid in {r.id for r in RULES} for rid, _, _ in mismatch)} 條（不改）：")
    for rid, lv, others in mismatch:
        print(f"  {rules[rid].pattern}  規則 N{lv}／JLPT_grammer {'、'.join(f'N{x}' for x in others)}")


if __name__ == "__main__":
    main()
