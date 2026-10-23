"""諺語比對（words.find_proverbs）的評估：前半句配不配得到、一般句子會不會誤配。

    python tools/proverb_eval.py

A. 諺語本身：字典的每條諺語斷詞後，在每個詞的邊界截斷（模擬只選到前半句），看配回哪一條：
   配回同一條／配到別條（同樣開頭的另一條，例如寫法不同的變體）／沒配到（太短、實詞不夠，照規則本來就不配）。
B. 一般句子：維基詞典例句（data/raw/zhwikt_ja.jsonl）＋文法說明的例句（grammar_notes.json），
   每句在每個詞的邊界截斷（模擬選取範圍停在句子中間），列出所有配到的諺語：
   - 前半句：原句接下去也照著諺語寫（或原句本來就在那裡結束）算對，否則是誤配
   - 照字面完全相同：列出來人工看（字面一樣，但可能是字面意思，不是諺語）
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from jplens.dictionary import Dictionary  # noqa: E402
from jplens.jlearn import analysis  # noqa: E402
from jplens.words import find_proverbs  # noqa: E402


def corpus() -> list[str]:
    out = []
    for line in open(ROOT / "data" / "raw" / "zhwikt_ja.jsonl", encoding="utf-8"):
        for s in json.loads(line).get("senses", []):
            for ex in s.get("examples", []):
                t = "".join((ex.get("text") or "").split())
                if t:
                    out.append(t)
    notes = json.loads((ROOT / "jplens" / "jlearn" / "grammar_notes.json").read_text(encoding="utf-8"))
    for lst in notes.values():
        for n in lst:
            ex = n.get("example") or ""
            ex = ex.split("（")[0].strip()
            if ex:
                out.append(ex)
    return list(dict.fromkeys(out))


def main():
    d = Dictionary()
    keys = d.proverbs()
    print(f"諺語、慣用句 {len(keys)} 條")

    # A. 前半句配回哪一條
    same = other = none = 0
    others = []
    for key in keys:
        toks = analysis.tokenize(key)
        for j in range(2, len(toks)):
            res = find_proverbs(toks[:j], d)
            hit = [f for i, _, f in res if i == 0]
            if not hit:
                none += 1
            elif hit[0] == key:
                same += 1
            else:
                other += 1
                others.append((key, "".join(t.surface for t in toks[:j]), hit[0]))
    total = same + other + none
    print(f"\nA. 諺語在詞的邊界截斷：{total} 種截法 → 配回同一條 {same}、配到別條 {other}、沒配到 {none}")
    for key, part, hit in others[:15]:
        print(f"   {part} → {hit}（原本 {key}）")

    # B. 一般句子在每個詞的邊界截斷
    sents = corpus()
    n_cut = 0
    wrong, right, literal = collections.Counter(), collections.Counter(), collections.Counter()
    examples = {}
    for s in sents:
        toks = analysis.tokenize(s)
        starts = [0]
        for t in toks:
            starts.append(starts[-1] + len(t.surface))
        for j in range(1, len(toks) + 1):
            n_cut += 1
            for i, e, form in find_proverbs(toks[:j], d):
                span = s[starts[i]:starts[e]]
                if form == span:
                    if e == j:  # 每個完整比對只在剛好截到那裡時算一次
                        literal[form] += 1
                        examples.setdefault(("lit", form), s)
                    continue
                rest_src, rest_key = s[starts[e]:], form[len(span):]
                ok = rest_src[:1] == rest_key[:1] or not rest_src or toks[e].is_punct()
                (right if ok else wrong)[(span, form)] += 1
                examples.setdefault((span, form), s)
    print(f"\nB. {len(sents)} 句、{n_cut} 種截法")
    print(f"   前半句：對 {sum(right.values())} 次（{len(right)} 種）、誤配 {sum(wrong.values())} 次（{len(wrong)} 種）")
    for (span, form), c in wrong.most_common(40):
        print(f"   ✗ {span} → {form} ×{c}｜{examples[(span, form)]}")
    for (span, form), c in right.most_common(10):
        print(f"   ✓ {span} → {form} ×{c}")
    print(f"   照字面完全相同：{sum(literal.values())} 次（{len(literal)} 種）")
    for form, c in literal.most_common(60):
        print(f"   = {form} ×{c}｜{examples[('lit', form)]}")


if __name__ == "__main__":
    main()
