"""讀音評估：維基詞典例句（data/raw/zhwikt_ja.jsonl，約 9,000 句，每句有人工標的振假名）跑 words.extract，
比對每個含漢字的單字讀音和例句的振假名。

    python tools/reading_eval.py 輸出.json          # 跑評估，錯的列表存成 JSON
    python tools/reading_eval.py 前.json 後.json     # 比較兩次的結果（改讀音相關的程式前後）

判斷方式：沒有活用的詞（原文＝寫法）讀音要和振假名完全相同；有活用的詞（言っ → 言う）只比原文和寫法相同的開頭
（言 → い）。振假名沒蓋到詞裡每個漢字的不評。例句的振假名偶爾也有錯，看結果時要逐條確認。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from jplens import words  # noqa: E402
from jplens.dictionary import Dictionary  # noqa: E402
from jplens.jlearn import analysis  # noqa: E402

SRC = ROOT / "data" / "raw" / "zhwikt_ja.jsonl"


def examples():
    seen = set()
    for line in open(SRC, encoding="utf-8"):
        for s in json.loads(line).get("senses", []):
            for ex in s.get("examples", []):
                text, ruby = ex.get("text", ""), ex.get("ruby")
                if text and ruby and text not in seen:
                    seen.add(text)
                    yield text, ruby


def gold_chars(text: str, ruby) -> list[str | None]:
    """每個字的讀音：振假名的一段（今日＝きょう）放在第一個字，後面的字是 ""；假名是自己；沒標的漢字是 None。"""
    out = [analysis.to_hira(c) if analysis._KANA_ONLY.match(c) else None for c in text]
    pos = 0
    for base, rd in ruby:
        i = text.find(base, pos)
        if i < 0:
            continue
        out[i] = analysis.to_hira(rd)
        for k in range(i + 1, i + len(base)):
            out[k] = ""
        pos = i + len(base)
    return out


def _has_kanji(s: str) -> bool:
    return any("一" <= c <= "鿿" or c == "々" for c in s)


def judge(text, gold, pos, surface, info):
    """(對不對, 振假名讀音)；不能評回傳 None。"""
    if not _has_kanji(info.word) or text[pos:pos + len(surface)] != surface or not info.reading:
        return None
    if surface == info.word:
        span = gold[pos:pos + len(surface)]
        if any(g is None for g in span):
            return None
        g = "".join(span)
        return analysis.to_hira(info.reading) == g, g
    k = 0
    while k < min(len(surface), len(info.word)) and surface[k] == info.word[k]:
        k += 1
    if k == 0 or not _has_kanji(surface[:k]):
        return None
    span = gold[pos:pos + k]
    if any(g is None for g in span):
        return None
    g = "".join(span)
    return analysis.to_hira(info.reading).startswith(g), g


def run(out_path: str):
    d = Dictionary()
    calls = []
    orig_word, orig_expr = words._word_info, words._expression

    def word_info(t, text, pos_in_text, dictionary):
        key, info = orig_word(t, text, pos_in_text, dictionary)
        calls.append((pos_in_text, t.surface, info))
        return key, info

    def expression(form, text, pos_in_text, dictionary, compound):
        info = orig_expr(form, text, pos_in_text, dictionary, compound)
        if compound:  # 複合詞的寫法就是原文；慣用語（気になる）不評
            calls.append((pos_in_text, form, info))
        return info

    words._word_info, words._expression = word_info, expression
    n = ok = 0
    wrong = []
    for text, ruby in examples():
        gold = gold_chars(text, ruby)
        calls.clear()
        words.extract([text], d)
        for pos, surface, info in calls:
            r = judge(text, gold, pos, surface, info)
            if r is None:
                continue
            n += 1
            ok += r[0]
            if not r[0]:
                wrong.append({"text": text, "word": info.word, "reading": info.reading, "gold": r[1]})
    print(f"評估 {n} 個詞，讀音相符 {ok}（{ok / max(n, 1):.2%}），不符 {n - ok}")
    Path(out_path).write_text(json.dumps({"n": n, "ok": ok, "wrong": wrong}, ensure_ascii=False, indent=0),
                              encoding="utf-8")


def compare(a_path: str, b_path: str):
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (a_path, b_path))
    ka = {(w["text"], w["word"], w["reading"]) for w in a["wrong"]}
    kb = {(w["text"], w["word"], w["reading"]) for w in b["wrong"]}
    gold = {(w["text"], w["word"]): w["gold"] for w in a["wrong"] + b["wrong"]}
    print(f"前：{a['ok']}/{a['n']}　後：{b['ok']}/{b['n']}")
    print(f"\n只有前面錯（修好或不再列出）{len(ka - kb)}：")
    for t, w, r in sorted(ka - kb, key=lambda x: x[1]):
        print(f"  {w} {r}（振假名 {gold[(t, w)]}）｜{t[:50]}")
    print(f"\n只有後面錯 {len(kb - ka)}：")
    for t, w, r in sorted(kb - ka, key=lambda x: x[1]):
        print(f"  {w} {r}（振假名 {gold[(t, w)]}）｜{t[:50]}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])
    else:
        run(sys.argv[1])
