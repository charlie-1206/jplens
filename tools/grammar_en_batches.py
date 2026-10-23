"""文法說明的英文版：分批交給 Gemini（網頁或 App）翻成英文：匯出批次 → 貼回結果 → 匯入檢查。

    python tools/grammar_en_batches.py export     # 產生 grammar_gen/en_batches/*.txt（預設每批 40 條）
    python tools/grammar_en_batches.py import     # 讀 grammar_gen/en_results/*.txt，檢查後存進 jplens/jlearn/grammar_en.json
    python tools/grammar_en_batches.py status     # 進度：哪些條目翻好了、哪些缺漏或沒通過檢查
    python tools/grammar_en_batches.py retry      # 缺漏、沒通過的條目重新做成一批（附上沒通過的原因）

要翻的（解釋的語言選 English 時顯示，lang.english()）：
- 手寫的規則（grammar_rules.py，編號 R:<規則 id>）：說明、接續；名稱裡有中文註解的（〜から（原因））連名稱一起
- JLPT_grammer 的條目（grammar_notes.json 的滑鼠提示，和 grammar_rules_extra.jsonl 的規則共用；編號 J:<key>）：
  說明、接續、例句的翻譯（例句的日文不動）
做法：
1. 把批次檔整個上傳（或貼上）給 Gemini，把它回覆的程式碼區塊內容存成 grammar_gen/en_results/<批次名>.txt。
2. Gemini 每條回一行 JSON；import 檢查編號、欄位、是不是英文（中文字太多的退回）、接續裡不能留中文的文法用語（動詞、名詞…）、
   名稱的日文部分不能被改。
grammar_gen/en_batches/ 可隨時重新產生（不進版控）；en_results/ 與 grammar_en.json 是翻譯成果，要進版控。
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from jplens.jlearn.grammar import EXTRA_PATH  # noqa: E402
from jplens.jlearn.grammar_rules import RULES  # noqa: E402

WORK = ROOT / "grammar_gen"
BATCHES = WORK / "en_batches"
RESULTS = WORK / "en_results"
OUT = ROOT / "jplens" / "jlearn" / "grammar_en.json"
NOTES = ROOT / "jplens" / "jlearn" / "grammar_notes.json"
_IDEO = re.compile(r"[一-鿿]")
_KANA = re.compile(r"[぀-ヿ]")
_NOTE = re.compile(r"（[^）]*[一-鿿][^）]*）")  # 名稱裡含漢字的括號註解（原因）（驚訝・意外・輕視）
# 接續欄翻好後不該留下的中文文法用語
ZH_TERMS = ("動詞", "名詞", "形容詞", "形容動詞", "普通形", "辭書形", "辞書形", "連用形", "原形", "詞幹", "語幹", "子句")

HEADER = """You are an editor of a Japanese grammar dictionary for English-speaking learners (JLPT N5–N1).
Below, each grammar point has an explanation written in Traditional Chinese. Translate it into English.

Rules:
1. Translate the meaning faithfully and concisely, in the style of a grammar textbook (e.g. "to suggest doing something together (let's ~)").
   Keep Japanese words, 「」 quotes and the ～/〜 placeholder as they are.
2. "formation": write it in English with the usual textbook terms, keeping the Japanese parts:
   e.g. 動詞ます形＋ましょう → "Verb (masu-stem) + ましょう", 名詞＋の＋あげく → "Noun + の + あげく",
   い形容詞 → "i-adjective", な形容詞 → "na-adjective", 普通形 → "plain form", 辞書形 → "dictionary form".
3. "example_en" (only for items that have an example): translate the Japanese example sentence into natural English.
   The Chinese translation in parentheses is only a reference.
4. "pattern" (only for items marked [pattern note]): the pattern name has a note in Chinese inside （）.
   Return the whole name with only that note translated into English, e.g. 〜から（原因） → 〜から（reason）. Do not change the Japanese.
5. Output ONE code block only, one JSON object per line, in the same order as the items, with no other text:
   {"id": "...", "meaning": "...", "formation": "..."[, "example_en": "..."][, "pattern": "..."]}

Example
Item:
J:N2-069  〜あげく（に）／〜あげくの  (N2)
  meaning: 歷經一番〜之後，結果卻〜（經過長久折騰，最終迎來往往是不好的結果）
  formation: 動詞た形／名詞＋の＋あげく（に）
  example: 散々迷ったあげく、結局何も買わなかった。（猶豫了老半天，結果最後什麼也沒買。）
Answer:
{"id": "J:N2-069", "meaning": "after a long process of ~, in the end ~ (the result after much trouble, usually a bad one)", "formation": "Verb (ta-form) / Noun + の + あげく（に）", "example_en": "After agonizing over it for ages, I ended up not buying anything."}

Items ({n} in total):
"""


def items() -> list[dict]:
    """要翻的條目：{id, pattern, level, meaning, formation, example（只有 J:）, note（名稱要一起翻）}。"""
    out = []
    for r in RULES:
        out.append({"id": f"R:{r.id}", "pattern": r.pattern, "level": r.level, "meaning": r.meaning,
                    "formation": r.formation, "note": bool(_NOTE.search(r.pattern))})
    jlpt = {}
    for lst in json.loads(NOTES.read_text(encoding="utf-8")).values():
        for n in lst:
            jlpt.setdefault(n["key"], n)
    for e in extra_rules():
        jlpt.setdefault(e["key"], e)  # 規則的說明取自 JLPT_grammer 同一條；滑鼠提示沒用到的也要翻
    for key, n in sorted(jlpt.items()):
        out.append({"id": f"J:{key}", "pattern": n["pattern"], "level": n["level"], "meaning": n.get("meaning", ""),
                    "formation": n.get("formation", ""), "example": n.get("example", ""),
                    "note": bool(_NOTE.search(n["pattern"]))})
    return out


def extra_rules() -> list[dict]:
    return [json.loads(ln) for ln in EXTRA_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]


def render(it: dict, why: str = "") -> str:
    lv = f"N{it['level']}" if it["level"] else "non-JLPT"
    lines = [f"{it['id']}  {it['pattern']}  ({lv})" + ("  [pattern note]" if it["note"] else "")]
    lines.append(f"  meaning: {it['meaning']}")
    lines.append(f"  formation: {it['formation']}")
    if it.get("example"):
        lines.append(f"  example: {it['example']}")
    if why:
        lines.append(f"  (previous answer was rejected: {why})")
    return "\n".join(lines)


def write_batch(name: str, part: list[dict], whys: dict | None = None):
    BATCHES.mkdir(parents=True, exist_ok=True)
    body = "\n".join(render(it, (whys or {}).get(it["id"], "")) for it in part)
    (BATCHES / f"{name}.txt").write_text(HEADER.replace("{n}", str(len(part))) + "\n" + body + "\n", encoding="utf-8")


def cmd_export(size: int):
    its = items()
    for i in range(0, len(its), size):
        write_batch(f"batch_{i // size + 1:03d}", its[i:i + size])
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = (len(its) + size - 1) // size
    print(f"{len(its)} 條、{n} 批 → {BATCHES}")
    print(f"把每批丟給 Gemini，回覆的程式碼區塊存成 {RESULTS}\\<同名>.txt，再執行 import")


# ---------------------------------------------------------------- 匯入、檢查
def read_answers() -> dict[str, dict]:
    """results 裡所有檔案的 JSON 行；同一個編號後面的檔案（retry）蓋前面的。"""
    out = {}
    for p in sorted(RESULTS.glob("*.txt")):
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip().rstrip(",")
            if not ln.startswith("{"):
                continue
            try:
                a = json.loads(ln)
            except ValueError:
                continue
            if isinstance(a, dict) and isinstance(a.get("id"), str):
                out[a["id"]] = a
    return out


def english(s: str) -> bool:
    """大部分是英文：拉丁字母比漢字多很多（日文的例字、「」引用可以有）。"""
    letters = sum(c.isascii() and c.isalpha() for c in s)
    return letters >= 3 and len(_IDEO.findall(s)) * 2 <= letters // 4 + 4


def check(it: dict, a: dict) -> str:
    """回傳沒通過的原因（空字串＝通過）。"""
    for k in ("meaning", "formation"):
        if not isinstance(a.get(k), str) or not a[k].strip():
            return f"{k} 是空的"
    if not english(a["meaning"]):
        return "meaning 不像英文"
    if not re.search(r"[A-Za-z]", a["formation"]) and it["formation"] and _IDEO.search(it["formation"]):
        return "formation 沒有翻"
    bad = [t for t in ZH_TERMS if t in a["formation"]]
    if bad:
        return f"formation 還有中文用語：{'、'.join(bad)}"
    if it.get("example"):
        if not isinstance(a.get("example_en"), str) or not english(a["example_en"]):
            return "example_en 是空的或不像英文"
    if it["note"]:
        p = a.get("pattern")
        if not isinstance(p, str):
            return "pattern 沒有回（名稱的中文註解要翻）"
        strip = lambda s: re.sub(r"（[^）]*）", "", s)
        if strip(p) != strip(it["pattern"]):
            return f"pattern 的日文部分被改了：{p}"
        if _NOTE.search(p) and not re.search(r"[A-Za-z]", p):
            return "pattern 的中文註解沒有翻"
    return ""


def evaluate() -> tuple[dict, dict, list]:
    """回傳（通過的 {id: 答案}, 沒通過的 {id: 原因}, 還沒回答的 [條目]）。"""
    answers = read_answers()
    ok, failed, missing = {}, {}, []
    for it in items():
        a = answers.get(it["id"])
        if a is None:
            missing.append(it)
            continue
        why = check(it, a)
        if why:
            failed[it["id"]] = why
        else:
            ok[it["id"]] = a
    return ok, failed, missing


def _japanese_part(example: str) -> str:
    """「日文。（中譯）」的日文部分。"""
    return re.sub(r"（[^）]*）\s*$", "", example).strip()


def cmd_import():
    ok, failed, missing = evaluate()
    by_id = {it["id"]: it for it in items()}
    rules, jlpt = {}, {}
    for i, a in ok.items():
        d = {"meaning": a["meaning"].strip(), "formation": a["formation"].strip()}
        if by_id[i]["note"]:
            d["pattern"] = a["pattern"].strip()
        if i.startswith("R:"):
            rules[i[2:]] = d
        else:
            d["example"] = f"{_japanese_part(by_id[i]['example'])}（{a['example_en'].strip()}）"
            jlpt[i[2:]] = d
    for e in extra_rules():  # Gemini 寫的規則：說明、接續就是 JLPT_grammer 那條的
        if e["key"] in jlpt:
            rules[e["id"]] = {k: v for k, v in jlpt[e["key"]].items() if k != "example"}
    OUT.write_text(json.dumps({"rules": rules, "jlpt": jlpt}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"通過 {len(ok)}、沒通過 {len(failed)}、還沒回答 {len(missing)} → {OUT}（規則 {len(rules)}、JLPT 條目 {len(jlpt)}）")
    for i, why in list(failed.items())[:30]:
        print(f"  ✗ {i}：{why}")


def cmd_status():
    ok, failed, missing = evaluate()
    total = len(ok) + len(failed) + len(missing)
    print(f"共 {total} 條：通過 {len(ok)}、沒通過 {len(failed)}、還沒回答 {len(missing)}")
    done = {p.stem for p in RESULTS.glob("*.txt")}
    todo = [p.stem for p in sorted(BATCHES.glob("batch_*.txt")) if p.stem not in done]
    if todo:
        print("還沒有結果檔的批次：" + "、".join(todo))


def cmd_retry(size: int):
    ok, failed, missing = evaluate()
    its = [it for it in items() if it["id"] in failed] + missing
    if not its:
        print("全部通過，不用重做")
        return
    n = len(list(BATCHES.glob("retry_*.txt"))) + 1
    for i in range(0, len(its), size):
        write_batch(f"retry_{n:02d}_{i // size + 1}", its[i:i + size], failed)
    print(f"{len(its)} 條（沒通過 {len(failed)}、還沒回答 {len(missing)}）→ {BATCHES}\\retry_{n:02d}_*.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import", "status", "retry"])
    ap.add_argument("--size", type=int, default=40)
    a = ap.parse_args()
    {"export": lambda: cmd_export(a.size), "import": cmd_import, "status": cmd_status,
     "retry": lambda: cmd_retry(a.size)}[a.cmd]()


if __name__ == "__main__":
    main()
