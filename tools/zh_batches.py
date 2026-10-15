"""把日檢單字表的詞分批交給 Gemini（網頁或 App）翻成中文：匯出批次 → 貼回結果 → 匯入檢查。

    python tools/zh_batches.py export            # 產生 dict_translate/batches/*.txt（預設每批 150 個詞）
    python tools/zh_batches.py export --size 100 --levels 3,2   # 每次都匯出全部，批次編號固定
    python tools/zh_batches.py import            # 讀 dict_translate/results/*.txt，檢查後存進 dict_translate/zh.jsonl
    python tools/zh_batches.py status            # 進度：哪些批次完成、哪些詞缺漏或格式不對
    python tools/zh_batches.py retry             # 把缺漏、格式不對的詞重新做成一批

做法：
1. 每個詞以 JMdict 的條目編號（ent_seq）當編號，附上 JMdict 每個意思的詞性與英文（當參考），
   要 Gemini 依日文原意、逐個意思翻成台灣繁體中文。逐個意思翻是為了保留詞性，查字時才能挑對意思。
2. 把批次檔整個上傳（或貼上）給 Gemini，把它回覆的程式碼區塊內容存成 dict_translate/results/<批次名>.txt。
3. import 會檢查編號、意思數量；有簡體字的行只列出來給人看，不自動轉換
   （OpenCC 會把正確的繁體改壞：掛斷 → 結束通話、傢伙 → 傢夥）。

dict_translate/batches/ 可隨時重新產生（不進版控）；results/ 與 zh.jsonl 是翻譯成果，要進版控。
"""

import argparse
import gzip
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jplens.jlearn import analysis  # noqa: E402
from jplens.words import _jlpt_lists  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zhconv import simplified_chars  # noqa: E402

WORK = ROOT / "dict_translate"
BATCHES = WORK / "batches"
RESULTS = WORK / "results"
OUT = WORK / "zh.jsonl"
MANIFEST = WORK / "batches" / "manifest.json"
JMDICT = ROOT / "data" / "raw" / "JMdict_e.gz"
MAX_SENSES = 5

PROMPT = """你是日語辭典的編輯。請把下面每個日文詞的「每一個意思」翻成台灣繁體中文。

規則：
1. 依日文原本的意思翻譯；英文只是參考，不要逐字翻英文。
2. 每個意思寫一個簡短的中文解釋（盡量 15 字以內），近義說法用「、」分隔。
3. 意思的數量和順序必須和題目完全相同（題目有 3 個意思就回 3 個，順序不變）。
4. 功能詞（助詞、補助動詞、接尾詞等）用一句話說明作用，例如「表示…」。
5. 只輸出一個程式碼區塊，每個詞一行，格式：編號｜意思1｜意思2｜…（用全形「｜」分隔），
   不要加任何其他說明文字。

範例
題目：
1428830 眺める（ながめる）
  ① 動詞：to look at; to gaze at; to watch; to stare at
  ② 動詞：to look out over; to get a view of; to admire (e.g. the scenery)
  ③ 動詞：to look on (from the sidelines); to stand by and watch
回答：
1428830｜注視、凝望｜眺望、遠眺｜旁觀、袖手旁觀

以下是題目（共 {n} 個詞）：
"""

# JMdict 詞性 → 簡短中文（只給 Gemini 參考）
_POS = [("'no'", "名詞"), ("expression", "慣用語"), ("adverb", "副詞"), ("adjective (keiyoushi)", "い形容詞"),
        ("keiyodoshi", "な形容詞"), ("'taru'", "な形容詞"), ("pre-noun", "連體詞"), ("prenominally", "連體詞"),
        ("suru verb", "する動詞"), ("verb", "動詞"), ("pronoun", "代名詞"), ("interjection", "感嘆詞"),
        ("conjunction", "接續詞"), ("counter", "助數詞"), ("prefix", "接頭詞"), ("suffix", "接尾詞"),
        ("particle", "助詞"), ("auxiliary", "助動詞"), ("numeric", "數詞"), ("noun", "名詞")]


def pos_label(pos: str) -> str:
    pos = pos.lower()
    labels = []
    for p in pos.split("|"):
        lab = next((zh for key, zh in _POS if key in p), "")
        if lab and lab not in labels:
            labels.append(lab)
    return "、".join(labels) or "其他"


def load_jmdict() -> dict[str, list[dict]]:
    """寫法或讀音 → JMdict 條目（ent_seq、寫法、讀音、常用度標記、各意思的詞性與英文）。"""
    index: dict[str, list[dict]] = {}
    with gzip.open(JMDICT) as f:
        for _, el in ET.iterparse(f):
            if el.tag != "entry":
                continue
            kebs = [k.text for k in el.iter("keb")]
            rebs = [r.text for r in el.iter("reb")]
            pri = [p.text for tag in ("ke_pri", "re_pri") for p in el.iter(tag) if p.text]
            common = bool(pri)
            senses, cur = [], ""
            for s in el.iter("sense"):
                pos = "|".join(p.text for p in s.iter("pos") if p.text)
                cur = pos or cur
                gl = "; ".join(g.text for g in s.iter("gloss") if g.text)
                if gl:
                    senses.append((cur, gl))
            e = {"id": el.findtext("ent_seq"), "kebs": kebs, "rebs": [analysis.to_hira(r) for r in rebs],
                 "rebs_raw": rebs, "common": common, "pri": pri, "senses": senses}
            el.clear()
            for k in kebs + rebs:
                index.setdefault(k, []).append(e)
    return index


_STOP = {"the", "and", "for", "with", "one", "something", "someone", "etc", "esp", "used", "being"}


def _en_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def match(index, word: str, reading: str, meaning: str = "") -> list[dict]:
    """日檢單字表的詞 → JMdict 條目：寫法相同且讀音相同；有好幾個時，取英文意思和單字表最像的，再取常用的。
    （單字表的「いくら」是「多少」，不能配到同樣唸 いくら 的「イクラ（鮭魚卵）」）
    單字表一次列了兩個同音詞時（コック：cook; tap），另一個條目涵蓋了其餘的意思就一起回傳。"""
    cands = index.get(word, [])
    rd = analysis.to_hira(reading)
    exact = [e for e in cands if rd in e["rebs"]] or cands
    if not exact:
        return []
    ref = _en_words(meaning)

    def overlap(e):
        return ref & _en_words(" ".join(g for _, g in e["senses"]))
    ranked = sorted(exact, key=lambda e: (len(overlap(e)), e["common"]), reverse=True)
    out, covered = [ranked[0]], overlap(ranked[0])
    for e in ranked[1:]:
        extra = overlap(e) - covered
        if extra:
            out.append(e)
            covered |= extra
    return out


def collect(levels: list[int]):
    print("讀取 JMdict（約 20 秒）…", flush=True)
    index = load_jmdict()
    items, seen, missing = [], set(), 0
    by_word = _jlpt_lists()[0]
    for level in levels:
        for word, entries in by_word.items():
            for je in entries:
                if je.level != level:
                    continue
                found = match(index, je.word, je.reading, je.meaning)
                if not found:
                    missing += 1
                    digest = hashlib.md5(f"{je.word}\t{je.reading}".encode()).hexdigest()
                    new = [{"id": f"J{int(digest, 16) % 10**7:07d}", "word": je.word,
                            "reading": je.reading, "level": level, "senses": [["", je.meaning]]}]
                else:
                    new = [{"id": e["id"], "word": je.word, "reading": je.reading, "level": level,
                            "senses": [list(s) for s in e["senses"][:MAX_SENSES]]} for e in found]
                for item in new:
                    if item["id"] not in seen:
                        seen.add(item["id"])
                        items.append(item)
    print(f"共 {len(items)} 個詞（JMdict 找不到、改用單字表英文的 {missing} 個）")
    return items


def render(items) -> str:
    lines = [PROMPT.format(n=len(items))]
    marks = "①②③④⑤⑥⑦⑧⑨⑩"
    for it in items:
        head = it["word"] if it["reading"] == it["word"] else f"{it['word']}（{it['reading']}）"
        lines.append(f"{it['id']} {head}")
        for k, (pos, gloss) in enumerate(it["senses"]):
            gloss = gloss if len(gloss) <= 110 else gloss[:107] + "…"
            lines.append(f"  {marks[k]} {pos_label(pos) + '：' if pos else ''}{gloss}")
    return "\n".join(lines) + "\n"


def load_manifest() -> dict:
    """{"items": 編號 → 詞（匯出過的都留著，匯入時用來核對）, "batches": 批次名 → [編號…]}"""
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"items": {}, "batches": {}}


def write_batches(items, size: int, prefix: str, replace: bool):
    """replace：先刪掉同類（同前綴）的舊批次檔，重新編號。"""
    BATCHES.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    if replace:
        for name in [k for k in manifest["batches"] if k.startswith(prefix)]:
            del manifest["batches"][name]
            (BATCHES / f"{name}.txt").unlink(missing_ok=True)
    names = []
    for n, i in enumerate(range(0, len(items), size), 1):
        part = items[i:i + size]
        name = f"{prefix}{n:03d}"
        (BATCHES / f"{name}.txt").write_text(render(part), encoding="utf-8")
        manifest["batches"][name] = [it["id"] for it in part]
        manifest["items"].update({it["id"]: it for it in part})
        names.append(name)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    if names:
        print(f"產生 {len(names)} 批：{names[0]} … {names[-1]}（{BATCHES}）")


def load_done() -> dict:
    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    return done


def save_done(done: dict):
    WORK.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()), encoding="utf-8")


def cmd_export(args):
    """每次都匯出全部的詞、順序固定，所以批次編號不會變（已存好的 results/batch_001.txt 仍然對得上）；
    哪些詞翻好了看 status，缺的用 retry 補。挑選 JMdict 條目的方式改過時，舊的翻譯若已不對應就刪掉。"""
    levels = [int(x) for x in args.levels.split(",")]
    items = collect(levels)
    current: dict[tuple, set] = {}
    for it in items:
        current.setdefault((it["word"], it["reading"]), set()).add(it["id"])
    done = load_done()
    stale = [r for r in done.values() if r["id"] not in current.get((r["word"], r["reading"]), {r["id"]})]
    if stale:
        for r in stale:
            del done[r["id"]]
            print(f"  {r['word']}（{r['reading']}）改配到別的 JMdict 條目，舊的翻譯刪除，之後用 retry 重翻")
        save_done(done)
    write_batches(items, args.size, "batch_", replace=True)


_LINE = re.compile(r"^\s*`*\s*(J?\d{5,8})\s*[｜|]\s*(.+?)\s*$")


def cmd_import(args):
    manifest = load_manifest()
    # 只收目前批次裡的編號：挑條目的方式改過時，舊編號（例如配錯的イクラ）不再收
    current = {i for ids in manifest["batches"].values() for i in ids}
    expect = {i: it for i, it in manifest["items"].items() if i in current}
    done = load_done()
    ok = bad = 0
    problems = []
    for f in sorted(RESULTS.glob("*.txt")):
        for raw in f.read_text(encoding="utf-8").splitlines():
            m = _LINE.match(raw)
            if not m:
                continue
            id_, rest = m.groups()
            senses = [s.strip() for s in re.split(r"[｜|]", rest) if s.strip()]
            it = expect.get(id_)
            if it is None:
                problems.append(f"{f.name}：編號 {id_} 不在目前的批次裡（可能是挑條目的方式改過，或打錯）")
                bad += 1
                continue
            if len(senses) != len(it["senses"]):
                problems.append(f"{f.name}：{id_} {it['word']} 應有 {len(it['senses'])} 個意思，回了 {len(senses)} 個")
                bad += 1
                continue
            sc = simplified_chars("".join(senses))
            if sc:
                problems.append(f"{f.name}：{id_} {it['word']} 可能有簡體字 {''.join(sc)}（已匯入，請人工確認）")
            done[id_] = {"id": id_, "word": it["word"], "reading": it["reading"], "level": it["level"],
                         "pos": [p for p, _ in it["senses"]], "en": [g for _, g in it["senses"]], "zh": senses,
                         "source": f.stem, "imported": datetime.now().isoformat(timespec="seconds")}
            ok += 1
    save_done(done)
    print(f"匯入 {ok} 行、有問題 {bad} 行；累計已翻譯 {len(done)} 個詞 → {OUT}")
    for p in problems[:30]:
        print("  " + p)
    if len(problems) > 30:
        print(f"  …還有 {len(problems) - 30} 個問題")


def cmd_status(args):
    manifest = load_manifest()
    done = load_done()
    print(f"已翻譯 {len(done)} 個詞（匯出過 {len(manifest['items'])} 個）")
    for name, ids in manifest["batches"].items():
        n = sum(1 for i in ids if i in done)
        mark = "完成" if n == len(ids) else ("未開始" if n == 0 else f"缺 {len(ids) - n} 個")
        print(f"  {name}：{n}/{len(ids)}　{mark}")


def cmd_retry(args):
    """把目前批次裡還沒成功匯入的詞（Gemini 漏掉、意思數量不對）重新做成 retry_xxx 批次。"""
    manifest = load_manifest()
    done = load_done()
    todo = list({i: manifest["items"][i] for ids in manifest["batches"].values() for i in ids
                 if i not in done}.values())
    if not todo:
        print("沒有缺漏的詞")
        return
    write_batches(todo, args.size, "retry_", replace=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--size", type=int, default=150, help="每批幾個詞（預設 150）")
    e.add_argument("--levels", default="5,4,3,2,1", help="要匯出的級數與順序，例如 3,2")
    sub.add_parser("import")
    sub.add_parser("status")
    r = sub.add_parser("retry")
    r.add_argument("--size", type=int, default=150)
    args = ap.parse_args()
    {"export": cmd_export, "import": cmd_import, "status": cmd_status, "retry": cmd_retry}[args.cmd](args)


if __name__ == "__main__":
    main()
