"""把 JLPT_grammer 裡現有規則還沒涵蓋的文法，分批交給 Gemini（網頁或 App）寫比對序列：匯出批次 → 貼回結果 → 匯入驗證。

    python tools/grammar_batches.py export            # 產生 grammar_gen/batches/*.txt（預設每批 30 條）
    python tools/grammar_batches.py import            # 讀 grammar_gen/results/*.txt，驗證後存進 jplens/jlearn/grammar_rules_extra.jsonl
    python tools/grammar_batches.py status            # 進度
    python tools/grammar_batches.py retry             # 沒通過、還沒回答的條目做成一批（附上錯誤原因與斷詞結果）

做法：
1. 每條附上例句的 Sudachi 斷詞結果（Gemini 跑不了 Sudachi，照著寫序列才會對），以及名稱相近的現有規則（讓它判斷是否已涵蓋）。
2. Gemini 每條回一行 JSON：序列、另外 2 句正例、2 句反例；太基本或已涵蓋的回 skip。
3. import 驗證（之後人工抽查，語意上的誤判寫進 grammar_gen/review.jsonl 退回）：序列能解析、要有指定文字的元素；例句與正例都比對得到、反例都比對不到；
   在語料（JLPT_grammer 所有例句＋video_translate 逐字稿，有的話）命中率不超過 MAX_RATE；
   不和現有規則在例句上比對到完全相同的範圍（重複）。通過的才寫進 grammar_rules_extra.jsonl。
4. 規則 id 由程式編（jg<級>_<編號>），名稱、級數、意思、接續、例句取自 JLPT_grammer，Gemini 只負責序列與測試句。

grammar_gen/batches/ 可隨時重新產生（不進版控）；results/ 與 grammar_rules_extra.jsonl 要進版控。
"""

import argparse
import json
import re
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jplens.jlearn import grammar  # noqa: E402
from jplens.jlearn.analysis import tokenize  # noqa: E402
from jplens.jlearn.grammar import EXTRA_PATH, compile_seq, find_spans  # noqa: E402
from jplens.jlearn.grammar_rules import RULES  # noqa: E402

SRC = ROOT / "JLPT_grammer"
WORK = ROOT / "grammar_gen"
BATCHES = WORK / "batches"
RESULTS = WORK / "results"
REPORT = WORK / "report.txt"
REVIEW = WORK / "review.jsonl"  # 人工（AI）抽查退回的序列：{"key", "seqs", "reason"[, "skip": true＝和別條重複]}；序列改了才重新驗證
TRANSCRIPTS = ROOT.parent / "video_translate" / "transcripts"  # 姊妹專案的真實逐字稿（只讀；沒有也能跑）
MAX_RATE = 0.05    # 語料命中率超過這個比例 → 退回（〜ている 約 11%、現有規則的中位數 0.3%）
WARN_RATE = 0.015  # 超過這個比例 → 列在報告裡給人抽查
ENTRY = re.compile(r"^(\d{3})\. 【(.+?)】\s*$")

PROMPT = """你是日語文法與自然語言處理的專家。我的程式用 Sudachi（SplitMode.C）把句子斷詞，再用「序列」比對文法句型。
請為下面每一條文法寫出比對序列。

## 序列的寫法
序列是以空白分隔的元素，每個元素比對「一個 Sudachi 詞」（題目附了例句的實際斷詞結果，請照著它寫）：

    て|で:接続助詞 しまう         「て」或「で」（接続助詞），後面接「しまう」
    :動詞;連用形 すぎる           任何動詞的連用形，後面接「すぎる」
    ない:助動詞;仮定形            「ない」（助動詞），活用形為仮定形
    か も しれる?                 結尾的 ? 表示這個元素可有可無
    ば * ほど                     * 比對任何一個詞（標點除外）
    ~過ぎる:動詞                  ~ 開頭：辭書形或標準寫法以此結尾的複合詞（食べ過ぎる）
    =ず:助動詞                    = 開頭：只比對表層形（「ません」的「ん」辭書形也是ず）
    よう に !思う|見える          ! 開頭：下一個詞「不是」這些（不佔位置）
    じゃ:助動詞 $                 $：句尾或標點之前（不佔位置）

- 文字部分比對「表層形／辭書形／標準寫法」任一個，所以寫辭書形就能比對到各種活用（「しまう」也比對到「しまっ」）。
- `:` 後是詞性，等於斷詞結果「詞性」欄任一段即可（動詞、助動詞、接続助詞、格助詞、非自立可能…）。
- `;` 後是活用形的開頭（未然形、連用形、終止形、連体形、仮定形、命令形、意志推量形…）。
- 「|」的選項沒寫詞性時，沿用右邊最近一個選項的條件：「ない|無い:形容詞」= 兩個都限形容詞。
  「|」分隔的是同一個詞的不同寫法，不能用來列多個詞性：「間|あいだ:名詞|:副詞可能」的「:副詞可能」是另一個選項，
  會比對到任何副詞可能的詞（上、前…）；詞性欄任一段符合即可，寫「間|あいだ:名詞」就好。
- 只有詞性、沒有文字的元素（:動詞;連用形）只當上下文，不算進句型範圍；每條序列至少要有一個指定文字的元素。
- 一條文法可以有好幾條序列（不同寫法、口語縮約、漢字／假名），任一條比對到就算。

## 現有規則的例子
{examples}

## 規則
1. 序列要「剛好」抓到這個句型：太寬會在一般句子大量誤判（例如只寫「の」），太窄會漏掉常見變化（活用、漢字／假名、口語縮約）。
   用詞性、活用形、`!`、`$` 把容易混淆的用法排除。
2. 太基本、比對出來只會洗版的（單一常見助詞或詞，例如 は、の、も、か、です、これ），或和題目附的「名稱相近的現有規則」其實是同一個句型的，
   回答 skip 並寫原因。
3. 每條另外寫 2 句正例（自然的日文，用到這個句型，最好和題目例句的活用不同）、2 句反例（含有相同或相似的字，但「不是」這個句型，
   序列不可以比對到）。
4. 只輸出一個程式碼區塊，每條一行 JSON，不要其他說明文字：
   {{"key": "N3-001", "seqs": ["序列1", "序列2"], "pos": ["正例1", "正例2"], "neg": ["反例1", "反例2"]}}
   {{"key": "N5-001", "skip": "原因"}}

## 題目
"""

EXAMPLE_RULES = ["teshimau", "nakereba", "tame_ni", "youni", "ni_kagitte", "zu", "kotoninaru", "bakari_only"]


# ---------------------------------------------------------------- 讀 JLPT_grammer

def load_entries() -> list[dict]:
    out = []
    for level in (5, 4, 3, 2, 1):
        cur = None
        for line in (SRC / f"jlpt_n{level}.txt").read_text(encoding="utf-8").splitlines():
            s = line.strip()
            m = ENTRY.match(s)
            if m:
                cur = {"key": f"N{level}-{m[1]}", "level": level, "pattern": m[2]}
                out.append(cur)
                continue
            # N2 有幾條標題寫成「〜といったらない -> 補充：〜とばかりに」，下一行才是真正的句型「【〜とばかりに】」
            # （N2-137 寫成「【〜にすぎない（…）】／【〜のみだ（唯有〜）】」：取最後一個）
            if s.startswith("【") and s.endswith("】") and cur is not None and "formation" not in cur:
                cur["pattern"] = re.findall(r"【(.+?)】", s)[-1]
                continue
            for k, tag in (("formation", "接續："), ("meaning", "語意："), ("example", "例句：")):
                if cur is not None and s.startswith(tag):
                    cur[k] = s[len(tag):]
    for e in out:
        ex = e.get("example", "")
        e["example_ja"] = re.split(r"[（(]", ex, 1)[0].strip()
    return out


def _names(p: str) -> set[str]:
    # 括號裡的讀音、漢字寫法、中文說明各自當一個名稱（「〜てください（請求・指示）」→ てください、請求、指示）
    p = re.sub(r"[（(]([^）)]*)[)）]", r"／\1", p)
    p = re.sub(r"[〜～]", "", p)
    return {x for x in re.split(r"[／/・、\s]", p) if x}


def related_rules(e) -> list:
    names = _names(e["pattern"])
    return [r for r in RULES if names & _names(r.pattern)]


def covered(e) -> bool:
    """現有規則已涵蓋：名稱相近的現有規則比對得到這條的例句。"""
    rel = related_rules(e)
    return bool(rel) and any(r in grammar.match_line(tokenize(e["example_ja"]), RULES) for r in rel)


def token_table(text: str) -> str:
    rows = []
    for t in tokenize(text):
        pos = ",".join(p for p in t.pos[:4] if p != "*")
        rows.append(f"  {t.surface}｜{t.lemma}｜{t.norm}｜{pos}｜{t.cform if t.cform != '*' else ''}")
    return "\n".join(rows)


def todo_entries() -> list[dict]:
    return [e for e in load_entries() if not covered(e)]


# ---------------------------------------------------------------- 匯出

def render_entry(e, note: str = "") -> str:
    lines = [f"【{e['key']}】{e['pattern']}", f"接續：{e.get('formation', '')}", f"語意：{e.get('meaning', '')}",
             f"例句：{e['example_ja']}", "例句斷詞（表層形｜辭書形｜標準寫法｜詞性｜活用形）：", token_table(e["example_ja"])]
    rel = related_rules(e)
    if rel:
        lines.append("名稱相近的現有規則：" + "；".join(f"{r.pattern}（{' / '.join(r.seqs)}）" for r in rel))
    if note:
        lines.append(note)
    return "\n".join(lines)


def prompt_header() -> str:
    by_id = {r.id: r for r in RULES}
    ex = "\n".join(f"- {by_id[i].pattern}：" + " / ".join(f"`{s}`" for s in by_id[i].seqs)
                   for i in EXAMPLE_RULES if i in by_id)
    return PROMPT.format(examples=ex)


def write_batch(name: str, blocks: list[str]):
    BATCHES.mkdir(parents=True, exist_ok=True)
    (BATCHES / f"{name}.txt").write_text(prompt_header() + "\n\n".join(blocks) + "\n", encoding="utf-8")


def cmd_export(size: int):
    todo = todo_entries()
    for old in BATCHES.glob("batch_*.txt"):
        old.unlink()
    n = 0
    for i in range(0, len(todo), size):
        n += 1
        write_batch(f"batch_{n:02d}", [render_entry(e) for e in todo[i:i + size]])
    print(f"現有規則已涵蓋 {len(load_entries()) - len(todo)} 條；其餘 {len(todo)} 條分成 {n} 批 → {BATCHES}")
    print("把批次檔交給 Gemini，回覆的程式碼區塊存成 grammar_gen/results/<批次名>.txt，再執行 import。")


# ---------------------------------------------------------------- 匯入與驗證

def read_answers() -> dict[str, dict]:
    """results/*.txt 裡的 JSON 行；同一條以檔名排序較後的為準（retry 的結果會蓋過原本的）。"""
    out = {}
    for p in sorted(RESULTS.glob("*.txt")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                a = json.loads(line)
            except ValueError:
                print(f"  {p.name}：無法解析的行：{line[:60]}")
                continue
            if "key" in a:
                a["_file"] = p.name
                out[a["key"]] = a
    return out


def corpus() -> list[tuple[str, list]]:
    sents = [e["example_ja"] for e in load_entries()]
    sents += [ex for r in RULES for ex in r.examples]
    if TRANSCRIPTS.is_dir():
        for p in TRANSCRIPTS.glob("*.txt"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    sents.append(re.sub(r"^\[[^\]]*\]\s*", "", line).split("｜", 1)[-1].strip())
    return [(s, tokenize(s)) for s in dict.fromkeys(sents) if s]


def _matches(seqs, text_or_tokens) -> bool:
    toks = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else text_or_tokens
    return any(find_spans(s, toks) for s in seqs)


def validate(e, a, corp) -> tuple[list[str], list[str], list[str]]:
    """回傳 (錯誤, 警告, 要附上斷詞結果給 Gemini 看的句子)。"""
    errors, warns, show = [], [], []
    seqs = a.get("seqs")
    if not isinstance(seqs, list) or not seqs or not all(isinstance(s, str) and s.strip() for s in seqs):
        return ["seqs 必須是非空的字串陣列"], [], []
    for s in seqs:
        try:
            elems = compile_seq(s)
        except Exception as ex:  # noqa: BLE001
            errors.append(f"序列無法解析：{s}（{ex}）")
            continue
        if not any(not el.kind and any(alt.has_text for alt in el.alts) for el in elems):
            errors.append(f"序列沒有任何指定文字的元素：{s}")
        # 「間|あいだ:名詞|:副詞可能」：最後的「:副詞可能」是另一個選項，會比對任何副詞可能的詞（上、前…）
        for raw, el in zip(s.split(), elems):
            if any(alt.has_text for alt in el.alts) and any(not alt.has_text for alt in el.alts):
                errors.append(f"元素「{raw}」混了有文字和只有詞性的選項，只有詞性的選項會比對到任何這種詞性的詞；"
                              "「|」分隔的是同一個詞的不同寫法，詞性只要寫一個（詞性欄任一段符合即可）")
    if errors:
        return errors, warns, show
    for label, sents, want in (("題目例句", [e["example_ja"]], True), ("正例", a.get("pos") or [], True),
                               ("反例", a.get("neg") or [], False)):
        for s in sents:
            if _matches(seqs, s) != want:
                errors.append(f"{label}{'比對不到' if want else '被比對到了'}：{s}")
                show.append(s)
    if len(a.get("pos") or []) < 2 or len(a.get("neg") or []) < 2:
        warns.append("正例或反例少於 2 句")
    own = {e["example_ja"], *(a.get("pos") or [])}
    hits = [s for s, toks in corp if s not in own and _matches(seqs, toks)]
    rate = len(hits) / max(1, len(corp))
    sample = "；".join(h[:30] for h in hits[:3])
    if rate > MAX_RATE:
        errors.append(f"太常比對到（語料 {len(corp)} 句裡 {len(hits)} 句，{rate:.1%}），可能誤判：{sample}")
    elif rate > WARN_RATE:
        warns.append(f"語料命中 {len(hits)} 句（{rate:.1%}），請抽查：{sample}")
    # 和現有規則在題目例句上比對到完全相同的範圍 → 重複
    toks = tokenize(e["example_ja"])
    mine = {(s0, s1) for s in seqs for s0, s1, _ in find_spans(s, toks)}
    for r in RULES:
        if any((s0, s1) in mine for sq in r.seqs for s0, s1, _ in find_spans(sq, toks)):
            errors.append(f"和現有規則「{r.pattern}」在例句比對到同一段，可能重複（是同一個句型就回答 skip）")
    # 加進去之後，現有規則的例句被這條（範圍更大的比對）蓋掉 → 同一個句型或序列太寬
    new = grammar.Rule("_new", e["pattern"], e["level"], "", "", tuple(seqs), ())
    for ex in dict.fromkeys([e["example_ja"], *(a.get("pos") or [])]):
        ex_toks = tokenize(ex)
        if _matches(seqs, ex_toks) and new not in grammar.match_line(ex_toks, RULES + [new]):
            by = [r.pattern for r in grammar.match_line(ex_toks, RULES)]
            errors.append(f"例句「{ex}」會被現有規則（{'、'.join(by)}）蓋掉，結果視窗不會顯示這條：是同一個句型就回答 skip")
    for r, ex, ex_toks in _rule_examples():
        if (_matches(seqs, ex_toks) and r in grammar.match_line(ex_toks, RULES)
                and r not in grammar.match_line(ex_toks, RULES + [new])):
            errors.append(f"會蓋掉現有規則「{r.pattern}」的例句「{ex}」：是同一個句型就回答 skip，不是就把序列改窄")
            show.append(ex)
    return errors, warns, show


@lru_cache(maxsize=None)
def _rule_examples() -> list:
    return [(r, ex, tokenize(ex)) for r in RULES for ex in r.examples]


def to_rule(e, a) -> dict:
    level, no = e["key"][1], e["key"].split("-")[1]
    return {"id": f"jg{level}_{no}", "key": e["key"], "pattern": e["pattern"], "level": e["level"],
            "meaning": e.get("meaning", ""), "formation": e.get("formation", ""), "seqs": a["seqs"],
            "examples": [e["example_ja"], *(a.get("pos") or [])]}


def evaluate():
    entries = {e["key"]: e for e in todo_entries()}
    review = {}
    if REVIEW.exists():
        for line in REVIEW.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                review[r["key"]] = r
    answers = read_answers()
    corp = corpus()
    ok, skipped, failed, warned = [], [], {}, {}
    for key, e in entries.items():
        a = answers.get(key)
        if a is None:
            continue
        if a.get("skip"):
            skipped.append((key, a["skip"]))
            continue
        errors, warns, show = validate(e, a, corp)
        if a.get("seqs") == review.get(key, {}).get("seqs"):
            if review[key].get("skip"):  # 抽查判定和別條重複：當成 skip，不必再請 Gemini 重寫
                skipped.append((key, "抽查：" + review[key]["reason"]))
                continue
            errors.append("抽查退回：" + review[key]["reason"])
        if errors:
            failed[key] = (errors, show, a)
        else:
            ok.append(to_rule(e, a))
            if warns:
                warned[key] = warns
    missing = [k for k in entries if k not in answers]
    ok, dups = _dedupe_levels(ok, answers)
    skipped += [(k, f"和 {q} 是同一個句型（JLPT_grammer 在不同級數各列一次），併進級數較簡單的 {q}") for k, q in dups]
    ok = _drop_hidden(ok, failed, answers)
    _check_claimed_dups(entries, ok, skipped, failed, answers)
    return entries, ok, skipped, failed, warned, missing


def _check_claimed_dups(entries, ok, skipped, failed, answers):
    """skip 的理由說「和〜重複／已涵蓋」，但點名的規則（N2-001 這類編號、「〜ている」這類現有規則名稱）其實比對不到這條的例句，
    結果視窗也沒有名稱相近的規則：Gemini 判斷錯就等於漏掉這個句型（現有的「〜てやまない」比對不到「祈ってやみません」）→ 退回。"""
    allr = RULES + [_as_rule(d) for d in ok]
    by_id = {r.id: r for r in allr}
    keep = []
    for k, why in skipped:
        named = []
        if re.search(r"重複|同一個句型|涵蓋", why):
            named = [by_id.get(f"jg{m[0]}_{m[1]}") for m in re.findall(r"N(\d)-(\d{3})", why)]
            named += [r for q in re.findall(r"「(〜[^」]+)」", why) for r in RULES if q.strip("〜") in r.pattern]
            named = [r for r in named if r is not None]
        ex = entries[k]["example_ja"]
        shown = grammar.match_line(tokenize(ex), allr) if named else []
        if named and not set(named) & set(shown) \
                and not any(_names(r.pattern) & _names(entries[k]["pattern"]) for r in shown):
            a = answers.get(k, {"skip": why})
            failed[k] = ([f"skip 的理由是「{why}」，但「{'、'.join(r.pattern for r in named)}」比對不到這條的例句「{ex}」"
                          f"（這句只會顯示：{'、'.join(r.pattern for r in shown) or '沒有任何文法'}）："
                          "請寫能比對到這句的序列；真的是同一個句型、只是寫法不同的話，序列要涵蓋這種寫法"], [ex], a)
        else:
            keep.append((k, why))
    skipped[:] = keep


def _as_rule(d: dict):
    return grammar.Rule(d["id"], d["pattern"], d["level"], d["meaning"], d["formation"], tuple(d["seqs"]), tuple(d["examples"]))


def _drop_hidden(ok: list[dict], failed: dict, answers: dict) -> list[dict]:
    """通過的規則一起加進去後，例句被別條（範圍更大、或同範圍但較難）蓋掉的：結果視窗看不到它，退回並寫明被誰蓋掉。
    一次退一條（被蓋掉最多句的先退），重算到沒有為止；被退的那條不再蓋別人。"""
    toks = {ex: tokenize(ex) for d in ok for ex in d["examples"]}
    while True:
        rules = {d["key"]: _as_rule(d) for d in ok}
        allr = RULES + list(rules.values())
        worst = None
        for d in ok:
            lost = [(ex, grammar.match_line(toks[ex], allr)) for ex in d["examples"]]
            lost = [(ex, m) for ex, m in lost if rules[d["key"]] not in m]
            if lost and (worst is None or len(lost) > len(worst[1])):
                worst = (d, lost)
        if worst is None:
            return ok
        d, lost = worst
        ex, by = lost[0]
        failed[d["key"]] = ([f"例句「{ex}」只會顯示（{'、'.join(r.pattern for r in by)}），這條被蓋掉：是同一個句型就回答 skip，不是就把序列改得更具體"],
                            [ex], answers[d["key"]])
        ok = [x for x in ok if x is not d]


def _dedupe_levels(ok: list[dict], answers: dict) -> tuple[list[dict], list[tuple[str, str]]]:
    """JLPT_grammer 同一個句型常在不同級數各列一次（〜まみれ N3／N1、〜のみならず N2／N1）。兩條都留的話比對到同一段，
    只會顯示較難的那條：名稱相近、又比對得到對方例句的，併成一條，名稱與級數用較簡單的（同級用編號小的）。"""
    toks = {r["key"]: tokenize(r["examples"][0]) for r in ok}
    order = sorted(ok, key=lambda r: (-r["level"], r["key"]))  # 較簡單的在前
    kept, dups = [], []
    for r in order:
        # 單向比對得到就算（寫法不同的兩條：一條涵蓋另一條的例句就會把它蓋掉）；名稱要相近（〜なり 和 〜なり〜なり 不相近）
        same = next((q for q in kept if _names(q["pattern"]) & _names(r["pattern"])
                     and (_matches(q["seqs"], toks[r["key"]]) or _matches(r["seqs"], toks[q["key"]]))), None)
        if same:
            # 併進較簡單的那條（較難的那條常寫得比較完整：寝るどころではない、仕事っぷり）。序列挑「兩條的例句都比對得到、
            # 兩條的反例都比對不到」的：較難那條的 → 較簡單那條的 → 聯集（〜のもとで 取聯集會失去 N1 版排除「机の下」的條件）
            exs = list(dict.fromkeys(same["examples"] + r["examples"]))
            negs = [n for k in (same["key"], r["key"]) for n in answers.get(k, {}).get("neg") or []]
            for cand in (r["seqs"], same["seqs"], list(dict.fromkeys(same["seqs"] + r["seqs"]))):
                if all(_matches(cand, x) for x in exs) and not any(_matches(cand, n) for n in negs):
                    same["seqs"], same["examples"] = list(cand), exs
                    break
            dups.append((r["key"], same["key"]))
        else:
            kept.append(dict(r))
    return [r for r in kept], dups


def cmd_import():
    entries, ok, skipped, failed, warned, missing = evaluate()
    EXTRA_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ok), encoding="utf-8")
    grammar.all_rules.cache_clear()
    # 加進去之後，每條的例句在「全部規則」一起比對時還看得到自己（沒被更完整的句型吃掉）
    hidden = [r.id for r in grammar.all_rules() if r.id.startswith("jg")
              and any(r not in grammar.match_line(tokenize(ex)) for ex in r.examples)]
    lines = [f"# 文法規則匯入報告（{datetime.now():%Y-%m-%d %H:%M}）",
             f"待處理 {len(entries)} 條：通過 {len(ok)}、skip {len(skipped)}、沒通過 {len(failed)}、還沒回答 {len(missing)}", ""]
    if failed:
        lines.append("## 沒通過（python tools/grammar_batches.py retry 會做成重試批次）")
        for k, (errs, _, a) in failed.items():
            lines += [f"- {k} {entries[k]['pattern']}（{a['_file']}）"] + [f"    {x}" for x in errs]
        lines.append("")
    if warned:
        lines.append("## 通過但請抽查")
        for k, ws in warned.items():
            lines += [f"- {k} {entries[k]['pattern']}"] + [f"    {x}" for x in ws]
        lines.append("")
    if hidden:
        lines += ["## 例句被別的規則（更完整的句型）蓋掉，結果視窗不會顯示這條（不一定是錯，請看一下）",
                  "  " + "、".join(hidden), ""]
    if skipped:
        lines.append("## skip")
        lines += [f"- {k} {entries[k]['pattern']}：{why}" for k, why in skipped]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:2]))
    print(f"通過的規則 → {EXTRA_PATH.relative_to(ROOT)}；詳細 → {REPORT.relative_to(ROOT)}")


def cmd_status():
    entries, ok, skipped, failed, warned, missing = evaluate()
    print(f"待處理 {len(entries)} 條：通過 {len(ok)}、skip {len(skipped)}、沒通過 {len(failed)}、還沒回答 {len(missing)}")
    for lv in (5, 4, 3, 2, 1):
        n = sum(1 for k in entries if k.startswith(f"N{lv}-"))
        d = sum(1 for r in ok if r["level"] == lv)
        print(f"  N{lv}：{n} 條，通過 {d}")


def cmd_retry(size: int):
    entries, ok, skipped, failed, warned, missing = evaluate()
    blocks = []
    for k, (errs, show, a) in failed.items():
        note = ["上次的回答：" + json.dumps({x: a[x] for x in ("seqs", "pos", "neg") if x in a}, ensure_ascii=False),
                "沒通過的原因：" + "；".join(errs)]
        for s in dict.fromkeys(show):
            if s != entries[k]["example_ja"]:
                note += [f"「{s}」的斷詞：", token_table(s)]
        blocks.append(render_entry(entries[k], "\n".join(note)))
    blocks += [render_entry(entries[k]) for k in missing]
    for old in BATCHES.glob("retry_*.txt"):
        old.unlink()
    for i in range(0, len(blocks), size):
        write_batch(f"retry_{i // size + 1:02d}", blocks[i:i + size])
    stamp = datetime.now().strftime("%m%d_%H%M")
    print(f"沒通過 {len(failed)} 條、還沒回答 {len(missing)} 條 → {BATCHES}/retry_*.txt")
    print(f"Gemini 的回覆請存成 grammar_gen/results/retry_{stamp}_01.txt 這類檔名（排在原本的結果後面，才會蓋過舊答案）。")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import", "status", "retry"])
    ap.add_argument("--size", type=int, default=30)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"export": lambda: cmd_export(args.size), "import": cmd_import, "status": cmd_status,
     "retry": lambda: cmd_retry(args.size)}[args.cmd]()


if __name__ == "__main__":
    main()
