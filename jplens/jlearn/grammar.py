"""文法規則比對：在 Sudachi 斷詞結果上找出常見的 N5～N1 文法。

規則寫在 grammar_rules.py。每條規則的「序列」是以空白分隔的元素，每個元素比對一個詞：

    て|で:接続助詞 しまう         「て」或「で」（接続助詞），後面接「しまう」
    :動詞;連用形 すぎる           任何動詞的連用形，後面接「すぎる」
    ない:助動詞;仮定形            「ない」（助動詞），活用形為仮定形
    か も しれる?                 結尾的 ? 表示這個元素可有可無
    ば * ほど                     * 比對任何一個詞（標點除外）
    ~過ぎる:動詞                  ~ 開頭：辭書形或標準寫法以此結尾的複合詞（食べ過ぎる）
    動詞~直す:動詞                ~ 前面寫「動詞」：去掉字尾後剩下的要是動詞連用形（練り直す 的 練り），
                                  「~方」才不會比對到 夕方、両方、粗方（あらかた）這類一般名詞
    =ず:助動詞                    = 開頭：只比對表層形（「ません」的「ん」辭書形也是ず）
    よう に !思う|見える          ! 開頭：下一個詞「不是」這些（不佔位置）
    じゃ:助動詞 $                 $：句尾或標點之前（不佔位置）

- 文字部分比對 表層形／辭書形／標準寫法 任一個（所以「しまう」也會比對到「しまっ」）。
- `:` 後是詞性，等於 Sudachi 6 欄詞性中的任一欄即可（動詞、助動詞、接続助詞…）。
- `;` 後是活用形的開頭（未然形、連用形、仮定形、命令形、意志推量形…）。
- 沒寫詞性的選項沿用右邊選項的條件：「ない|無い:形容詞」= 兩個都限形容詞。
- 只有詞性條件、沒有指定文字的元素（例如 :動詞;連用形）只當上下文，不算進句型範圍。
- 同一句裡，被另一個比對結果完全包住的比對會被移除（例如「〜なければならない」
  會吃掉「〜ば」），讓列出來的都是最完整的句型。
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .analysis import Token, tokenize

EXTRA_PATH = Path(__file__).with_name("grammar_rules_extra.jsonl")  # JLPT_grammer 補充的規則（tools/grammar_batches.py）


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: str       # 顯示用名稱，例如「〜てしまう」
    level: int | None  # 5～1；None = 級外（老人語、方言等角色語）
    meaning: str
    formation: str
    seqs: tuple        # 一或多個序列字串，任一個比對成功即可
    examples: tuple    # 測試用例句（每句都必須比對得到）

    @property
    def jlpt(self) -> str:
        return f"N{self.level}" if self.level else "級外"


@dataclass
class GrammarHit:
    rule: Rule
    line_ids: list[int]


# ---------------------------------------------------------------- 序列比對

@dataclass(frozen=True)
class _Alt:
    words: frozenset = frozenset()  # 空 = 不限
    pos: str = ""
    form: str = ""
    any: bool = False
    suffix: str = ""
    surface_only: bool = False
    head_verb: bool = False  # 字尾前面要是動詞連用形（動詞~直す）

    @property
    def has_text(self) -> bool:
        return bool(self.words or self.suffix or self.any)

    def match(self, t: Token) -> bool:
        if self.any:
            return not t.is_punct()
        if self.suffix:
            if compound_head(self, t) is None:
                return False
        elif self.words:
            cands = {t.surface} if self.surface_only else {t.surface, t.lemma, t.norm}
            if not cands & self.words:
                return False
        if self.pos and self.pos not in t.pos:
            return False
        if self.form and not t.cform.startswith(self.form):
            return False
        return True


# 動詞連用形的最後一個假名（い段、え段）：読み、使い、考え、食べ
_REN_KANA = set("いきぎしじちぢにひびぴみりえけげせぜてでねへべぺめれ")


@lru_cache(maxsize=4096)
def _is_verb_stem(head: str) -> bool:
    """head 是不是動詞連用形（複合詞去掉字尾剩下的部分）：結尾是い段／え段的假名（読み、考え），
    或整段被 Sudachi 斷成一個動詞連用形（見方 的 見、練り）。夕、両、粗（形容詞語幹）都不是。"""
    if not head:
        return False
    if head[-1] in _REN_KANA:
        return True
    toks = tokenize(head)
    return len(toks) == 1 and toks[0].pos1 == "動詞" and toks[0].cform.startswith("連用形")


def compound_head(alt: "_Alt", t: Token) -> str | None:
    """字尾元素（~直す）比對得到這個詞時，回傳去掉字尾剩下的部分（練り直す → 練り）；比對不到回傳 None。"""
    for w in (t.lemma, t.norm):
        if w.endswith(alt.suffix) and w != alt.suffix:
            head = w[:-len(alt.suffix)]
            if not alt.head_verb or _is_verb_stem(head):
                return head
    return None


@dataclass(frozen=True)
class _Elem:
    alts: tuple
    optional: bool = False
    kind: str = ""      # "" 一般；"end" 句尾；"not" 否定前瞻

    @property
    def context(self) -> bool:
        """只有詞性條件的元素：當上下文用，不算進句型的範圍。"""
        return not self.kind and not any(a.has_text for a in self.alts)

    def match(self, t: Token) -> bool:
        return any(a.match(t) for a in self.alts)


def _parse_alt(s: str) -> _Alt:
    if s == "*":
        return _Alt(any=True)
    form = pos = ""
    if ";" in s:
        s, form = s.split(";", 1)
    if ":" in s:
        s, pos = s.split(":", 1)
    if s.startswith("~"):
        return _Alt(pos=pos, form=form, suffix=s[1:])
    if s.startswith("動詞~") and len(s) > 3:
        return _Alt(pos=pos, form=form, suffix=s[3:], head_verb=True)
    if s.startswith("="):
        return _Alt(frozenset([s[1:]]), pos, form, surface_only=True)
    return _Alt(frozenset([s]) if s else frozenset(), pos, form)


@lru_cache(maxsize=None)
def compile_seq(seq: str) -> tuple:
    elems = []
    for raw in seq.split():
        if raw == "$":
            elems.append(_Elem((), kind="end"))
            continue
        kind = ""
        if raw.startswith("!") and len(raw) > 1:
            kind, raw = "not", raw[1:]
        optional = raw.endswith("?") and len(raw) > 1
        raw = raw[:-1] if optional else raw
        # 「a|b」：同一個詞的多種寫法；「:動詞;連用形|:形容詞;語幹」：多種詞性條件。
        # 由右往左處理：沒寫條件的選項沿用右邊最近一個選項的詞性與活用形
        alts, pos, form = [], "", ""
        for part in reversed(raw.split("|")):
            alt = _parse_alt(part)
            if alt.pos or alt.form:
                pos, form = alt.pos, alt.form
            elif not alt.any:
                alt = _Alt(alt.words, pos, form, suffix=alt.suffix, surface_only=alt.surface_only,
                           head_verb=alt.head_verb)
            alts.append(alt)
        elems.append(_Elem(tuple(reversed(alts)), optional, kind))
    return tuple(elems)


def _match_at(elems, tokens, ei, ti, core):
    """回傳 (結束位置, 句型本體的詞位置)；失敗回傳 None。可省略的元素先嘗試比對、不行再略過。"""
    if ei == len(elems):
        return ti, core
    e = elems[ei]
    if e.kind == "end":
        at_end = ti == len(tokens) or tokens[ti].is_punct()
        return _match_at(elems, tokens, ei + 1, ti, core) if at_end else None
    if e.kind == "not":
        if ti < len(tokens) and e.match(tokens[ti]):
            return None
        return _match_at(elems, tokens, ei + 1, ti, core)
    if ti < len(tokens) and e.match(tokens[ti]):
        res = _match_at(elems, tokens, ei + 1, ti + 1, core if e.context else core + (ti,))
        if res is not None:
            return res
    if e.optional:
        return _match_at(elems, tokens, ei + 1, ti, core)
    return None


def find_spans(seq: str, tokens: list[Token]) -> list[tuple[int, int, int]]:
    """回傳每個比對結果的 (開始, 結束, 具體程度)。上下文元素不算在範圍內；
    具體程度 = 有指定文字的元素數（只靠詞性比對的規則，例如「命令形」，為 0）。"""
    elems = compile_seq(seq)
    spans = []
    for start in range(len(tokens)):
        res = _match_at(elems, tokens, 0, start, ())
        if res is None or res[0] == start:
            continue
        end, core = res
        spans.append((min(core), max(core) + 1, len(core)) if core else (start, end, 0))
    return spans


def rule_compound_head(rule: Rule, t: Token) -> str | None:
    """規則是靠字尾元素（~直す）比對到 t 這一個詞時（t 是複合詞：練り直す、読み方），回傳前半（練り、読み）；否則 None。"""
    for seq in rule.seqs:
        for el in compile_seq(seq):
            for alt in el.alts:
                if alt.suffix and alt.match(t):
                    return compound_head(alt, t)
    return None


def match_line(tokens: list[Token], rules=None) -> list[Rule]:
    """回傳這一句比對到的規則（已移除被更完整句型包住的比對），依出現位置排序。"""
    out = []
    for r, _, _ in match_line_spans(tokens, rules):
        if r not in out:
            out.append(r)
    return out


def match_line_spans(tokens: list[Token], rules=None) -> list[tuple[Rule, int, int]]:
    """同 match_line，但回傳 (規則, 開始, 結束)；同一規則在一句中可出現多次。"""
    rules = rules if rules is not None else all_rules()
    hits = []  # (start, end, 具體程度, rule)
    for r in rules:
        for seq in r.seqs:
            hits += [(s, e, k, r) for s, e, k in find_spans(seq, tokens)]

    def covered(h, other):
        (s, e, k, r), (s2, e2, k2, r2) = h, other
        if r2 is r or not (s2 <= s and e <= e2):
            return False
        if (s2, e2) != (s, e):
            return True
        # 範圍完全相同：保留較難的那條；同級時保留較具體的（「〜なさい」勝過「命令形」）
        lv, lv2 = r.level or 6, r2.level or 6
        return lv2 < lv or (lv2 == lv and k2 > k)

    kept = [h for h in hits if not any(covered(h, o) for o in hits)]
    kept.sort(key=lambda h: (h[0], -h[1]))
    return [(r, s, e) for s, e, _, r in kept]


def _extra_rules() -> list[Rule]:
    """JLPT_grammer 補充的規則（序列由 Gemini 寫、tools/grammar_batches.py 驗證過才寫進這個檔）。"""
    import json
    if not EXTRA_PATH.exists():
        return []
    out = []
    for line in EXTRA_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Rule(d["id"], d["pattern"], d["level"], d["meaning"], d["formation"],
                            tuple(d["seqs"]), tuple(d["examples"])))
    return out


@lru_cache(maxsize=1)
def all_rules() -> tuple:
    from .grammar_rules import RULES
    rules = [*RULES, *_extra_rules()]
    ids = [r.id for r in rules]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"文法規則 id 重複：{sorted(dup)}")
    return tuple(rules)


def find_grammar(entries: list[dict]) -> list[GrammarHit]:
    """整部影片的文法：依第一次出現的順序排列，同一文法只列一次，附上所有出現的句子。"""
    from .analysis import analyze
    return analyze(entries).grammar
