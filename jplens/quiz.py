"""單字測驗的出題：每題 1 個答案＋3 個混淆選項。

目的（使用者 2026-10-21）：答案一定要是唯一的（選項裡不能有意思一樣的字）；混淆選項只要讓人「些微混淆」，
不能光看漢字就猜到，不需要像老師出題那麼難。

- 唯一（一定要遵守）：混淆選項和答案的意思片段（「、」「；」切開的每個說法）
  ① 字串相同或互相包含（2 個字以上）、② 語意模型（bge-small-zh）的相似度 ≥ SAME_SIM、
  ③ 使用者在測驗裡按過「這個也對」的配對、④ 同一個寫法（選單字時選項會長得一樣）——任一成立就不用。
  ② 的門檻（10-21 在使用者 303 個字上看過分數帶）：兩個字的短說法同義時分數不高（即刻／即時 0.74、開端／開頭 0.73、
  起點／起端 0.73），相關但不同義的是 0.71（桌子／椅子）、無關的約 0.5。目的是答案唯一、多排除沒關係，取 0.72：
  每個字平均排除 5.6 個（最多 27 個，300 個字裡還剩 275 個可以挑）。
  解釋的語言是 English 時（10-23）：① 的英文以詞為單位、詞尾變化算同一個（faint／faintly）；② 改比這個字的中文意思
  （zh_meanings：紀錄或字典的中文；bge-small-zh 比英文分不出同義），① 也加比中文意思。
- 些微混淆（加分，不是硬規則）：選意思時，選項的中文裡有答案的漢字（毛布 → 有「毛」的意思），
  才不會一看漢字就知道；選單字時，選項的漢字出現在題目的中文裡、和答案共用漢字、讀音相近。
- 單字本太少（湊不滿 3 個）：從日檢單字表補，同樣檢查①②。

語意模型：data/models/bge-small-zh/model_quantized.onnx（int8，24 MB，CPU 每個片段約 0.7 毫秒）；
向量存在 cache/meaning_vectors.npz，只有新的片段才要算。開測驗時才載入，不影響平常查字。
"""

import json
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import lang
from .config import APP_DIR, ROOT
from .kanji_variants import TO_TW

MODEL_DIR = APP_DIR / "data" / "models" / "bge-small-zh"
VEC_CACHE = ROOT / "cache" / "meaning_vectors.npz"
SAME_NAME = "quiz_same.json"  # 「這個也對」記住的配對，放在單字本資料夾
SAME_SIM = 0.72
SHOWN_SENSES = 2  # 選項上顯示前幾個意思
N_OPTIONS = 4

_PAREN = re.compile(r"[（(][^）)]*[）)]")
_SPLIT = re.compile(r"[、，,；;/／]")
_SPLIT_CJK = re.compile(r"[、，,；;/／\s]+")
_CJK = re.compile(r"[㐀-鿿]")  # i18n: 不翻
_KANA = re.compile(r"^[぀-ヿー・]+$")  # i18n: 不翻
_STRIP = " 　…。．."
_SUFFIX = ("的", "地")  # 迷人的、唐突地 → 迷人、唐突；只去掉結尾一個，而且至少留兩個字（目的、墓地、道地 不動）  # i18n: 不翻


def meanings(rec: dict) -> list[str]:
    """這個字的意思（解釋的語言優先，lang.rec_meanings）。"""
    return lang.rec_meanings(rec)[0]


def shown(rec: dict) -> str:
    """選項上顯示的意思：前兩個意思。"""
    return "；".join(meanings(rec)[:SHOWN_SENSES])


def glosses(rec: dict) -> list[str]:
    """意思切成一個個說法（去掉括號裡的補充）：「戳、刺；撞（球）」→ [戳, 刺, 撞]。也讀的意思也算。"""
    return split_glosses(meanings(rec) + [mean for _, mean in rec.get("also") or []]) or \
        ["".join(meanings(rec)) or rec.get("word", "")]


def split_glosses(texts: list[str]) -> list[str]:
    out = []
    for sense in texts:
        sense = _PAREN.sub("", sense)
        # 單字 App 自己輸入的解釋常用空白隔開（頻繁地 經常地）；英文不能照空白切
        for g in (_SPLIT_CJK if _CJK.search(sense) else _SPLIT).split(sense):
            g = g.strip(_STRIP)
            if len(g) >= 3 and g.endswith(_SUFFIX):
                g = g[:-1]
            if g:
                out.append(g)
    return list(dict.fromkeys(out))


def kanji(s: str) -> set[str]:
    """s 裡的漢字，日文新字體換成繁體字（気 → 氣），日文單字和中文意思才比得起來（重んじる 的 重 ／ 加重）。"""
    return {TO_TW.get(c, c) for c in _CJK.findall(s)}


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _strings_overlap(ga: list[str], gb: list[str]) -> bool:
    """單字 App 原本的規則：說法相同，或 2 個字以上互相包含。英文以詞為單位比（_en_overlap）。"""
    for x in ga:
        for y in gb:
            if x == y:
                return True
            if _CJK.search(x) or _CJK.search(y) or _KANA_ANY.search(x) or _KANA_ANY.search(y):
                if len(x) >= 2 and len(y) >= 2 and (x in y or y in x):
                    return True
            elif _en_overlap(_en_words(x), _en_words(y)):
                return True
    return False


_EN_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_KANA_ANY = re.compile(r"[぀-ヿ]")  # i18n: 不翻
# 只有這些詞的說法（to do、to be、one's）太空泛：只算完全相同，不算「包含」
_EN_STOP = {"to", "a", "an", "the", "be", "do", "get", "have", "make", "one", "one's", "oneself", "someone", "something",
            "of", "in", "on", "at", "with", "for", "from", "by", "as", "up", "out", "and", "or", "not", "e", "g", "etc"}


def _stem(w: str) -> str:
    """粗略的詞幹：faintly／faint、ignoring／ignore、slowly／slow 比得起來（只用來判斷同義，不顯示）。"""
    for suf in ("ly", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not (suf == "s" and w.endswith("ss")):
            w = w[:-len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def _en_words(g: str) -> tuple[str, ...]:
    ws = _EN_WORD.findall(g.lower())
    while ws and ws[0] in ("to", "a", "an", "the"):  # to eat／eat、a look／look
        ws = ws[1:]
    return tuple(_stem(w) for w in ws)


_EN_STOP_STEMS = {_stem(w) for w in _EN_STOP}


def _en_overlap(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """英文說法：詞幹相同，或短的整段出現在長的裡面（burst／burst open；pop 不算在 population 裡）。"""
    if not a or not b:
        return False
    if a == b:
        return True
    s, lg = (a, b) if len(a) <= len(b) else (b, a)
    if all(w in _EN_STOP_STEMS for w in s):
        return False
    return any(lg[i:i + len(s)] == s for i in range(len(lg) - len(s) + 1))


def _reading_close(a: str, b: str) -> bool:
    if not a or not b or a == b:
        return False
    return (len(a) >= 2 and len(b) >= 2 and (a[:2] == b[:2] or a[-2:] == b[-2:])) or \
        (len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1)


# ---------------------------------------------------------------- 語意模型
class Semantic:
    """bge-small-zh：意思片段 → 單位向量。向量快取在 cache/meaning_vectors.npz。"""

    def __init__(self, model_dir: Path = MODEL_DIR, cache: Path = VEC_CACHE):
        self.model_dir, self.cache = Path(model_dir), Path(cache)
        self._sess = self._tok = None
        self._vec: dict[str, np.ndarray] = {}
        self._dirty = False
        self._lock = threading.Lock()
        if self.cache.exists():
            try:
                z = np.load(self.cache, allow_pickle=False)
                self._vec = dict(zip(z["texts"].tolist(), z["vecs"].astype(np.float32)))
            except Exception:
                self._vec = {}

    @property
    def ok(self) -> bool:
        return (self.model_dir / "model_quantized.onnx").exists() and (self.model_dir / "tokenizer.json").exists()

    def _load(self):
        if self._sess is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            tok.enable_padding()
            tok.enable_truncation(64)
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            self._sess = ort.InferenceSession(str(self.model_dir / "model_quantized.onnx"), so,
                                              providers=["CPUExecutionProvider"])
            self._tok = tok

    def vectors(self, texts: list[str]) -> dict[str, np.ndarray]:
        """片段 → 向量；沒算過的才跑模型。"""
        with self._lock:
            new = [t for t in dict.fromkeys(texts) if t not in self._vec]
            if new and self.ok:
                self._load()
                names = {i.name for i in self._sess.get_inputs()}
                for k in range(0, len(new), 64):
                    enc = self._tok.encode_batch(new[k:k + 64])
                    ids = np.array([e.ids for e in enc], np.int64)
                    feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], np.int64)}
                    if "token_type_ids" in names:
                        feed["token_type_ids"] = np.zeros_like(ids)
                    h = self._sess.run(None, feed)[0][:, 0]  # bge 用 [CLS]
                    h /= np.linalg.norm(h, axis=1, keepdims=True)
                    self._vec.update(zip(new[k:k + 64], h.astype(np.float32)))
                self._dirty = True
            return {t: self._vec[t] for t in texts if t in self._vec}

    def save(self):
        with self._lock:
            if not self._dirty:
                return
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            texts = list(self._vec)
            tmp = self.cache.with_name(self.cache.stem + ".tmp.npz")
            np.savez(tmp, texts=np.array(texts), vecs=np.stack([self._vec[t] for t in texts]).astype(np.float16))
            tmp.replace(self.cache)
            self._dirty = False

    def unload(self):
        """測驗結束：放掉模型（約 100 MB），向量留著。"""
        self.save()
        with self._lock:  # 背景出題可能正在 vectors() 裡用模型
            self._sess = self._tok = None


# ---------------------------------------------------------------- 「這個也對」
class SamePairs:
    """使用者在測驗裡按「這個也對」的配對（單字的 key_of），存在單字本資料夾的 quiz_same.json。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.pairs = {pair_key(a, b) for a, b in json.loads(self.path.read_text(encoding="utf-8"))}
        except (OSError, ValueError, TypeError):
            self.pairs = set()

    def has(self, a: str, b: str) -> bool:
        return pair_key(a, b) in self.pairs

    def add(self, a: str, b: str):
        self.pairs.add(pair_key(a, b))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self.pairs), ensure_ascii=False, indent=0), encoding="utf-8")


# ---------------------------------------------------------------- 出題
@dataclass
class Question:
    target: dict
    options: list[dict]          # 4 個紀錄（含答案），已打亂
    mode: str                    # "meaning"：看單字選意思；"word"：看意思選單字
    answer: int = field(init=False)

    def __post_init__(self):
        self.answer = next(i for i, o in enumerate(self.options) if o is self.target)

    @property
    def prompt(self) -> str:
        return self.target.get("word", "") if self.mode == "meaning" else shown(self.target)

    def label(self, i: int) -> str:
        o = self.options[i]
        return shown(o) if self.mode == "meaning" else o.get("word", "")


class QuizMaker:
    """pool：可以拿來當混淆選項的單字（整本單字本）；key：紀錄 → 唯一的鍵（vocab.key_of）。"""

    def __init__(self, pool: list[dict], key, semantic: Semantic | None = None, same: SamePairs | None = None,
                 extra=None, rand: random.Random | None = None, zh_of=None):
        self.key, self.sem, self.same = key, semantic, same
        self.extra = extra  # 單字本不夠時補選項：extra(target, n) → 紀錄列表（日檢單字表）
        self.rand = rand or random.Random()
        # 解釋的語言是 English 時，語意模型改比中文意思（zh_meanings）；中文模式就是顯示的意思
        self.zh_of = zh_of if lang.english() else None
        self.pool = [r for r in pool if meanings(r) and r.get("word")]
        self._g, self._z, self._v = {}, {}, {}  # 顯示語言的說法、給語意模型比的說法、說法的向量
        self._prepare(self.pool)

    def _prepare(self, recs):
        """算紀錄的說法與向量（補進來的紀錄是新物件，id 可能和已回收的重複，所以一律重算）。"""
        for r in recs:
            self._g[id(r)] = glosses(r)
            self._z[id(r)] = split_glosses(self.zh_of(r)) if self.zh_of else self._g[id(r)]
        if self.sem is not None and self.sem.ok:
            self._v.update(self.sem.vectors([g for r in recs for g in self._z[id(r)]]))

    def _mat(self, rec) -> np.ndarray | None:
        vs = [self._v[g] for g in self._z[id(rec)] if g in self._v]
        return np.stack(vs) if vs else None

    def same_meaning(self, a: dict, b: dict) -> bool:
        """a、b 的意思有沒有重疊（不能出現在同一題）。"""
        if self.key(a) == self.key(b) or a.get("word") == b.get("word") or shown(a) == shown(b):
            return True
        if self.same is not None and self.same.has(self.key(a), self.key(b)):
            return True
        if _strings_overlap(self._g[id(a)], self._g[id(b)]):
            return True
        if self.zh_of is not None and _strings_overlap(self._z[id(a)], self._z[id(b)]):
            return True
        ma, mb = self._mat(a), self._mat(b)
        return ma is not None and mb is not None and float((ma @ mb.T).max()) >= SAME_SIM

    def _confusion(self, target: dict, cand: dict, mode: str) -> float:
        """些微混淆的分數（越高越像）。"""
        tw, cw = target.get("word", ""), cand.get("word", "")
        s = 0.0
        if mode == "meaning":  # 看單字選意思：選項的中文裡有答案的漢字，才不會一看漢字就知道
            if kanji(tw) & kanji(shown(cand)):
                s += 2
            if kanji(tw) & kanji(cw):
                s += 1
        else:  # 看意思選單字：選項的漢字出現在題目的中文裡、共用漢字、讀音相近、同樣是假名
            if kanji(cw) & kanji(shown(target)):
                s += 2
            if kanji(tw) & kanji(cw):
                s += 1
            if _reading_close(target.get("reading", ""), cand.get("reading", "")):
                s += 1
            if bool(_KANA.match(tw)) == bool(_KANA.match(cw)):
                s += 0.5
        if target.get("pos") and target.get("pos") == cand.get("pos"):
            s += 0.5
        return s

    def distractors(self, target: dict, mode: str, k: int = N_OPTIONS - 1) -> list[dict]:
        if id(target) not in self._g:
            self._prepare([target])
        ok = [c for c in self.pool if c is not target and not self.same_meaning(target, c)]
        if len(ok) < k and self.extra is not None:
            more = [r for r in self.extra(target, 40) if r.get("word") and meanings(r)]
            self._prepare(more)
            for r in more:  # 一個一個加，補進來的之間也要比（同寫法的兩個會在下面被去掉，湊不滿 k 個整題就沒了）
                if len(ok) >= k:
                    break
                if not self.same_meaning(target, r) and not any(self.same_meaning(r, o) for o in ok):
                    ok.append(r)
        # 選項之間也不能意思一樣（兩個錯的選項同義沒關係，但選單字時兩個一樣的寫法會看不出差別）
        scored = [(self._confusion(target, c, mode) + self.rand.random() * 1.5, c) for c in ok]
        scored.sort(key=lambda x: -x[0])
        out = []
        for _, c in scored:
            if all(c.get("word") != o.get("word") and shown(c) != shown(o) for o in out):
                out.append(c)
            if len(out) == k:
                break
        return out

    def question(self, target: dict, mode: str) -> Question | None:
        ds = self.distractors(target, mode)
        if len(ds) < N_OPTIONS - 1:
            return None
        opts = ds + [target]
        self.rand.shuffle(opts)
        return Question(target, opts, mode)

    def make(self, targets: list[dict], mode: str, n: int) -> list[Question]:
        """從 targets 隨機出 n 題（不重複）；湊不出選項的字略過。mode="mixed" 每題隨機。"""
        order = list(targets)
        self.rand.shuffle(order)
        out = []
        for t in order:
            if len(out) >= n:
                break
            if not meanings(t):
                continue
            q = self.question(t, self.rand.choice(["meaning", "word"]) if mode == "mixed" else mode)
            if q:
                out.append(q)
        if self.sem is not None:
            self.sem.save()
        return out


def zh_meanings(dictionary):
    """解釋的語言是 English 時，給語意模型比的中文意思：紀錄裡有就用（中文模式時記的字），沒有就查字典。回傳 zh_of(rec)。
    bge-small-zh 比英文說法分不出同義（10-23 在使用者 327 個字上：中文判斷同義的配對分數中位數 0.70、其他 0.62），
    英文模型要另外下載，所以借用字典的中文意思；英文說法本身另外用 _strings_overlap 比。"""
    cache = {}

    def zh_of(rec: dict) -> list[str]:
        if rec.get("meaning_zh"):
            return list(rec["meaning_zh"])
        k = (rec.get("word", ""), rec.get("reading", ""))
        if k not in cache:
            try:
                cache[k] = dictionary.lookup([k[0]], k[1], zh=True).zh if dictionary is not None else []
            except Exception:
                cache[k] = []
        return cache[k]
    return zh_of


def jlpt_fill(dictionary):
    """單字本不夠時，從日檢單字表補混淆選項（和答案同級或相鄰級）：回傳 extra(target, n)。"""
    from .words import _jlpt_lists

    def extra(target: dict, n: int) -> list[dict]:
        by_word = _jlpt_lists()[0]
        entries = [e for es in by_word.values() for e in es]
        lv = {"N5": 5, "N4": 4, "N3": 3, "N2": 2, "N1": 1}.get(target.get("jlpt", ""), 3)
        near = [e for e in entries if abs(e.level - lv) <= 1] or entries
        out = []
        for e in random.sample(near, min(len(near), n * 2)):
            zh = []
            if dictionary is not None:
                try:
                    m = dictionary.lookup([e.word], e.reading, None)
                    zh = (m.en if lang.english() else m.zh)[:3]
                except Exception:
                    zh = []
            if zh:
                out.append({"word": e.word, "reading": e.reading, "pos": "", "jlpt": f"N{e.level}",
                            "meaning_en" if lang.english() else "meaning_zh": zh})
            if len(out) >= n:
                break
        return out
    return extra
