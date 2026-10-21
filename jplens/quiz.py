"""單字測驗的出題：每題 1 個答案＋3 個混淆選項。

目的（使用者 2026-10-21）：答案一定要是唯一的（選項裡不能有意思一樣的字）；混淆選項只要讓人「些微混淆」，
不能光看漢字就猜到，不需要像老師出題那麼難。

- 唯一（一定要遵守）：混淆選項和答案的意思片段（「、」「；」切開的每個說法）
  ① 字串相同或互相包含（2 個字以上）、② 語意模型（bge-small-zh）的相似度 ≥ SAME_SIM、
  ③ 使用者在測驗裡按過「這個也對」的配對、④ 同一個寫法（選單字時選項會長得一樣）——任一成立就不用。
  ② 的門檻（10-21 在使用者 303 個字上看過分數帶）：兩個字的短說法同義時分數不高（即刻／即時 0.74、開端／開頭 0.73、
  起點／起端 0.73），相關但不同義的是 0.71（桌子／椅子）、無關的約 0.5。目的是答案唯一、多排除沒關係，取 0.72：
  每個字平均排除 5.6 個（最多 27 個，300 個字裡還剩 275 個可以挑）。
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
_CJK = re.compile(r"[㐀-鿿]")
_KANA = re.compile(r"^[぀-ヿー・]+$")
_STRIP = " 　…。．."
_SUFFIX = ("的", "地")  # 迷人的、唐突地 → 迷人、唐突；只去掉結尾一個，而且至少留兩個字（目的、墓地、道地 不動）


def meanings(rec: dict) -> list[str]:
    """這個字的意思（中文優先，沒有中文用英文）。"""
    return list(rec.get("meaning_zh") or []) or list(rec.get("meaning_en") or [])


def shown(rec: dict) -> str:
    """選項上顯示的意思：前兩個意思。"""
    return "；".join(meanings(rec)[:SHOWN_SENSES])


def glosses(rec: dict) -> list[str]:
    """意思切成一個個說法（去掉括號裡的補充）：「戳、刺；撞（球）」→ [戳, 刺, 撞]。也讀的意思也算。"""
    out = []
    texts = meanings(rec) + [mean for _, mean in rec.get("also") or []]
    for sense in texts:
        sense = _PAREN.sub("", sense)
        # 單字 App 自己輸入的解釋常用空白隔開（頻繁地 經常地）；英文不能照空白切
        for g in (_SPLIT_CJK if _CJK.search(sense) else _SPLIT).split(sense):
            g = g.strip(_STRIP)
            if len(g) >= 3 and g.endswith(_SUFFIX):
                g = g[:-1]
            if g:
                out.append(g)
    return list(dict.fromkeys(out)) or ["".join(meanings(rec)) or rec.get("word", "")]


def kanji(s: str) -> set[str]:
    """s 裡的漢字，日文新字體換成繁體字（気 → 氣），日文單字和中文意思才比得起來（重んじる 的 重 ／ 加重）。"""
    return {TO_TW.get(c, c) for c in _CJK.findall(s)}


def pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _strings_overlap(ga: list[str], gb: list[str]) -> bool:
    """單字 App 原本的規則：說法相同，或 2 個字以上互相包含。"""
    return any(x == y or (len(x) >= 2 and len(y) >= 2 and (x in y or y in x)) for x in ga for y in gb)


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
                 extra=None, rand: random.Random | None = None):
        self.key, self.sem, self.same = key, semantic, same
        self.extra = extra  # 單字本不夠時補選項：extra(target, n) → 紀錄列表（日檢單字表）
        self.rand = rand or random.Random()
        self.pool = [r for r in pool if meanings(r) and r.get("word")]
        self._g = {id(r): glosses(r) for r in self.pool}
        self._prepare(self.pool)

    def _prepare(self, recs):
        if self.sem is not None and self.sem.ok:
            texts = [g for r in recs for g in self._g.setdefault(id(r), glosses(r))]
            self._v = getattr(self, "_v", {})
            self._v.update(self.sem.vectors(texts))
        else:
            self._v = getattr(self, "_v", {})

    def _mat(self, rec) -> np.ndarray | None:
        vs = [self._v[g] for g in self._g[id(rec)] if g in self._v]
        return np.stack(vs) if vs else None

    def same_meaning(self, a: dict, b: dict) -> bool:
        """a、b 的意思有沒有重疊（不能出現在同一題）。"""
        if self.key(a) == self.key(b) or a.get("word") == b.get("word") or shown(a) == shown(b):
            return True
        if self.same is not None and self.same.has(self.key(a), self.key(b)):
            return True
        ga, gb = self._g[id(a)], self._g[id(b)]
        if _strings_overlap(ga, gb):
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
            self._g[id(target)] = glosses(target)
            self._prepare([target])
        ok = [c for c in self.pool if c is not target and not self.same_meaning(target, c)]
        if len(ok) < k and self.extra is not None:
            more = [r for r in self.extra(target, 40) if r.get("word") and meanings(r)]
            for r in more:
                self._g[id(r)] = glosses(r)
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
                    zh = dictionary.lookup([e.word], e.reading, None).zh[:3]
                except Exception:
                    zh = []
            if zh:
                out.append({"word": e.word, "reading": e.reading, "pos": "", "jlpt": f"N{e.level}", "meaning_zh": zh})
            if len(out) >= n:
                break
        return out
    return extra
