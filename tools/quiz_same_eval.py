"""單字測驗「意思重疊、不能同題」的判斷：英文模式和中文模式（10-21 在使用者的字上校準過）比較。

    python tools/quiz_same_eval.py [單字本.jsonl]     # 預設 config 的單字本；要中文、英文意思都有的字（中文模式時記的）

每兩個字比一次，印出各種做法排除的配對數（每個字平均排除幾個）、涵蓋多少中文模式判斷同義的配對，
和英文模式漏掉的配對抽樣（中文判斷本來就從寬，漏掉的大多不是真的同義，要人看）。
「英文模式・字典中文」把紀錄裡的中文拿掉、只靠字典查（模擬英文模式記的字）。向量快取放暫存資料夾，不動 cache/。
"""
import itertools
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import json  # noqa: E402

from jplens import lang, quiz  # noqa: E402
from jplens.config import Config  # noqa: E402
from jplens.dictionary import Dictionary  # noqa: E402
from jplens.vocab import key_of  # noqa: E402


def judge(recs, sem, english: bool, zh_of=None) -> set[tuple[int, int]]:
    lang.set_explain("en" if english else "zh-TW")
    qm = quiz.QuizMaker(recs, key_of, sem, None, zh_of=zh_of)
    return {(i, j) for i, j in itertools.combinations(range(len(recs)), 2) if qm.same_meaning(recs[i], recs[j])}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Config.load().vocab_path / "saved_words.jsonl"
    recs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    recs = [r for r in recs if r.get("meaning_zh") and r.get("meaning_en") and r.get("pos") != "文法" and r.get("word")]
    n = len(recs)
    print(f"{n} 個字、{n * (n - 1) // 2} 對")
    sem = quiz.Semantic(cache=Path(tempfile.mkdtemp()) / "vec.npz")
    if not sem.ok:
        print("沒有 bge-small-zh 模型")
        return
    dic = Dictionary()
    ref = judge(recs, sem, False)
    no_zh = [{k: v for k, v in r.items() if k != "meaning_zh"} for r in recs]
    runs = {
        "中文模式（參考）": ref,
        "英文模式・舊（bge-small-zh 比英文）": judge(recs, sem, True),
        "英文模式・紀錄的中文": judge(recs, sem, True, quiz.zh_meanings(dic)),
        "英文模式・字典中文": judge(no_zh, sem, True, quiz.zh_meanings(dic)),
    }
    lang.set_explain("en")
    en_str = {(i, j) for i, j in itertools.combinations(range(n), 2)
              if quiz._strings_overlap(quiz.glosses(recs[i]), quiz.glosses(recs[j]))}
    runs["只比英文說法"] = en_str
    for name, s in runs.items():
        print(f"{name:24s} 排除 {len(s):5d} 對（每字 {2 * len(s) / n:4.1f}）  涵蓋中文判斷 {len(s & ref)}/{len(ref)}")
    lang.set_explain("en")
    new = runs["英文模式・字典中文"]
    for title, pairs in (("中文判斷同義、英文模式沒排除", ref - new), ("英文模式多排除的", new - ref)):
        pairs = sorted(pairs)
        print(f"\n{title}：{len(pairs)} 對，抽 15 對")
        for i, j in random.Random(0).sample(pairs, min(15, len(pairs))):
            print(f"  {recs[i]['word']}｜{recs[j]['word']}：{quiz.shown(recs[i])}  ‖  {quiz.shown(recs[j])}")


if __name__ == "__main__":
    main()
