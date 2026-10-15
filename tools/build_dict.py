"""把下載的原始字典建成查詢用的 SQLite（data/dict.sqlite）。

    python tools/build_dict.py

來源（放在 data/raw/，不進版控）：
- zhwikt_ja.jsonl：中文維基詞典的日語詞條，kaikki.org 以 wiktextract 擷取
  https://kaikki.org/zhwiktionary/日語/ （CC BY-SA）。簡繁混雜，含簡體字的解釋才轉成台灣繁體（tools/zhconv.py）。
- JMdict_e.gz：EDRDG 的 JMdict 英文版 http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz （CC BY-SA）
- tomoshi/tomoshi-dict-open.db：Tomoshi 日語詞典的開放資料（Y1Z，CC BY-SA 4.0），JMdict 逐個意思的繁體中文翻譯。
  https://github.com/tomoshi-app/tomoshi-dict-data/releases 下載 .db.zst 後用 zstd -d 解壓縮。
  依 JMdict 條目編號對應，而且英文意思要完全相同才用（兩邊的 JMdict 版本不同）；沒有這個檔案也能建。
"""

import gzip
import json
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zhconv import to_traditional  # noqa: E402
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "dict.sqlite"
TOMOSHI = RAW / "tomoshi" / "tomoshi-dict-open.db"
MAX_SENSES = 6
MAX_SENSES_JM = 10  # JMdict 一個條目常橫跨多種詞性，多留一些給查詢時依詞性挑選

sys.path.insert(0, str(ROOT))
from jplens.jlearn import analysis  # noqa: E402

SCHEMA = """
CREATE TABLE entries (id INTEGER PRIMARY KEY, src TEXT, word TEXT, readings TEXT, pos TEXT, senses TEXT,
                      sense_pos TEXT, seq TEXT, zh TEXT, common TEXT, sense_misc TEXT);
CREATE TABLE keys (key TEXT, entry_id INTEGER);
CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
"""


def _reading_from_ruby(form: str, ruby: list) -> str:
    """把 [['喫','きっ'],['茶','さ']] 的振假名套回原字串：喫茶店 → きっさてん。"""
    out, i = "", 0
    for base, rd in ruby:
        j = form.find(base, i)
        if j < 0:
            return ""
        out += form[i:j] + rd
        i = j + len(base)
    return analysis.to_hira(out + form[i:])


def wikt_entries():
    redirects = []  # (別名, 目標詞)
    with open(RAW / "zhwikt_ja.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            word = d["word"]
            if d.get("redirects"):
                redirects += [(word, r) for r in d["redirects"]]
            senses = []
            for s in d.get("senses", []):
                # 含換行的解釋是漢字條目的級別說明（「峠\n（常用漢字）」）或整段貼上的字典原文，不適合顯示
                gl = "；".join(g.strip() for g in s.get("glosses", []) if g.strip() and "\n" not in g)
                if gl and "no-gloss" not in s.get("tags", []):
                    senses.append(to_traditional(gl))
            if not senses:
                continue
            readings, aliases = [], []
            if analysis._KANA_ONLY.match(word):
                readings.append(analysis.to_hira(word))
            for fm in d.get("forms", []):
                tags = fm.get("tags", [])
                if "canonical" in tags and fm.get("ruby"):
                    rd = _reading_from_ruby(fm["form"], fm["ruby"])
                    if rd:
                        readings.append(rd)
                elif "alternative" in tags and fm.get("form"):
                    aliases.append(fm["form"])
            pos = d.get("pos_title") or d.get("pos", "")
            yield {"word": word, "readings": list(dict.fromkeys(readings)), "pos": pos,
                   "senses": senses[:MAX_SENSES], "sense_pos": [pos] * len(senses[:MAX_SENSES]),
                   "keys": [word] + aliases}
    yield {"redirects": redirects}


TOMOSHI_NOTE = re.compile(r"[（(][^（）()]*(?:書寫|寫作|寫法)[^（）()]*[）)]")


def tomoshi_gloss(text: str) -> str:
    """「（通常以假名書寫）記住；記；記憶；銘記；背下來」→「記住、記、記憶、銘記」：
    去掉寫法說明、最多留 4 個說法，和 Gemini 翻譯一樣用「、」分隔。"""
    items = [TOMOSHI_NOTE.sub("", t).strip() for t in re.split(r"[；;]", text)]
    return "、".join([t for t in items if t][:4])


def tomoshi_senses():
    """JMdict 條目編號 → [(英文, 繁中)…]（依 JMdict 原本的意思順序）。沒下載 Tomoshi 資料就回傳空的。"""
    if not TOMOSHI.exists():
        print(f"沒有 {TOMOSHI}，中文只用維基詞典")
        return {}
    db = sqlite3.connect(f"file:{TOMOSHI}?mode=ro", uri=True)
    zh = {eid: json.loads(data)["senses"] for eid, data in db.execute("SELECT entry_id, data FROM zh_defs_zhtw")}
    out = {}
    for eid, data in db.execute("SELECT id, data FROM entries"):
        senses = json.loads(data)["senses"]
        z = zh.get(eid, {})
        out[eid] = [("; ".join(g["text"] for g in s["glosses"] if g["lang"] == "eng"),
                     tomoshi_gloss("；".join(g["text"] for g in z.get(str(k), {}).get("glosses", []))))
                    for k, s in enumerate(senses)]
    db.close()
    return out


def jmdict_entries(tomoshi: dict):
    with gzip.open(RAW / "JMdict_e.gz") as f:
        for _, el in ET.iterparse(f):
            if el.tag != "entry":
                continue
            kebs = [k.text for k in el.iter("keb")]
            rebs = [r.text for r in el.iter("reb")]
            # 常用的讀音（JMdict 的 re_pri：ichi1、news1、spec1…）。「要」的かなめ有標、よう沒有；查字時用來列「也讀」
            common = [analysis.to_hira(r.findtext("reb")) for r in el.iter("r_ele") if r.find("re_pri") is not None]
            seq = el.findtext("ent_seq")  # JMdict 條目編號：Gemini 翻譯（dict_translate/zh.jsonl）用它對應
            tm = tomoshi.get(seq, [])
            senses, sense_pos, zh, misc, cur = [], [], [], [], ""
            for k, s in enumerate(el.iter("sense")):
                pos = "|".join(p.text for p in s.iter("pos") if p.text)
                cur = pos or cur  # JMdict 的詞性沒寫時沿用上一個詞義的
                gl = "; ".join(g.text for g in s.iter("gloss") if g.text)
                if gl:
                    senses.append(gl)
                    sense_pos.append(cur)
                    # Tomoshi 的 JMdict 版本不同：英文意思完全相同才用它的中文，避免意思錯位
                    zh.append(tm[k][1] if k < len(tm) and tm[k][0] == gl else "")
                    # 詞義標籤（abbreviation、archaic、slang…）：之後可用來把冷僻的意思排到後面
                    misc.append([m.text for m in s.iter("misc") if m.text])
            el.clear()
            if senses:
                yield {"word": (kebs or rebs)[0], "readings": [analysis.to_hira(r) for r in rebs],
                       "pos": sense_pos[0], "senses": senses[:MAX_SENSES_JM],
                       "sense_pos": sense_pos[:MAX_SENSES_JM], "keys": kebs + rebs, "seq": seq,
                       "zh": zh[:MAX_SENSES_JM] if any(zh) else None,
                       "common": list(dict.fromkeys(common)) or None,
                       "sense_misc": misc[:MAX_SENSES_JM] if any(misc) else None}


def main():
    t = time.time()
    tmp = OUT.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)
    next_id = 1

    def add(src, e):
        nonlocal next_id
        def js(v):
            return json.dumps(v, ensure_ascii=False) if v else None
        db.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (next_id, src, e["word"], json.dumps(e["readings"], ensure_ascii=False), e["pos"],
                    json.dumps(e["senses"], ensure_ascii=False), json.dumps(e["sense_pos"], ensure_ascii=False),
                    e.get("seq"), js(e.get("zh")), js(e.get("common")), js(e.get("sense_misc"))))
        db.executemany("INSERT INTO keys VALUES (?,?)", [(k, next_id) for k in dict.fromkeys(e["keys"])])
        next_id += 1

    n_wikt = 0
    for e in wikt_entries():
        if "redirects" in e:
            # 軟重定向（ヤバい → やばい）：別名直接指到目標詞的所有條目
            db.executemany("INSERT INTO keys SELECT ?, entry_id FROM keys WHERE key = ?",
                           [(alias, target) for alias, target in e["redirects"] if alias != target])
            continue
        add("wikt", e)
        n_wikt += 1
    n_jm = n_tm = 0
    for e in jmdict_entries(tomoshi_senses()):
        add("jmdict", e)
        n_jm += 1
        n_tm += bool(e["zh"])
    db.execute("CREATE INDEX keys_key ON keys(key)")
    db.executemany("INSERT INTO meta VALUES (?,?)", [
        ("built", time.strftime("%Y-%m-%d %H:%M")),
        ("attribution", "中文意思：Tomoshi 開放資料（Y1Z，CC BY-SA 4.0）、維基詞典（CC BY-SA，經 kaikki.org/wiktextract 擷取）；"
                        "英文意思：JMdict（EDRDG，CC BY-SA）"),
    ])
    db.commit()
    db.close()
    try:
        OUT.unlink(missing_ok=True)
    except PermissionError:
        sys.exit(f"{OUT} 正在被使用（查字工具開著？）。關掉程式後把 {tmp.name} 改名成 {OUT.name} 即可，不用重建。")
    tmp.rename(OUT)
    print(f"維基詞典 {n_wikt} 筆、JMdict {n_jm} 筆（其中 {n_tm} 筆有 Tomoshi 中文）→ {OUT}"
          f"（{OUT.stat().st_size / 1e6:.0f} MB，{time.time() - t:.0f} 秒）")


if __name__ == "__main__":
    main()
