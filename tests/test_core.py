"""單元測試（不載入 OCR 模型）。

    python -m pytest tests                    # 有 pytest 時
    python -c "import tests.test_core as m; [getattr(m,k)() for k in dir(m) if k.startswith('test_')]; print('ok')"
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jplens.dictionary import Dictionary, token_cats  # noqa: E402
from jplens.recognize import Box, group_boxes  # noqa: E402
from jplens.vocab import VocabStore, key_of  # noqa: E402
from jplens.words import WordInfo, _sentence_at, extract, visible  # noqa: E402


def _w(word, level=None, proper=False, reading=""):
    return WordInfo(word=word, reading=reading or word, pos="名詞", level=level, proper=proper, sentence="")


def test_group_vertical_columns_right_to_left():
    # 三欄直排（由右而左讀），欄距小於字寬 → 同一個區塊
    cols = [Box(200, 10, 230, 300), Box(160, 10, 190, 250), Box(120, 10, 150, 200)]
    groups = group_boxes(list(reversed(cols)))
    assert len(groups) == 1
    assert [b.x0 for b in groups[0]] == [200, 160, 120]


def test_group_horizontal_lines_and_separate_blocks():
    lines = [Box(10, 50, 400, 80), Box(10, 10, 400, 40), Box(10, 300, 300, 330)]
    groups = group_boxes(lines)
    assert [[b.y0 for b in g] for g in groups] == [[10, 50], [300]]


def test_merge_chunks():
    from jplens.recognize import merge_chunks
    # 真實字幕切成重疊的段落，邊緣半個字讀成別的字（！、Ｄ）
    parts = ["部長代理の臨時！", "Ｄ臨時リーダーを", "ダーを務めさせて", "させていただき", "ただきます。頑", "・頑張ります！"]
    assert merge_chunks(parts) == "部長代理の臨時リーダーを務めさせていただきます。頑張ります！"
    assert merge_chunks(["委員会の人が、", "人が、部活会議", "活会議に出す書"]) == "委員会の人が、部活会議に出す書"
    assert merge_chunks(["一段だけ"]) == "一段だけ"
    # 兩段把重疊區讀成不同的字（遺憾／遠慮）：沒有橋只能直接相接，有接縫重讀的結果就用它當橋
    parts = ["のはまことに遺憾を", "に遠慮なのですが。"]
    assert merge_chunks(parts) == "のはまことに遺憾をに遠慮なのですが。"
    assert merge_chunks(parts, {0: "ことに遺憾なの"}) == "のはまことに遺憾なのですが。"


def test_text_selection():
    from jplens.layout import TLine, select
    # 三欄直排，閱讀順序由右而左（已排好）
    lines = [TLine(Box(200, 0, 230, 300), True), TLine(Box(160, 0, 190, 300), True),
             TLine(Box(120, 0, 150, 200), True)]
    segs = select(lines, (215, 100), (135, 50))
    assert [(s.line, s.a, s.b) for s in segs] == [(0, 100, 300), (1, 0, 300), (2, 0, 50)]
    back = select(lines, (135, 50), (215, 100))  # 反向拖曳結果相同
    assert [(s.line, s.a, s.b) for s in back] == [(s.line, s.a, s.b) for s in segs]
    assert select(lines, (400, 400), (135, 50)) == []  # 起點不在文字上
    one = select(lines, (175, 40), (175, 120))  # 同一欄中間一段
    assert [(s.line, s.a, s.b) for s in one] == [(1, 40, 120)]


def test_layout_ruby_and_overlap():
    from jplens import layout
    col = Box(100, 0, 140, 400)
    ruby = Box(136, 50, 150, 110)            # 正文右側的振假名（框和正文重疊一點）
    left = Box(60, 0, 104, 300)              # 左邊一欄，框和右欄重疊 4 像素
    old, layout.RULES = layout.RULES, True   # 振假名規則預設關閉（直排小說改用 inkcols），這裡測規則本身
    try:
        lines = layout.build_layout([col, ruby, left])
    finally:
        layout.RULES = old
    assert len(lines) == 2                   # 振假名被濾掉
    right_, left_ = lines                    # 由右而左
    assert right_.box.x0 >= left_.box.x1     # 重疊的部分在中間分開


def test_visible_filter():
    ws = [_w("a", 5), _w("b", 3), _w("c", 1), _w("d", None), _w("e", None, proper=True)]
    assert [w.word for w in visible(ws, 3, False)] == ["b", "c", "d"]
    assert [w.word for w in visible(ws, 5, True)] == ["a", "b", "c", "d", "e"]
    assert [w.word for w in visible(ws, 1, False)] == ["c", "d"]


def test_sentence_at():
    text = "昨日は雨だった。今日は晴れて、気持ちがいい。"
    assert _sentence_at(text, text.index("晴")) == "今日は晴れて、気持ちがいい。"
    assert _sentence_at(text, 0) == "昨日は雨だった。"
    text = "彼は「もう帰る。」と言った。"
    assert _sentence_at(text, text.index("帰")) == "彼は「もう帰る。」"  # 右引號留在同一句


def test_empty_list_falls_back_to_basic_words():
    # 何（代名詞）、する（太基本）平常不列；整次查詢一個單字都沒有時照樣列出
    assert [w.word for w in extract(["何をする"], Dictionary()) if not w.is_grammar] == ["何", "する"]
    assert "する" not in [w.word for w in extract(["勉強をする"], Dictionary())]


def test_other_readings():
    d = Dictionary()
    if not d._has_column("common"):
        return  # 舊版字典（tools/build_dict.py 重建後才有）
    assert [r for r, _ in d.other_readings("要", "よう")] == ["かなめ"]
    assert [r for r, _ in d.other_readings("辛い", "つらい")] == ["からい"]
    assert d.other_readings("上手", "じょうず") == []   # うわて、かみて 不是常用讀音
    assert d.other_readings("下手", "へた") == []       # したて 是常用但不在日檢單字表（多字的詞）
    assert d.other_readings("かなめ", "かなめ") == []


def test_rare_senses_last():
    d = Dictionary()
    if not d._has_column("sense_misc"):
        return  # 舊版字典
    m = d.lookup(["泥"], "どろ")  # 「安卓系統」是網路用語
    assert m.rare_zh >= 1 and "Android" in m.en[-1] and "安卓" in m.zh[-1]
    m = d.lookup(["莫大"], "ばくだい")  # 「胖大海」附學名
    assert m.rare_en == 1 and m.en[0].startswith("vast")
    assert d.lookup(["気"], "き").rare_zh == 0


def test_jlpt_reading_check():
    d = Dictionary()
    by = {w.word: w for w in extract(["私を信じてる。そのとき、ただ楽しい。結構なことです。"], d)}
    assert "信じる" in by and by["信じる"].level == 3  # 不是 Sudachi 辭書形的「信ずる」（N2）
    toki = next(w for w in by.values() if w.reading == "とき")  # 表上「時」有 じ（～時）和 とき，要取 とき
    assert toki.word == "時" and toki.level == 5
    assert toki.sentence == "そのとき、ただ楽しい。"
    assert "朱鷺" not in toki.meaning  # 不是同音的「朱鷺」


def test_reading_by_context():
    d = Dictionary()
    by = {w.word: w for w in extract(["彼が来たとき。そんな風に思った。車を止めなかった。何でも言ってくれ。"], d)}
    assert by["言う"].reading == "いう" and by["言う"].level == 5  # Sudachi 單獨念「言う」是 ゆう
    assert by["来る"].reading == "くる" and by["来る"].level == 5  # 不是「来る（きたる）N1」
    assert by["風"].reading == "ふう"  # 不是 かぜ
    assert by["止める"].reading == "とめる"  # 不是 やめる
    by = {w.word: w for w in extract(["そんな風に言うな。入学の準備をした。"], d)}
    assert by["言う"].reading == "いう" and by["言う"].level == 5  # 原文就是辭書形：Sudachi 給 ゆう（JMdict 同一條目）
    assert by["入学"].reading == "にゅうがく" and by["準備"].reading == "じゅんび"  # 表上的讀音欄寫「にゅうがくする」


def test_expressions():
    d = Dictionary()
    if not d.ok:
        return

    def words(s):
        return {w.word: w for w in extract([s], d)}
    by = words("その話がずっと気になっていたのです")
    assert "気になる" in by and by["気になる"].pos == "慣用語" and "worry" in " ".join(by["気になる"].en)
    assert "気" not in by and "なる" not in by  # 組成部分不另外列
    assert "聞き捨てならない" in words("その一言は聞き捨てなりませんね。")  # 否定形
    by = words("彼女は小さくため息をついた。")
    assert "ため息をつく" in by and "ため息" in by and "つく" not in by  # 兩個字以上的名詞照列
    assert "胸を張る" in words("胸張って歩けばいい")  # 省略的を
    assert words("経験では彼に一日の長がある。")["一日の長"].reading == "いちじつのちょう"  # 用字典的讀音
    by = words("ボードゲームの知識")
    assert by["ボードゲーム"].pos == "複合詞" and "ボード" not in by and "ゲーム" not in by
    # 不該當成慣用說法的
    assert "do so" in " ".join(words("そうすれば分かる")["そうする"].en)  # 不是同音的「奏する（上奏）」
    for s, bad in (("こうして迷ってしまった", "こうする"), ("たった一つだけ", "唯一"),
                   ("話しましょうか", "説明し"), ("ともすると忘れがちだ", "ともにする"), ("つけることにします", "ことにする"), ("そう言われると", "そう言う")):
        assert bad not in words(s), (s, bad)
    dekiru = words("突破出来てた")["出来る"]  # 表上寫假名「できる」
    assert dekiru.level == 5 and "able" in " ".join(dekiru.en)
    # 2026-10-21：動詞＋文語助動詞、慣用語中間的助詞被換掉
    for s, want in (("その名を世に知らしめた", "知らしめる"), ("思わず笑った", "思わず"), ("気もつかなかった", "気がつく"),
                    ("腹は立つ", "腹が立つ"), ("手には負えない", "手に負えない")):
        assert want in words(s), (s, want)
    assert not {"すべし", "せず"} & set(words("すぐに出発すべし。何もせず待つ"))  # する＋助動詞是文法


def test_norm_variants():
    """Sudachi 的標準寫法只在是同一個詞時才用（2026-10-21：乗っかる 被查成 乗る）。"""
    d = Dictionary()
    if not d.ok:
        return

    def words(s):
        return {w.word for w in extract([s], d)}
    assert "乗っかる" in words("補正が乗っかる") and "乗る" not in words("補正が乗っかる")  # 口語變體是另一個詞
    assert "抑える" in words("出費を抑える")                       # 字典裡和 押さえる 是不同條目
    assert "引っ張る" in words("ひっぱる")                         # 同一條目的假名寫法 → 標準寫法
    assert "大分" in words("だいぶ寒い") and "言う" in words("ことにいうものは")  # Sudachi 單獨念漢字會念錯，不能比讀音
    assert "言う" in words("胸張って言える")                       # 可能動詞照舊列原本的動詞
    assert "若い" in words("若き獅子") and "愛する" in words("愛されたい")  # 文語、〜す 的舊形列現代的辭書形


def test_grammar_points():
    d = Dictionary()
    ws = extract(["明日も雨かもしれないと思ったんです"], d)
    gram = [w for w in ws if w.is_grammar]
    assert [w.word for w in gram] == ["〜かもしれない", "〜と思う", "〜のだ／〜んです"]
    assert all(not w.is_grammar for w in ws[:len(ws) - len(gram)])  # 單字在前、文法在後
    assert "思う" not in {w.word for w in ws}  # 組成句型的詞不另外列成單字
    omou = gram[1]
    assert omou.level == 4 and omou.zh and "と思う" in omou.formation and omou.sentence.endswith("思ったんです")
    # 同一段字比對到兩條同級規則時只留名稱相符的：「ので」不另外列「〜のだ／〜んです」
    names = {w.word for w in extract(["なくしてしまったので、行かなければならない。"], d) if w.is_grammar}
    assert "〜ので" in names and "〜のだ／〜んです" not in names
    # 同一條文法在幾段文字裡只列一次
    assert sum(w.word == "〜と思う" for w in extract(["そう思う。", "いいと思った。"], d)) == 1


def test_grammar_in_compound_word():
    """文法靠字尾比對到一個複合詞（練り直す、読み方）時，詞本身照樣列，前半的動詞也列；
    「〜方」不比對 粗方（あらかた）、夕方 這類前半不是動詞的名詞（使用者 2026-10-18 回報）。"""
    d = Dictionary()
    names = [w.word for w in extract(["計画を練り直した"], d)]
    assert {"練り直す", "練る", "〜なおす（直す）"} <= set(names)
    names = [w.word for w in extract(["塗り直した"], d)]
    assert "塗る" in names  # 單獨的「塗り」Sudachi 斷成名詞，要接ます再斷
    assert "やる" in [w.word for w in extract(["やり直したい"], d)]  # 標準寫法是「遣り直す」，前半照原文寫
    names = [w.word for w in extract(["この漢字の読み方"], d)]
    assert {"読み方", "読む", "〜方（かた）"} <= set(names)
    for text, word in (("あらかた終わった", "あらかた"), ("夕方に帰る", "夕方"), ("両方とも", "両方"), ("地方の町", "地方")):
        ws = extract([text], d)
        assert word in [w.word for w in ws] and "〜方（かた）" not in [w.word for w in ws], text


def test_gemini_translation_first():
    import json
    d = Dictionary()
    if not d.ok:
        return
    with tempfile.TemporaryDirectory() as tmp:
        zh = Path(tmp) / "zh.jsonl"
        seq = d._db().execute("SELECT e.seq FROM keys k JOIN entries e ON e.id = k.entry_id "
                              "WHERE k.key = '甘い' AND e.src = 'jmdict' ORDER BY e.id").fetchone()[0]
        zh.write_text(json.dumps({"id": seq, "word": "甘い", "reading": "あまい",
                                  "zh": ["甜的", "芳香的", "淡的", "縱容的", "天真的"]}, ensure_ascii=False) + "\n",
                      encoding="utf-8")
        g = Dictionary(zh_path=zh)
        assert g.lookup(["甘い"], "あまい", token_cats(("形容詞", "一般", "*", "*", "形容詞", "連用形"))).zh[0] == "甜的"
        # 沒翻譯的詞照舊用維基詞典
        assert g.lookup(["喫茶店"], "きっさてん").zh == Dictionary(zh_path=Path(tmp) / "none").lookup(
            ["喫茶店"], "きっさてん").zh


def test_meaning_by_pos():
    d = Dictionary()
    if not d.ok:
        return
    by = {w.word: w for w in extract(["それでいいだろう。ただ楽しい。結構なことです。"], d)}
    assert "連用形" not in by["いい"].meaning and "穀物" not in by["いい"].meaning  # 不用動詞「言う」、名詞「飯」
    assert "好" in by["いい"].meaning or "good" in by["いい"].meaning  # 有 Gemini 翻譯時是中文
    assert "ordinary" not in by["ただ"].meaning  # 副詞／接續詞，不是名詞「普通的」
    assert "構造" not in by["結構"].zh[0]  # な形容詞優先，不是名詞「構造」


def test_vocab_add_dedupe_and_export():
    with tempfile.TemporaryDirectory() as d:
        store = VocabStore(Path(d) / "saved.jsonl")
        w = WordInfo(word="眺める", reading="ながめる", pos="一段動詞", level=3, proper=False,
                     sentence="窓の外を眺める。", zh=["觀看", "眺望"], en=["to view"])
        assert store.add(w) is True
        assert store.add(w) is False  # 重複不再寫入
        kana = WordInfo(word="やばい", reading="やばい", pos="い形容詞", level=None, proper=False, sentence="",
                        en=["dangerous"])
        store.add(kana)
        assert VocabStore(store.path).has(w)  # 重新開啟仍記得
        out = Path(d) / "export.txt"
        assert store.export_app_backup(out) == 2
        raw = out.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and not raw.endswith(b"\n") and b"\r\n" not in raw
        data = json.loads(raw)
        assert list(data) == ["version", "date", "words"] and data["version"] == 2
        a, b = data["words"]
        assert list(a) == ["id", "term", "type", "definition", "comment", "isMarked", "date",
                           "reading", "example", "jlpt", "uid"]
        assert (a["term"], a["type"], a["definition"], a["comment"]) == ("眺める", "v.", "觀看；眺望", "")
        assert (a["reading"], a["example"], a["jlpt"], a["uid"]) == ("ながめる", "窓の外を眺める。", "N3", "jplens:眺める|ながめる")
        assert (b["reading"], b["definition"], b["type"], b["jlpt"]) == ("", "dangerous", "adj.", "")  # 假名詞讀音留空、沒中文用英文

        # 單字本視窗：重點、已學會、改讀音、刪除與復原；isMarked 照 marked 匯出
        key = key_of(store.records()[0])
        store.update(key, marked=True, learned=True)
        assert store.export_app_backup(out, [r for r in store.records() if r.get("marked")]) == 1
        assert json.loads(out.read_bytes())["words"][0]["isMarked"] == 1
        try:
            store.update(key, word="やばい", reading="やばい")  # 和另一筆重複
            raise AssertionError("應該擋下重複")
        except ValueError:
            pass
        idx, rec = store.delete(key)
        assert not store.has(w) and len(store.records()) == 1
        store.restore(idx, rec)
        assert store.has(w) and store.records()[0]["learned"] is True
        assert Path(str(store.path) + ".bak").exists()  # 第一次重寫前備份原檔


def test_vocab_also_readings():
    """「也讀」記進單字本、匯出時接在解釋後面；以前記錄的（沒有 also 欄位）用字典現查補上（2026-10-21 使用者要求）。"""
    with tempfile.TemporaryDirectory() as d:
        store = VocabStore(Path(d) / "saved.jsonl")
        w = WordInfo(word="要", reading="よう", pos="名詞", level=3, proper=False, sentence="要は気持ちだ。",
                     zh=["要點"], also=[("かなめ", "樞紐；關鍵")])
        store.add(w)
        rec = store.records()[0]
        assert rec["also"] == [["かなめ", "樞紐；關鍵"]] and store.also_of(rec) == [("かなめ", "樞紐；關鍵")]
        out = Path(d) / "export.txt"
        store.export_app_backup(out)
        a = json.loads(out.read_bytes())["words"][0]
        # 讀音接在後面、意思接上「讀〇〇的意思」（2026-10-21 使用者要求改成這樣）
        assert a["definition"] == "要點；讀かなめ的意思：樞紐、關鍵" and a["reading"] == "よう；かなめ" and a["uid"] == "jplens:要|よう"
        assert store.reading_with_also(rec) == "よう；かなめ" and store.has(w) and not store.add(w)  # 紀錄的 reading 不變，查重複照舊
        old = {"word": "要", "reading": "よう", "pos": "名詞", "jlpt": "N3", "meaning_zh": ["要點"]}  # 舊紀錄
        assert store.also_of(old) == []  # 沒有字典就不補
        store.dictionary = Dictionary()
        assert [rd for rd, _ in store.also_of(old)] == [rd for rd, _ in Dictionary().other_readings("要", "よう")]


def test_vocab_import_clear_add():
    """匯入（單字 App 備份、完整備份）、清空與復原、手動新增（2026-10-21）。"""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = VocabStore(d / "a" / "saved_words.jsonl")
        src.add(WordInfo(word="要", reading="よう", pos="名詞", level=3, proper=False, sentence="要は気持ちだ。",
                         zh=["要點"], also=[("かなめ", "樞紐；關鍵")]))
        src.update(key_of(src.records()[0]), marked=True)
        src.export_app_backup(d / "app.txt")
        dst = VocabStore(d / "b" / "saved_words.jsonl")
        assert dst.import_file(d / "app.txt") == {"added": 1, "updated": 0, "skipped": 0}
        r = dst.records()[0]  # 匯出時接在後面的也讀拆回來
        assert (r["word"], r["reading"], r["meaning_zh"], r["also"], r["marked"]) == \
            ("要", "よう", ["要點"], [["かなめ", "樞紐、關鍵"]], True)
        assert dst.import_file(d / "app.txt")["skipped"] == 1  # 再匯一次：已經有了
        # 單字 App 自己的字：沒有 uid、解釋用空白隔開、同一個字出現兩次
        app = {"version": 2, "words": [
            {"term": "時々", "type": "adv.", "definition": "有時 偶爾", "isMarked": 1, "date": 1700000000000,
             "reading": "ときどき", "example": "", "jlpt": "N5", "comment": "筆記", "uid": None},
            {"term": "時々", "definition": "重複", "reading": "ときどき"},
            {"term": "", "definition": "沒有單字"}]}
        (d / "app2.json").write_text(json.dumps(app, ensure_ascii=False), encoding="utf-8")
        assert dst.import_file(d / "app2.json") == {"added": 1, "updated": 0, "skipped": 1}
        t = next(r for r in dst.records() if r["word"] == "時々")
        assert t["pos"] == "副詞" and t["jlpt"] == "N5" and t["comment"] == "筆記" and t["marked"]
        try:
            (d / "v3.json").write_text('{"version": 3, "words": []}', encoding="utf-8")
            dst.import_file(d / "v3.json")
            assert False, "version 3 應該拒絕"
        except ValueError:
            pass
        # 完整備份來回
        dst.export_jsonl(d / "full.jsonl")
        assert dst.add_record({"word": "新しい", "reading": "あたらしい", "meaning_zh": ["新的"]})
        assert not dst.add_record({"word": "新しい", "reading": "あたらしい"})
        old, backup = dst.clear()
        assert dst.records() == [] and len(old) == 3 and backup.exists()
        dst.add_record({"word": "後で", "reading": "あとで", "meaning_zh": ["之後"]})
        dst.restore_all(old)
        assert [r["word"] for r in dst.records()] == ["要", "時々", "新しい", "後で"]
        dst.clear()
        assert dst.import_file(d / "full.jsonl")["added"] == 2


def test_quiz_unique_options():
    """測驗：選項的意思不能重疊（同一個說法、語意相近、使用者按過「這個也對」），選項之間不能長得一樣。"""
    import random
    from jplens import quiz

    def rec(w, rd, zh, pos="名詞"):
        return {"word": w, "reading": rd, "meaning_zh": zh, "pos": pos}
    pool = [rec("即座", "そくざ", ["即刻、當場"]), rec("早急", "さっきゅう", ["即時、迅速"]),
            rec("毛布", "もうふ", ["毛毯、毯子"]), rec("羽毛", "うもう", ["羽毛"]), rec("布団", "ふとん", ["被褥、棉被"]),
            rec("机", "つくえ", ["桌子"]), rec("椅子", "いす", ["椅子"]), rec("窓", "まど", ["窗戶"]),
            rec("緒", "いとぐち", ["開端、起點"]), rec("端", "はし", ["開頭、起端；邊緣"]),
            rec("要", "よう", ["要點"]), rec("要", "かなめ", ["樞紐"])]
    with tempfile.TemporaryDirectory() as d:
        same = quiz.SamePairs(Path(d) / quiz.SAME_NAME)
        sem = quiz.Semantic(cache=Path(d) / "vec.npz")
        qm = quiz.QuizMaker(pool, key_of, sem if sem.ok else None, same, rand=random.Random(1))
        by = {(r["word"], r["reading"]): r for r in pool}
        assert qm.same_meaning(by["要", "よう"], by["要", "かなめ"])  # 同寫法：選單字時看不出差別
        if sem.ok:  # 用字不同、意思相近（字串比對抓不到）
            assert qm.same_meaning(by["即座", "そくざ"], by["早急", "さっきゅう"])
            assert qm.same_meaning(by["緒", "いとぐち"], by["端", "はし"])
            assert not qm.same_meaning(by["机", "つくえ"], by["窓", "まど"])
        assert not qm.same_meaning(by["机", "つくえ"], by["椅子", "いす"])
        same.add(key_of(by["机", "つくえ"]), key_of(by["椅子", "いす"]))
        assert qm.same_meaning(by["机", "つくえ"], by["椅子", "いす"])
        assert quiz.SamePairs(Path(d) / quiz.SAME_NAME).has(key_of(by["椅子", "いす"]), key_of(by["机", "つくえ"]))
        for mode in ("meaning", "word"):
            for q in qm.make(pool, mode, 20):
                labels = [q.label(i) for i in range(4)]
                assert len(set(labels)) == 4 and q.options[q.answer] is q.target
                assert not any(qm.same_meaning(q.target, o) for o in q.options if o is not q.target), labels
        # 毛布（看單字選意思）：混淆選項優先挑有「毛」「布」的意思，不能光看漢字就猜到
        hits = sum(any("毛" in quiz.shown(o) for o in qm.question(by["毛布", "もうふ"], "meaning").options
                       if o is not by["毛布", "もうふ"]) for _ in range(10))
        assert hits >= 8
        # 單字本太少：從 extra 補
        small = quiz.QuizMaker(pool[:2], key_of, None, None, extra=lambda t, n: pool[5:8], rand=random.Random(2))
        assert small.question(pool[0], "meaning") is not None
        # 補的選項之間同寫法：一個一個比，不會因為去重湊不滿 3 個而整題沒了
        dup = [{"word": "鉛筆", "reading": "えんぴつ", "meaning_zh": [f"鉛筆{i}"]} for i in range(3)] + pool[5:8]
        small = quiz.QuizMaker(pool[:1], key_of, None, None, extra=lambda t, n: dup, rand=random.Random(3))
        assert small.question(pool[0], "meaning") is not None
    # 意思片段：結尾的「的／地」去掉一個、至少留兩個字（目的、墓地 不能變成 目、墓）
    assert quiz.glosses({"meaning_zh": ["目的；墓地", "迷人的、唐突地", "漫無目的地", "地位較低者"]}) == \
        ["目的", "墓地", "迷人", "唐突", "漫無目的", "地位較低者"]
    # 日文新字體和繁體中文比得起來（色気 的 気 ／ 香氣）
    assert quiz.kanji("色気") & quiz.kanji("香氣、芳香") and quiz.kanji("発表") & quiz.kanji("發表")


def test_sentence_record():
    with tempfile.TemporaryDirectory() as d:
        store = VocabStore(Path(d) / "saved.jsonl")
        assert store.add_sentence("もう少しゆっくり話しましょうか。", ["ゆっくり", "話す"]) is True
        assert store.add_sentence("もう少しゆっくり話しましょうか。", []) is False  # 重複不再寫入
        assert store.add_sentence("話しましょうか", []) is True  # 反白的一部分算不同句
        recs = VocabStore(store.path).sentences()
        assert [r["text"] for r in recs] == ["もう少しゆっくり話しましょうか。", "話しましょうか"]
        assert recs[0]["words"] == ["ゆっくり", "話す"] and recs[0]["saved_at"]
        assert store.records() == []  # 不影響單字本
        store.add_sentence("一行目\n二行目", [])
        out = Path(d) / "sentences.txt"
        assert store.export_sentences(out) == 3
        assert out.read_bytes().decode("utf-8") == (
            "1. もう少しゆっくり話しましょうか。\n2. 話しましょうか\n3. 一行目 二行目\n")  # 多行接成一行、無 BOM、LF


def test_dictionary_lookup():
    d = Dictionary()
    if not d.ok:
        print("略過：沒有 data/dict.sqlite")
        return
    m = d.lookup(["喫茶店"], "きっさてん")
    assert m.zh and "咖啡" in "".join(m.zh)
    assert m.en
    # 同一寫法多個讀音：依讀音挑 JMdict 條目
    assert "spicy" in " ".join(d.lookup(["辛い"], "からい").en).lower()
    assert "painful" in " ".join(d.lookup(["辛い"], "つらい").en).lower()
    # 句尾句號已去掉
    assert all(not s.endswith("。") for s in d.lookup(["分かる"], "わかる").zh)


def test_extract_words():
    d = Dictionary()
    ws = extract(["彼女は窓の外を眺めながら、小さくため息をついた。"], d)
    by = {w.word: w for w in ws}
    assert "眺める" in by and by["眺める"].level == 3 and by["眺める"].reading == "ながめる"
    assert "は" not in by and "の" not in by  # 助詞不列
    assert "窓の外を眺めながら" in by["眺める"].sentence


def test_manga_bubble_pick():
    from jplens.bubbles import Bubble, at, pick
    right = Bubble(200, 10, 240, 120, 0.9)   # 右邊的對話框（先讀）
    left = Bubble(100, 10, 140, 90, 0.9)
    small = Bubble(205, 20, 215, 30, 0.5)    # 疊在右邊對話框裡的小塊
    blocks = [left, right]
    assert pick(blocks, (90, 0, 250, 130)) == [right, left]  # 由右而左
    assert pick(blocks, (190, 0, 250, 130)) == [right]
    assert pick(blocks, (200, 10, 240, 40)) == []  # 只框到一小段：不算，改用一般框選
    assert at(blocks, 220, 60) == [right] and at(blocks, 145, 50) == [left]  # 點在字旁邊的空白也算
    assert at(blocks, 170, 50) == []
    assert at([right, small], 210, 25) == [small]  # 重疊時取最小的
    w, h = 250, 125
    assert right.crop(w, h) == (194, 4, 246, 125)  # 外擴 5%＋4 像素，不超出圖


# ---------------------------------------------------------------- 文法規則與日檢單字表（原本在 video_translate 的 tests/test_analysis.py）

def _rules_in(text):
    from jplens.jlearn.analysis import tokenize
    from jplens.jlearn.grammar import match_line
    return [r.id for r in match_line(tokenize(text))]


def test_every_grammar_rule_matches_its_examples():
    from jplens.jlearn.analysis import tokenize
    from jplens.jlearn.grammar import all_rules, match_line
    failed = [(r.id, ex) for r in all_rules() for ex in r.examples if r not in match_line(tokenize(ex))]
    assert not failed, failed
    assert len(all_rules()) >= 150


def test_grammar_false_positives():
    assert "zu" not in _rules_in("もう手の打ちようもありません。")          # ません的ん辭書形也是ず
    assert "tearu" not in _rules_in("先生であっても聞き捨てなりません")   # 這個「で」是助動詞
    assert "youni" not in _rules_in("少し無理があるように思うのです。")        # 是「ようだ」不是目的
    assert "tomo" not in _rules_in("少なくとも、私はそう思う")              # 少なくとも是固定副詞
    assert "kagiri" not in _rules_in("時間には限りがあるから")             # 名詞「限り」
    assert "imperative" not in _rules_in("話を聞いてあげなさい。")           # なさい 已由〜なさい涵蓋
    assert "ga_contrast" not in _rules_in("今のうちに楽しんでおくがよかろう。")
    assert "nitsurete" not in _rules_in("友達を家に連れてくる。")             # 漢字「連れ」是帶人來的動詞


def test_grammar_keeps_most_complete_pattern():
    ids = _rules_in("明日までに出さなければならない。")
    assert "nakereba" in ids and "ba" not in ids
    ids = _rules_in("部屋の壁に飾りたいと思いまして")
    assert "tai" in ids and "to_omou" in ids                                # 前導的詞性條件不會吞掉〜たい
    assert _rules_in("全部食べちゃった。") == ["teshimau"]


def test_jlpt_lookup():
    from jplens.jlearn.analysis import lookup_jlpt
    assert lookup_jlpt("わかる", "分かる").level == 5           # 原文寫假名，表上寫漢字
    assert lookup_jlpt("率直").level == 2
    assert lookup_jlpt("見紛う") is None                        # 不在表上 → 級外


def test_mangaocr_onnx_helpers():
    import numpy as np
    from jplens import mangaocr_onnx as mo
    a = np.array([[0.1, 0.9, 0.5, 0.7], [3.0, 1.0, 2.0, 0.0]], np.float32)
    assert mo._topk(a, 3).tolist() == [[1, 3, 2], [0, 2, 1]]           # 同 torch.topk：由大到小
    assert mo.post_process("あ い…ABC1") == "あい．．．ＡＢＣ１"          # 同 manga_ocr.ocr.post_process
    x = mo.preprocess(np.full((30, 90), 255, np.uint8))
    assert x.shape == (3, 224, 224) and abs(float(x.max()) - 1.0) < 1e-6


def test_ruby_hints():
    # 書上標的振假名（ruby.py 讀出來的）當讀音提示
    d = Dictionary()

    def by(s, ruby):
        return {w.word: w for w in extract([s], d, ruby) if not w.is_grammar}
    assert by("辛いカレーを食べた", [("辛", "から")])["辛い"].reading == "からい"     # Sudachi 給 つらい
    assert by("彼は楽しそうに喋っている", [("喋", "しゃべ")])["喋る"].reading == "しゃべる"
    w = by("綴理が笑った", [("綴理", "つづり")])["綴理"]                            # Sudachi 拆成 綴｜理
    assert w.reading == "つづり" and w.proper and "綴" not in by("綴理が笑った", [("綴理", "つづり")])
    # 字典查不到、Sudachi 又認識的詞不用（振假名可能讀錯）；和 Sudachi 一樣的不改
    assert by("相手を慮って", [("慮", "おもんぴか")])["慮る"].reading == "おもんぱかる"   # 讀錯的振假名
    assert by("相手を慮って", [("慮", "おもんばか")])["慮る"].reading == "おもんばかる"   # 字典有的另一個讀音
    assert by("雑魚は黙ってて", [("雑魚", "ざこ")])["雑魚"].reading == "ざこ"


def test_grammar_notes():
    # JLPT_grammer 的說明（tools/grammar_notes.py 產生）：規則編號都存在、文法項目帶著說明
    from jplens.jlearn import grammar
    from jplens.words import _grammar_notes
    notes = _grammar_notes()
    ids = {r.id for r in grammar.all_rules()}
    assert notes and set(notes) <= ids
    w = next(w for w in extract(["このレストランは値段が高いわりには、あまり美味しくない。"], Dictionary())
             if w.is_grammar and "わりに" in w.word)
    assert w.notes and "わりに" in w.notes[0]["pattern"] and w.notes[0]["example"]


def test_fix_angle_bracket_ku():
    from jplens.jlearn.textutil import fix_common_ocr
    assert fix_common_ocr("ちょくちょ〈") == "ちょくちょく"          # 「く」被讀成「〈」
    assert fix_common_ocr("よ〈ある話") == "よくある話"
    assert fix_common_ocr("画〈字〉") == "画〈字〉"                    # 成對的括號不動
    assert fix_common_ocr("太郎〈17歳〉") == "太郎〈17歳〉"
    assert fix_common_ocr("字：〈コマ") == "字：〈コマ"                # 前面不是平假名


def test_translation_cache_key():
    from jplens.ai import cache_key
    # 只差在引號（含半邊）、句尾標點、刪節號寫法、空白的算同一句
    same = ["「追いついたよ", "「追いついたよ」", "追いついたよ。", "『追いついたよ…』", "追いついたよ...", " 追いついたよ "]
    assert len({cache_key(t) for t in same}) == 1
    # 疑問句、差一個字的不算同一句
    assert cache_key("追いついたよ？") != cache_key("追いついたよ")
    assert cache_key("行ける") != cache_key("行けない")
