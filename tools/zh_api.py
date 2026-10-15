"""背景呼叫 Gemini API，把日檢以外的 JMdict 條目翻成中文（日檢的詞由使用者手動分批翻，見 zh_batches.py）。

    python tools/zh_api.py plan                  # 只估算：要翻幾個詞、幾批、多久（不呼叫 API）
    python tools/zh_api.py run                   # 開始翻：每批 150 個詞，每 3 分鐘一批，翻完或停止為止
    python tools/zh_api.py run --max-batches 1   # 只翻一批（試品質、看 token 用量）
    python tools/zh_api.py status                # 進度
    python tools/zh_api.py stop                  # 請背景程式在這一批結束後停下來

做法：
1. 順序：JMdict 標記為常用的條目先翻（依常用度排序），--scope all 時再翻其餘條目。
   日檢單字表的詞（手動批次 batch_／retry_ 裡的）和已經翻過的都跳過。
2. 題目格式、翻譯指示和手動批次完全相同（zh_batches.render），回覆用同樣的規則檢查：編號、意思數量。
3. 通過檢查的寫進 dict_translate/zh_api.jsonl（每批附加一次，查字工具會自動重讀）；
   缺漏或意思數量不對的詞排到下一批重翻，失敗兩次就跳過。原始回覆存在 dict_translate/api_raw/。
4. 伺服器忙碌（5xx）退避重試；超過每分鐘額度等 2 分鐘；每日額度用完先換備援模型，都用完就每小時再試一次。
   紀錄在 logs/zh_api.log。可以隨時中斷，重新執行會從沒翻的詞繼續。
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zh_batches import (MAX_SENSES, OUT, WORK, _LINE, load_done, load_jmdict,  # noqa: E402
                        load_manifest, render)
from zhconv import simplified_chars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
API_OUT = WORK / "zh_api.jsonl"
RAW = WORK / "api_raw"
STATE = WORK / "api_state.json"
STOP = WORK / "api_stop"
LOG = ROOT / "logs" / "zh_api.log"
MODELS = ["gemini-3.5-flash", "gemini-3.8-flash"]  # 和 video_translate 相同；不用 Flash-Lite（品質優先）
MAX_FAILS = 2
_exhausted: dict[str, float] = {}  # 模型 → 每日額度用完的時間（1 小時後再試）


def log(msg: str):
    line = f"{datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def rank(e) -> int:
    """常用度：nf01～nf48（每 500 詞一級）最準；只有 news1 等標記的當作 nf24 左右；沒有標記的排最後。"""
    nf = [int(p[2:]) for p in e["pri"] if p.startswith("nf")]
    if nf:
        return min(nf)
    if any(p.endswith("1") for p in e["pri"]):
        return 24
    return 48 if e["pri"] else 99


def candidates(scope: str) -> list[dict]:
    print("讀取 JMdict（約 20 秒）…", flush=True)
    index = load_jmdict()
    entries = {e["id"]: e for es in index.values() for e in es}
    manifest = load_manifest()
    manual = {i for name, ids in manifest["batches"].items() for i in ids}  # 手動批次（日檢）的詞
    out = []
    for e in entries.values():
        if e["id"] in manual or not e["senses"] or (scope == "common" and not e["common"]):
            continue
        word = e["kebs"][0] if e["kebs"] else e["rebs_raw"][0]
        out.append({"id": e["id"], "word": word, "reading": e["rebs_raw"][0], "level": None,
                    "senses": [list(s) for s in e["senses"][:MAX_SENSES]], "rank": rank(e)})
    out.sort(key=lambda it: (it["rank"], int(it["id"])))
    return out


def load_api_done() -> dict:
    done = {}
    if API_OUT.exists():
        for line in API_OUT.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line) if line.strip() else None
            except ValueError:
                r = None
            if r:
                done[r["id"]] = r
    return done


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"batch": 0, "fails": {}, "tokens_in": 0, "tokens_out": 0}


def save_state(state: dict):
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def todo_list(scope: str):
    items = candidates(scope)
    done = set(load_done()) | set(load_api_done())
    fails = load_state()["fails"]
    todo = [it for it in items if it["id"] not in done and fails.get(it["id"], 0) < MAX_FAILS]
    return items, todo


def cmd_plan(args):
    items, todo = todo_list(args.scope)
    n_batches = -(-len(todo) // args.size)
    sample = render(todo[:args.size]) if todo else ""
    print(f"範圍 {args.scope}：共 {len(items)} 個條目（不含日檢），還沒翻 {len(todo)} 個")
    print(f"每批 {args.size} 個 → {n_batches} 批；每 {args.interval} 秒一批，約 {n_batches * args.interval / 3600:.1f} 小時")
    print(f"第一批題目約 {len(sample)} 字元（實際 token 數以第一批 API 回報為準）")
    by_rank = {}
    for it in todo:
        key = "常用（nf01～24）" if it["rank"] <= 24 else ("常用（其他）" if it["rank"] < 99 else "非常用")
        by_rank[key] = by_rank.get(key, 0) + 1
    for k, v in by_rank.items():
        print(f"  {k}：{v} 個")


def call(client, prompt: str, state: dict):
    """回傳 (文字, 模型)；None = 應該停止（每日額度都用完以外的錯誤，或收到停止指令）。"""
    from google.genai import errors, types
    config = types.GenerateContentConfig(max_output_tokens=32768)
    exhausted = _exhausted
    busy = 0
    while not STOP.exists():
        avail = [m for m in MODELS if time.time() - exhausted.get(m, 0) > 3600]
        if not avail:
            log("所有模型的每日額度都用完，1 小時後再試")
            if not wait(3600):
                return None
            continue
        model = avail[0]
        try:
            started = time.time()
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            u = resp.usage_metadata
            if u:
                state["tokens_in"] += u.prompt_token_count or 0
                state["tokens_out"] += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
                log(f"  {model} {time.time() - started:.0f} 秒，輸入 {u.prompt_token_count} / 輸出 "
                    f"{u.candidates_token_count} / 思考 {u.thoughts_token_count or 0} tokens")
            cand = (resp.candidates or [None])[0]
            if cand is not None and str(getattr(cand, "finish_reason", "")).endswith("MAX_TOKENS"):
                log("  回覆超過長度上限被截斷，只收完整的行")
            return resp.text or "", model
        except errors.APIError as e:
            msg = (e.message or "")[:200]
            if e.code == 429 and re.search(r"per ?day|PerDay|daily", str(e), re.I):
                log(f"  {model} 每日額度用完：{msg}")
                exhausted[model] = time.time()
            elif e.code == 429:
                log(f"  {model} 超過每分鐘額度，等 2 分鐘：{msg}")
                if not wait(120):
                    return None
            elif e.code in (500, 502, 503, 504):
                busy += 1
                log(f"  {model} 伺服器忙碌（{e.code}），第 {busy} 次，等 {60 * min(busy, 5)} 秒")
                if busy % 3 == 0 and len(avail) > 1:
                    exhausted[model] = time.time() - 3000  # 這個模型先休息 10 分鐘，改用備援
                if not wait(60 * min(busy, 5)):
                    return None
            else:
                log(f"  {model} 錯誤 {e.code}，停止：{msg}")
                return None
        except Exception as e:  # 網路斷線等
            log(f"  連線錯誤，2 分鐘後重試：{str(e)[:200]}")
            if not wait(120):
                return None
    return None


def wait(seconds: float) -> bool:
    """等待；收到停止指令就回傳 False。"""
    end = time.time() + seconds
    while time.time() < end:
        if STOP.exists():
            return False
        time.sleep(min(5, max(0, end - time.time())))
    return True


def parse(text: str, part: list[dict], batch: str):
    want = {it["id"]: it for it in part}
    rows, problems = {}, []
    for raw in text.splitlines():
        m = _LINE.match(raw)
        if not m or m.group(1) not in want:
            continue
        id_, rest = m.groups()
        it = want[id_]
        senses = [s.strip() for s in re.split(r"[｜|]", rest) if s.strip()]
        if len(senses) != len(it["senses"]):
            problems.append(f"{id_} {it['word']} 應有 {len(it['senses'])} 個意思，回了 {len(senses)} 個")
            continue
        sc = simplified_chars("".join(senses))
        if sc:
            problems.append(f"{id_} {it['word']} 可能有簡體字 {''.join(sc)}（已收）")
        rows[id_] = {"id": id_, "word": it["word"], "reading": it["reading"], "level": None,
                     "pos": [p for p, _ in it["senses"]], "en": [g for _, g in it["senses"]], "zh": senses,
                     "source": batch, "imported": datetime.now().isoformat(timespec="seconds")}
    return rows, problems


def cmd_run(args):
    from google import genai
    STOP.unlink(missing_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    _, todo = todo_list(args.scope)
    log(f"開始：範圍 {args.scope}，還沒翻 {len(todo)} 個詞，每批 {args.size} 個，每 {args.interval} 秒一批")
    client = genai.Client()
    state = load_state()
    count = 0
    while todo and not STOP.exists() and (not args.max_batches or count < args.max_batches):
        started = time.time()
        part = todo[:args.size]
        state["batch"] += 1
        batch = f"jm_{state['batch']:05d}"
        result = call(client, render(part), state)
        if result is None:
            state["batch"] -= 1
            break
        text, model = result
        (RAW / f"{batch}.txt").write_text(text, encoding="utf-8")
        rows, problems = parse(text, part, batch)
        if rows:
            with API_OUT.open("a", encoding="utf-8") as f:  # 一次寫完一批，查字工具不會讀到半批
                f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()))
        for it in part:
            if it["id"] not in rows:
                state["fails"][it["id"]] = state["fails"].get(it["id"], 0) + 1
        save_state(state)
        count += 1
        log(f"{batch}（{model}）：收 {len(rows)}/{len(part)} 個；累計輸入 {state['tokens_in']:,} / "
            f"輸出 {state['tokens_out']:,} tokens")
        for p in problems[:10]:
            log("  " + p)
        # 缺漏的詞排到下一批前面重翻（失敗兩次就跳過）
        retry = [it for it in part if it["id"] not in rows and state["fails"][it["id"]] < MAX_FAILS]
        todo = retry + todo[len(part):]
        if todo and (not args.max_batches or count < args.max_batches):
            if not wait(max(0, args.interval - (time.time() - started))):
                break
    log("收到停止指令，結束" if STOP.exists() else ("全部翻完" if not todo else f"結束（這次翻了 {count} 批）"))
    STOP.unlink(missing_ok=True)


def cmd_status(args):
    done = load_api_done()
    state = load_state()
    skipped = sum(1 for v in state["fails"].values() if v >= MAX_FAILS)
    print(f"API 已翻 {len(done)} 個詞（{state['batch']} 批）；手動已翻 {len(load_done())} 個詞（{OUT.name}）")
    print(f"累計 token：輸入 {state['tokens_in']:,} / 輸出（含思考）{state['tokens_out']:,}；失敗兩次跳過 {skipped} 個")
    if LOG.exists():
        print("最近的紀錄：")
        for line in LOG.read_text(encoding="utf-8").splitlines()[-8:]:
            print("  " + line)


def cmd_stop(args):
    STOP.write_text("stop", encoding="utf-8")
    print("已送出停止指令：背景程式會在這一批結束（或等待中）時停下來")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--scope", choices=["common", "all"], default="common",
                       help="common：只翻 JMdict 標記為常用的條目（預設）；all：全部")
        p.add_argument("--size", type=int, default=150, help="每批幾個詞（預設 150）")
        p.add_argument("--interval", type=int, default=180, help="每批間隔幾秒（預設 180）")
        if name == "run":
            p.add_argument("--max-batches", type=int, default=0, help="最多翻幾批（0 = 不限）")
    sub.add_parser("status")
    sub.add_parser("stop")
    args = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "status": cmd_status, "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    main()
