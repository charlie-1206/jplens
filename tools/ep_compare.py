"""比較 ONNX Runtime 的加速方式（CUDA／DirectML／Windows ML 的 TensorRT-RTX、OpenVINO／CPU）：速度和 box_eval 準確度。

    python tools/ep_compare.py cuda 輸出.json        # 專案環境（onnxruntime-gpu）
    另一個裝 onnxruntime-windowsml 的環境（要 Python 3.11 以上）：
    python tools/ep_compare.py dml 輸出.json
    python tools/ep_compare.py trtrtx 輸出.json     # Windows ML 的 NVIDIA TensorRT-RTX（Windows Update 裝的）
    python tools/ep_compare.py openvino 輸出.json   # Windows ML 的 Intel OpenVINO
    python tools/ep_compare.py cpu 輸出.json
    python tools/ep_compare.py 比較 a.json b.json …  # 把幾次的結果排成表；manga-ocr 的輸出和第一個檔案比

程式裡寫死要 CUDA 的地方（gpu.cuda_ready、各個 InferenceSession 的 providers）在這裡換成指定的加速方式，
其他照 app 原本的流程跑 tests/eval/box_eval.py，同時計時：整個螢幕的文字偵測（按下 Alt）、manga-ocr（放開滑鼠）、
漫畫文字區塊偵測（漫畫模式）。跑完再把每張圖、每批辨識各重跑幾次取中位數（第一次可能含編譯時間，另外記）。
"""
import glob
import json
import os
import runpy
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

PLUGIN = {  # Windows ML 的加速元件（Windows Update 裝在 WindowsApps）：(EP 名稱, 套件名稱開頭, DLL)
    "trtrtx": ("NvTensorRTRTXExecutionProvider", "Microsoft.WinML.NVIDIA.TRT-RTX.EP",
               "onnxruntime_providers_nv_tensorrt_rtx.dll"),
    "openvino": ("OpenVINOExecutionProvider", "Microsoft.WinML.Intel.OpenVINO.EP", "onnxruntime_providers_openvino*.dll"),
}


def _plugin_dll(prefix: str, pattern: str) -> str:
    # WindowsApps 一般使用者不能列出內容，但套件自己的資料夾讀得到：用 Get-AppxPackage 問安裝位置
    import subprocess
    locs = subprocess.run(["powershell", "-NoProfile", "-Command",
                           f"(Get-AppxPackage -Name '{prefix}*' | Where-Object Architecture -eq 'X64').InstallLocation"],
                          capture_output=True, text=True).stdout.splitlines()
    hits = sorted(h for loc in locs if loc.strip() for h in glob.glob(str(Path(loc) / "**" / pattern), recursive=True))
    if not hits:
        sys.exit(f"找不到 {prefix} 的 {pattern}（Windows Update 還沒裝？）")
    return hits[-1]


def install(ep: str):
    """把程式裡「要 CUDA」的 session 換成 ep。cpu：照 app 沒有顯示卡時的流程。"""
    import onnxruntime as ort

    from jplens import gpu
    if ep in ("cuda", "dml"):
        return  # 程式本來就支援（gpu.backend()：CUDA 版 onnxruntime-gpu、DirectML 版 onnxruntime-directml）
    if ep == "cpu":
        gpu._backend = "cpu"
        return
    gpu.cuda_ready = lambda: True  # Windows ML 的加速元件：假裝是 CUDA，下面把 session 換掉
    name = None
    if ep in PLUGIN:
        name, prefix, pattern = PLUGIN[ep]
        ort.register_execution_provider_library(name, _plugin_dll(prefix, pattern))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == name] if name else []
    if name:
        print("加速裝置：", [(d.ep_name, d.device.type, d.ep_vendor) for d in devices])
        devices = [d for d in devices if str(d.device.type).endswith("GPU")] or devices
    base = ort.InferenceSession
    configured = []

    class Session(base):
        def __init__(self, path, sess_options=None, providers=None, **kw):
            wants = any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in providers or [])
            self._fake = wants
            if wants:
                so = sess_options or ort.SessionOptions()
                if ep == "dml":
                    so.enable_mem_pattern = False  # DirectML 的要求
                    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                else:
                    if ep == "trtrtx" and Path(path).parent.name == "ogkalu":
                        # TensorRT-RTX 遇到變動尺寸建不起引擎（TopK）；bubbles.detect 一律送 1×SIZE×SIZE
                        from jplens.bubbles import SIZE
                        for k, v in (("N", 1), ("H", SIZE), ("W", SIZE)):
                            so.add_free_dimension_override_by_name(k, v)
                    if not any(x is so for x in configured):  # manga-ocr 的編碼器、解碼器共用同一個 SessionOptions，只能加一次
                        so.add_provider_for_devices(devices[:1], {})
                        configured.append(so)
                    providers = None
                sess_options = so
            super().__init__(path, sess_options, providers=providers, **kw)

        def get_providers(self):
            real = super().get_providers()
            # 程式用「CUDAExecutionProvider 在不在裡面」判斷有沒有用到顯示卡；manga-ocr 用第一個決定資料放哪
            return real[:1] + ["CUDAExecutionProvider"] + real[1:] if self._fake and real[0] != "CPUExecutionProvider" else real

    ort.InferenceSession = Session
    if ep == "trtrtx":  # TensorRT-RTX 用 CUDA 的記憶體：manga-ocr 的中間結果留在顯示卡上（不然每一步都要來回複製）
        from jplens import mangaocr_onnx
        init = mangaocr_onnx.MangaOcrOnnx.__init__

        def init_cuda(self, *a, **kw):
            init(self, *a, **kw)
            self.device = "cuda"
        mangaocr_onnx.MangaOcrOnnx.__init__ = init_cuda


def bench(ep: str, out: str):
    t0 = time.perf_counter()
    install(ep)
    import cv2
    import numpy as np

    from jplens import bubbles, recognize
    times = {"detect": [], "rec": [], "bubbles": []}
    calls = {"rec": [], "screens": []}

    def timed(key, f):
        def g(*a, **kw):
            t = time.perf_counter()
            r = f(*a, **kw)
            times[key].append(time.perf_counter() - t)
            return r
        return g

    rb, ds, bd = recognize.Recognizer.rec_batch, recognize.Recognizer.detect_screen, bubbles.BubbleDetector.detect

    def rec_batch(self, grays):
        r = rb(self, grays)
        calls["rec"].append(([g.copy() for g in grays], r))
        return r
    recognize.Recognizer.rec_batch = timed("rec", rec_batch)
    recognize.Recognizer.detect_screen = timed("detect", ds)
    bubbles.BubbleDetector.detect = timed("bubbles", bd)

    t1 = time.perf_counter()
    ns = runpy.run_path(str(ROOT / "tests" / "eval" / "box_eval.py"), run_name="box_eval")
    t_eval = time.perf_counter() - t1
    rec, det = ns["rec"], ns["det"]
    print(f"\n偵測：{rec.det_device}；辨識：{rec.rec_device}；漫畫區塊：{det.device}")
    first = {k: (v[0] if v else 0) for k, v in times.items()}

    # 重跑：每張圖偵測 3 次、每批辨識 2 次，取中位數
    recognize.Recognizer.rec_batch, recognize.Recognizer.detect_screen, bubbles.BubbleDetector.detect = rb, ds, bd
    imgs = [cv2.imread(f) for f in ns["cache"]]
    med = {"detect": [], "rec": [], "bubbles": []}
    for img in imgs:
        for key, f in (("detect", lambda: rec.detect_screen(img)), ("bubbles", lambda: det.detect(img))):
            ts = []
            for _ in range(3):
                t = time.perf_counter()
                f()
                ts.append(time.perf_counter() - t)
            med[key].append(statistics.median(ts))
    texts = []
    for grays, r in calls["rec"]:
        ts = []
        for _ in range(2):
            t = time.perf_counter()
            again = rec.rec_batch(grays)
            ts.append(time.perf_counter() - t)
        med["rec"].append(statistics.median(ts))
        texts.append({"n": len(grays), "first": r, "again": again})
    res = {"ep": ep, "ort": __import__("onnxruntime").__version__, "python": sys.version.split()[0],
           "devices": [rec.det_device, rec.rec_device, det.device],
           "load_and_eval_sec": round(t_eval, 2), "total_sec": round(time.perf_counter() - t0, 2),
           "first_call_sec": {k: round(v, 3) for k, v in first.items()},
           "median_ms": {k: round(1000 * statistics.median(v), 1) for k, v in med.items() if v},
           "sum_ms": {k: round(1000 * sum(v), 1) for k, v in med.items()},
           "rec_lines": sum(t["n"] for t in texts), "texts": texts,
           "screens": [list(img.shape[:2]) for img in imgs]}
    # 拿基準（cuda）那次辨識的同一批圖重跑：偵測結果不同時送進 manga-ocr 的行也不同，只有這樣才能逐行比
    replay = Path(out).with_name("ep_replay.npz")
    if ep == "cuda":
        flat = [g for grays, _ in calls["rec"] for g in grays]
        np.savez_compressed(replay, sizes=np.array([len(g) for g, _ in calls["rec"]]), **{f"g{i}": g for i, g in enumerate(flat)})
    elif replay.exists():
        z = np.load(replay)
        flat = [z[f"g{i}"] for i in range(int(z["sizes"].sum()))]
        res["replay"] = rec.rec_batch(flat)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("texts", "screens")}, ensure_ascii=False))


def compare(paths):
    rs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    base = [t for x in rs[0]["texts"] for t in x["first"]]
    same = lambda r: r.get("replay") or [t for x in r["texts"] for t in x["first"]]  # noqa: E731
    print("| 項目 | " + " | ".join(r["ep"] for r in rs) + " |")
    print("| --- |" + " --- |" * len(rs))
    rows = [("ONNX Runtime", lambda r: r["ort"]),
            ("偵測／辨識／漫畫區塊", lambda r: "／".join(r["devices"])),
            ("載入＋跑完 box_eval（秒）", lambda r: r["load_and_eval_sec"]),
            ("第一次整個螢幕偵測（秒）", lambda r: r["first_call_sec"]["detect"]),
            ("第一次 manga-ocr（秒）", lambda r: r["first_call_sec"]["rec"]),
            ("整個螢幕偵測 中位數（ms）", lambda r: r["median_ms"]["detect"]),
            ("manga-ocr 每批 中位數（ms）", lambda r: r["median_ms"]["rec"]),
            ("manga-ocr 全部（ms）", lambda r: r["sum_ms"]["rec"]),
            ("漫畫區塊偵測 中位數（ms）", lambda r: r["median_ms"]["bubbles"]),
            (f"同一批圖 manga-ocr 輸出和 {rs[0]['ep']} 相同", lambda r: "{}/{}".format(
                sum(a == b for a, b in zip(base, same(r))), len(base))),
            ("同一批重跑結果相同", lambda r: "{}/{}".format(sum(x["first"] == x["again"] for x in r["texts"]), len(r["texts"])))]
    for name, f in rows:
        print(f"| {name} | " + " | ".join(str(f(r)) for r in rs) + " |")
    for r in rs[1:]:
        diff = [(a, b) for a, b in zip(base, same(r)) if a != b]
        for a, b in diff[:8]:
            print(f"  {r['ep']}：{a} → {b}")


if __name__ == "__main__":
    if sys.argv[1] == "比較":
        compare(sys.argv[2:])
    else:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        bench(sys.argv[1], sys.argv[2])
