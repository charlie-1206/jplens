# 日文即時螢幕查字工具（jplens）

在 Windows 上**按住 Alt、用滑鼠拖曳**選取畫面上的日文（遊戲、動畫暫停畫面、漫畫、小說，橫排直排都可以），
馬上列出選到的單字：讀音、日檢級數、詞性、繁體中文意思，下面再列出句子裡的**文法句型**（例如「〜と思う」「〜かもしれない」）。
想記的字、文法、句子按「＋ 記錄」存進單字本，可以搜尋、篩選、標重點，再匯出到單字 App。

- 全部在自己的電腦上跑（文字辨識用 [manga-ocr](https://github.com/kha-white/manga-ocr)、斷詞用 Sudachi），不用網路、不用 API。
- 有 NVIDIA 顯示卡時放開滑鼠到出結果約 0.1～0.5 秒；沒有也能用（CPU，慢很多）。
- 直排小說、振假名、漫畫對話框都有特別處理。

## 下載就能用的版本

到 [Releases](../../releases) 下載壓縮檔，解壓縮後雙擊 `jplens.exe`（不用裝 Python）。右下角系統匣出現紅色「字」圖示、跳出「就緒」通知就可以用了。
Windows 10／11 64 位元。第一次開啟時 Windows SmartScreen 可能會警告（執行檔沒有簽章），按「其他資訊」→「仍要執行」。

## 怎麼用

三種選取方式（按住 Alt＋按一下右鍵、結果視窗標題列的按鈕，或系統匣選單切換）：
- **反白**：像在文書軟體裡選字，沿著文字從第一個字拖到最後一個字，只選你要的那句，跨行、直排都可以。
- **框選**：拖曳出一個矩形，查矩形裡所有的文字。
- **漫畫**：用漫畫專用的模型找出對話框裡的字，整塊讀，不會漏欄。按住 Alt **框住對話框**，或按住 Alt **雙擊對話框**。
  只適合漫畫，字幕、小說、遊戲請用另外兩種。

| 動作 | 說明 |
| --- | --- |
| 按住 Alt＋左鍵拖曳 | 選取要查的文字，放開就開始辨識 |
| 按住 Alt＋按一下右鍵 | 切換選取方式（反白 → 框選 → 漫畫） |
| 結果視窗的級數選單 | 只顯示這個級數以上的單字（級外的字一律顯示） |
| 滑鼠停在單字上 | 完整的中英文意思、出現的句子；文法顯示說明、接續、例句 |
| ＋ 記錄／＋ 記錄原句 | 存進單字本；先反白原文的一部分，就只記錄那一段 |
| 📌 | 釘選：點畫面其他地方時不自動關閉 |
| Esc、✕、點別的地方 | 關閉結果視窗 |

**系統匣選單**（「字」圖示，左鍵或右鍵；可能被收在工作列右下角的「^」裡）：
- **單字本**：單字／文法／句子分開看；搜尋、篩選（級數、詞性、記錄時間、已學會、重點單字）、排序；
  星號標**重點單字**、標記**已學會**、編輯、刪除（可以復原）。
- **匯出**：單字 → 單字 App 備份檔（重點單字會標成 App 的重點）；句子 → 編號的文字檔。匯出前可以選範圍。
- **設定**：熱鍵（Alt／Ctrl＋Shift／Ctrl＋Alt）、預設選取方式、結果視窗的字級與寬度、單字本資料夾、開機自動啟動等，改了馬上生效。
- **保存上一次查詢的畫面**：讀錯時存下來並填正確答案，當測試資料。

**遊戲裡沒反應？** 很多遊戲（例如絕區零、鳴潮）以系統管理員身分執行，Windows 不讓一般權限的程式收到在遊戲視窗上的按鍵。
從系統匣選「以系統管理員身分重新啟動」，或在設定 → 一般打開「以系統管理員身分執行」。遊戲請用視窗模式或無邊框視窗。

## 從原始碼執行

Windows、Python 3.10。

```bash
pip install -r requirements.txt
```

repo 裡沒有模型和字典（太大），要先準備：

1. **字典** `data/dict.sqlite`：下載下面三個檔案放到 `data/raw/`，再執行 `python tools/build_dict.py`（約 1～2 分鐘）。
   - `JMdict_e.gz`：<http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz>
   - `zhwikt_ja.jsonl`：中文維基詞典的日語詞條，<https://kaikki.org/zhwiktionary/日語/>（JSONL）
   - `tomoshi/tomoshi-dict-open.db`：[Tomoshi 開放資料](https://github.com/tomoshi-app/tomoshi-dict-data/releases) 的 `.db.zst`，用 `zstd -d` 解壓縮（沒有也能建，中文意思會少很多）
2. **文字辨識模型**：`python tools/export_manga_ocr_onnx.py`，把 manga-ocr 匯出成 ONNX 到 `data/models/manga-ocr-onnx/`（約 460 MB）。
   只有匯出時需要 `pip install torch transformers manga-ocr`，之後執行查字工具不需要。
3. **漫畫模式的模型**（要用漫畫模式才需要）：[ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector/tree/main)
   的 `detector.onnx`（NVIDIA 顯示卡用）或 `detector_int8.onnx`（CPU 用），連同 `config.json`、`preprocessor_config.json` 放到 `data/models/ogkalu/`。
4. （選用，加速）文字偵測模型改吃 uint8：`pip install onnx` 後 `python tools/export_det_onnx.py`。沒做就用 RapidOCR 附的模型。

有 NVIDIA 顯示卡時要找得到 CUDA 12＋cuDNN 9 的程式庫：放在 `data/cuda/`、pip 的 `nvidia-*` 套件，或裝了 PyTorch（cu12 版）時借用它的
（`jplens/gpu.py`）。找不到會用 CPU。

啟動：雙擊 `run_realtime_words_translate.bat`（不開主控台視窗），或要看訊息時：

```bash
python -m jplens
```

### 打包成執行檔

```bash
pip install pyinstaller
python packaging/build.py            # 帶 CUDA／cuDNN（約 3.5 GB）
python packaging/build.py --no-cuda  # 不帶（沒有 CUDA 的電腦只能用 CPU）
```

輸出在 `dist/jplens/`，整個資料夾就是可以給別人的版本；打包完會自動跑 `jplens.exe --selftest` 確認模型、字典都帶進去了。

## 檔案

| 位置 | 內容 |
| --- | --- |
| `vocab/saved_words.jsonl` | 單字本：一行一個字，含讀音、級數、中英文意思、原句、記錄時間、重點、已學會 |
| `vocab/saved_sentences.jsonl` | 記錄的原句 |
| `config.json` | 設定（設定視窗會寫入） |
| `logs/jplens.log` | 執行紀錄（每次查詢的時間、錯誤訊息） |
| `jplens/` | 程式（`app.py` 主程式、`layout.py`／`inkcols.py` 版面、`recognize.py` 辨識、`words.py` 單字與文法、`manager.py` 單字本、`settings_ui.py` 設定） |
| `tools/` | 建字典、匯出模型、文法規則與字典翻譯的批次工具 |
| `tests/` | `python -c "import tests.test_core as m; [getattr(m,k)() for k in dir(m) if k.startswith('test_')]"`（需要字典）；`tests/eval/` 是辨識準確率的評估（需要自己的截圖） |

打包版的 `config.json`、`vocab/`、`logs/` 在 `jplens.exe` 旁邊。

## 中文意思從哪裡來

依序：Gemini 翻譯（日檢單字表的詞，逐個 JMdict 意思翻譯，`dict_translate/zh.jsonl`）→ Tomoshi 開放資料 → 中文維基詞典；
都沒有才顯示 JMdict 的英文意思（標示〔英〕）。文法規則 531 條：228 條是專案原本的規則，其餘 303 條由 Gemini 依 JLPT 句型清單寫成、經自動驗證與人工抽查。

## 授權與來源

程式碼以 Apache License 2.0 授權（[LICENSE](LICENSE)）。字典等資料的授權和程式碼不同（多為 CC BY-SA 4.0），完整清單見 [NOTICE.md](NOTICE.md)。

- 英文意思、詞性：[JMdict](https://www.edrdg.org/jmdict/j_jmdict.html)（EDRDG，CC BY-SA 4.0）
- 中文意思：[Tomoshi 開放資料](https://github.com/tomoshi-app/tomoshi-dict-data)（Tomoshi (Y1Z)，CC BY-SA 4.0）、
  [中文維基詞典](https://zh.wiktionary.org/)（CC BY-SA，經 [kaikki.org](https://kaikki.org/zhwiktionary/) 擷取）、Gemini 翻譯（譯自 JMdict，CC BY-SA 4.0）
- 日檢單字表（`data/jlpt/`）：Jonathan Waller 的 JLPT 單字表（CC BY），經 open-anki-jlpt-decks（MIT）取得
- 文字辨識：[manga-ocr](https://github.com/kha-white/manga-ocr)（Apache-2.0）；文字偵測：PP-OCR／[RapidOCR](https://github.com/RapidAI/RapidOCR)（Apache-2.0）
- 漫畫模式：[ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)（Apache-2.0）
- 斷詞：[SudachiPy](https://github.com/WorksApplications/sudachi.rs)、SudachiDict（Apache-2.0）；介面：Qt／PySide6（LGPL-3.0）
