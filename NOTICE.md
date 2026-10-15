# 螢幕查字

Copyright 2026 charlie-1206

本專案的程式碼以 Apache License 2.0 授權（見 `LICENSE`）。

# 第三方資料、模型與套件的來源與授權

下面列出本專案使用、或打包時會一起散布的第三方內容。
**資料檔的授權和程式碼不同**，轉傳、修改時請照各自的條款標示來源。

## 字典資料（CC BY-SA）

| 內容 | 來源 | 授權 | 本專案裡的位置 |
| --- | --- | --- | --- |
| 英文意思、讀音、詞性、詞義標籤 | [JMdict](https://www.edrdg.org/jmdict/j_jmdict.html)，The Electronic Dictionary Research and Development Group（EDRDG） | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)（[EDRDG 授權說明](https://www.edrdg.org/edrdg/licence.html)） | `data/dict.sqlite`（由 `tools/build_dict.py` 建立，不在 repo 裡） |
| 中文意思（JMdict 的中文翻譯） | [Tomoshi 開放資料](https://github.com/tomoshi-app/tomoshi-dict-data)，Tomoshi (Y1Z)；JMdict 衍生作品 | CC BY-SA 4.0 | `data/dict.sqlite` |
| 中文意思（維基詞典的日語詞條） | [中文維基詞典](https://zh.wiktionary.org/)，經 [kaikki.org](https://kaikki.org/zhwiktionary/)（wiktextract）擷取 | CC BY-SA 4.0 及 GFDL | `data/dict.sqlite` |
| 中文意思（日檢單字表的詞由 Gemini 逐個 JMdict 意思翻譯） | 翻譯的是 JMdict 的英文意思，屬 JMdict 衍生作品 | CC BY-SA 4.0 | `dict_translate/` |

- 依 CC BY-SA 4.0，散布 `data/dict.sqlite`、`dict_translate/` 或它們的修改版時，必須標示上面的來源，並以相同授權（CC BY-SA 4.0）釋出。
- 本專案**不是** Tomoshi 的官方產品，也沒有得到 Tomoshi 的背書；不使用 Tomoshi 的名稱或標誌暗示任何關係。
- 字典建好後，來源說明存在 `data/dict.sqlite` 的 `meta` 表，結果視窗下方也會顯示。

## 日檢單字表（CC BY）

| 內容 | 來源 | 授權 | 位置 |
| --- | --- | --- | --- |
| JLPT N5～N1 單字表 | Jonathan Waller 整理（[tanos.co.uk](https://www.tanos.co.uk/jlpt/)） | CC BY | `data/jlpt/*.csv` |
| 上述單字表的 CSV 整理版 | [jamsinclair/open-anki-jlpt-decks](https://github.com/jamsinclair/open-anki-jlpt-decks) | MIT | `data/jlpt/*.csv` |

詳見 `data/jlpt/README.md`。日本語能力試驗官方自 2010 年起不公布單字表，級數僅供參考。

## 文法資料

`JLPT_grammer/`、`grammar_gen/results/`、`jplens/jlearn/grammar_rules_extra.jsonl`、`jplens/jlearn/grammar_notes.json`
是本專案作者請 Gemini 整理、產生後再以程式驗證的內容，與程式碼採用相同授權（Apache-2.0，見 `LICENSE`）。

## 模型（不在 repo 裡；打包時會附上）

| 模型 | 來源 | 授權 | 備註 |
| --- | --- | --- | --- |
| manga-ocr（文字辨識） | [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr)、[kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) | Apache-2.0 | 本專案以 `tools/export_manga_ocr_onnx.py` 轉成 ONNX（屬修改；散布時附 Apache-2.0 全文並註明有修改） |
| PP-OCR 文字偵測（經 RapidOCR） | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[RapidOCR](https://github.com/RapidAI/RapidOCR) | Apache-2.0 | |
| 漫畫文字區塊偵測 | [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) | Apache-2.0 | |

## Python 套件

| 套件 | 授權 |
| --- | --- |
| PySide6（Qt for Python） | LGPL-3.0（打包散布時以動態連結方式使用，附 LGPL 全文與 Qt 原始碼的取得方式，使用者可以替換） |
| ONNX Runtime | MIT |
| RapidOCR（rapidocr-onnxruntime） | Apache-2.0 |
| SudachiPy、SudachiDict-core | Apache-2.0 |
| OpenCV（opencv-python） | Apache-2.0 |
| NumPy | BSD-3-Clause |
| Pillow | MIT-CMU（HPND） |
| mss | MIT |
| jaconv | MIT |
| huggingface_hub | Apache-2.0 |
| OpenCC | Apache-2.0 |

NVIDIA CUDA／cuDNN 的程式庫（有 NVIDIA 顯示卡時使用）依 NVIDIA 的授權條款，只在打包散布時附上可散布的部分，不放在 repo 裡。
