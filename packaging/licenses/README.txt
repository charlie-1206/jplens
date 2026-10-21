打包版（jplens.exe）附帶的第三方程式庫授權說明
================================================

本程式的程式碼以 Apache License 2.0 授權（上一層的 LICENSE）；字典、模型等資料的來源與授權見上一層的 NOTICE.md。
這個資料夾是執行檔裡附帶的第三方程式庫的授權。

Qt for Python（PySide6 6.12.0）與 Qt 6
--------------------------------------
- 授權：GNU Lesser General Public License v3（LGPL-3.0-only），全文見 LGPL-3.0.txt；LGPL 引用的 GPL v3 全文見 GPL-3.0.txt。
- 本程式以動態連結方式使用 Qt：Qt 的程式庫是 _internal\PySide6\ 裡獨立的 DLL（Qt6Core.dll、Qt6Gui.dll、Qt6Widgets.dll 等）
  與 _internal\shiboken6\。
- 你可以用相容版本的 Qt／PySide6 替換這些 DLL；也可以直接從原始碼執行本程式（見 https://github.com/charlie-1206/jplens ），
  用 pip 安裝任何版本的 PySide6。
- Qt 與 PySide6 的原始碼：https://code.qt.io/cgit/qt/qtbase.git/ 、https://code.qt.io/cgit/pyside/pyside-setup.git/
  （版本 6.12.0；也可以從 https://download.qt.io/official_releases/ 取得）。

NVIDIA CUDA 與 cuDNN（_internal\data\cuda\）
-------------------------------------------
- cudart64_12.dll、cublas64_12.dll、cublasLt64_12.dll、cufft64_11.dll：NVIDIA CUDA Toolkit 的可散布元件，
  依 NVIDIA CUDA Toolkit End User License Agreement 散布：https://docs.nvidia.com/cuda/eula/index.html
- cudnn*64_9.dll：NVIDIA cuDNN，依 NVIDIA cuDNN Software License Agreement 隨應用程式散布：
  https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html
- 這些檔案未經修改，著作權屬於 NVIDIA Corporation。只在有 NVIDIA 顯示卡時使用；不需要時可以整個 cuda 資料夾刪掉（會改用 CPU）。

ONNX Runtime（_internal\onnxruntime\）
-------------------------------------
- MIT License，Copyright (c) Microsoft Corporation。授權與它使用的第三方元件見 onnxruntime-LICENSE.txt、onnxruntime-ThirdPartyNotices.txt。

Microsoft DirectML（只有 DirectML 版：_internal\onnxruntime\capi\DirectML.dll）
---------------------------------------------------------------------------
- onnxruntime-directml 套件附帶的 DirectML 可散布元件，未經修改，著作權屬於 Microsoft Corporation。
- 授權條款見 Microsoft.AI.DirectML 套件的授權：https://www.nuget.org/packages/Microsoft.AI.DirectML/ （License）。

FFmpeg（Qt Multimedia 附帶：_internal\PySide6\avcodec-63.dll、avformat-63.dll、avutil-61.dll、swresample-7.dll、swscale-10.dll）
--------------------------------------------------------------------------------------------------------------
- 單字本的朗讀用到 Qt 的語音、多媒體模組，PySide6 一併附帶 Qt 官方編譯的 FFmpeg 程式庫（LGPL 設定編譯，不含 GPL 元件）。
- 授權：GNU Lesser General Public License v2.1 或之後的版本（LGPL-2.1-or-later；可依 LGPL v3 散布，全文見 LGPL-3.0.txt）。
- 以動態連結方式使用、未經修改，可以替換成相容版本的 DLL。原始碼：https://ffmpeg.org/download.html ；
  Qt 的說明：https://doc.qt.io/qt-6/qtmultimedia-attribution-ffmpeg.html
- 本程式自己的程式碼仍是 Apache License 2.0；以上 LGPL 只適用於這些第三方程式庫本身。

其他 Python 套件
----------------
NumPy（BSD-3-Clause）、OpenCV（Apache-2.0）、Pillow（MIT-CMU）、mss（MIT）、dxcam（MIT）、RapidOCR（Apache-2.0）、
SudachiPy 與 SudachiDict（Apache-2.0）、pyclipper（MIT）、Shapely（BSD-3-Clause）。Python 本身：PSF License。
