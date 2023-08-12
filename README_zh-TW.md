# faster-whisper-transcribe

`faster-whisper-transcribe` 是一個針對從線上通話和會議錄製的視頻或音頻文件進行字幕轉錄的項目。它還可以通過將字幕合併到輸入視頻中來生成 .mkv 文件。

## 安裝

**GPU 執行需求:**
- 該項目需要 NVIDIA 的 cuBLAS 11.x 和 cuDNN 8.x 庫進行 GPU 執行。有關詳細的安裝說明，請參考 [CTranslate2 文檔](https://opennmt.net/CTranslate2/installation.html)。
- 或者，您可以使用以下命令安裝所需的庫：
```
conda install cudatoolkit=11.8 cudnn
```

1. 創建環境:

```bash
conda create -n faster-whisper python=3.8 -y
conda activate faster-whisper
```

2. 克隆此存儲庫:

```bash
git clone https://github.com/RxChi1d/faster-whisper-transcribe.git
cd faster-whisper-transcribe
```

3. 安裝所需的包:

```bash
conda install cudatoolkit=11.8 cudnn
pip install -r requirements.txt
```

## 使用方法

要使用 `faster-whisper-transcribe`，使用以下選項執行 `transcribe.py` 腳本：

```
python transcribe.py [選項]
```

### 選項

- `--input_path (-i)`: 輸入視頻或音頻文件的路徑。這個選項是必需的。
  
- `--output_path (-o)`: 輸出轉錄的路徑。如果未提供，輸出將以默認名稱保存在當前目錄中。

- `--merge_srt (-s)`: 將生成的 SRT（字幕）文件合併到輸入視頻中的標誌。默認為 `True`。

- `--beam_size (-b)`: 模型的 Beam 大小。默認為 `5`。

- `--model_size (-z)`: 指定要使用的模型大小。可選項包括 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 和 'large-v2'。默認為 'large-v2'。

- `--device_type (-d)`: 模型運行的設備類型。可選項包括 'auto', 'cuda', 和 'cpu'。默認為 'cuda'。

- `--device_index (-x)`: 模型的設備索引，當有多個 GPU 可用時非常有用。默認為 `0`。

- `--compute_type (-c)`: 指定模型的計算類型。選項包括 'default', 'float16', 'int8_float16', 和 'int8'。默認為 'float16'。

- `--cpu_threads (-t)`: 指定要使用的 CPU 線程數。默認為可用的 CPU 核心數。

- `--language (-l)`: 模型的語言。選項包括 "auto", "en", "zh", "ja", "fr", 和 "de"。默認為 'en'。

- `--word_level_timestamps (-w)`: 在輸出中啟用單詞級時間戳的標誌。默認為 `False`。

- `--vad_filter (-f)`: 啟用語音活動檢測 (VAD) 過濾器的標誌。這有助於過濾出非語音部分。默認為 `True`。

- `--vad_filter_min_silence_duration_ms (-g)`: VAD 過濾器的最小沉默持續時間（毫秒）。默認為 `50` 毫秒。

- `--verbose (-v)`: 啟用詳細日誌模式的標誌，這在執行期間提供詳細日誌。默認為 `True`。

- `--max_gap_ms_between_two_sentence (-mg)`: 允許兩句之間的最大間隔（毫秒）。默認為 `200` 毫秒。

- `--max_length (-ml)`: 句子的最大長度。默認為 `35` 個單詞。



## 參考
感謝以下項目：
1. [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper)
2. [lewangdev/autotranslate](https://github.com/lewangdev/autotranslate)


