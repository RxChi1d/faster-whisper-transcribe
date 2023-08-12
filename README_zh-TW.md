
# Whisper 語音轉錄專案

此存儲庫包含使用 Whisper ASR 系統進行音頻轉錄的工具和腳本。

## 內容

- `faster_whisper_transcribe.py`: 使用 Whisper 進行音頻轉錄的主腳本。
- `requirements.txt`: 本項目所需的 Python 軟件包列表。

## 依賴項

- GPU 執行需要在系統上安裝 NVIDIA 庫 cuBLAS 11.x 和 cuDNN 8.x。請參閱 [CTranslate2 文檔](https://opennmt.net/CTranslate2/installation.html)。或者，您可以使用 `conda install cudatoolkit=11.8 cudnn` 來安裝所需的庫。您也可以使用以下命令安裝所需的庫：
```
conda install cudatoolkit=11.8 cudnn
```

## 安裝

1. 克隆此存儲庫：

```bash
git clone https://github.com/RxChi1d/faster-whisper-transcribe.git
cd faster-whisper-transcribe
```

2. 安裝所需的軟件包：

```bash
pip install -r requirements.txt
```

## 使用方法

要使用 `faster_whisper_transcribe.py` 腳本進行音頻轉錄，運行：

```bash
python faster_whisper_transcribe.py <選項>
```


### Options for `faster_whisper_transcribe.py`

- `--video_path`: Path to the .mkv video file.
- `--output_path`: Path for the output. Default: None.
- `--merge_srt`: Flag to merge SRT with video. Default: True.
- `--beam_size`: Beam size for the model. Default: 5.
- `--model_size`: Size of the model to use. Choices include: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'. Default: 'large-v2'.
- `--device_type`: Device type for the model. Choices: 'auto', 'cuda', 'cpu'. Default: 'cuda'.
- `--device_index`: Device index for the model. Default: 1.
- `--compute_type`: Compute type for the model. Choices: 'default', 'float16', 'int8_float16', 'int8'. Default: 'float16'.
- `--cpu_threads`: Number of CPU threads. Default: number of system's CPU threads.
- `--language`: Language for the model. Choices: 'auto', 'en', 'zh', 'ja', 'fr', 'de'. Default: 'en'.
- `--word_level_timestamps`: Flag for word level timestamps. Default: False.
- `--vad_filter`: Flag for VAD filter. Default: True.
- `--vad_filter_min_silence_duration_ms`: Minimum silence duration in ms for VAD filter. Default: 50.
- `--verbose`: Flag for verbose mode. Default: True.
- `--max_gap_ms_between_two_sentence`: Maximum gap in ms between two sentences. Default: 200.


## 參考資料

- 此項目使用 [faster-whisper](https://github.com/guillaumekln/faster-whisper) 存儲庫進行高效的 Whisper ASR 執行。
- 對於自動翻譯轉錄，此項目引用了 [autotranslate](https://github.com/lewangdev/autotranslate) 存儲庫。

