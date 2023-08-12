
# Whisper Transcription Project

This repository contains tools and scripts for audio transcription using the Whisper ASR system.

## Contents

- `faster_whisper_transcribe.py`: Main script for transcribing audio using Whisper.
- `requirements.txt`: List of required Python packages for this project.

## Dependencies

- GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html). Alternatively, you can use `conda install cudatoolkit=11.8 cudnn` to install the required libraries. You can also install the required libraries using the following command:
```
conda install cudatoolkit=11.8 cudnn
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/RxChi1d/faster-whisper-transcribe.git
cd faster-whisper-transcribe
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To transcribe audio using the `faster_whisper_transcribe.py` script, run:

```bash
python faster_whisper_transcribe.py <OPTIONS>
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


## References

- This project uses the [faster-whisper](https://github.com/guillaumekln/faster-whisper) repository for efficient Whisper ASR execution.
- For automatic translation of transcriptions, this project references the [autotranslate](https://github.com/lewangdev/autotranslate) repository.

