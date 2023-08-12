
# faster-whisper-transcribe

`faster-whisper-transcribe` is a project focus on transcribing the subtitile from video or audio file which is recorded from online calls and meetings. It is also able to produce the .mkv file by merging the input video with subtitle.

## Installation

**GPU Execution Requirements:**
- This project requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x for GPU execution. For detailed installation instructions, please refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html).
- Alternatively, you can use the following command to install the required libraries:
```
conda install cudatoolkit=11.8 cudnn
```

1. Create the environment:

```bash
conda create -n faster-whisper python=3.8 -y
conda activate faster-whisper
```

2. Clone this repository:

```bash
git clone https://github.com/RxChi1d/faster-whisper-transcribe.git
cd faster-whisper-transcribe
```

3. Install the required packages:

```bash
conda install cudatoolkit=11.8 cudnn
pip install -r requirements.txt
```

## Usage

To use `faster-whisper-transcribe`, execute the `transcribe.py` script with the following options:

```
python transcribe.py [options]
```

### Options

- `--input_path (-i)`: Path to the input video or audio file. This option is required.
  
- `--output_path (-o)`: Path for the output transcription. If not provided, the output will be saved with a default name in the current directory.

- `--merge_srt (-s)`: Flag to merge the generated SRT (subtitle) file with the input video. Default is `True`.

- `--beam_size (-b)`: Beam size for the model. Default is `5`.

- `--model_size (-z)`: Specifies the size of the model to use. Choices include 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', and 'large-v2'. Default is 'large-v2'.

- `--device_type (-d)`: Device type on which the model runs. Choices include 'auto', 'cuda', and 'cpu'. Default is 'cuda'.

- `--device_index (-x)`: Device index for the model, useful when multiple GPUs are available. Default is `0`.

- `--compute_type (-c)`: Specifies the compute type for the model. Choices include 'default', 'float16', 'int8_float16', and 'int8'. Default is 'float16'.

- `--cpu_threads (-t)`: Specifies the number of CPU threads to be used. Default is the number of CPU cores available.

- `--language (-l)`: Language for the model. Choices include "auto", "en", "zh", "ja", "fr", and "de". Default is 'en'.

- `--word_level_timestamps (-w)`: Flag to enable word level timestamps in the output. Default is `False`.

- `--vad_filter (-f)`: Flag to enable Voice Activity Detection (VAD) filter. This helps in filtering out non-speech segments. Default is `True`.

- `--vad_filter_min_silence_duration_ms (-g)`: Minimum silence duration (in milliseconds) for the VAD filter. Default is `50` ms.

- `--verbose (-v)`: Flag to enable verbose mode, which provides detailed logs during execution. Default is `True`.

- `--max_gap_ms_between_two_sentence (-mg)`: Specifies the maximum gap (in milliseconds) allowed between two sentences. Default is `200` ms.

- `--max_length (-ml)`: Specifies the maximum length of a sentence. Default is `35` words.



## References
Thanks for the following projects:  
1. [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper)
2. [lewangdev/autotranslate](https://github.com/lewangdev/autotranslate)

