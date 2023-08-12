import os
from faster_whisper import WhisperModel
import subprocess
import argparse
import ast


def str2bool(v):
    """Converts string representation of bool into bool type."""
    return ast.literal_eval(str(v).capitalize())


def classify_file_by_ffmpeg(filepath):
    try:
        # Use ffmpeg to get file information
        result = subprocess.run(["ffmpeg", "-i", filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # If ffmpeg processes it without errors, we classify based on the output
        output = result.stderr.lower()
        if "video:" in output:
            return "video"
        elif "audio:" in output:
            return "audio"
        else:
            return "other"
    except Exception as e:
        return "other"
    

# Extract the audio from mkv file by ffmpeg
def extract_audio(video_path, audio_path=None):
    # If audio_path is None, the audio will be saved in the same folder as the video and the extension is .wav
    if audio_path is None:
        audio_path = os.path.splitext(video_path)[0] + '.wav'

    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "-loglevel", "panic", audio_path])
    
    return audio_path


# Convert seconds to hours, minutes, seconds, and milliseconds
def seconds_to_time_format(s):
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    seconds = s // 1
    milliseconds = round((s % 1) * 1000)

    # Return the formatted string
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


# Merge words/segments to sentences
def merge_fragments(fragments, gap_ms=0.2, max_length=None):
    new_fragments = []
    new_fragment = {}
    length = len(fragments)
    for i, fragment in enumerate(fragments):
        start = fragment['start']
        end = fragment['end']
        text = fragment['text']

        if new_fragment.get('start', None) is None:
            new_fragment['start'] = start
        if new_fragment.get('end', None) is None:
            new_fragment['end'] = end
        if new_fragment.get('text', None) is None:
            new_fragment['text'] = ""

        # Check if the combined text exceeds the max_length
        combined_text_length = len(new_fragment['text']) + len(text)
        if max_length is not None and combined_text_length > max_length:
            new_fragments.append(new_fragment)
            new_fragment = dict(start=start, end=end, text=text)
            continue

        if start - new_fragment['end'] > gap_ms:
            new_fragments.append(new_fragment)
            new_fragment = dict(start=start, end=end, text=text)
            continue

        new_fragment['end'] = end

        delimiter = '' if text.startswith('-') else ' '
        new_fragment['text'] = f"{new_fragment['text']}{delimiter}{text.lstrip()}"

        # End of a sentence when symbols found: [.?]
        if text.endswith('.') or text.endswith('?') or i == length-1:
            new_fragments.append(new_fragment)
            new_fragment = {}
    return new_fragments


# Detect the subtitle from mkv file by whisper
def detect_subtitle(audio_path, beam_size=5, model_size='large-v2', device_type="cuda", device_index=1, compute_type="float16", cpu_threads=os.cpu_count(), language="en", word_level_timestamps=True, vad_filter=False, vad_filter_min_silence_duration_ms=50, verbose=True):
    # Load model
    model = WhisperModel(model_size, device=device_type, device_index=device_index,
                         cpu_threads=cpu_threads, compute_type=compute_type)    

    segments, info = model.transcribe(audio_path, beam_size=beam_size,
                                      language=None if language == "auto" else language,
                                      word_timestamps=word_level_timestamps,
                                      vad_filter=vad_filter,
                                      vad_parameters=dict(min_silence_duration_ms=vad_filter_min_silence_duration_ms))

    fragments = []

    for segment in segments:
        # print(f"[{seconds_to_time_format(segment.start)} --> {seconds_to_time_format(segment.end)}] {segment.text}")
        if word_level_timestamps:
            for word in segment.words:
                if verbose:
                    ts_start = seconds_to_time_format(word.start)
                    ts_end = seconds_to_time_format(word.end)
                    print(f"[{ts_start} --> {ts_end}] {word.word}")
                fragments.append(
                    dict(start=word.start, end=word.end, text=word.word))
        else:
            if verbose:
                ts_start = seconds_to_time_format(segment.start)
                ts_end = seconds_to_time_format(segment.end)
                print(f"[{ts_start} --> {ts_end}] {segment.text}")
            fragments.append(
                dict(start=segment.start, end=segment.end, text=segment.text))

    return fragments


# Save as srt
def export_srt(new_fragments, srt_transcript_file_name):
    with open(srt_transcript_file_name, 'w') as f:
        for sentence_idx, fragment in enumerate(new_fragments):
            ts_start = seconds_to_time_format(fragment['start'])
            ts_end = seconds_to_time_format(fragment['end'])
            text = fragment['text']
            # print(f"[{ts_start} --> {ts_end}] {text}")
            f.write(f"{sentence_idx + 1}\n")
            f.write(f"{ts_start} --> {ts_end}\n")
            f.write(f"{text.strip()}\n\n")


# Merge the mkv and srt file to a new mkv file
def merge_video_subtitle(video_path, srt_path, output_path=None, lang='English'):
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '_merged.mkv'

    lang_map = {
        'en': ('English', 'eng'),
        'zh': ('zh-TW', 'cht'),
        'ja': ('Japanese', 'jpn'),
        'fr': ('French', 'fre'),
        'de': ('German', 'ger')
    }
    
    ffmpeg_language, ffmpeg_title = lang_map[lang]
    
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", srt_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "srt",
        "-map", "0",
        "-map", "1",
        "-metadata:s:s:0", f"language={ffmpeg_language}",
        "-metadata:s:s:0", f"title={ffmpeg_title}",
        "-disposition:s:s:0", "default",
        output_path
    ]

    subprocess.run(command)


if __name__ == '__main__':
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Whisper WebUI Parameters")

    # Add arguments
    parser.add_argument('--input_path', '-i', help='Path to the input video file or audio file', required=True)
    parser.add_argument('--output_path', '-o', default=None, help='Path for the output')
    parser.add_argument('--merge_srt', '-s', type=str2bool, default=True, help='Flag to merge SRT with video')
    parser.add_argument('--beam_size', '-b', type=int, default=5, help='Beam size for the model')
    parser.add_argument('--model_size', '-z', choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'], default='large-v2', help='Size of the model to use')
    parser.add_argument('--device_type', '-d', choices=['auto', 'cuda', 'cpu'], default='cuda', help='Device type for the model')
    parser.add_argument('--device_index', '-x', type=int, default=1, help='Device index for the model')
    parser.add_argument('--compute_type', '-c', choices=['default', 'float16', 'int8_float16', 'int8'], default='float16', help='Compute type for the model')
    parser.add_argument('--cpu_threads', '-t', type=int, default=os.cpu_count(), help='Number of CPU threads')
    parser.add_argument('--language', '-l', choices=["auto", "en", "zh", "ja", "fr", "de"], default='en', help='Language for the model')
    parser.add_argument('--word_level_timestamps', '-w', type=str2bool, default=False, help='Flag for word level timestamps')
    parser.add_argument('--vad_filter', '-f', type=str2bool, default=True, help='Flag for VAD filter')
    parser.add_argument('--vad_filter_min_silence_duration_ms', '-g', type=int, default=50, help='Minimum silence duration in ms for VAD filter')
    parser.add_argument('--verbose', '-v', type=str2bool, default=True, help='Flag for verbose mode')
    parser.add_argument('--max_gap_ms_between_two_sentence', '-mg', type=int, default=200, help='Maximum gap in ms between two sentences')
    parser.add_argument('--max_length', '-ml', type=int, default=35, help='Maximum length of a sentence')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Assign argparse values to the original variable names
    input_path = args.input_path
    output_path = args.output_path
    merge_srt = args.merge_srt
    beam_size = args.beam_size
    model_size = args.model_size
    device_type = args.device_type
    device_index = args.device_index
    compute_type = args.compute_type
    cpu_threads = args.cpu_threads
    language = args.language
    word_level_timestamps = args.word_level_timestamps
    vad_filter = args.vad_filter
    vad_filter_min_silence_duration_ms = args.vad_filter_min_silence_duration_ms
    verbose = args.verbose
    max_gap_ms_between_two_sentence = args.max_gap_ms_between_two_sentence
    max_length = args.max_length

    # detect the file type is video or audio
    file_type = classify_file_by_ffmpeg(input_path)
    
    if file_type == "video":
        audio_path = extract_audio(input_path)
    elif file_type == "audio":
        audio_path = input_path
    else:
        raise ValueError("Unsupported file type")
    
    
    fragments = detect_subtitle(audio_path, beam_size=beam_size, model_size=model_size,
                                device_type=device_type, device_index=device_index, 
                                compute_type=compute_type, cpu_threads=cpu_threads, 
                                language=language, word_level_timestamps=word_level_timestamps, 
                                vad_filter=vad_filter, 
                                vad_filter_min_silence_duration_ms=vad_filter_min_silence_duration_ms,
                                verbose=verbose)

    new_fragments = merge_fragments(
        fragments, max_gap_ms_between_two_sentence/1000.0, max_length=max_length)

    # Get the srt path
    srt_path = os.path.splitext(input_path)[0] + ".srt"

    # Export the srt file
    export_srt(new_fragments, srt_path)

    if file_type == "video":
        # remove the audio file
        os.remove(audio_path)
        
        if merge_srt:
            # Merge the video and srt file
            merge_video_subtitle(input_path, srt_path, output_path=output_path, lang=language)
