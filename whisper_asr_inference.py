import torch
from transformers import pipeline
from datasets import load_dataset
import librosa.display
import numpy as np
from IPython.display import Audio
import json
import os
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# set directory containing audio files
AUDIO_PATH_TO_ROOT_DIR = 'trials/outputs/test-primock'
# set out directory for whisper transcripts
# transcript file name will include audio file name reference : "<audio-file-name>-stt.txt"
TRANSCIPT_PATH = 'trials/outputs/test-primock/transcripts/'
_in_test_mode = 0

# load pre-trained whisper checkpoint:
# checkpoints available : tiny, base, small, medium, large , etc.
# https://huggingface.co/openai/whisper-medium

# config for long form audio
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-medium", device=device,
    max_new_tokens=256,
    generate_kwargs={"task": "transcribe"},
    chunk_length_s=30,
    batch_size=8
)
# Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])

def whisper_transcribe(path_to_audio_files, transcript_out_dir):
    AUDIO_PATH_TO_ROOT_DIR = path_to_audio_files
    TRANSCIPT_PATH = transcript_out_dir
    if _in_test_mode == 1:
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        audio_file_list = [item['path'] for item in dataset[:]['audio']]
    else:
        audio_file_list = os.listdir(AUDIO_PATH_TO_ROOT_DIR)
        audio_file_list.sort()

    for a_file in tqdm(audio_file_list[:2]):
        if a_file.endswith('.wav') or a_file.endswith('.mp3') or a_file.endswith('.flac'):
            file = os.path.join(AUDIO_PATH_TO_ROOT_DIR,a_file)
            array, sampling_rate = librosa.load(file)
            # model inference
            transcript_text = pipe(array.copy())
            out_file_name = a_file.split(".")[0]
            with open(TRANSCIPT_PATH+out_file_name+"-stt.txt", "w") as f:
                json.dump(transcript_text['text'], f)
            f.close()
    return 'transciption completed. files in:'+transcript_out_dir




