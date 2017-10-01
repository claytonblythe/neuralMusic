import librosa
import os
import random
from tqdm import *

## Utilties for my CNN genre classification project
def save_random_clips(base_path, save_path, snip_length):
    filenames = iter(f for f in os.listdir(base_path))
    for filename in tqdm(filenames):
        y, sr = librosa.load(base_path + filename, mono=True, sr=None)
        song_duration = librosa.core.get_duration(y, sr)
        random_offset = random.uniform(0, song_duration - snip_length)
        y, sr, = librosa.load(base_path + filename, mono=True, offset=random_offset, duration=float(snip_length), sr=None)
        librosa.output.write_wav(y=y, sr=sr, path=save_path + filename.strip('.mp3') + '.wav')
