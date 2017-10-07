import librosa
import os
import random
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
## Author: Clayton Blythe <claytondblythe@gmail.com>
## Utilties for my CNN genre classification project

# Save randomized short clips from the .mp3 files in the base_path directory
def save_random_clips(base_path, save_path, snip_length):
    directories = iter(f for f in os.listdir(base_path))
    for directory in directories:
        filenames = iter(f for f in os.listdir(base_path + directory + '/'))
        for filename in filenames:
            y, sr = librosa.load(base_path + directory + '/' + filename, mono=True,  sr=None)
            song_duration = librosa.core.get_duration(y, sr)
            random_offset = random.uniform(0,song_duration - 5.96)
            y, sr = librosa.load(base_path + directory + '/' + filename, mono=True,  offset=random_offset, duration= 5.94, sr=None)
            librosa.output.write_wav(y=y, sr=sr, path=save_path + filename[:-4] + '.wav')

# Save melspectrogram tensors for every file in some base_path directory to some save_path
## Note: this creates 512 bins (128*4) for the frequency component
def save_spectrogram_tensors(base_path, save_path):
    filenames = iter(f for f in os.listdir(base_path))
    for filename in tqdm(filenames):
        y, sr = librosa.load(base_path + filename, mono=True, sr=None)
        S = librosa.feature.melspectrogram(y=y, n_mels=128*4, fmax=8000)
        S.tofile(save_path + filename[:-4])

# Save random samples into spectrogram figures.. this takes alot of storage so be warned
# def save_spectrograms(base_path, save_path):
    # filenames = iter(f for f in os.listdir(base_path))
    # for filename in tqdm(filenames):
        # y, sr = librosa.load(base_path + filename, mono=True, sr=None)
        # S = librosa.feature.melspectrogram(y=y, n_mels=128*4, fmax=8000)
        # plt.figure(figsize=(10,8))
        # librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                # y_axis='mel', fmax=8000,
                # x_axis='time')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel Spectrogram')
        # plt.tight_layout()
        # plt.savefig(save_path + filename[:-4] + '.png')
