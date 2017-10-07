import librosa
import os
import random
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import utils
import torch.utils.data as data
## Author: Clayton Blythe <claytondblythe@gmail.com>
## Utilties for my CNN genre classification project

# Save genre master list for referencing
def save_genre_master_list(base_path, save_path):
    tracks = utils.load(base_path + 'tracks.csv')
    track_genres_df = tracks['track']['genre_top']
    track_genres_df.index = [str(item).zfill(6) for item in track_genres_df.index]
    track_genres_df.to_csv(save_path + 'genre_master_list.csv', header=False)

# Save randomized short clips from the .mp3 files in the base_path directory
def save_random_clips(base_path, save_path, snip_length):
    directories = [f for f in os.listdir(base_path)]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for directory in tqdm(directories):
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
    filenames = [f for f in os.listdir(base_path)]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for filename in tqdm(filenames):
        y, sr = librosa.load(base_path + filename, mono=True, sr=None)
        S = librosa.feature.melspectrogram(y=y, n_mels=128*4, fmax=8000)
        S.tofile(save_path + filename[:-4])

# Save csv for spectrogram_tensor genre labels
def save_tensor_labels(base_path, save_path):
    tensor_files = os.listdir(base_path)
    df = pd.read_csv(save_path + 'genre_master_list.csv', dtype=object, header=None, names=['tensor_name', 'genre_top'])
    new_df = df[df['tensor_name'].isin(tensor_files)]
    new_df.to_csv(save_path + 'tensor_genres.csv')

# FMA Dataset class
class FmaDataset(data.Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.
    Arguments:
        A CSV file path
        Path to torch tensors folder
        Torch Tensor transforms to do
    """

    def __init__(self, csv_path, tensor_path, transform=None):
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['tensor_name'].apply(lambda x: os.path.isfile(tensor_path + x)).all(), \
"Some tensors referenced in the CSV file were not found"

        self.mlb = MultiLabelBinarizer()
        self.tensor_path = tensor_path
        self.transform = transform

        self.X_train = tmp_df['tensor_name']
        self.y_train = self.mlb.fit_transform(tmp_df['genre_top']).astype(np.float32)

    def __getitem__(self, index):
        tensor = np.fromfile(self.tensor_path + self.X_train[index])
        if self.transform is not None:
            tensor = self.transform(tensor)
        label = torch.from_numpy(self.y_train[index])
        return tensor, label

    def __len__(self):
        return len(self.X_train.index)

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
