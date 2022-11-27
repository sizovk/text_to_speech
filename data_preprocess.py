import numpy as np
import os
import librosa
import pandas as pd
import pyworld as pw
import torch
from torchaudio.transforms import Spectrogram
from tqdm import tqdm

N_FFT = 1024
HOP_LENGTH = 256

stft = Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH)

metadata = pd.read_csv("./data/LJSpeech-1.1/metadata.csv", sep='|', header=None)
wavs = [f"./data/LJSpeech-1.1/wavs/{wav_path}.wav" for wav_path in metadata.iloc[:,0].to_list()]

os.makedirs("./data/pitch", exist_ok=True)
os.makedirs("./data/energy", exist_ok=True)

pitch_mean = []
pitch_std = []
energy_mean = []
energy_std = []

for i in tqdm(range(len(metadata))):
    wav, sr = librosa.load(wavs[i])

    pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=HOP_LENGTH / sr * 1000)
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)
    pitch_mean.append(np.mean(pitch))
    pitch_std.append(np.std(pitch))

    energy = torch.norm(stft(torch.from_numpy(wav)), dim=0).numpy()
    energy_mean.append(np.mean(energy))
    energy_std.append(np.std(energy))

    np.save(os.path.join("./data/pitch", "ljspeech-pitch-%05d.npy" % (i+1)), pitch)
    np.save(os.path.join("./data/energy", "ljspeech-energy-%05d.npy" % (i+1)), energy)
pitch_mean = np.mean(pitch_mean)
pitch_std = np.mean(pitch_std)
energy_mean = np.mean(energy_mean)
energy_std = np.mean(energy_std)

# normalization pitch and energy
for i in tqdm(range(len(metadata))):
    pitch = np.load(os.path.join("./data/pitch", "ljspeech-pitch-%05d.npy" % (i+1)))
    energy = np.load(os.path.join("./data/energy", "ljspeech-energy-%05d.npy" % (i+1)))
    pitch = (pitch - pitch_mean) / pitch_std
    energy = (energy - energy_mean) / energy_std
    np.save(os.path.join("./data/pitch", "ljspeech-pitch-%05d.npy" % (i+1)), pitch)
    np.save(os.path.join("./data/energy", "ljspeech-energy-%05d.npy" % (i+1)), energy)
