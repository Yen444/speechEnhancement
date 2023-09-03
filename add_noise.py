from datasets import load_dataset, Dataset, Audio, load_from_disk
import numpy as np
import json
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import random


current_dir = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(current_dir, 'data'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'data/background_noise'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'data/white_noise'), exist_ok=True)

os.makedirs(os.path.join(current_dir, 'data/background_noise/train'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'data/background_noise/test'), exist_ok=True)

os.makedirs(os.path.join(current_dir, 'data/white_noise/train'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'data/white_noise/test'), exist_ok=True)

np.random.seed(14)

def add_white_noise(input_signal, snr_dB):
   snr_linear = 10**(snr_dB/10)
   signal_power = np.mean(input_signal**2)
   std_noise = np.sqrt(signal_power/snr_linear)
   awgn_noise = np.random.normal(0, std_noise, len(input_signal))
   noisy_signal = input_signal + awgn_noise
   return noisy_signal

def add_background_noise(background_sound, input_signal, snr_db):
   # normalize signal power to 1 (0dB) and amplify by desired power afterwards
   """signal_power = np.mean(input_signal**2)
   noise_signal = background_sound / np.sqrt(signal_power)
   noise_amp_linear = 10**(-snr_db/20)
   noise_signal = noise_signal*noise_amp_linear"""
   
   signal_power = np.mean(input_signal**2)
   snr_linear = 10**(snr_db/10)
   # sdt of background noise
   std_noise_real = np.sqrt(np.mean(background_sound**2))
   # sdt of require background noise basing on snr
   std_noise_required = np.sqrt(signal_power/snr_linear)
   noise_signal = background_sound*(std_noise_required/std_noise_real)
   
   # pad or crop background sound and add to input signal
   if input_signal.shape[0] > noise_signal.shape[0]:
      padded_signal = np.zeros_like(input_signal)
      padded_signal[:noise_signal.shape[0]] = noise_signal
      corrupted_signal = input_signal + padded_signal
   else:
      noise_signal = noise_signal[:input_signal.shape[0]]
      corrupted_signal = input_signal + noise_signal
   return corrupted_signal 

def main():
   # Background noise files
   wave_files = glob.glob('/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/tran/asr_timit/ESC-50-master/audio/*.wav')
   # to be added files
   test_dataset_files = glob.glob('/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/data/timit/test/*/*/*.wav')
   train_dataset_files = glob.glob('/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/data/timit/train/*/*/*.wav')
   # adding white noise
   noise_power_list = [-10,0,10,20]
   for noise_power in noise_power_list:
      for file in test_dataset_files:
         #print(file)
         input_signal, sr = librosa.load(file)
         noisy_signal = add_white_noise(input_signal, noise_power)
         out_dir = os.path.join(current_dir, 'data/white_noise/test', str(noise_power))
         if not os.path.exists(out_dir):
            os.makedirs(out_dir)
         filename = os.path.join(out_dir, str(noise_power) + os.path.basename(file))
         sf.write(filename, noisy_signal, sr)
      for file in train_dataset_files:
         input_signal, sr = librosa.load(file)
         noisy_signal = add_white_noise(input_signal, noise_power)
         out_dir = os.path.join(current_dir, 'data/white_noise/train', str(noise_power))
         if not os.path.exists(out_dir):
            os.makedirs(out_dir)
         filename = os.path.join(out_dir, str(noise_power) + os.path.basename(file))
         sf.write(filename, noisy_signal, sr)
   # adding background noise
   # choosing randomly background noise
   
   for noise_power in noise_power_list:
      for file in test_dataset_files:
         input_signal, sr = librosa.load(file)
         idx = random.randint(0, len(wave_files)-1)
         background_signal, _ = librosa.load(wave_files[idx])
         noisy_signal = add_background_noise(background_signal, input_signal, noise_power)
         out_dir = os.path.join(current_dir, 'data/background_noise/test', str(noise_power))
         if not os.path.exists(out_dir):
            os.makedirs(out_dir)
         filename = os.path.join(out_dir, str(noise_power) + os.path.basename(file))
         sf.write(filename, noisy_signal, sr)
      for file in train_dataset_files:
         input_signal, sr = librosa.load(file)
         idx = random.randint(0, len(wave_files)-1)
         background_signal, _ = librosa.load(wave_files[idx])
         noisy_signal = add_background_noise(background_signal, input_signal, noise_power)
         out_dir = os.path.join(current_dir, 'data/background_noise/train', str(noise_power))
         if not os.path.exists(out_dir):
            os.makedirs(out_dir)
         filename = os.path.join(out_dir, str(noise_power) + os.path.basename(file))
         sf.write(filename, noisy_signal, sr)
      

if __name__ == '__main__':
   main()
