import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Load the WAV file
file_path = "/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/tran/SpeechEnhancement/data/background_noise/test/-10/-10sa1.wav"
sample_rate, audio_data = wavfile.read(file_path)

# Calculate the spectrogram
frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate)

# Plot the spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.title('Spectrogram of WAV File')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(0, 5000)  # Adjust the frequency range as needed
plt.xlim(0, len(audio_data) / sample_rate)  # Adjust the time range as needed
plt.show()
plt.savefig('/mount/arbeitsdaten/studenten1/team-lab-phonetics/2023/student_directories/tran/SpeechEnhancement/spectrograms/spec')

def plot_spectrogram():
   pass