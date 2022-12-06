#%%
import wave
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%%
samplerate = 44100
image_path = 'image2.jpeg'
img = cv2.imread(image_path)[..., ::-1].copy()

img = cv2.GaussianBlur(img, (9, 9), 10)
plt.imshow(img)
img = cv2.resize(img, (1000, 100))
h, w = img.shape[:2]
h, w

#%%
total_duration = 5
row_duration = total_duration / w
row_duration
#%%
row_rate = int(row_duration * samplerate)
row_rate
#%%
audio = np.zeros(row_rate * w, 'f')
frequency = np.ones((h, row_rate), 'f')
t = np.linspace(0, 1, row_rate)
frequency_max = 0.1
for i in range(len(frequency)):
    frequency[i] = np.sin(t * 2 * np.pi * frequency_max * (i + 1) / len(frequency))
plt.imshow(frequency)
plt.show()
#%%
for row_id in range(w):
    brightness = img[:, row_id].mean(-1)
    value = (brightness * frequency.T).mean(-1)

    audio[row_id * row_rate: (row_id + 1) * row_rate] = value
#%%
kernel = np.ones(100)
kernel /= kernel.sum()
audio = np.convolve(audio, kernel)

volume = 0.1  # range [0.0, 1.0]
fs = 44100  # sampling rate, Hz, must be integer
audio_ = (audio * volume  * (2 ** 15 - 1)).astype("int16")

print(audio_)
with wave.open("sound1.wav", "w") as f:
    # 2 Channels.
    f.setnchannels(1)
    # 2 bytes per sample.
    f.setsampwidth(1)
    f.setframerate(samplerate)
    f.writeframes(audio_.tobytes())


# %%

from sound_to_midi.monophonic import wave_to_midi
midi = wave_to_midi(audio * volume, samplerate)

# %%
