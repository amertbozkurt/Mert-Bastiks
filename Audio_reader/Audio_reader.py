from scipy.io import wavfile
import numpy as np
import scipy.fft as fft
from matplotlib import pyplot as plt

# +
from pydub import AudioSegment

# files                                                                       
src = "destiny.mp3"
dst = "destiny.wav"

# convert wav to mp3                                                            
audSeg = AudioSegment.from_mp3(src)
audSeg.export(dst, format="wav")

# +
samplerate, data = wavfile.read('destiny.wav')
n_samples = np.shape(data)[0]
step = 1/samplerate

import numpy as np
print(np.shape(data))
print(samplerate)


# -

## FFT function
def audio_fft(data, step):
    n_samples=np.shape(data)[0]
    channel_fft=fft.fft(data,axis=0)
    freqs = fft.fftfreq(n_samples,step)

    power_spectrum = np.abs(channel_fft)**2 / n_samples
    
    return freqs, power_spectrum


# +
# full FFT
freqs, power_spectrum = audio_fft(data,step)
fig, ax = plt.subplots()

ax.plot(freqs, power_spectrum)
ax.set_xlim([-200,200])

# +
#buffering
buffering_window=int(0.1*samplerate)
data_tail=n_samples%buffering_window
print(data_tail)
cropped_data=np.delete(data,np.s_[n_samples-data_tail:n_samples],axis=0) #Crop data for easy buffering
print(np.shape(cropped_data))
channel_array=cropped_data[:,0].reshape(buffering_window,-1) #Looking at one channel
print(np.shape(channel_array))

freqs, power_spectrum = audio_fft(channel_array,step)

max_freq=freqs[np.argmax(power_spectrum,axis=0)]

plt.plot(max_freq)

fig, ax = plt.subplots()

ax.plot(freqs, power_spectrum[:,0:3])
ax.set_xlim([-200,200])
# -

12899250/22050


