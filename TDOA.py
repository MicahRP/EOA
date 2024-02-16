import numpy as np

import matplotlib.pyplot as plt

def xcorr(x, y):
    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(y) + 1, len(x))
    return lags, corr


# frequency is the number of times a wave repeats a second
frequency = 1000
noisy_freq = 100
num_samples = 2750
# The sampling rate of the analog to digital convert
sampling_rate = 48000.0


# sine_wave = [np.sin(2 * np.pi * frequency * x1 / sampling_rate) for x1 in range(num_samples)]
# plt.subplot(3,1,1)
# plt.title("Original sine wave")
# # Need to add empty space, else everything looks scrunched up!
# plt.subplots_adjust(hspace=.5)
# plt.plot(sine_wave[:1000])
# plt.subplot(3,1,2)
# plt.title("Sin wave shifted 34 units")
# plt.plot(sine_wave[34:])
# plt.subplot(3,1,3)
# plt.title("Cross correlation")
# corl_data = np.correlate(sine_wave[:1000],sine_wave[34:], mode='full')
# lags = np.arange(-len(sine_wave[34:]) + 1, len(sine_wave[:1000]))
# plt.plot(lags, corl_data)
# plt.show()
# print(lags[np.argmax(corl_data)])

#Create the sine wave and noise
sine_wave = [np.sin(2 * np.pi * frequency * x1 / sampling_rate) for x1 in range(num_samples)]
sine_noise = [(10 * ((abs(abs(x1 - 1250) - 1250))/1250)) * np.sin(2 * np.pi * noisy_freq * x1/  sampling_rate) for x1 in range(num_samples)]
#Convert them to numpy arrays
sine_wave = np.array(sine_wave)
sine_noise = np.array(sine_noise)
# Add them to create a noisy signal
combined_signal = sine_wave + sine_noise



# plt.subplot(3,1,1)
# plt.title("Original sine wave")
# # Need to add empty space, else everything looks scrunched up!
# plt.subplots_adjust(hspace=.5)
# plt.plot(sine_wave[:2500])
# plt.subplot(3,1,2)
# plt.title("Noisy wave")
# plt.plot(sine_noise[:2500])
# plt.subplot(3,1,3)
# plt.title("Original + Noise")
# plt.plot(combined_signal[:2500])
# plt.show()

# plt.subplot(3,1,1)
# plt.title("Original sine wave")
# plt.subplots_adjust(hspace=.5)
# plt.plot(sine_wave[250:])
# plt.subplot(3,1,2)
# plt.title("Noisy wave")
# plt.plot(sine_noise[250:])
# plt.subplot(3,1,3)
# plt.title("Original + Noise")
# plt.plot(combined_signal[250:])
# plt.show()

plt.subplot(2,1,1)
plt.title("Noise Signal")
plt.xlabel("Time steps")
plt.ylabel("Amplitude")
plt.subplots_adjust(hspace=.5)
plt.plot(combined_signal[:2500])
plt.subplot(2,1,2)
plt.title("Noise delayed 250 steps")
plt.xlabel("Time steps")
plt.ylabel("Amplitude")
plt.plot(combined_signal[250:])
plt.show()

[lags, conv_data] = xcorr(combined_signal[:2500],combined_signal[250:])
plt.plot(lags, conv_data)
plt.xlim(-1500, 1500)
plt.axvline(x = 250, color = 'r',)
plt.show()

print(lags[np.argmax(conv_data)])