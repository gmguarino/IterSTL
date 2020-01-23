import numpy as np
import matplotlib.pyplot as plt
from RobuSTL import bilateral_window, bilateral_filtering, estimate_period, \
    extract_seasonality, remove_norm_mean, seasonal_difference, extract_trend

np.random.seed(4)
signal = []

freq = 0.8
print("T = ", 1 / freq)
fs = 100
w = 2 * np.pi * freq
trend = 0.002
for i in range(500):
    switch = np.random.rand()
    if switch < 0.98:
        signal.append(trend * i + np.sin(w * i / fs) + np.random.rand() * 0.05)
    else:
        signal.append(trend * i + np.sin(w * i / fs) + np.random.rand() * 3)
signal = np.asarray(signal)
plt.figure()
plt.plot(signal)
plt.figure()
H = 3
K = 2
delta_d = 1
delta_i = 1
t = 146
j = np.linspace(t - H, t + H, 2 * H + 1, dtype=int)
window, yj = bilateral_window(signal, t, j, delta_d, delta_i)
plt.plot(yj)
plt.plot(window)

plt.figure()
filtered = bilateral_filtering(signal, H, delta_d, delta_i)
plt.plot(filtered)

T = estimate_period(signal, fs=fs)

print(seasonal_difference(signal, T).shape, signal.size - T*fs)


new_signal, trend = extract_trend(filtered, T, lambda_1=1.0, lambda_2=0.5)
plt.figure()
plt.plot(trend)
plt.plot(new_signal)
plt.show()
