import numpy as np
import matplotlib.pyplot as plt
from RobuSTL import bilateral_window, bilateral_filtering, estimate_period, \
    extract_seasonality, adjust_season, seasonal_difference, extract_trend, get_remainder

np.random.seed(4)
signal = []
print("starting")
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
H = 3
K = 2
delta_d = 1
delta_i = 1
# t = 146
# j = np.linspace(t - H, t + H, 2 * H + 1, dtype=int)
# window, yj = bilateral_window(signal, t, j, delta_d, delta_i)
# plt.plot(yj)
# plt.plot(window)
#
# plt.figure()
# filtered = bilateral_filtering(signal, H, delta_d, delta_i)
# plt.plot(filtered)
#
T = estimate_period(signal, fs=fs)
#
# print(seasonal_difference(signal, T).shape, signal.size - T * fs)
#
new_signal, trend = extract_trend(signal, T, lambda_1=1.0, lambda_2=0.5)
# plt.figure()
# plt.plot(trend)
# plt.plot(new_signal)
season = extract_seasonality(new_signal, fs, H, K, delta_d, delta_i)
plt.figure()
plt.plot(season)
plt.plot(new_signal)

filter_params = {"H": 3, "delta_d": 1, "delta_i": 1}
season_params = {"H": 3, "K": 2, "delta_d": 1, "delta_i": 1}

(remainder, filtered, season, trend) = get_remainder(signal, fs, filter_params, season_params, lambda_1=1.0,
                                                     lambda_2=0.5)
plt.figure()
plt.plot(remainder, label="remainder")
plt.plot(season, label="season")
plt.legend()

plt.figure()
plt.plot(trend, label="trend")
plt.plot(filtered, label="filtered")

plt.legend()

plt.show()
