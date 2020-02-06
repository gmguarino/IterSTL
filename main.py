import numpy as np
import matplotlib.pyplot as plt
from IterSTL import estimate_period, ISTL, RSTL

np.random.seed(5)

signal = []
print("Generating Signal ...")
freq = 0.8
print("Signal period is T = ", 1 / freq)
fs = 100
w = 2 * np.pi * freq
trend = 0.003
for i in range(500):
    switch = np.random.rand()
    base_signal = trend * i + np.sin(w * i / fs)
    if switch < 0.97:
        signal.append(base_signal + np.random.rand() * 0.1)
    else:
        signal.append(base_signal + np.random.rand() * 3)
signal = np.asarray(signal)

H = 3
K = 2
delta_d = 1
delta_i = 1
T = estimate_period(signal, fs=fs)
print(f"Calculated period is: {T} samples")

filter_params = {"H": 5, "delta_d": 0.5, "delta_i": 0.5}
season_params = {"H": 3, "K": 2, "delta_d": 1, "delta_i": 1}
print("Finding Anomalies...")
# (remainder, filtered, season, trend) = RSTL(signal, fs, filter_params, season_params, lambda_1=1.0, lambda_2=0.5)
(remainder, filtered, season, trend) = ISTL(signal, fs, filter_params, decimation_rate=0.4, lambda_1=1.0, lambda_2=0.5)
results = (signal, filtered, season, trend, remainder)
colours = ['red', 'blue', 'green', 'black', 'violet']

labels = ("signal", "filtered", "season", "trend", "remainder")
fig, axes = plt.subplots(5, 1, sharex='col')
for i in range(5):
    axes[i].plot(results[i], color=colours[i])
    axes[i].set_ylabel(labels[i])
    # axes[i].set_ylim([-1.5, 5])
axes[-1].set_xlabel("Sample Number")
axes[-1].plot(np.diff(np.diff(results[-1]) ** 2) ** 2, color='orange', alpha=0.5)
detection = np.diff(np.diff(results[-1]) ** 2) ** 2
anomalies = np.where(abs(sample_list[-1]) > 0)[0]
print(sample_list[-1], anomalies)
axes[-1].vlines(anomalies, detection.min(), detection.max(), color='black')
axes[0].vlines(anomalies, results[0].min(), results[0].max(), color='black')

plt.show()
