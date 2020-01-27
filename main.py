import numpy as np
import matplotlib.pyplot as plt
from IterSTL import estimate_period, ISTL

# np.random.seed(10)

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

print("Finding Anomalies...")
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
axes[-1].plot(np.diff(results[-1]) ** 2, color='orange', alpha=0.5)
plt.show()
