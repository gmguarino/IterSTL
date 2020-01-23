import numpy as np
from scipy.signal import periodogram
from scipy.sparse import diags, eye, vstack
from scipy.sparse.linalg import norm, spsolve, lsqr
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from l1 import l1


def bilateral_window(y, t, J, delta_d, delta_i):
    filtered = []
    y_j = []
    for j in J:
        filtered.append(
            np.exp(-abs(j - t) ** 2 / (2 * delta_d ** 2)) * np.exp(-abs(y[j] - y[t]) ** 2 / (2 * delta_i ** 2)))
        y_j.append(y[j])
    return np.array(filtered) / max(filtered), np.array(y_j)


def bilateral_filtering(time_series, H, delta_d, delta_i):
    N = time_series.size
    filtered = []
    for t in range(N):
        if t < H:
            J = np.arange(0, t + H, 1, dtype=int)
        elif N - t <= H:
            J = np.arange(t - H, N, 1, dtype=int)
        else:
            J = np.arange(t - H, t + H, 1, dtype=int)
        window, yj = bilateral_window(time_series, t, J, delta_d, delta_i)
        filtered.append(np.sum(window * yj))
    return np.array(filtered)


def estimate_period(timeseries, fs):
    frq, psd = periodogram(timeseries, fs=fs)
    interpolator = interp1d(frq, psd, kind='quadratic')
    frequencies = np.linspace(0, max(frq) // 4, 8 * len(frq))
    powerspectrum = interpolator(frequencies)
    max_index = np.where(powerspectrum == np.amax(powerspectrum))
    main_frq = frequencies[max_index[0][0]]
    period = int(round(1 / main_frq * fs))
    return period


def extract_seasonality(timeseries, fs, K, H, delta_d, delta_i):
    N = timeseries.size
    T = round(estimate_period(timeseries, fs), 2)
    season = []
    for t in range(N):
        if t < K:
            t_dash = np.arange(0, t - T, T, dtype=int)
        else:
            t_dash = np.arange(t - K * T, t - T, T, dtype=int)
        t_window = []
        for _t_dash in t_dash:
            if _t_dash < H:
                J = np.arange(0, _t_dash + H, 1, dtype=int)
            elif N - _t_dash <= H:
                J = np.arange(_t_dash - H, N, 1, dtype=int)
            else:
                J = np.arange(_t_dash - H, _t_dash + H, 1, dtype=int)
            window, yj = bilateral_window(timeseries, _t_dash, J, delta_d, delta_i)
            t_window.append(np.sum(window * yj))
        season.append(np.sum(t_window))
    return np.array(season)


def remove_norm_mean(season, fs):
    N = season.size
    T = round(estimate_period(season, fs), 2)
    prefactor = 1 / (T * np.floor(N / T))
    tau1 = prefactor * np.sum(season[0: int(1 / prefactor)])
    season = season - tau1
    season /= season.max()
    return season


def seasonal_difference(timeseries, period_samples):
    g = timeseries[period_samples:] - timeseries[: -period_samples]
    return g.reshape((g.size, 1))


def grad2relative(grad):


    def grad2relative_single(trend_idx):
        if trend_idx < 0:
            return 0
        else:
            return np.sum(grad[:trend_idx])

    trend_idxs = np.arange(-1, len(grad))
    relative = list(map(grad2relative_single, trend_idxs))
    return np.array(relative)


def extract_trend(timeseries, period_samples, lambda_1=1.0, lambda_2=0.5):
    N = timeseries.size
    g = seasonal_difference(timeseries, period_samples)
    T = period_samples
    Identity = eye(N - 1)
    diagonals = np.ones((T, N - T))
    offsets = np.arange(0, T)
    M = diags(diagonals, offsets=offsets, shape=(N - T, N - 1))
    diagonals_d = np.concatenate([np.ones((1, N - 2)), -1 * np.ones((1, N - 2))])
    D = diags(diagonals_d, offsets=[0, 1], shape=(N - 2, N - 1))
    P = vstack([M, lambda_1 * Identity, lambda_2 * D])
    q = vstack([g, np.zeros((2 * N - 3, 1))])
    grad_trend = l1(P, q)
    trends = grad2relative(grad_trend)
    new_timeseries = timeseries - trends
    return new_timeseries, trends

# TODO: compatibility with sparse matrices for the l1 norm
