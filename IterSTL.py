import numpy as np
from numpy.random import randint
from scipy.signal import periodogram
from scipy.sparse import diags, eye, vstack
from scipy.interpolate import interp1d
from l1 import l1


# TODO: FIX TREND


def bilateral_window(y, t, J, delta_d, delta_i):
    filtered = []
    y_j = []
    # print(len(J))
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
        t_dash = np.arange(int(max(0, t - K * T)), t - T, T, dtype=int)
        t_window = []
        for _t_dash in t_dash:
            J = np.arange(max([0, _t_dash - H]), min([N, _t_dash + H]), 1, dtype=int)
            window, yj = bilateral_window(timeseries, _t_dash, J, delta_d, delta_i)
            t_window.append(np.sum(window * yj))
        season.append(np.sum(t_window))
    return np.array(season)


def correction_factor(season, fs):
    N = season.size
    T = round(estimate_period(season, fs), 2)
    prefactor = 1 / (T * np.floor(N / T))
    tau1 = prefactor * np.sum(season[0: int(1 / prefactor)])
    return tau1


def adjust_season(season, fs):
    tau1 = correction_factor(season, fs)
    return season - tau1


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


def adjust_trend(trend, tau1):
    return trend + tau1


def get_remainder(timeseries, fs, filter_params, season_params, lambda_1=10.0, lambda_2=0.5):
    for i in range(5):
        H, delta_d, delta_i = (filter_params["H"], filter_params["delta_d"], filter_params["delta_i"])
        filtered = bilateral_filtering(timeseries, H, delta_d, delta_i)
        T = estimate_period(filtered, fs=fs)
        filtered, trend = extract_trend(filtered, T, lambda_1=lambda_1, lambda_2=lambda_2)
        H, K, delta_d, delta_i = (
            season_params["H"], season_params["K"], season_params["delta_d"], season_params["delta_i"])
        season = extract_seasonality(filtered, fs, H, K, delta_d, delta_i)
        timeseries = filtered - season
    return timeseries, filtered, season, trend


def define_rejection(timeseries):
    diff = np.diff(timeseries)
    rejection_params = {"mu": diff.mean(), "sep": diff.std() * 2}
    return rejection_params


def reject_point(timeseries, idx):
    diff = np.diff(timeseries)

    if abs(diff[idx - 1] - diff.mean()) > diff.std() * 2:
        return True
    else:
        return False


def IterativeSeason(timeseries, decimation_rate=0.5, prob_dist=None):
    N = len(timeseries)
    x_t = randint(N)
    n_iter = int(round(N * decimation_rate))
    idxs, values = (np.zeros(n_iter, dtype=int), np.zeros(n_iter))
    idxs[n_iter - 1] = N - 1
    values[n_iter - 1] = timeseries[N - 1]
    it = 0
    while True:

        # for it in range(1, n_iter - 1):
        xd = randint(N)
        if xd != 0 and xd != N - 1 and xd not in idxs and not reject_point(timeseries, xd):
            it += 1

            idxs[it] = xd
            values[it] = timeseries[xd]
        if it >= n_iter - 2:
            break
    sorted_idxs = np.argsort(idxs)
    idxs = idxs[sorted_idxs]
    values = values[sorted_idxs]
    interpolator = interp1d(idxs, values, kind='cubic')
    new_idxs = np.linspace(0, N - 1, N)
    season = interpolator(new_idxs)
    return season
    # return sorted(idxs), timeseries[sorted(idxs)]


def ISTL(timeseries, fs, filter_params, decimation_rate=0.5, lambda_1=10.0, lambda_2=0.5):
    H, delta_d, delta_i = (filter_params["H"], filter_params["delta_d"], filter_params["delta_i"])
    filtered = timeseries  # bilateral_filtering(timeseries, H, delta_d, delta_i)
    T = estimate_period(filtered, fs=fs)
    if T > timeseries.size // 2:
        T = min([timeseries.size // 2, T // 2])
    filtered, trend = extract_trend(filtered, T, lambda_1=lambda_1, lambda_2=lambda_2)
    season = IterativeSeason(filtered, decimation_rate=decimation_rate, prob_dist=None)
    remainder = filtered - season
    return remainder, filtered, season, trend
