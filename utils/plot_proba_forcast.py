import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


def plot_proba_forcast(
        x: np.ndarray,
        pred: np.ndarray,
        prediction_intervals: Tuple = (50.0, 90.0),
        show_mean: bool = False,
        color: str = "b",
        label: str = None,
        *args,
        **kwargs,):
    label_prefix = "" if label is None else label + "-"

    for c in prediction_intervals:
        assert 0.0 <= c <= 100.0

    ps = [50.0] + [
        50.0 + f * c / 2.0
        for c in prediction_intervals
        for f in [-1.0, +1.0]
    ]
    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    ps_data = [np.quantile(pred, q=p/100.0, axis=0) for p in percentiles_sorted]
    i_p50 = len(percentiles_sorted) // 2
    p50_data = ps_data[i_p50]
    plt.plot(x, np.median(pred, axis=0), color=color, ls='-', label=f"{label_prefix}median")

    if show_mean:
        mean_data = np.mean(pred, axis=0)
        plt.plot(x, mean_data, color=color, ls=':', label=f"{label_prefix}mean")

    for i in range(len(percentiles_sorted) // 2):
        ptile = percentiles_sorted[i]
        alpha = alpha_for_percentile(ptile)
        plt.fill_between(
            x,
            ps_data[i],
            ps_data[-i - 1],
            facecolor=color,
            alpha=alpha,
            interpolate=True,
            *args,
            **kwargs,
        )