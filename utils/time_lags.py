"""
Generate a list of lags that are appropriate for the given frequency.

By default, all frequencies have lags: [1, 2, 3, 4, 5, 6, 7]
"""

from functools import reduce
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import torch


def _make_lags(middle: int, delta: int):
    """
    Create lags around a middle point including +/- delta
    """
    return np.arange(middle-delta, middle+delta+1).tolist()

def MinuteLags(multiple, num_cycles=3):
    """
    3 hours lags
    """
    return [_make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)]

def HourLags(multiple, num_cycles=7):
    """
    7 days lags
    """
    return [_make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)]

def DayLags(multiple, num_cycles=4):
    """
    4 weeks lags + last month
    """
    return [_make_lags(k * 7 // multiple, 1) for k in range(1, num_cycles + 1)] + \
           [_make_lags(30 // multiple, 1)]

def BusinessDayLags(multiple, num_cycles=4):
    """
    4 weeks lags
    """
    return [_make_lags(k * 5 // multiple, 1) for k in range(1, num_cycles + 1)]

def WeekLags(multiple, num_cycles=3):
    """
    3 years lags + 4, 8, 12 weeks
    """
    return [_make_lags(k * 52 // multiple, 1) for k in range(1, num_cycles + 1)] + \
           [[4 // multiple, 8 // multiple, 12 // multiple]]

def MonthLags(multiple, num_cycles=3):
    """
    3 years lags
    """
    return [_make_lags(k * 12 // multiple, 1) for k in range(1, num_cycles + 1)]

def getLagsForFreq(freq):
    """
    supported freq:
    > * M: monthly
    > * W: weekly
    > * D: daily
    > * B: business days
    > * H: hourly
    > * T: minutely
    """

    offset = to_offset(freq)

    if offset.name == "M":
        lags = MonthLags(offset.n)
    elif offset.name == "W-SUN" or offset.name == "W-MON":
        lags = WeekLags(offset.n)
    elif offset.name == "D":
        lags = DayLags(offset.n) + WeekLags(offset.n / 7.0)
    elif offset.name == "B":
        lags = BusinessDayLags(offset.n)
    elif offset.name == "H":
        lags = (
            HourLags(offset.n)
            + DayLags(offset.n / 24.0)
            + WeekLags(offset.n / (24.0 * 7))
        )
    elif offset.name == "T":
        lags = (
            MinuteLags(offset.n)
            + HourLags(offset.n / 60.0)
            + DayLags(offset.n / (60.0 * 24))
            + WeekLags(offset.n / (60.0 * 24 * 7))
        )
    else:
        raise Exception("invalid frequency")

    # flatten lags
    return [1, 2, 3, 4, 5, 6, 7] + list(sorted(map(lambda x: int(x), reduce(lambda x, y: x + y, lags))))


def get_lagged_subsequences_by_freq(freq: str,
                                    sequence: torch.Tensor,
                                    sequence_len: int):
    # TODO: 需要想好怎么使用这个lags，hist_len一定要足够大
    indices = getLagsForFreq(freq)

    lagged_values = []
    for lag_index in indices:
        begin_index = -lag_index - sequence_len
        if (-begin_index > sequence.shape[1]):
            # protect from length overflow
            break
        end_index = -lag_index if lag_index > 0 else None
        lagged_values.append(sequence[:, begin_index: end_index, ...])
    return torch.stack(lagged_values, dim=-1)

def get_lagged_subsequences_by_default(sequence: torch.Tensor,
                                       sequence_len: int,
                                       subsequence_len: int,
                                       mode: bool):

    lagged_values = []
    if mode == True:
        # Train
        for i in range(1, sequence_len - subsequence_len + 1) :
            begin_index = -i - subsequence_len
            end_index = -i
            lagged_values.append(sequence[:, begin_index: end_index, ...])
    else:
        for i in range(0, sequence_len - subsequence_len) :
            begin_index = -i - subsequence_len
            end_index = -i if i > 0 else None
            lagged_values.append(sequence[:, begin_index: end_index, ...])

    return torch.stack(lagged_values, dim=-1)