"""all time features are encoded as value between [-0.5, 0.5]"""

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


def SecondOfMinute(index: pd.DatetimeIndex) -> np.ndarray:
    return index.second / 59.0 - 0.5


def MinuteOfHour(index: pd.DatetimeIndex) -> np.ndarray:
    return index.minute / 59.0 - 0.5


def HourOfDay(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour / 23.0 - 0.5


def DayOfWeek(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.dayofweek - 1) / 6.0 - 0.5


def DayOfMonth(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.day - 1) / 30.0 - 0.5


def DayOfYear(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.dayofyear - 1) / 365.0 - 0.5


def MonthOfYear(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.month - 1) / 11.0 - 0.5


def WeekOfYear(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.weekofyear - 1) / 52.0 - 0.5


def ConstantAge(index: pd.DatetimeIndex) -> np.ndarray:
    return (np.arange(index.shape[0]) - index.shape[0] // 2) / index.shape[0]


def timeFeatures(dates: pd.DatetimeIndex, freq: str='H') -> np.ndarray:
    """
    supported freq:
    > * Y: yearly = [age]
    > * M: monthly = [age, month]
    > * W: weekly = [age, day of month, week of year]
    > * D: daily = [age, day of week, day of month, day of year]
    > * B: business days = [age, day of week, day of month, day of year]
    > * H: hourly = [age, hour of day, day of week, day of month, day of year]
    > * T: minutely = [age, minute of hour, hour of day, day of week, day of month, day of year]
    > * S: secondly = [age, second of minute, minute of hour, hour of day, day of week, day of month, day of year]
    """
    features_by_offsets = {
        offsets.YearEnd: [ConstantAge],
        offsets.MonthEnd: [ConstantAge, MonthOfYear],
        offsets.Week: [ConstantAge, DayOfMonth, WeekOfYear],
        offsets.Day: [ConstantAge, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [ConstantAge, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [ConstantAge, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [ConstantAge, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [ConstantAge, SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
    }

    offset = to_offset(freq)

    time_features_from_frequency = []
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            time_features_from_frequency = feature_classes

    return np.vstack([feat(dates) for feat in time_features_from_frequency]).transpose(1, 0)