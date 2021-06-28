"""all time features are encoded as value between [-0.5, 0.5]"""

import numpy as np
import pandas as pd


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

