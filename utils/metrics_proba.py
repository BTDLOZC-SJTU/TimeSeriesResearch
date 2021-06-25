"""
概率预测相较于点预测，多了一维采样维度
-> 点预测 [batch_size, pred_len, feature_dim]
-> 概率预测 [batch_size, num_samples, pred_len, feature_dim]
"""

import numpy as np


def MAE(pred, true):
    pred = np.median(pred, axis=1)
    return np.mean(np.abs((pred - true)))

def MSE(pred, true):
    pred = np.median(pred, axis=1)
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    pred = np.median(pred, axis=1)
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    pred = np.median(pred, axis=1)
    return np.mean(np.square((pred - true) / true))

def RSE(pred, true):
    pred = np.median(pred, axis=1)
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def metric(pred, true, metric_params=None):
    if metric_params is None:
        metric_params = ['mae', 'mse', 'rmse', 'mape', 'mspe']
    metrics = []
    metrics_dict = {'mae': MAE, 'mse': MSE, 'rmse': RMSE, 'mape': MAPE, 'mspe': MSPE, 'rse': RSE}
    for p in metric_params:
        metrics.append(metrics_dict[p](pred, true))

    return metrics