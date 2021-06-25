import numpy as np


def MAE(pred, true):
    return np.mean(np.abs((pred - true)))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    # Pearson相关系数
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def metric(pred, true, metric_params=None):
    if metric_params is None:
        metric_params = ['mae', 'mse', 'rmse', 'mape', 'mspe']
    metrics = dict()
    metrics_dict = {'mae': MAE, 'mse': MSE, 'rmse': RMSE, 'mape': MAPE, 'mspe': MSPE, 'rse': RSE, 'corr': CORR}
    for p in metric_params:
        metrics[p.upper()] = metrics_dict[p](pred, true)

    return metrics