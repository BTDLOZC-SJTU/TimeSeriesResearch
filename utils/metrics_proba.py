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

def COVER(pred, true, q):
    p_ub, p_lb = 0.5 + q / 2, 0.5 - q / 2
    pred_ub = np.quantile(pred, q=p_ub, axis=1)
    pred_lb = np.quantile(pred, q=p_lb, axis=1)
    return np.mean((pred_lb < true) & (true <= pred_ub))

def QUANT(pred, true, q):
    pred = np.median(pred, axis=1)
    return 2 * np.sum(np.abs((pred - true) * ((pred <= true) - q)))

def metric(pred, true, metric_params=None):
    if metric_params is None:
        metric_params = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'coverage', 'quantile_loss']
    metrics = dict()
    metrics_dict = {'mae': MAE, 'mse': MSE, 'rmse': RMSE, 'mape': MAPE, 'mspe': MSPE, 'rse': RSE, 'coverage': COVER, 'quantile_loss': QUANT}
    for p in metric_params:
        if p == 'coverage':
            for q in [0.1, 0.5, 0.9]:
                metrics[f"Coverage[{str(q)}]"] = metrics_dict[p](pred, true, q)
        elif p == 'quantile_loss':
            for q in [0.1, 0.5, 0.9]:
                metrics[f"QuantileLoss[{str(q)}]"] = metrics_dict[p](pred, true, q)
        else:
            metrics[p.upper()] = metrics_dict[p](pred, true)

    return metrics