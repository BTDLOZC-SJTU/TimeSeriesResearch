# Time Series Research with Torch

这个开源项目主要是对经典的时间序列预测算法论文进行复现，模型主要参考自[GluonTS](https://github.com/awslabs/gluon-ts)，框架主要参考自[Informer](https://github.com/zhouhaoyi/Informer2020)。

## 建立原因

相较于mxnet和TF，Torch框架中的神经网络层需要提前指定输入维度：
```python
# 建立线性层 TensorFlow vs PyTorch
tf.keras.Dense(units=output_size) # 不需要提前指定输入维度
torch.nn.Linear(in_features=input_size, out_features=output_size) # 需要提前指定输入维度
```
这对于单一模型来说不会存在问题，我们可以对每个模型作针对性的特征工程，然后将数据输入即可。但在一个API统一的框架中可能会导致模型复用及其困难，因为用户并不知道自己调用的模型中封装了什么特征工程，所以也无法预知网络最底层的输入维度。

[PyTorchTS](https://github.com/zalandoresearch/pytorch-ts)是一位大佬根据GluonTS框架实现的基于PyTorch的时间序列预测框架，其数据加载、转换和模型的测试都非常漂亮，但由于PyTorch的这个特性，导致用户在调用时需要指定**input_size**参数：
```python
# PyTorchTS框架中DeepAR模型的调用
estimator = DeepAREstimator(
    distr_output=ImplicitQuantileOutput(output_domain="Positive"),
    cell_type='GRU',
    input_size=62, # 输入维度指定, 且只能指定为62, 但对没有深入了解框架的用户意义不明
    num_cells=64,
    num_layers=3,
    ...)
```
这个input_size=62并不是指用户输入的时间序列的维度，而是经过多个特征构造和转换后到达RNN单元的Tensor维度，这就需要用户提前在草稿纸上推导出变换后的数据维度，并当做评估器的输入，然而这不是一件容易的事情(复杂的多项式关系-_-||)，并且也丢失了神经网络的端到端的黑箱特性。

因此，希望能够实现一种更黑箱的框架，并做一些model和trick上的研究，这就是这个项目建立的原因啦。

## 数据加载

项目中的Benchmark数据来源于[multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)，并额外添加了人工生成的较为简单的时间序列，用于检测模型的正确性

| Dataset | Dimension | Frequency | Start Date |
| :----: | :----: | :----: | :----: |
| Electricity | 321 | H | 2012-01-01 00:00:00 |
| Exchange Rate | 8 | B | 1990-01-01 00:00:00 |
| Solar Energy | 137 | 10min | 2006-01-01 00:00:00 |
| Traffic | 862 | H | 2015-01-01 00:00:00 |
| Artificial | 1 | H | 2013-11-28 18:00:00 |

![time-series data show](/images/data_show.png)

## 时间特征

#### 时间特征生成

项目中时间特征完全由数据的时间频率决定，每个模型中都预先设定了频率特征长度参照，用户可以设定**embedding_size**参数来控制时间特征的Embedding维度，各时间频率对应特征如下表
| Frequency | Length | Feature Generated |
| :----: | :----: | :---- |
| Y (yearly) | 1 | age |
| M (monthly) | 2 | age, month |
| W (weekly) | 3 | age, day of month, week of year |
| D (daily) | 4 | age, day of week, day of month, day of year |
| B (business days) | 4 | age, day of week, day of month, day of year |
| H (hourly) | 5 | age, hour of day, day of week, day of month, day of year |
| T (minutely) | 6 | age, minute of hour, hour of day, day of week, day of month, day of year |
| S (secondly) | 7 | age, second of minute, minute of hour, hour of day, day of week, day of month, day of year |

其中，特征均归一化到[-0.5, 0.5]，部分情况下可能超过该值（闰年等），在模型训练时可以不再额外进行归一化，统一了特征全局归一化和局部归一化的差异。

#### 时间特征滞后处理

部分RNN-based模型采用自回归的方式进行预测，经典做法是将单一观测变量$x_t$当作输入与上一时刻输出的状态隐向量$h_{t-1}$进行运算得到下一时刻的隐向量$h_{t}$并获取输出$x_{t + 1}$。但如果观测变量仅包含一个时间点的观测值，自回归的效果就会随着预测长度的增加迅速变差。此时就需要对时间特征进行之后处理，让单个观测向量包含多个观测值，提升长时间点预测效果。

![diff lag](/images/diff_lag.png)

本项目中编写了两种时间特征滞后处理
1. *Default*: 设定一个滞后长度cntx_len，当前时间点的输入向量额外包含cntx_len个历史观测值;
2. *By_Freq*: 根据时间频率特征选择滞后观测点，以小时H为例，选择将在[24, 48, 72, ...]等带有周期性质的滞后时间点中产生。但前提是历史观测长度大于最长滞后长度，否则会导致输入Tensor维度不一致的错误，目前当历史观测长度大于769时，能够保证不出现上述错误;


## 概率输出

传统基于机器学习(XGBoost, ARIMA等)的时序预测模型通常是点预测模型，仅能作出数值上的绝对估计。然而，现实数据中存在低噪声区域和高噪声区域，尤其是对股票数据等高信噪比数据预测且需要根据预测采取行动的情景下，单一的预测数值可能参考意义不大，此时便需要**通过概率建模来预测未来的不确定性**，从而更好地指导未来行为。
![point](/images/point_modeling.png)
![proba](/images/probabilistic_modeling.png)
