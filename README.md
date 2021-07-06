# Time Series Research with Torch

这个开源项目主要是对经典的时间序列预测算法论文进行复现，模型主要参考自[GluonTS](https://github.com/awslabs/gluon-ts)，框架主要参考自[Informer](https://github.com/zhouhaoyi/Informer2020)。

**建立原因**
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