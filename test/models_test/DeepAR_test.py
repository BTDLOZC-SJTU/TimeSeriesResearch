from utils.tools import dotdict
from exp.exp_DeepAR import Exp_DeepAR
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

args = dotdict()

args.model = 'deepar'

# args.data_path = "../data/electricity/electricity.txt.gz"
# args.data_path = "../data/exchange_rate/exchange_rate.txt.gz"
args.data_path = "../data/simple_sin/simple_sin.csv"
args.freq = 'H'
args.start_date = "2014-01-01"
args.cols=[0]

args.hist_len = 96
args.pred_len = 24
args.c_in = 1  # input target feature dimension
args.c_out = 1  # output target feature dimension
args.d_model = 16  # model dimension
args.use_time_freq = False

args.num_layers = 2
args.hidden_size = 40
args.embedding_dim = 10
args.cell_type = 'GRU'
args.dropout_rate = 0.0
args.num_parallel_samples = 100

args.batch_size = 64
args.learning_rate = 0.001
args.num_workers = 0

args.train_epochs = 20
args.patience = 5
args.checkpoints = "DeepAR_checkpoints"

args.use_gpu = True if torch.cuda.is_available() else False


Exp = Exp_DeepAR

exp = Exp(args)

print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train()

print('>>>>>>>start testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.test()