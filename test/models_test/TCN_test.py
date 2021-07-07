import json
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils.tools import dotdict
from exp.exp_TCN import Exp_TCN


args = dotdict()

args.model = 'tcn'

with open("../data/data_info.json",'r', encoding='utf8') as f:
    data_info = json.load(f)

# available select: "electricity", "exchange_rate", "solar-energy", "traffic", "artificial"
args.data = "artificial"

args.data_path = data_info[args.data]["data_path"]
args.freq = data_info[args.data]["freq"]
args.start_date = data_info[args.data]["start_date"]
args.cols=[0]
args.scale = True

args.hist_len = 96
args.pred_len = 24
args.c_in = 1  # input target feature dimension
args.c_out = 1  # output target feature dimension

args.num_hidden_dimension = [150, 150, 150, 150]
args.dilation_base = 2
args.kernel_size = 2
args.dropout_rate = 0.2
args.embedding_dim = 10


args.batch_size = 64
args.learning_rate = 0.001
args.loss = 'mse'
args.num_workers = 0

args.train_epochs = 20
args.patience = 5
args.checkpoints = "TCN_checkpoints"

args.use_gpu = True if torch.cuda.is_available() else False


Exp = Exp_TCN

exp = Exp(args)

print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train()

print('>>>>>>>start testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.test()