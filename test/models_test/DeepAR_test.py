import json
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils.tools import dotdict
from exp.exp_DeepAR import Exp_DeepAR


args = dotdict()

args.model = 'deepar'

with open("../data/data_info.json",'r', encoding='utf8') as f:
    data_info = json.load(f)

# available select: "electricity", "exchange_rate", "solar-energy", "traffic", "artificial"
args.data = "exchange_rate"

args.data_path = data_info[args.data]["data_path"]
args.freq = data_info[args.data]["freq"]
args.start_date = data_info[args.data]["start_date"]

args.cols=[0]

args.hist_len = 96
args.cntx_len = 24
args.pred_len = 24
args.c_in = 1  # input target feature dimension
args.c_out = 1  # output target feature dimension
args.d_model = 16  # model dimension

args.num_layers = 2
args.hidden_size = 40
args.embedding_dim = 10
args.cell_type = 'GRU'
args.dropout_rate = 0.0
args.num_parallel_samples = 100

args.batch_size = 64
args.learning_rate = 0.001
args.num_workers = 0

args.train_epochs = 1
args.patience = 5
args.checkpoints = "DeepAR_checkpoints"

args.use_gpu = True if torch.cuda.is_available() else False


Exp = Exp_DeepAR

exp = Exp(args)

print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train()

print('>>>>>>>start testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.test()