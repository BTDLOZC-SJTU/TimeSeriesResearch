import json
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils.tools import dotdict
from exp.exp_DeepFactor import Exp_DeepFactor


args = dotdict()

args.model = 'deepfactor'

with open("../data/data_info.json",'r', encoding='utf8') as f:
    data_info = json.load(f)

# available select: "electricity", "exchange_rate", "solar-energy", "traffic", "artificial"
args.data = "electricity"

args.data_path = data_info[args.data]["data_path"]
args.freq = data_info[args.data]["freq"]
args.start_date = data_info[args.data]["start_date"]

args.cols = [0]
args.scale = True


args.c_in = 1  # input target feature dimension
args.c_out = 1  # output target feature dimension
args.d_model = 16  # model dimension
args.hist_len = 96
args.pred_len = 24
args.num_hidden_global = 50
args.num_layers_global = 1
args.num_factors = 5
args.num_hidden_local = 5
args.num_layers_local = 1
args.embedding_dim = 10
args.cell_type = 'GRU'

args.use_time_freq = True # in deepfactor, cannot be False

args.batch_size = 64
args.learning_rate = 0.01
args.num_workers = 0

args.train_epochs = 50
args.patience = 10
args.checkpoints = "DeepFactor_checkpoints"

args.use_gpu = True if torch.cuda.is_available() else False

Exp = Exp_DeepFactor

exp = Exp(args)

print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train()

print('>>>>>>>start testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.test()
