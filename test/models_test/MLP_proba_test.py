from utils.tools import dotdict
from exp.exp_MLP_proba import Exp_MLP_proba
import torch

args = dotdict()

args.model = 'mlp_proba'

# args.data_path = "../data/electricity/electricity.txt.gz"
# args.data_path = "../data/exchange_rate/exchange_rate.txt.gz"
# args.data_path = "../data/simple_sin/simple_sin.csv"
args.freq = 'H'
args.start_date = "2014-01-01"
args.cols=[0]

args.hist_len = 96
args.pred_len = 24

args.num_hidden_dimension = [512, 256, 128]

args.batch_size = 64
args.learning_rate = 0.001
args.num_workers = 0

args.train_epochs = 10

args.use_gpu = True if torch.cuda.is_available() else False


Exp = Exp_MLP_proba

exp = Exp(args)

print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train()

print('>>>>>>>start testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.test()
