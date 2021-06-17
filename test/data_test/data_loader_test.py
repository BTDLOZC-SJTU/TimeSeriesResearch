from data.data_loader import *


if __name__ == '__main__':
    data_set = Dataset_TS(data_path="../data/electricity/electricity.txt.gz",
                      freq='H',
                      start_date="2014-01-01",
                      flag='train',
                      hist_len=96,
                      pred_len=24)
    print(len(data_set))