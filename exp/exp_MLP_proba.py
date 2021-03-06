from exp.exp_basic import Exp_Basic
from models.MLP_proba.MLP_proba_network import MLP_proba
from data.data_loader import Dataset_TS
from utils.plot_proba_forcast import plot_proba_forcast
from utils.metrics_proba import metric
from utils.tools import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


class Exp_MLP_proba(Exp_Basic):
    def __init__(self, args):
        super(Exp_MLP_proba, self).__init__(args)

    def _build_model(self):
        assert self.args.model == 'mlp_proba'

        model = MLP_proba(
            self.args.c_in,
            self.args.c_out,
            self.args.d_model,
            self.args.num_hidden_dimension,
            self.args.hist_len,
            self.args.pred_len,
            self.args.freq,
            self.args.use_time_feat
        ).float().to(self.args.device)

        return model

    def _get_data(self, flag):
        args = self.args

        data_set = Dataset_TS(data_path=self.args.data_path,
                          freq=self.args.freq,
                          start_date=self.args.start_date,
                          flag=flag,
                          hist_len=self.args.hist_len,
                          pred_len=self.args.pred_len,
                          cols=self.args.cols)

        print(flag, len(data_set))

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        distr = self.model(batch_x, batch_x_mark, batch_y_mark)
        loss = -distr.log_prob(batch_y)

        return loss.mean()

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            loss = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                loss = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                train_loss.append(loss.item())

                if (i + 1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, num_samples=100):
        test_data, test_loader = self._get_data(flag='pred')

        self.model.eval()

        pres = []
        preds = []
        trues = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.args.device)
            batch_y_mark = batch_y_mark.float().to(self.args.device)
            distr = self.model(batch_x, batch_x_mark, batch_y_mark)

            pres.append(batch_x.detach().cpu().numpy())
            preds.append(distr.sample([num_samples]).squeeze(1).cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

        pres = np.array(pres).squeeze(1)
        preds = np.array(preds)
        trues = np.array(trues).squeeze(1)

        metrics = metric(preds, trues)
        print('mse:{}, mae:{}'.format(metrics['MSE'], metrics['MAE']))
        print(metrics)

        i = 0
        plt.figure()
        plt.plot(np.arange(pres.shape[1] - 1), pres[i, : -1, -1], label='GroundTruth')
        plot_proba_forcast(np.arange(pres.shape[1] - preds.shape[2], pres.shape[1]) - 1, preds[i, :, :, -1])
        plt.legend()
        plt.show()