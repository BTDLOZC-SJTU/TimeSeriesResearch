from exp.exp_basic import Exp_Basic
from models.Transformer.Transformer_network import Transformer
from data.data_loader import Dataset_TS
from utils.metrics import metric
from utils.tools import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


class Exp_Transformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Transformer, self).__init__(args)

    def _build_model(self):
        assert self.args.model == 'transformer'

        model = Transformer(
            self.args.c_in,
            self.args.c_out,
            self.args.hist_len,
            self.args.cntx_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_head,
            self.args.num_encoder_layers,
            self.args.num_decoder_layers,
            self.args.dim_feedforward,
            self.args.dropout_rate,
            self.args.activation,
            self.args.embedding_dim,
            self.args.freq
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
                              cols=self.args.cols,
                              scale=self.args.scale)

        print(flag, len(data_set))

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            # Data = Dataset_Pred
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

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, ture = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            loss = criterion(pred.detach().cpu(), ture.detach().cpu())
            total_loss.append(loss)
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
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                pred, ture = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                loss = criterion(pred, ture)
                train_loss.append(loss.item())

                if (i + 1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self):
        # test_data, test_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag='pred')

        self.model.eval()

        pres = []
        preds = []
        trues = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pres.append(batch_x.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        pres = np.array(pres)
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        pres = pres.reshape(-1, pres.shape[-2], pres.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        metrics = metric(preds, trues)
        print('mse:{}, mae:{}'.format(metrics['MSE'], metrics['MAE']))
        print(metrics)

        for i in range(8):
            plt.figure()
            plt.plot(np.arange(len(pres[i, : -1, -1])), pres[i, : -1, -1], label='GroundTruth')
            plt.plot(np.arange(len(pres[i, :, -1]) - len(preds[i, :, -1]), len(pres[i, :, -1])) - 1, preds[i, :, -1], label='Prediction')
            plt.legend()
            plt.show()