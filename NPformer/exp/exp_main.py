from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, SCformer, Preformer, Preformer_wo_MS, Decom_FullAttention, Decom_ProbAttention, Decom_LogAttention, SCformer_de2, LogTrans, NPformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as panda

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Preformer': Preformer,
            'Preformer_wo_MS': Preformer_wo_MS,
            'SCformer': SCformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'LogTrans': LogTrans,
            'Decom_Log': Decom_LogAttention,
            'Decom_Prob': Decom_ProbAttention,
            'Decom_Full': Decom_FullAttention,
            'SCformer_de2': SCformer_de2,
            'NPformer': NPformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, sc):
        total_loss = []
        self.model.eval()
        if sc == 1:
            preds = []
            trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if sc == 1:
                    pre = pred.detach().numpy()
                    tru = true.detach().numpy()
                    preds.append(pre)
                    trues.append(tru)

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)

        if sc == 1:
            preds = np.array(preds)
            trues = np.array(trues)
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('test mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)  # (32, 96, 1)

                batch_y = batch_y.float().to(self.device)  # (32, 144, 1)
                batch_x_mark = batch_x_mark.float().to(self.device)  # (32, 96, 3)
                batch_y_mark = batch_y_mark.float().to(self.device)  # (32, 144, 3)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # dec_inp每个batch中前48位是正常值，后96位用0填充

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, 0)
            test_loss = self.vali(test_data, test_loader, criterion, 0)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    #
    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     self.model.eval()
    #     # ？
    #     all_mse_min = 100
    #     all_index_min = 0
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                 else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #
    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()
    #
    #             pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
    #             true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
    #
    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 1 == 0:
    #                 print(i)
    #                 mse_min = 100
    #                 index_min = 0
    #                 for k in range(true.shape[0]):
    #                     current_mse = np.mean((pred[k, :, -1] - true[k, :, -1]) ** 2)
    #                     if current_mse < mse_min:
    #                         index_min = k
    #                         mse_min = current_mse
    #                 print(index_min)
    #                 if mse_min < all_mse_min:
    #                     all_mse_min = mse_min
    #                     all_index_min = i
    #                 input = batch_x.detach().cpu().numpy()
    #                 gt = np.concatenate((input[index_min, :, -1], true[index_min, :, -1]), axis=0)
    #                 pd = np.concatenate((input[index_min, :, -1], pred[index_min, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    #
    #             # if i == 130:
    #             #     k = 1
    #             #     input = batch_x.detach().cpu().numpy()
    #             #     gt = np.concatenate((input[k, :, -1], true[k, :, -1]), axis=0)
    #             #     pd = np.concatenate((input[k, :, -1], pred[k, :, -1]), axis=0)
    #             #     visual(gt, pd, os.path.join(folder_path, str(i) + "_" + str(k) + '.pdf'))
    #             #
    #             # if i == 151:
    #             #     k = 11
    #             #     input = batch_x.detach().cpu().numpy()
    #             #     gt = np.concatenate((input[k, :, -1], true[k, :, -1]), axis=0)
    #             #     pd = np.concatenate((input[k, :, -1], pred[k, :, -1]), axis=0)
    #             #     visual(gt, pd, os.path.join(folder_path, str(i) + "_" + str(k) + '.pdf'))
    #             #
    #             # if i == 123:
    #             #     k = 29
    #             #     input = batch_x.detach().cpu().numpy()
    #             #     gt = np.concatenate((input[k, :, -1], true[k, :, -1]), axis=0)
    #             #     pd = np.concatenate((input[k, :, -1], pred[k, :, -1]), axis=0)
    #             #     visual(gt, pd, os.path.join(folder_path, str(i) + "_" + str(k) + '.pdf'))
    #             #
    #             # if i == 124:
    #             #     k = 7
    #             #     input = batch_x.detach().cpu().numpy()
    #             #     gt = np.concatenate((input[k, :, -1], true[k, :, -1]), axis=0)
    #             #     pd = np.concatenate((input[k, :, -1], pred[k, :, -1]), axis=0)
    #             #     visual(gt, pd, os.path.join(folder_path, str(i) + "_" + str(k) + '.pdf'))
    #
    #             # if i == 228:
    #             #     k = 14
    #             #     input = batch_x.detach().cpu().numpy()
    #             #     gt = np.concatenate((input[k, :, -1], true[k, :, -1]), axis=0)
    #             #     pd = np.concatenate((input[k, :, -1], pred[k, :, -1]), axis=0)
    #             #     visual_one(gt, pd, os.path.join(folder_path, str(i) + "_" + str(k) + '.pdf'), -2, -1.2, 0.1)
    #
    #
    #     print("最小mse对应的i为" + str(all_index_min))
    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     print('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)
    #
    #     # result save
    #     # folder_path = './results/' + setting + '/'
    #     # if not os.path.exists(folder_path):
    #     #     os.makedirs(folder_path)
    #     #
    #     # mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     # print('mse:{}, mae:{}'.format(mse, mae))
    #     # f = open("result.txt", 'a')
    #     # f.write(setting + "  \n")
    #     # f.write('mse:{}, mae:{}'.format(mse, mae))
    #     # f.write('\n')
    #     # f.write('\n')
    #     # f.close()
    #     #
    #     # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     # np.save(folder_path + 'pred.npy', preds)
    #     # np.save(folder_path + 'true.npy', trues)
    #
    #     return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        batch = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                batch.append(batch_x.detach().cpu().numpy())
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        batch = np.array(batch)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        batch = batch.reshape(-1, batch.shape[-2], batch.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        truth = trues[-1, :, -1]
        prediction = preds[-1, :, -1]
        before_x = batch[-1, :, -1]
        truth = np.concatenate((before_x, truth), axis=0)
        prediction = np.concatenate((before_x, prediction), axis=0)
        truth = truth.reshape(1, truth.shape[0])
        prediction = prediction.reshape(1, prediction.shape[0])
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        dataFrame = panda.DataFrame({'truth': truth[0, :], 'prediction': prediction[0, :]})
        dataFrame.to_csv(r'' + folder_path + 'result.csv', index=False)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        truth = []
        batch = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                batch_y = batch_y[:, -self.args.pred_len:, -1:]
                outputs = outputs[:, :, -1:]
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(pred)
                truth.append(batch_y)
                # batch.append(batch_x.detach().cpu().numpy())

        r = np.array(preds)
        truth = np.array(truth)
        # truth = truth[::10, ...]
        # r = r[::10, ...]
        tru = truth.reshape(-1, truth.shape[-2])
        pre = r.reshape(-1, r.shape[-2])
        tru = tru.reshape(1, -1)
        pre = pre.reshape(1, -1)

        # result save
        folder_path = './prediction/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        dataFrame = panda.DataFrame({'truth': tru[0, :], 'prediction': pre[0, :]})
        dataFrame.to_csv(r'' + folder_path + 'res.csv', index=False)

        # np.save(folder_path + 'real_prediction.npy', preds)

        return