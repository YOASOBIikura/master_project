import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

class Params():

    def __init__(self, description='Transformer family for Time Series Forecasting', is_training=1, model_id='Traffic', model='NPformer', data='custom', root_path='data',
                 data_path='traffic.csv',features='M', target='OT', freq='h', checkpoints='checkpoints', seq_len=96, label_len=48, pred_len=96, bucket_size=4, version='Fourier',
                 n_hashes=8, enc_in=862, dec_in=862, c_out=862, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=25, output_stl=False,
                 factor=1, distil=True, dropout=0.05, embed='timeF', activation='gelu', output_attention=False, do_predict=False, temp=1,
                 num_workers=0, itr=2, train_epochs=20, batch_size=32, patience=5, learning_rate=0.0001, des='test', loss='mse',
                 lradj='type1', use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0', p_hidden_dims=[128, 128], p_hidden_layers=2,
                 conv_kernel=[24, 48], decomp_kernel=[24, 48], isometric_kernel=[24, 48], en_conv_kernel=[24, 48], en_decomp_kernel=[25, 49],
                 en_isometric_kernel=[9, 5]):
        self.en_conv_kernel = en_conv_kernel
        self.en_decomp_kernel = en_decomp_kernel
        self.en_isometric_kernel = en_isometric_kernel
        self.conv_kernel = conv_kernel
        self.decomp_kernel = decomp_kernel
        self.isometric_kernel = isometric_kernel
        self.p_hidden_dims = p_hidden_dims
        self.p_hidden_layers = p_hidden_layers
        self.temp = temp
        self.output_stl = output_stl
        self.version = version
        self.description = description
        self.is_training = is_training
        self.model_id = model_id
        self.model = model
        self.data = data
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.freq = freq
        self.checkpoints = checkpoints
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.factor = factor
        self.distil = distil
        self.dropout = dropout
        self.embed = embed
        self.activation = activation
        self.output_attention = output_attention
        self.do_predict = do_predict
        self.num_workers = num_workers
        self.itr = itr
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.des = des
        self.loss = loss
        self.lradj = lradj
        self.use_amp = use_amp
        self.use_gpu = use_gpu
        self.gpu = gpu
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices

args = Params()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if args.pred_len == 24:
    args.conv_kernel = [6, 12]
    args.decomp_kernel = [6, 12]
    args.isometric_kernel = [6, 12]
elif args.pred_len == 96:
    args.conv_kernel = [24, 48]
    args.decomp_kernel = [24, 48]
    args.isometric_kernel = [24, 48]
elif args.pred_len == 192:
    args.conv_kernel = [12, 24]
    args.decomp_kernel = [12, 24]
    args.isometric_kernel = [12, 24]
elif args.pred_len == 336:
    args.conv_kernel = [16, 24]
    args.decomp_kernel = [16, 24]
    args.isometric_kernel = [16, 24]

decomp_kernel = []  # kernel of decomposition operation
isometric_kernel = []  # kernel of isometric convolution
for ii in args.conv_kernel:
    if ii%2 == 0:   # the kernel of decomposition operation must be odd
        decomp_kernel.append(ii+1)
        isometric_kernel.append((args.seq_len + args.pred_len+ii) // ii)
    else:
        decomp_kernel.append(ii)
        isometric_kernel.append((args.seq_len + args.pred_len+ii-1) // ii)
args.isometric_kernel = isometric_kernel  # kernel of isometric convolution
args.decomp_kernel = decomp_kernel   # kernel of decomposition operation

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()