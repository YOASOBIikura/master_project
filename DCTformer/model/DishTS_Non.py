import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        init = args.dish_init  # 'standard', 'avg' or 'uniform'
        configs = args
        activate = True
        n_series = args.c_out  # number of series
        lookback = args.seq_len  # lookback length
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2) / lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2) / lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(
                torch.ones(n_series, lookback, 2) / lookback + torch.rand(n_series, lookback, 2) / lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate
        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)
        self.std_enc = None
        self.mean_enc = None
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

    def forward(self, batch_x, mode='forward', dec_inp=None):

        if mode == 'forward':

            x_raw = batch_x.clone().detach()
            # Normalization
            self.mean_enc = x_raw.mean(1, keepdim=True).detach()  # B x 1 x E
            batch_x = x_raw - self.mean_enc
            self.std_enc = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
            batch_x = batch_x / self.std_enc
            dec_new = torch.cat([batch_x[:, :self.label_len, :], torch.zeros_like(dec_inp[:, -self.pred_len:, :])],
                                  dim=1).to(batch_x.device).clone()

            tau = self.tau_learner(x_raw, self.std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
            delta = self.delta_learner(x_raw, self.mean_enc)  # B x S x E, B x 1 x E -> B x S

            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_new = None if dec_inp is None else self.forward_process(dec_new)

            return batch_x, dec_new, tau, delta
        elif mode == 'inverse':
            batch_y = batch_x * self.std_enc + self.mean_enc
            batch_y = self.inverse_process(batch_y)
            return batch_y

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2, 0, 1)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :] #
        self.xil = torch.sum(torch.pow(batch_x - self.phil, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)

    def forward_process(self, batch_input):
        # print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil) / torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst

    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih