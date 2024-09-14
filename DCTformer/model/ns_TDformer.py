from layers.TDformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, AttentionLayer, series_decomp, series_decomp_multi
import torch.nn as nn
import torch
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp
from layers.Attention import FourierAttention, FullAttention, DSFourierAttention
from layers.MultiWaveletCorrelation import WaveletAttention
from layers.RevIN import RevIN
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
    """
    Transformer for seasonality, MLP for trend
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.output_stl = configs.output_stl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # self.enc_seasonal_embedding = DataEmbedding_wo_pos(c_in=configs.enc_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq,
        #                                             dropout=configs.dropout)
        # self.dec_seasonal_embedding = DataEmbedding_wo_pos(c_in=configs.enc_in, d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq,
        #                                             dropout=configs.dropout)
        self.enc_seasonal_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed,
                                                            configs.freq,
                                                            configs.dropout)
        self.dec_seasonal_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed,
                                                            configs.freq,
                                                            configs.dropout)

        # Encoder
        if configs.version == 'Wavelet':
            enc_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len,
                                                  seq_len_kv=configs.seq_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                  seq_len_kv=configs.seq_len // 2 + configs.pred_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = WaveletAttention(in_channels=configs.d_model,
                                                   out_channels=configs.d_model,
                                                   seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                   seq_len_kv=configs.seq_len,
                                                   ich=configs.d_model,
                                                   T=configs.temp,
                                                   activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Fourier':
            enc_self_attention = DSFourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = DSFourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = DSFourierAttention(T=configs.temp, activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Time':
            enc_self_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_self_attention = FullAttention(True, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_cross_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention)

        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        enc_self_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.seasonal_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        dec_self_attention,
                        configs.d_model),
                    AttentionLayer(
                        dec_cross_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # Encoder
        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )

        self.revin_trend = RevIN(configs.enc_in).to(self.device)

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                    hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

    def forward(self, tau, delta, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_raw = x_enc.clone().detach()
        # Normalization
        # mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        # x_enc = x_enc - mean_enc
        # std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        # x_enc = x_enc / std_enc
        #
        # tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        # delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        seasonal_enc, trend_enc = self.decomp(x_enc)
        seasonal_dec, trend_dec = self.decomp(x_dec)
        seasonal_dec = F.pad(seasonal_dec[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # seasonal
        enc_out = self.enc_seasonal_embedding(seasonal_enc, None)
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_seasonal_embedding(seasonal_dec, None)
        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=None)
        seasonal_out = seasonal_out[:, -self.pred_len:, :]
        # seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        # trend
        trend_enc = self.revin_trend(trend_enc, 'norm')
        trend_out = self.trend(trend_enc.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.revin_trend(trend_out, 'denorm')

        # final
        dec_out = trend_out + seasonal_ratio * seasonal_out

        # De-normalization
        # dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out, (attn_e, attn_d)
        elif self.output_stl:
            return dec_out, trend_enc, seasonal_enc, trend_out, seasonal_ratio * seasonal_out
        else:
            return dec_out