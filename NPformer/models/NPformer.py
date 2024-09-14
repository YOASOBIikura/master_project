import torch.nn as nn
import torch
from layers.RevIN import RevIN
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.SMFourierAttention import FourierAttention, PreFourierAttention, DSFourierAttention
from layers.NPformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer, series_decomp, series_decomp_multi
from layers.SelfAttention_NP_Family import MultiScaleAttentionLayer
from layers.local_global import MIC

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
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
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
        self.enc_seasonal_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.dec_seasonal_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    MultiScaleAttentionLayer(
                        FourierAttention(False, configs.factor, attention_dropout=configs.dropout,
                                         output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            conv_layer=MIC(feature_size=configs.d_model, n_heads=configs.n_heads, decomp_kernel=configs.en_decomp_kernel,
                           conv_kernel=configs.en_conv_kernel, isometric_kernel=configs.en_isometric_kernel, device=self.device)
        )

        # Decoder
        self.seasonal_decoder = Decoder(
            [
                DecoderLayer(
                    MultiScaleAttentionLayer(
                        FourierAttention(True, configs.factor, attention_dropout=configs.dropout,
                                         output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    MultiScaleAttentionLayer(
                        PreFourierAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            conv_layer=MIC(feature_size=configs.d_model, n_heads=configs.n_heads,
                           decomp_kernel=configs.decomp_kernel,conv_kernel=configs.conv_kernel, isometric_kernel=configs.isometric_kernel,
                           device=self.device)
        )

        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        ).to(self.device)


        self.revin_trend = RevIN(configs.enc_in).to(self.device)

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_raw = x_enc.clone().detach()
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E u
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        tau = self.tau_learner(x_raw, std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # decomp init
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(self.device)  # cuda()
        seasonal_enc, trend_enc = self.decomp(x_enc)
        # season, _ = self.decomp(x_raw)
        seasonal_dec = F.pad(seasonal_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # seasonal
        enc_out = self.enc_seasonal_embedding(seasonal_enc, x_mark_enc)
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_seasonal_embedding(seasonal_dec, x_mark_dec)
        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                     tau=tau, delta=None)
        seasonal_out = seasonal_out[:, -self.pred_len:, :]
        seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        # trend
        trend_enc = self.revin_trend(trend_enc, 'norm')
        trend_out = self.trend(trend_enc.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.revin_trend(trend_out, 'denorm')

        # final
        dec_out = trend_out + seasonal_ratio * seasonal_out

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out, (attn_e, attn_d)
        elif self.output_stl:
            return dec_out, trend_enc, seasonal_enc, trend_out, seasonal_ratio * seasonal_out
        else:
            return dec_out



