from layers.TDformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, AttentionLayer, series_decomp, series_decomp_multi
import torch.nn as nn
import torch
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp
from layers.Attention import DCTAttention, DCT_NonAttention
from layers.RevIN import RevIN
from layers.local_global import MIC
import torch.nn.functional as F


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

        self.std_enc = None
        self.mean_enc = None


        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        self.enc_seasonal_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)
        self.dec_seasonal_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                         configs.dropout)


        # DCT_NonAttention
        enc_self_attention = DCT_NonAttention(activation=configs.activation,
                                          output_attention=configs.output_attention)

        dec_self_attention = DCT_NonAttention(activation=configs.activation,
                                          output_attention=configs.output_attention)

        dec_cross_attention = DCT_NonAttention(activation=configs.activation,
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
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            conv_layer=MIC(feature_size=configs.d_model, n_heads=configs.n_heads, decomp_kernel=configs.en_decomp_kernel,
                           conv_kernel=configs.en_conv_kernel, isometric_kernel=configs.en_isometric_kernel, device=self.device)
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
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            conv_layer=MIC(feature_size=configs.d_model, n_heads=configs.n_heads,
                           decomp_kernel=configs.decomp_kernel, conv_kernel=configs.conv_kernel, isometric_kernel=configs.isometric_kernel,
                           device=self.device)
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

    def forward(self, tau, delta, x_enc, x_mark_enc, x_dec, x_mark_dec,
                    enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # series decomposition
        seasonal_enc, trend_enc = self.decomp(x_enc)
        seasonal_dec, _ = self.decomp(x_dec)
        seasonal_dec = F.pad(seasonal_dec[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # seasonal
        enc_out = self.enc_seasonal_embedding(seasonal_enc, None)
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_seasonal_embedding(seasonal_dec, None)
        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask,
                                                     cross_mask=dec_enc_mask, tau=tau, delta=delta)
        seasonal_out = seasonal_out[:, -self.pred_len:, :]
        seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        # trend
        trend_enc = self.revin_trend(trend_enc, 'norm')
        trend_out = self.trend(trend_enc.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.revin_trend(trend_out, 'denorm')

        # output
        dec_out = trend_out + seasonal_ratio * seasonal_out

        if self.output_attention:
            return dec_out, (attn_e, attn_d)
        elif self.output_stl:
            return dec_out, trend_enc, seasonal_enc, trend_out, seasonal_ratio * seasonal_out
        else:
            return dec_out

    def dishTS_non(self, x_enc):
        x_raw = x_enc.clone().detach()
        # Normalization
        self.mean_enc = x_raw.mean(1, keepdim=True).detach()  # B x 1 x E
        enc_new = x_raw - self.mean_enc
        self.std_enc = torch.sqrt(torch.var(enc_new, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        enc_new = enc_new / self.std_enc
        tau = self.tau_learner(x_raw, self.std_enc).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, self.mean_enc)  # B x S x E, B x 1 x E -> B x S
        return enc_new, tau, delta

    def dishTS_non_seasonl(self, seasonal_x):
        self.preget(seasonal_x)
        seasonal_new = self.forward_process(seasonal_x)
        return seasonal_new

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2, 0, 1)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
        self.xil = torch.sum(torch.pow(batch_x - self.phil, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)


    def forward_process(self, batch_input):
        # print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil) / torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst

    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih

    