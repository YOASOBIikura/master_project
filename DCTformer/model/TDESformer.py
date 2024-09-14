import torch.nn as nn
import torch
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.Attention import FourierAttention, FullAttention
from layers.RevIN import RevIN
import torch.nn.functional as F
from layers.ETSlayers import Encoder, Decoder, EncoderLayer, DecoderLayer, series_decomp, GrowthLayer, AttentionLayer

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.output_stl = configs.output_stl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(c_in=configs.enc_in, d_model=configs.d_model,
                                                           embed_type=configs.embed, freq=configs.freq,
                                                           dropout=configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(c_in=configs.enc_in, d_model=configs.d_model,
                                                           embed_type=configs.embed, freq=configs.freq,
                                                           dropout=configs.dropout)

        enc_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                              output_attention=configs.output_attention)
        dec_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                              output_attention=configs.output_attention)
        dec_cross_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                               output_attention=configs.output_attention)

        enc_es_attention = GrowthLayer(d_model=configs.d_model, nhead=configs.n_heads, dropout=configs.dropout,
                                       output_attention=configs.output_attention)
        dec_es1_attention = GrowthLayer(d_model=configs.d_model, nhead=configs.n_heads, dropout=configs.dropout,
                                        output_attention=configs.output_attention)
        dec_es2_attention = GrowthLayer(d_model=configs.d_model, nhead=configs.n_heads, dropout=configs.dropout,
                                        output_attention=configs.output_attention)

        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        enc_self_attention,
                        configs.d_model),
                    enc_es_attention,
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
                    dec_es1_attention,
                    dec_es2_attention,
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    c_out=configs.c_out
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # Encoder
        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len+configs.label_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )

        self.Projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        self.revin_trend = RevIN(configs.enc_in).to(self.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask, tau=None, delta=None)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part, attn_d = self.seasonal_decoder(dec_out, enc_out, trend_init, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 tau=None, delta=None)
        # seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        seasonal_ratio = seasonal_init.abs().mean(dim=1) / seasonal_part.abs().mean(dim=1)
        seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        # trend
        # trend_enc = self.revin_trend(trend_part, 'norm')
        trend_out = self.trend(trend_part.permute(0, 2, 1)).permute(0, 2, 1)
        # trend_out = self.revin_trend(trend_out, 'denorm')

        # final
        seasonal_part = seasonal_part[:, -self.pred_len:, :]
        dec_out = trend_out + seasonal_ratio * seasonal_part

        if self.output_attention:
            return dec_out, (attn_e, attn_d)
        elif self.output_stl:
            return dec_out, trend_out, seasonal_init, trend_out, seasonal_ratio * seasonal_part
        else:
            return dec_out
