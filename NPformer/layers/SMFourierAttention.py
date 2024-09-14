import torch
import torch.nn as nn
import math
from math import sqrt


class FourierAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, flag=True):
        super(FourierAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.head_flag = flag

    def forward(self, queries, keys, values, attn_mask):
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        L_s = self.factor  # segment length
        scale = self.scale or 1. / sqrt(L_s * D_q)
        assert L_k == L_v
        assert D_q == D_k == D_v
        assert L_s <= L_q
        assert L_s <= L_v
        addition_len_q = L_q % L_s
        addition_len_v = L_v % L_s

        if self.head_flag:  # drop first
            queries = queries[:, addition_len_q:, ...]
            keys = keys[:, addition_len_v:, ...]
            values = values[:, addition_len_v:, ...]
            addition_Q = queries[:, :addition_len_q, ...] if addition_len_q != 0 else None
        else:  # drop last
            queries = queries[:, :-addition_len_q, ...] if addition_len_q != 0 else queries
            keys = keys[:, :-addition_len_v, ...] if addition_len_v != 0 else keys
            values = values[:, :-addition_len_v, ...] if addition_len_v != 0 else values
            addition_Q = queries[:, -addition_len_q:, ...] if addition_len_q != 0 else None

        seg_queries = queries.reshape(B, -1, L_s, H, D_q)  # (b, 5, l_s, h, d_q)
        seg_keys = keys.reshape(B, -1, L_s, H, D_q)  # (b, 3, l_s, h, d_q)
        seg_values = values.reshape(B, -1, L_s, H, D_v)  # (b, 3, l_s, h, d_v)

        seg_queries_fft = torch.fft.rfft(seg_queries, dim=-1, norm='ortho')
        seg_keys_fft = torch.fft.rfft(seg_keys, dim=-1, norm='ortho')
        seg_values_fft = torch.fft.rfft(seg_values, dim=-1, norm='ortho')

        correlation_scores = torch.einsum('bmlhd,bnlhd->bhmn', seg_queries_fft, torch.conj(seg_keys_fft)) / math.sqrt(D_q)
        A = torch.softmax(scale * correlation_scores.abs(), dim=-1)
        A = torch.complex(A, torch.zeros_like(A))
        V = torch.einsum('bhmn,bnlhd->bmlhd', A, seg_values_fft)
        V = torch.fft.irfft(V, n=D_v, dim=-1, norm='ortho')

        V = V.reshape(B, -1, H, D_v)
        V = self.dropout(V)
        if self.head_flag:
            if addition_Q is not None:
                V = torch.cat([addition_Q, V], 1)
        else:
            if addition_Q is not None:
                V = torch.cat([V, addition_Q], 1)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class PreFourierAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, flag=True):
        super(PreFourierAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.head_flag = flag

    def forward(self, queries, keys, values, attn_mask):
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        L_s = self.factor  # segment length
        scale = self.scale or 1. / sqrt(L_s * D_q)
        assert L_k == L_v
        assert D_q == D_k == D_v
        assert L_s <= L_q
        assert L_s <= L_v
        addition_len_q = L_q % L_s
        addition_len_v = L_v % L_s

        if self.head_flag:  # drop first
            queries = queries[:, addition_len_q:, ...]
            keys = keys[:, addition_len_v:, ...]
            values = values[:, addition_len_v:, ...]
            addition_Q = queries[:, :addition_len_q, ...] if addition_len_q != 0 else None
        else:  # drop last
            queries = queries[:, :-addition_len_q, ...] if addition_len_q != 0 else queries
            keys = keys[:, :-addition_len_v, ...] if addition_len_v != 0 else keys
            values = values[:, :-addition_len_v, ...] if addition_len_v != 0 else values
            addition_Q = queries[:, -addition_len_q:, ...] if addition_len_q != 0 else None

        seg_queries = queries.reshape(B, -1, L_s, H, D_q)  # (b, 5, l_s, h, d_q)
        seg_keys = keys.reshape(B, -1, L_s, H, D_q)  # (b, 3, l_s, h, d_q)
        seg_keys_pre = seg_keys[:, :-1, ...]  # (b, 2, l_s, h, d_q)
        seg_values = values.reshape(B, -1, L_s, H, D_v)  # (b, 3, l_s, h, d_v)
        seg_values_aft = seg_values[:, 1:, ...]  # (b, 2, l_s, h, d_v)

        _, s, _, _, _ = seg_keys_pre.shape
        if s != 0:
            seg_queries_fft = torch.fft.rfft(seg_queries, dim=-1, norm='ortho')
            seg_keys_pre_fft = torch.fft.rfft(seg_keys_pre, dim=-1, norm='ortho')
            seg_values_aft_fft = torch.fft.rfft(seg_values_aft, dim=-1, norm='ortho')

            correlation_scores = torch.einsum('bmlhd,bnlhd->bhmn', seg_queries_fft,
                                              torch.conj(seg_keys_pre_fft)) / math.sqrt(D_q)
            A = torch.softmax(scale * correlation_scores.abs(), dim=-1)
            A = torch.complex(A, torch.zeros_like(A))
            tmp_V = torch.einsum('bhmn,bnlhd->bmlhd', A, seg_values_aft_fft)
            tmp_V = torch.fft.irfft(tmp_V, n=D_v, dim=-1, norm='ortho')
            V = torch.roll(tmp_V, shifts=1, dims=1)
        else:
            V = torch.zeros_like(seg_queries, device=seg_queries.device)

        V = V.reshape(B, -1, H, D_v)
        V = self.dropout(V)
        if self.head_flag:
            if addition_Q is not None:
                V = torch.cat([addition_Q, V], 1)
        else:
            if addition_Q is not None:
                V = torch.cat([V, addition_Q], 1)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

"""
HS Fourier Attention
"""
class DSFourierAttention(nn.Module):
    def __init__(self, T=1, activation='softmax', output_attention=False):
        super(DSFourierAttention, self).__init__()
        print(' fourier attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = 'softmax'
        self.output_attention = output_attention
        self.T = T

    def forward(self, q, k, v, mask, tau, delta):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        _, S, H, E = k.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        xq_ft_ = torch.fft.rfft(xq, dim=-1, norm='ortho')
        xk_ft_ = torch.fft.rfft(xk, dim=-1, norm='ortho')
        xv_ft_ = torch.fft.rfft(xv, dim=-1, norm='ortho')

        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, torch.conj(xk_ft_)) / sqrt(E)

        if self.activation == 'softmax':
            xqk_ft = torch.softmax(xqk_ft.abs() / self.T, dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear':
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear_norm':
            mins_real = xqk_ft.real.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real = xqk_ft.real - mins_real
            sums_real = xqk_ft_real.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real /= sums_real

            mins_imag = xqk_ft.imag.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_imag = xqk_ft.imag - mins_imag
            sums_imag = xqk_ft_imag.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_imag /= sums_imag

            xqkv_ft_real = torch.einsum("bhxy,bhey->bhex", xqk_ft_real, xv_ft_.real)
            xqkv_ft_imag = torch.einsum("bhxy,bhey->bhex", xqk_ft_imag, xv_ft_.imag)
            xqkv_ft = torch.complex(xqkv_ft_real, xqkv_ft_imag)

        elif self.activation == 'linear_norm_abs':
            xqk_ft = xqk_ft.abs() / xqk_ft.abs().sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear_norm_real':
            mins_real = xqk_ft.real.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real = xqk_ft.real - mins_real
            sums_real = xqk_ft_real.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real /= sums_real

            xqk_ft = torch.complex(xqk_ft_real, torch.zeros_like(xqk_ft_real))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        out = torch.fft.irfft(xqkv_ft, n=L, dim=-1, norm='ortho')
        out = out * tau + delta
        out = out.permute(0, 3, 1, 2)

        if self.output_attention == False:
            return (out, None)
        else:
            return (out, (xqk_ft_real, xqk_ft_imag))







