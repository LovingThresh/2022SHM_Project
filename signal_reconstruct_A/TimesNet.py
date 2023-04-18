import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

from utils import dict2cls


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                    ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = x[:, -(length - (self.seq_len + self.pred_len)):, :]
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                   for _ in range(configs.e_layers)])
        # configs.freq其实无所谓
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.adjust_linear = nn.Linear(
            configs.enc_in, configs.c_out, bias=True)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        # x_enc_ = self.adjust_linear(x_enc)
        x_extend = x_enc[:, :, :1] + x_enc[:, :, 1:]
        x_enc = torch.cat([x_enc, x_extend], dim=-1)

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + torch.tensor(1e-5, device=x_enc.device))
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc=None):

        dec_out = self.forecast(x_enc, x_mark_enc)

        # return torch.sum(dec_out[:, -self.pred_len:, :], dim=-1, keepdim=True)
        return dec_out[:, -self.pred_len:, :]


# dic = {"seq_len": 256, "pred_len": 256, "freq": 'h', 'enc_in': 3, 'dec_in': 3, 'd_model': 128, 'embed': 'fixed',
#        'dropout': 0.1, 'e_layers': 2, 'c_out': 3, 'd_ff': 256, 'num_kernels': 6, 'top_k': 5}
# TimesNet_model_cfg = dict2cls(dic)
# TimesNet_model = TimesNet(TimesNet_model_cfg)
# pred = TimesNet_model(torch.randn(1, 256, 2))
#
# from thop import profile
# flops, params = profile(TimesNet_model, inputs=(torch.randn(1, 256, 4), None))
