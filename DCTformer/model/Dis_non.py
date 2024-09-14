import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, forecast_model, norm_model, args):
        super(Model, self).__init__()
        self.args = args
        self.fm = forecast_model
        self.nm = norm_model

    def forward(self, batch_x, dec_inp):
        if self.nm is not None:
            batch_x, dec_inp, tau, delta = self.nm(batch_x, 'forward', dec_inp)
        
        if 'former' in self.args.model:
            forecast = self.fm(tau, delta, batch_x, None, dec_inp, None)
        else:
            forecast = self.fm(batch_x)

        if self.nm is not None:
            forecast = self.nm(forecast, 'inverse')
        
        return forecast
