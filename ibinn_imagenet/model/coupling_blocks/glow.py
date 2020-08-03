from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import special_ortho_group

class GlowCouplingBlock(nn.Module):

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=2.):
        super().__init__()

        channels = dims_in[0][0]
        if dims_c:
            raise ValueError('does not support conditioning yet')

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.splits = [self.split_len1, self.split_len2]

        self.in_channels = channels
        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = False
        condition_length = 0

        self.s1 = subnet_constructor(self.split_len1, 2 * self.split_len2)
        self.s2 = subnet_constructor(self.split_len2, 2 * self.split_len1)

        self.last_jac = None

    def log_e(self, s):
        return self.clamp * torch.tanh(0.2 * s)

    def affine(self, x, a, rev=False):
        ch = x.shape[1]
        sub_jac = self.log_e(a[:,:ch])
        if not rev:
            return (x * torch.exp(sub_jac) + a[:,ch:],
                    torch.sum(sub_jac, dim=(1,2,3)))
        else:
            return ((x - a[:,ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=(1,2,3)))

    def forward(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if not rev:
            a1 = self.s1(x1)
            y2, j2 = self.affine(x2, a1)

            a2 = self.s2(y2)
            y1, j1 = self.affine(x1, a2)

        else: # names of x and y are swapped!
            a2 = self.s2(x2)
            y1, j1 = self.affine(x1, a2, rev=True)

            a1 = self.s1(y1)
            y2, j2 = self.affine(x2, a1, rev=True)

        self.last_jac = j1 + j2
        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims


class AIO_GlowCouplingBlock(GlowCouplingBlock):

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None,
                                           clamp=2.,
                                           act_norm=1.,
                                           act_norm_type='SOFTPLUS',
                                           permute_soft=False):

        super().__init__(dims_in, dims_c=dims_c, subnet_constructor=subnet_constructor, clamp=clamp)

        if act_norm_type == 'SIGMOID':
            act_norm = np.log(act_norm)
            self.actnorm_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif act_norm_type == 'SOFTPLUS':
            act_norm = 10. * act_norm
            self.softplus = nn.Softplus(beta=0.5)
            self.actnorm_activation = (lambda a: 0.1 * self.softplus(a))
        elif act_norm_type == 'EXP':
            act_norm = np.log(act_norm)
            self.actnorm_activation = (lambda a: torch.exp(a))
        else:
            raise ValueError('Please, SIGMOID, SOFTPLUS or EXP, as actnorm type')

        assert act_norm > 0., "please, this is not allowed. don't do it. take it... and go."
        channels = self.in_channels

        self.act_norm = nn.Parameter(torch.ones(1, channels, 1, 1) * float(act_norm))
        self.act_offset = nn.Parameter(torch.zeros(1, channels, 1, 1))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels,channels))
            for i,j in enumerate(np.random.permutation(channels)):
                w[i,j] = 1.
        w_inv = w.T

        self.w = nn.Parameter(torch.FloatTensor(w).view(channels, channels, 1, 1),
                              requires_grad=False)
        self.w_inv = nn.Parameter(torch.FloatTensor(w_inv).view(channels, channels, 1, 1),
                              requires_grad=False)

    def permute(self, x, rev=False):
        scale = self.actnorm_activation( self.act_norm)
        if rev:
            return (F.conv2d(x, self.w_inv) - self.act_offset) / scale
        else:
            return F.conv2d(x * scale + self.act_offset, self.w)

    def forward(self, x, c=[], rev=False):
        if rev:
            x = [self.permute(x[0], rev=True)]

        x_out = super().forward(x, c=[], rev=rev)[0]

        if not rev:
            x_out = self.permute(x_out, rev=False)

        n_pixels = x_out.shape[2] * x_out.shape[3]
        self.last_jac += ((-1)**rev * n_pixels) * (torch.log(self.actnorm_activation(self.act_norm) + 1e-12).sum())
        return [x_out]


if __name__ == '__main__':
    import numpy as np
    N = 8
    c = 32
    x = torch.FloatTensor(128, c, N, N)
    x.normal_(0,1)

    def constr(c_in, c_out):
        layer = torch.nn.Conv2d(c_in, c_out, 1)
        layer.weight.data *= 0.
        layer.bias.data *= 0.
        return layer

    actnorm = 5.26
    layer = AIO_GlowCouplingBlock([(c, N, N)],
                                  subnet_constructor=constr,
                                  clamp=2.,
                                  act_norm=actnorm,
                                  permute_soft=True)

    transf = layer([x])
    jac = layer.jacobian([x])
    x_inv = layer(transf, rev=True)[0]

    err = torch.abs(x - x_inv)

    print(transf[0].shape)
    print(jac.mean().item(), np.log(actnorm) * x.numel() / 128)
    print(err.max().item())
    print(err.mean().item())


