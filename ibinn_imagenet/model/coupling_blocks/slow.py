import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group

class AIO_SlowCouplingBlock(nn.Module):
    ''' This elegant coupling block was invented by Jakob Kruse '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor=None,
                 clamp=2.,
                 gin_block=False,
                 act_norm=1.,
                 act_norm_type='SOFTPLUS',
                 permute_soft=False,
                 learned_householder_permutation=0,
                 welling_permutation=False):

        super().__init__()

        channels = dims_in[0][0]
        if dims_c:
            raise ValueError('does not support conditioning yet')

        self.split_len1 = channels - channels // 2
        self.split_len2 = channels // 2
        self.splits = [self.split_len1, self.split_len2]

        self.in_channels = channels
        self.clamp = clamp
        self.GIN = gin_block
        self.welling_perm = welling_permutation
        self.householder = learned_householder_permutation

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
        self.act_norm = nn.Parameter(torch.ones(1, self.in_channels, 1, 1) * float(act_norm))
        self.act_offset = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))


        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels,channels))
            for i,j in enumerate(np.random.permutation(channels)):
                w[i,j] = 1.

        if self.householder:
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w = None
            self.w_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w = nn.Parameter(torch.FloatTensor(w).view(channels, channels, 1, 1),
                                  requires_grad=False)
            self.w_inv = nn.Parameter(torch.FloatTensor(w.T).view(channels, channels, 1, 1),
                                  requires_grad=False)

        self.conditional = False
        condition_length = 0

        self.s = subnet_constructor(self.split_len1, 2 * self.split_len2)
        self.last_jac = None

    def construct_householder_permutation(self):
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).cuda() - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        return w.unsqueeze(2).unsqueeze(3), w.t().contiguous().unsqueeze(2).unsqueeze(3)

    def log_e(self, s):
        s = self.clamp * torch.tanh(0.1 * s)
        if self.GIN:
            s -= torch.mean(s, dim=(1,2,3), keepdim=True)
        return s

    def permute(self, x, rev=False):
        scale = self.actnorm_activation( self.act_norm)
        if rev:
            return (F.conv2d(x, self.w_inv) - self.act_offset) / scale
        else:
            return F.conv2d(x * scale + self.act_offset, self.w)

    def pre_permute(self, x, rev=False):
        if rev:
            return F.conv2d(x, self.w)
        else:
            return F.conv2d(x, self.w_inv)

    def affine(self, x, a, rev=False):
        ch = x.shape[1]
        sub_jac = self.log_e(a[:,:ch])
        if not rev:
            return (x * torch.exp(sub_jac) + 0.1 * a[:,ch:],
                    torch.sum(sub_jac, dim=(1,2,3)))
        else:
            return ((x - 0.1 * a[:,ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=(1,2,3)))

    def forward(self, x, c=[], rev=False):
        if self.householder:
            self.w, self.w_inv = self.construct_householder_permutation()

        if rev:
            x = [self.permute(x[0], rev=True)]
        elif self.welling_perm:
            x = [self.pre_permute(x[0], rev=False)]

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if not rev:
            a1 = self.s(x1)
            x2, j2 = self.affine(x2, a1)
        else: # names of x and y are swapped!
            a1 = self.s(x1)
            x2, j2 = self.affine(x2, a1, rev=True)

        self.last_jac = j2
        x_out = torch.cat((x1, x2), 1)

        n_pixels = x_out.shape[2] * x_out.shape[3]
        self.last_jac += ((-1)**rev * n_pixels) * (torch.log(self.actnorm_activation(self.act_norm) + 1e-12).sum())

        if not rev:
            x_out = self.permute(x_out, rev=False)
        elif self.welling_perm:
            x_out = self.pre_permute(x_out, rev=True)

        return [x_out]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims

if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    np.set_printoptions(precision=2, linewidth=300)

    N = 8
    c = 48
    x = torch.FloatTensor(128, c, N, N)
    z = torch.FloatTensor(128, c, N, N)
    x.normal_(0,1)
    z.normal_(0,1)

    def constr(c_in, c_out):
        layer = torch.nn.Conv2d(c_in, c_out, 1)
        layer.weight.data *= 0
        layer.bias.data *= 0
        return layer

    actnorm = 1.25
    layer = AIO_SlowCouplingBlock([(c, N, N)],
                                  subnet_constructor=constr,
                                  clamp=2.,
                                  gin_block=False,
                                  act_norm=actnorm,
                                  permute_soft=True,
                                  learned_householder_permutation=3,
                                  welling_permutation=False)

    transf = layer([x])
    jac = layer.jacobian([x])
    x_inv = layer(transf, rev=True)[0]

    err = torch.abs(x - x_inv)

    mean_jac = (jac.mean()/ x.numel() * 128).item()
    print('jac true/actual', np.log(actnorm), mean_jac)

    print(err.max().item())
    print(err.mean().item())

    print('see if householder refelction trains')

    print('before:')
    print(layer.vk_householder[0].data.cpu().numpy()[:10])

    optim = torch.optim.SGD([layer.vk_householder], lr=1.0)
    for i in tqdm(range(100)):
        loss = z - layer([x])[0]
        loss = torch.mean(loss**2)
        loss.backward()
        optim.step()
        optim.zero_grad()

    print('after:')
    print(layer.vk_householder[0].data.cpu().numpy()[:10])
