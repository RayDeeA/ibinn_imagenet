from enum import Enum

import torch
import torch.nn as nn

from ibinn_imagenet.utils.synced_batchnorm.synced_batchnorm import SynchronizedBatchNorm2d

import numpy as np

class CouplingType(Enum):
    GLOW = 1
    SLOW = 2

class InvertibleArchitecture(nn.Module):

    def construct_inn(self, input, *kwargs):
        pass

    def forward(self, input):
        pass

    def _partial_zeros_weights_init(self, n_dims_non_zeros, n_dims_total):

        def internal(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:

                non_zeros_partial = m.weight[:, :n_dims_non_zeros, :, :]
                zeros_partial = m.weight[:, n_dims_non_zeros:n_dims_total, :, :]

                torch.nn.init.kaiming_normal_(non_zeros_partial)
                torch.nn.init.zeros_(zeros_partial)

                try:
                    torch.nn.init.zeros_(m.bias)
                except:
                    pass

        return internal

    def _weights_init(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n) * 0.0001)
            # m.weight.data.normal_(0, 0.005)
            torch.nn.init.kaiming_normal_(m.weight)

            # torch.nn.init.xavier_uniform(m.weight)
            #m.weight.data *= 0.1
            try:
                torch.nn.init.zeros_(m.bias)
            except:
                pass
        if type(m) == nn.BatchNorm2d or type(m) == SynchronizedBatchNorm2d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def _weights_init_linear(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n) * 0.0001)
            # m.weight.data.normal_(0, 0.005)
            torch.nn.init.kaiming_normal_(m.weight)

            # torch.nn.init.xavier_uniform(m.weight)
            #m.weight.data *= 0.1
            try:
                torch.nn.init.zeros_(m.bias)
            except:
                pass
        if type(m) == nn.BatchNorm2d or type(m) == SynchronizedBatchNorm2d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def _random_orthogonal(self, n, scalar=0.25):
        w = np.zeros((n, n))
        for i, j in enumerate(np.random.permutation(n)):
            w[i, j] = scalar
        return torch.FloatTensor(w)

    def _conv_1x1(self, n_kernels):
        def conv_1x1_internal(cin, cout):

            layers = nn.Sequential(
                nn.Conv2d(cin, n_kernels, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(n_kernels, cout, 1, padding=0),
            )

            layers.apply(self._weights_init)

            return layers

        return conv_1x1_internal

    def _conv_1x1_partial_zero_init(self, n_dims_non_zeros, n_dims_total, n_kernels):
        def conv_1x1_zero_init_internal(cin, cout):

            layers = nn.Sequential(
                nn.Conv2d(cin, n_kernels, 1, padding=0),
                nn.ReLU(),
                nn.Conv2d(n_kernels, cout, 1, padding=0),
            )

            layers.apply(self._partial_zeros_weights_init(n_dims_non_zeros, n_dims_total))

            return layers

        return conv_1x1_zero_init_internal
