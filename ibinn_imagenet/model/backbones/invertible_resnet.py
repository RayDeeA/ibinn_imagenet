from enum import Enum

import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from ..coupling_blocks.slow import AIO_SlowCouplingBlock
from ..coupling_blocks.glow import AIO_GlowCouplingBlock
from ..coupling_blocks.downsampling import AIO_DownsampleCouplingBlock

from .. import InvertibleArchitecture
from .. import CouplingType


class InvertibleResNet(InvertibleArchitecture):

    class BlockType(Enum):
        BASIC = 1
        BOTTLENECK = 2

    def __init__(
            self,
            base_width,
            coupling_type=CouplingType.SLOW,
            block_type=BlockType.BOTTLENECK,
            clamp=1.,
            act_norm=0.25,
            permute_soft=True,
            welling=False,
            householder=0,
            act_norm_type='SOFTPLUS',
            blocks=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            dilations=[1, 1, 1, 1],
            synchronized_batchnorm=False,
            skip_connection=False
    ):
        super().__init__()

        self.model = None

        self.base_width = base_width
        self.block_type = block_type

        self.clamp = clamp
        self.act_norm = act_norm

        self.slow_coupling_args = \
        {
            'subnet_constructor': None,
            'clamp': self.clamp,
            'act_norm': self.act_norm,
            'act_norm_type': act_norm_type,
            'gin_block': False,
            'permute_soft': permute_soft,
            'welling_permutation': welling,
            'learned_householder_permutation': householder
        }

        self.glow_coupling_args = \
        {
            'subnet_constructor': None,
            'clamp': self.clamp,
            'act_norm': self.act_norm,
            'act_norm_type': act_norm_type,
            'permute_soft': permute_soft
        }
        self.down_coupling_args = \
        {
            'clamp': self.clamp,
            'act_norm': self.act_norm,
            'act_norm_type': act_norm_type,
            'permute_soft': permute_soft
        }
        self.downsampling_layer = AIO_DownsampleCouplingBlock
        self.coupling_args = self.glow_coupling_args if coupling_type == CouplingType.GLOW else self.slow_coupling_args
        self.coupling_layer = AIO_GlowCouplingBlock if coupling_type == CouplingType.GLOW else AIO_SlowCouplingBlock

        self.block = 0
        self.blocks = blocks
        self.strides = strides
        self.dilations = dilations

        if synchronized_batchnorm:
            from ...utils.synced_batchnorm.synced_batchnorm import SynchronizedBatchNorm2d

        self.BatchNorm = SynchronizedBatchNorm2d if synchronized_batchnorm else nn.BatchNorm2d
        self.skip_connection = skip_connection

        self.channels = 3
        self.dilation = 1
        self.groups = 1

    def _basic_residual_block(self, planes, stride=1, groups=1, base_width=64, dilation=1):

        def basic_residual_block_internal(cin, cout):
            width = int(planes * (base_width / 64.)) * groups

            layers = nn.Sequential(
                self.BatchNorm(cin, track_running_stats=True, momentum=0.05),
                nn.ReLU(),

                nn.Conv2d(cin, width, kernel_size=3, padding=1, bias=False),
                self.BatchNorm(width, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, dilation=dilation, bias=False),
                self.BatchNorm(width, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(width, cout, 1, padding=0)
            )

            layers.apply(self._weights_init)

            return layers

        return basic_residual_block_internal

    def _bottleneck_residual_block(self, planes, stride=1, groups=1, base_width=64, dilation=1):

        def get_bottleneck_internal(cin, cout):
            width = int(planes * (base_width / 64.)) * groups

            layers = nn.Sequential(
                self.BatchNorm(cin, track_running_stats=True, momentum=0.05),
                nn.ReLU(),

                nn.Conv2d(cin, width, kernel_size=1, padding=0, bias=False),
                self.BatchNorm(width, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, dilation=dilation, bias=False),
                self.BatchNorm(width, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(width, width * 4, kernel_size=1, padding=0, bias=False),
                self.BatchNorm(width * 4, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(width * 4, cout, kernel_size=1, padding=0),
            )

            layers.apply(self._weights_init)

            return layers

        return get_bottleneck_internal

    def _entry_flow(self, input):

        nodes = []

        def _entry_flow_block(cin, cout):

            layers = nn.Sequential(
                nn.Conv2d(cin, self.base_width, kernel_size=7, stride=1, padding=3, dilation=1, bias=False),
                self.BatchNorm(self.base_width, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(self.base_width, cout, 1, padding=0)
            )

            layers.apply(self._weights_init)

            return layers

        def _entry_flow_block_strided(cin, cout):
            layers = nn.Sequential(
                nn.Conv2d(cin, self.base_width, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                self.BatchNorm(self.base_width, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(self.base_width, self.base_width, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                self.BatchNorm(self.base_width, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(self.base_width, cout, 1, padding=0)
            )

            layers.apply(self._weights_init)

            return layers

        def _entry_flow_block_strided_beta(cin, cout):
            layers = nn.Sequential(
                nn.Conv2d(cin, self.base_width, kernel_size=3, stride=1, padding=1, bias=False),
                self.BatchNorm(self.base_width, track_running_stats=True, momentum=0.05),
                nn.ReLU(),
                nn.Conv2d(self.base_width, cout, kernel_size=1, stride=1, padding=0, bias=False),
            )

            layers.apply(self._weights_init)

            return layers

        entry_flow = Ff.Node(input, self.downsampling_layer, dict(self.down_coupling_args, subnet_constructor_low_res=_entry_flow_block_strided_beta, subnet_constructor_strided=_entry_flow_block_strided), name='Strided entry_flow')
        self.channels *= 4  # Downsampling by factor 2 leads to 4 time increase of channels
        nodes.append(entry_flow)

        entry_flow_pool = Ff.Node(entry_flow, Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance': 0.5}, name='downsampling entry_flow')
        nodes.append(entry_flow_pool)
        self.channels *= 4

        entry_flow_pool = Ff.Node(entry_flow_pool, self.coupling_layer, dict(self.coupling_args, subnet_constructor=self._conv_1x1(int(self.channels))), name='downsampling entry_flow 1x1')
        nodes.append(entry_flow_pool)

        return nodes

    def _residual_flow(self, input, planes, blocks, stride, dilation, create_skip_connection=False):

        self.block += 1

        nodes = []
        split_nodes = []

        if stride == 2:
            bottleneck_strided = self._bottleneck_residual_block(planes, 2, groups=self.groups, base_width=self.base_width, dilation=dilation)
            bottleneck = self._bottleneck_residual_block(planes, 1, groups=self.groups, base_width=self.base_width, dilation=dilation)
            middle_flow = Ff.Node(input, self.downsampling_layer, dict(self.down_coupling_args, subnet_constructor_low_res=bottleneck, subnet_constructor_strided=bottleneck_strided), name='Strided entry_flow')
            self.channels *= 4
            nodes.append(middle_flow)
        else:
            bottleneck = self._bottleneck_residual_block(planes, 1, groups=self.groups, base_width=self.base_width, dilation=dilation)
            middle_flow = Ff.Node(input,  self.coupling_layer, dict(self.coupling_args, subnet_constructor=bottleneck), name=f'middle_flow %s_1 ' % self.block)
            nodes.append(middle_flow)

        low_level_node = None
        for i in range(1, blocks):

            if create_skip_connection and i == blocks - 1:

                n_downstream_channels = int(self.channels * (3/4))

                n_skip_channels = int(self.channels-n_downstream_channels)
                self.channels = n_downstream_channels

                split = Ff.Node(middle_flow, Fm.Split1D, {'dim': 0, 'split_size_or_sections': (n_downstream_channels, n_skip_channels)}, name=f'middle_flow split %s_%s ' % (self.block, (1+i)))
                split_nodes.append(split)

                bottleneck = self._bottleneck_residual_block(planes, 1, self.groups, self.base_width, dilation)
                middle_flow = Ff.Node(split.out0,  self.coupling_layer, dict(self.coupling_args, subnet_constructor=bottleneck), name=f'middle_flow %s_%s ' % (self.block, (1+i)))
                nodes.append(middle_flow)

                low_level_node = split.out1

            else:
                bottleneck = self._bottleneck_residual_block(planes, 1, self.groups, self.base_width, dilation)
                middle_flow = Ff.Node(middle_flow,  self.coupling_layer, dict(self.coupling_args, subnet_constructor=bottleneck), name=f'middle_flow %s_%s ' % (self.block, (1+i)))
                nodes.append(middle_flow)

        return nodes, split_nodes, low_level_node

    def construct_inn(self, input):

        i_resnet_nodes = []
        split_nodes = []

        i_resnet_nodes += self._entry_flow(input)

        middle_flow_low_level_node = None
        if self.skip_connection:
            middle_flow_nodes, middle_flow_split_nodes, middle_flow_low_level_node = self._residual_flow(i_resnet_nodes[-1], 64, self.blocks[0], self.strides[0], self.dilations[0], create_skip_connection=self.skip_connection)
            i_resnet_nodes += middle_flow_nodes
            split_nodes += middle_flow_split_nodes
        else:
            i_resnet_nodes += self._residual_flow(i_resnet_nodes[-1], 64, self.blocks[0], self.strides[0], self.dilations[0])[0]

        i_resnet_nodes += self._residual_flow(i_resnet_nodes[-1], 128, self.blocks[1], self.strides[1], self.dilations[1])[0]
        i_resnet_nodes += self._residual_flow(i_resnet_nodes[-1], 256, self.blocks[2], self.strides[2], self.dilations[2])[0]
        i_resnet_nodes += self._residual_flow(i_resnet_nodes[-1], 512, self.blocks[3], self.strides[3], self.dilations[3])[0]

        return i_resnet_nodes, split_nodes, middle_flow_low_level_node

