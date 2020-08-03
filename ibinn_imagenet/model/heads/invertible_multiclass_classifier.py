import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from .. import InvertibleArchitecture
from .. import CouplingType

from ..coupling_blocks.slow import AIO_SlowCouplingBlock
from ..coupling_blocks.glow import AIO_GlowCouplingBlock

from ...utils.dct_transform import DCTPooling2d

class InvertibleMulticlassClassifier(InvertibleArchitecture):

    def __init__(
            self,
            fc_width,
            n_loss_dims_1d,
            n_total_dims_1d,
            coupling_type=CouplingType.SLOW,
            clamp=1.,
            act_norm=0.25,
            act_norm_type='SOFTPLUS',
            permute_soft=True
    ):
        super().__init__()

        self.fc_width = fc_width

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
        }

        self.glow_coupling_args = \
        {
            'subnet_constructor': None,
            'clamp': self.clamp,
            'act_norm': self.act_norm,
            'act_norm_type': act_norm_type,
            'permute_soft': permute_soft
        }

        self.coupling_args = self.glow_coupling_args if coupling_type == CouplingType.GLOW else self.slow_coupling_args
        self.coupling_layer = AIO_GlowCouplingBlock if coupling_type == CouplingType.GLOW else AIO_SlowCouplingBlock

        self.clamp = clamp

        self.n_loss_dims_1d = n_loss_dims_1d
        self.n_total_dims_1d = n_total_dims_1d

        self.model = None

    def scoring_block(self):

        def scoring_block_internal(cin, cout):
            block = nn.Sequential(
                nn.Conv2d(cin, self.fc_width, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.fc_width, cout, kernel_size=1, padding=0, bias=False)
            )

            block.apply(self._weights_init_linear)

            return block

        return scoring_block_internal

    def construct_inn(self, input):

        nodes = []
        split_nodes = []

        dctpooling = Ff.Node(input.out0, DCTPooling2d, {'rebalance': 0.5}, name='DCT')
        nodes.append(dctpooling)

        split_node = Ff.Node(dctpooling.out0, Fm.Split1D, {'split_size_or_sections': (self.n_loss_dims_1d, self.n_total_dims_1d - self.n_loss_dims_1d), 'dim': 0}, name='exit_flow split')
        split_nodes.append(split_node)

        output_node = Ff.OutputNode(split_node.out1, name="out_conv")
        nodes.append(output_node)

        random_permute = Ff.Node(split_node.out0, Fm.PermuteRandom, {'seed': 0}, name=f'PERM_FC_{0} 1')
        nodes.append(random_permute)

        return nodes, split_nodes

