import torch
import torch.nn as nn
import numpy as np

import FrEIA.framework as Ff

from .. import InvertibleArchitecture

import time

class InvertibleImagenetClassifier(InvertibleArchitecture):

    def __init__(self, lr, mu_init, mu_conv_init, mu_low_rank_k, input_dims, n_classes, n_loss_dims_1d, n_total_dims_1d, backbone: InvertibleArchitecture, head: InvertibleArchitecture, finetune_mu=False):
        super().__init__()

        self.model = None

        self.lr = lr

        self.n_classes = n_classes

        self.backbone = backbone
        self.head = head

        self.construct_inn(Ff.InputNode(input_dims[0], input_dims[1], input_dims[2], name='input'), backbone, head)

        self.n_total_dims_1d = n_total_dims_1d
        self.n_loss_dims_1d = n_loss_dims_1d

        init_scale = mu_init / np.sqrt(2 * (n_loss_dims_1d // n_classes))
        self.mu_fc = nn.Parameter(torch.zeros(1, n_classes, n_loss_dims_1d))
        for k in range(n_loss_dims_1d // n_classes):
            self.mu_fc.data[0, :, n_classes * k: n_classes * (k + 1)] = init_scale * torch.eye(n_classes)

        self.mu_low_rank_k = mu_low_rank_k

        if self.mu_low_rank_k > 0:
            mu_conv_dims = n_total_dims_1d - n_loss_dims_1d

            self.mu_t = nn.Parameter(mu_conv_init * torch.randn(self.n_classes, self.mu_low_rank_k).cuda())
            self.mu_m = nn.Parameter(mu_conv_init * torch.randn(self.mu_low_rank_k, mu_conv_dims).cuda())
        else:
            self.mu_conv = nn.Parameter(mu_conv_init * torch.randn(1, n_classes, n_total_dims_1d - n_loss_dims_1d))

        self.train_mu  = True
        self.train_phi = False
        self.train_inn = True
        self.model_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))

        self.finetune_mu = finetune_mu
        if finetune_mu:
            self.optimizer_params = [{'params': self.model_parameters, 'lr': 0 * self.lr, 'weight_decay':0.}]
        else:
            self.optimizer_params = [{'params': self.model_parameters, 'lr': 1 * self.lr, 'weight_decay':0.}]

        if self.train_mu:
            self.optimizer_params.append({'params': [self.mu_fc],
                                     'lr': 1. * self.lr,
                                     'weight_decay': 0.})

            self.optimizer_params.append({'params': [self.mu_m, self.mu_t] if self.mu_low_rank_k > 0 else [self.mu_conv],
                                     'lr': 1. * self.lr,
                                     'weight_decay': 0.})
        if self.train_phi:
            self.optimizer_params.append({'params': [self.phi],
                                     'lr': 1. * self.lr,
                                     'weight_decay': 0.})

        self.optimizer = torch.optim.SGD(self.optimizer_params, self.lr, momentum=0.9, weight_decay=1e-5)

    def construct_inn(self, input, backbone: InvertibleArchitecture, head: InvertibleArchitecture):

        nodes = []
        split_nodes = []

        nodes.append(input)

        backbone_nodes, backbone_split_nodes, skip_connections = backbone.construct_inn(nodes[-1])

        nodes += backbone_nodes

        if skip_connections:
            print("HAS SKIP CONNECTION")
            head_nodes, head_split_nodes = head.construct_inn(nodes[-1], skip_connections)
            split_nodes += backbone_split_nodes
        else:
            head_nodes, head_split_nodes = head.construct_inn(nodes[-1])

        nodes.append(Ff.OutputNode(head_nodes[-1], name='out_fc'))

        nodes += head_nodes
        split_nodes += head_split_nodes

        self.model = Ff.ReversibleGraphNet(nodes + split_nodes, verbose=True)
        print(self.model)

        return nodes

    def calc_mu_conv(self):
        self.mu_conv = torch.mm(self.mu_t, self.mu_m).unsqueeze(0)

    def cluster_distances(self, z, mu):
        z_i_z_i = torch.sum(z**2, dim=1, keepdim=True) # batchsize x 1
        mu_j_mu_j = torch.sum(mu**2, dim=2)       # 1 x n_classes
        z_i_mu_j = torch.mm(z, mu.squeeze().t())  # batchsize x n_classes

        return -2 * z_i_mu_j + z_i_z_i + mu_j_mu_j

    def forward(self, x, y=None):
        if self.finetune_mu:
            with torch.no_grad():
                z_fc, z_conv = self.model(x)
                jac = self.model.log_jacobian(run_forward=False)
        else:
            z_fc, z_conv = self.model(x)
            jac = self.model.log_jacobian(run_forward=False)

        if self.mu_low_rank_k > 0:
            self.calc_mu_conv()

        cluster_distances = self.cluster_distances(z_fc, self.mu_fc)
        cluster_distances += self.cluster_distances(z_conv, self.mu_conv)

        losses = {'nll_joint_tr': ((- torch.logsumexp(- 0.5 * cluster_distances, dim=1)) - jac) / self.n_total_dims_1d, 'logits_tr': - 0.5 * cluster_distances}

        if y is not None:
            losses['nll_class_tr'] = ((0.5 * torch.sum(cluster_distances * y, dim=1)) - jac) / self.n_total_dims_1d
            losses['cat_ce_tr'] = - torch.sum((torch.log_softmax(- 0.5 * cluster_distances, dim=1)) * y, dim=1)
            losses['acc_tr'] = torch.mean((torch.argmax(y, dim=1) == torch.argmax(-cluster_distances, dim=1)).float())

            for lname in ['nll_joint_tr', 'nll_class_tr', 'cat_ce_tr', 'acc_tr']:
                losses[lname] = torch.mean(losses[lname])

        return losses

    def mu_pairwise_dist(self):
        distances = []
        for mu in [self.mu_fc, self.mu_conv]:
            mu_i_mu_j = mu.squeeze().mm(mu.squeeze().t())
            mu_i_mu_i = torch.sum(mu.squeeze()**2, 1, keepdim=True).expand(self.n_classes, self.n_classes)

            dist = mu_i_mu_i + mu_i_mu_i.t() - 2 * mu_i_mu_j
            dist = torch.masked_select(dist, (1 - torch.eye(self.n_classes).cuda()).byte()).clamp(min=0.)
            distances.append(dist)
        return distances[0] + distances[1]

    def validate(self, x, y):
        with torch.no_grad():

            losses = self.forward(x, y)

            nll_joint, nll_class, cat_ce, acc = (losses['nll_joint_tr'], losses['nll_class_tr'], losses['cat_ce_tr'], losses['acc_tr'])
            mu_dist = torch.mean(torch.sqrt(self.mu_pairwise_dist()))

        return {'nll_joint_val': nll_joint,
                'nll_class_val': nll_class,
                'cat_ce_val':    cat_ce,
                'acc_val':       acc,
                'delta_mu_val':  mu_dist}

    def sample(self, y, temperature=1.):
        z = temperature * torch.randn(y.shape[0], self.n_loss_dims_1d).cuda()
        mu = torch.sum(y.view(-1, self.n_classes, 1) * self.mu, dim=1)
        z = z + mu
        return self.inn(z, rev=True)

    def save(self, fname):
        if self.mu_low_rank_k > 0:
            torch.save({'inn': self.model.state_dict(),
                        'mu': self.mu_fc,
                        'mu_t': self.mu_t,
                        'mu_m': self.mu_m,
                        'opt': self.optimizer.state_dict()}, fname)
        else:
            torch.save({'inn': self.model.state_dict(),
                        'mu': self.mu_fc,
                        'mu_conv': self.mu_conv,
                        'opt': self.optimizer.state_dict()}, fname)


    def load(self, fname):
        data = torch.load(fname)
        self.model.load_state_dict(data['inn'], strict=True)
        self.mu_fc.data.copy_(data['mu'].data)

        if self.mu_low_rank_k > 0:
            self.mu_t.data.copy_(data['mu_t'].data)
            self.mu_m.data.copy_(data['mu_m'].data)
        else:
            self.mu_conv.data.copy_(data['mu_conv'].data)

        try:
            self.optimizer.load_state_dict(data['opt'])
        except:
            print('loading the optimizer went wrong, skipping')

        self.calc_mu_conv()


