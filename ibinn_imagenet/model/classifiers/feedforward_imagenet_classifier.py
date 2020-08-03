import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

class FeedForwardImagenetClassifier(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.model = None

        self.lr = 0.0
        self.n_classes = 1000

        self.mu_fc = nn.Parameter(torch.zeros(1, self.n_classes, 1))
        self.mu_conv = nn.Parameter(torch.zeros(1, self.n_classes, 1))

        self.model = torchvision.models.resnet50(pretrained=True)
        self.model_parameters = list(self.model.parameters())
        self.n_total_dims_1d = 1

    def cluster_distances(self, z, mu):
        return torch.zeros(z.shape[0], mu.shape[1]).to(z.device)

    def forward_l2_attack(self, x, y=None):
        return self.model(x.cuda())

    def forward(self, x, y=None):
        z = self.model(x)

        model_conv = nn.Sequential(*list(self.model.children())[:-2])
        z_conv = model_conv(x)
        bs = z_conv.shape[0]
        z_conv = z_conv.view((bs, -1))
        z_fc = torch.zeros((bs, 1)).cuda()

        losses = {'nll_joint_tr': torch.tensor([0.0]).to(x.device),
                  'logits_tr': z, 'z_conv': z_conv, 'z_fc': z_fc}

        losses['nll_class_tr'] = torch.tensor([0.0]).cuda()
        losses['cat_ce_tr'] = torch.tensor([0.0]).cuda()
        losses['acc_tr'] = torch.tensor([0.0]).cuda()

        if y is not None:
            cluster_distances = torch.zeros(y.shape).to(y.device)
            losses['nll_class_tr'] = 0.0
            losses['cat_ce_tr'] = - torch.sum((torch.log_softmax(z, dim=1)) * y, dim=1)
            losses['acc_tr'] = torch.mean((torch.argmax(y, dim=1) == torch.argmax(z, dim=1)).float())

        for lname in ['nll_joint_tr', 'nll_class_tr', 'cat_ce_tr', 'acc_tr']:
            losses[lname] = torch.mean(losses[lname])

        return losses

    def mu_pairwise_dist(self):
        return torch.tensor([0.0]).cuda()

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
