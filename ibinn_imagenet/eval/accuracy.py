import torch

from torchvision.transforms import transforms as T

import numpy as np

def accuracy(model, data):

    loader = data.val_loader

    accuracies = []

    for val_batch, (x, y) in enumerate(loader):

        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            val_losses = model.validate(x, y)

            accuracies.append(val_losses['acc_val'].item())

    accuracy = 100. * np.mean(accuracies)

    print('ACCURACY %5.2f' % (accuracy))

    return accuracy


def accuracy_10_crop(model, data, ff = False):

    accuracies = []

    for val_batch, (x, y) in enumerate(data.val_loader_10_crop):

        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()

            bs, ncrops, c, h, w = x.shape

            x = x.view(-1, c, h, w)

            if ff:
                logits = model.model(x)
                probs = torch.softmax(logits, dim=1)
            else:
                z_fc, z_conv = model.model(x)

                if model.mu_low_rank_k > 0:
                    model.calc_mu_conv()

                cluster_distances = model.cluster_distances(z_fc, model.mu_fc)
                cluster_distances += model.cluster_distances(z_conv, model.mu_conv)

                logits = -0.5 * cluster_distances

                probs = torch.softmax(logits, dim=1)

            probs = probs.view(bs, ncrops, -1).mean(1)

            accuracy = torch.mean((torch.argmax(y, dim=1) == torch.argmax(probs, dim=1)).float())

            accuracies.append(accuracy.item())

    accuracy = 100. * np.mean(accuracies)

    print('ACCURACY %5.2f' % (accuracy))

    return accuracy
