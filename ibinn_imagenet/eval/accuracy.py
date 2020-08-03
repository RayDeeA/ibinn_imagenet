import torch

from torchvision.transforms import transforms as T

import numpy as np

def accuracy(model, data):

    loader = data.val_loader

    accuracies = []
    accuracies_top_5 = []

    for val_batch, (x, y) in enumerate(loader):

        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            val_losses = model.validate(x, y)

            accuracy = val_losses['acc_val'].item()
            accuracies.append(accuracy)
            
            z_fc, z_conv = model.model(x)
            model.calc_mu_conv()
            
            cluster_distances = model.cluster_distances(z_fc, model.mu_fc)
            cluster_distances += model.cluster_distances(z_conv, model.mu_conv)

            y_ = -cluster_distances

            y_top_k = y_.topk(k=5, dim=1)[1]
            y = torch.argmax(y, dim=1)
            y_t = y.unsqueeze(0).t()
            accuracy_top_5 = torch.mean(y_top_k.eq(y_t).float().sum(dim=1)).item()
            accuracies_top_5.append(accuracy_top_5)

    accuracy = 100. * np.mean(accuracies)
    accuracy_top_5 = 100. * np.mean(accuracies_top_5)

    print('ACCURACY %5.2f' % (accuracy))
    print('ACCURACY TOP 5 %5.2f' % (accuracy_top_5))

    return accuracy


def accuracy_10_crop(model, data, ff = False):

    accuracies = []
    accuracies_top_5 = []

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

            y_top_k = probs.topk(k=5, dim=1)[1]
            y = torch.argmax(y, dim=1)
            y_t = y.unsqueeze(0).t()
            accuracy_top_5 = torch.mean(y_top_k.eq(y_t).float().sum(dim=1)).item()

            accuracies_top_5.append(accuracy_top_5)

            accuracies.append(accuracy.item())

    accuracy = 100. * np.mean(accuracies)
    print('ACCURACY %5.2f' % (accuracy))
    accuracy_top_5 = 100. * np.mean(accuracies_top_5)
    print('ACCURACY TOP 5 %5.2f' % (accuracy_top_5))

    return accuracy
