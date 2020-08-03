import os
import sys

from tqdm import tqdm
import numpy as np
import torch.nn

from ..model.backbones.invertible_resnet import InvertibleResNet
from ..model.heads.invertible_multiclass_classifier import InvertibleMulticlassClassifier
from ..model.classifiers.invertible_imagenet_classifier import InvertibleImagenetClassifier
from ..model.classifiers.feedforward_imagenet_classifier import FeedForwardImagenetClassifier
from ..model import CouplingType

def average_batch_norm(model, data, count_max=1000, model_file_path=None):

    try:
        model.load(model_file_path + ".avg")
        print("loaded averaged model")
    except:
        print(f"averaging batch norm ({count_max} batches)")
        # because of FrEIA, there are so many layers and layers of subnetworks...
        reset_counter = 0
        for node in model.model.children():
            for module in node.children():
                for subnet in module.children():
                    for layer in subnet.children():
                        if type(layer) == torch.nn.BatchNorm2d:
                            layer.reset_running_stats()
                            layer.momentum = None
                            reset_counter += 1

        model.train()

        with torch.no_grad():
            count = 0
            for x, l in tqdm(data.train_loader, total=min(count_max, len(data.train_loader)), ncols=120, ascii=True):
                x = x.cuda()
                z = model.model(x)
                if count >= count_max - 1:
                    break
                print(f'{count}/{min(count_max, len(data.val_loader))}', end='\r', flush=True)
                count += 1

        print(f'>>> Reset and averaged {reset_counter} instances of nn.BatchNorm2d')

        model.save(model_file_path + "avg")

    model.eval()

def construct_feed_forward(args, verbose=False):
    return FeedForwardImagenetClassifier().cuda()

def construct_inn(args, verbose=False):
    if not verbose:
        print('>>> Constructing network')
        sys.stdout = open(os.devnull, "w")

    coupling_type = CouplingType.GLOW if args['model_coupling_type_name'] == "GLOW" else CouplingType.SLOW

    backbone = InvertibleResNet(
        base_width=64,
        coupling_type=coupling_type,
        block_type=InvertibleResNet.BlockType.BASIC,
        clamp=args['model_clamp'],
        act_norm=args['model_act_norm'],
        act_norm_type=args['model_act_norm_type'],
        permute_soft=args['model_soft_permutation'],
        welling=args['model_welling'],
        householder=int(args['model_householder']),
        blocks=eval(args['model_blocks']),
        strides=eval(args['model_strides']),
        dilations=eval(args['model_dilations']),
        synchronized_batchnorm=args['model_synchronized_batchnorm']
    )

    head = InvertibleMulticlassClassifier(
        fc_width =                  int(args['model_fc_width']),
        n_loss_dims_1d =            int(args['model_n_loss_dims_1d']),
        n_total_dims_1d =           3 * 224 * 224,
        clamp =                     float(args['model_clamp']),
        coupling_type =             coupling_type,
        act_norm=                   float(args['model_act_norm']),
        act_norm_type=              args['model_act_norm_type'],
        permute_soft=               False
    )

    inn = InvertibleImagenetClassifier(
        lr =                        float(args['training_lr']),
        mu_init =                   float(args['training_mu_init']),
        mu_conv_init =              float(args['training_mu_conv_init']),
        input_dims =                (3, 224, 224),
        n_classes =                 1000,
        n_total_dims_1d =           3 * 224 * 224,
        mu_low_rank_k =             int(args['training_mu_low_rank_k']),
        n_loss_dims_1d =            int(args['model_n_loss_dims_1d']),
        backbone =                  backbone,
        head =                      head,
    )

    inn = inn.cuda()

    sys.stdout = sys.__stdout__

    return inn


mu_img = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
std_img = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]

def to_np_image(x, de_normalize=True):

    x = x.detach().cpu().numpy().squeeze()
    assert len(x.shape) == 3

    x = x.transpose((1, 2, 0))
    if de_normalize:
        x = x * std_img + mu_img
    x = np.clip(x, 0., 1.)

    return x
