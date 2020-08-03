import sys
from time import time
from os.path import join

import click

import numpy as np
import torch.optim

from ..data.imagenet import Imagenet

from ..model.backbones.invertible_resnet import InvertibleResNet
from ..model.heads.invertible_multiclass_classifier import InvertibleMulticlassClassifier

from ..model.classifiers.invertible_imagenet_classifier import InvertibleImagenetClassifier

from ..model import CouplingType


@click.command()
@click.option('--checkpoints_out_folder', default='default_out', help='Base directory for all outputs')
@click.option('--data_root_folder_train', default='../data/imagenet', help='Root folder for training dataset')
@click.option('--data_root_folder_val', default='../data/imagenet', help='Root folder for validation dataset')
@click.option('--data_batch_size', default=16, help='Batch size')
@click.option('--model_n_loss_dims_1d', default=1024, help='Size of latent space vector')
@click.option('--model_coupling_type_name', default='Slow', help='Type of coupling block: GLOW/SLOW')
@click.option('--model_clamp', default=0.7, help='Clamp output of coupling block')
@click.option('--model_act_norm', default=0.7, help='Scale output of coupling block')
@click.option('--model_act_norm_type', default='SOFTPLUS', help='Type of activation normalization of coupling block outputs: SOFTPLUS/SIGMOID/RELU')
@click.option('--model_soft_permutation', is_flag=True, help='Enable soft permutation? True/False')
@click.option('--model_welling', is_flag=True, help='Enable welling permutation? True/False')
@click.option('--model_householder', default=0, help='Use householder permutation? True/False')
@click.option('--model_blocks', default='[3,4,6,3]', help='Coupling layers per blocks')
@click.option('--model_strides', default='[1,2,2,2]', help='Strides per block type')
@click.option('--model_dilations', default='[1,1,1,1]', help='Dilations per block type')
@click.option('--model_synchronized_batchnorm', is_flag=True, help='Enable global BatchNorm computation')
@click.option('--model_fc_width', default=1024, help='Width of fully-connected layer')
@click.option('--training_lr', default=0.07, help='Learning rate')
@click.option('--training_lr_mu', default=0.07, help='Learning rate for parameter mu')
@click.option('--training_mu_init', default=3.5, help='Initial value for parameter mu')
@click.option('--training_mu_conv_init', default=0.01, help='Learning rate for disjoined information')
@click.option('--training_mu_low_rank_k', default=128, help='Low-rank approximation')
@click.option('--training_n_epochs', default=30, help='Number of training epochs')
@click.option('--training_train_nll', is_flag=True, help='Train with negative log likelihood? True/False')
@click.option('--training_beta', default='0.5', help='Parameter beta')
@click.option('--training_burn_in_iterations', default=10, help='Number of burn in iterations with lower learning rate')
@click.option('--checkpoints_interval_log', default=1000, help='Interval to save logs')
@click.option('--checkpoints_interval_checkpoint', default=25000, help='Interval to save checkpoints')
@click.option('--checkpoints_checkpoint_when_crash', is_flag=True, help='Save checkpoint if training crashes? True/False')
@click.option('--checkpoints_resume_checkpoint', is_flag=True, help='Continue from last checkpoint?')
@click.option('--checkpoints_cooling_step', default=1, help='Lower learning rate 1e-10 per step')
@click.option('--checkpoints_finetune_mu', is_flag=True, help='Only finetune latent space')
@click.option('--checkpoints_extension', default='', help='Custom extension to the output model file for ablations')
def train(**args):

    data = Imagenet(args['data_root_folder_train'], args['data_root_folder_val'], int(args['data_batch_size']))

    extension = args['checkpoints_extension']

    output_scale = 16
    skip_connection = False

    n_loss_dims_1d = int(args['model_n_loss_dims_1d'])
    n_total_dims_1d = int(3 * data.img_crop_size[0] * data.img_crop_size[1])

    coupling_type_name = args['model_coupling_type_name']
    coupling_type = CouplingType.GLOW if coupling_type_name == "GLOW" else CouplingType.SLOW

    finetune_mu = args['checkpoints_finetune_mu']
    print(finetune_mu)
    backbone = InvertibleResNet(
        64,
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
        synchronized_batchnorm=args['model_synchronized_batchnorm'],
        skip_connection=skip_connection
    )

    head = InvertibleMulticlassClassifier(
        int(args['model_fc_width']),
        n_loss_dims_1d,
        n_total_dims_1d,
        coupling_type=coupling_type,
        clamp=float(args['model_clamp']),
        act_norm=float(args['model_act_norm']),
        act_norm_type=args['model_act_norm_type'],
        permute_soft=args['model_soft_permutation']
    )

    inn = InvertibleImagenetClassifier(
        float(args['training_lr']) * 10 ** (-float(args['checkpoints_cooling_step'])),
        float(args['training_mu_init']),
        float(args['training_mu_conv_init']),
        args['training_mu_low_rank_k'],
        (3, data.img_crop_size[0], data.img_crop_size[1]),
        data.n_classes,
        n_loss_dims_1d,
        n_total_dims_1d,
        backbone, head,
        finetune_mu = finetune_mu
    )
    inn.cuda()

    inn_parallel = torch.nn.DataParallel(inn)
    n_batches_per_epoch = (len(data.train_data.imgs)//data.batch_size)
    N_epochs = int(args['training_n_epochs'])

    train_nll = args['training_train_nll']
    beta = args['training_beta']
    if not beta =='infinity':
        beta = float(beta)

    def truncate(n, decimals=10):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    if finetune_mu:
        last_lr = float(args['training_lr']) * 10 ** (-(max(0,float(args['checkpoints_cooling_step']))))
    else:
        last_lr = float(args['training_lr']) * 10 ** (-(max(0,float(args['checkpoints_cooling_step'])-1)))
    current_lr = float(args['training_lr']) * 10 ** (-(float(args['checkpoints_cooling_step'])))

    last_lr = truncate(last_lr)
    current_lr = truncate(current_lr)

    if not beta == 'infinity':
        if train_nll:
            beta_nll = 1. / (1 + beta)
            beta_cat_ce = 1. * beta / (1 + beta)
        else:
            beta_nll, beta_cat_ce = 0., 1.

    interval_log = int(args['checkpoints_interval_log'])
    # interval_checkpoint = int(args['checkpoints_interval_checkpoint'])

    save_on_crash = args['checkpoints_checkpoint_when_crash']
    out_folder = args['checkpoints_out_folder']
    print('out_folder', out_folder)

    finetune_mu_extension = '_finetune_mu' if finetune_mu else ""
    save_name = '_lr-' + str(current_lr)  + \
    '_nll-' + str(args['training_train_nll']) + \
    '_beta-' + str(args['training_beta']) + \
    '_mbs-' + str(args['data_batch_size']) + \
    '_ct-' + str(args['model_coupling_type_name']) + \
    '_cl-' + str(args['model_clamp']) + \
    '_an-' + str(args['model_act_norm']) + \
    '_blocks-' + str(args['model_blocks']) + \
    'strides-' + str(args['model_strides']) + \
    'dilations-' + str(args['model_dilations']) + \
    '_os-' + str(output_scale) + \
    '_ld-' + str(args['model_fc_width']) + \
    '_k_' + str(args['training_mu_low_rank_k']) + \
    '_ext_' + extension

    save_name = save_name + finetune_mu_extension

    print(save_name, flush=True)

    load_name = '_lr-' + str(last_lr) + \
    '_nll-' + str(args['training_train_nll']) + \
    '_beta-' + str(args['training_beta']) + \
    '_mbs-' + str(args['data_batch_size']) + \
    '_ct-' + str(args['model_coupling_type_name']) + \
    '_cl-' + str(args['model_clamp']) + \
    '_an-' + str(args['model_act_norm']) + \
    '_blocks-' + str(args['model_blocks']) + \
    'strides-' + str(args['model_strides']) + \
    'dilations-' + str(args['model_dilations']) + \
    '_os-' + str(output_scale) + \
    '_ld-' + str(args['model_fc_width']) + \
    '_k_' + str(args['training_mu_low_rank_k']) + \
    '_ext_' + extension

    print(load_name, flush=True)

    csv = join(out_folder, 'log_train.csv')
    csv_val = join(out_folder, 'log_val.csv')

    resume = args['checkpoints_resume_checkpoint']

    plot_columns = ['time', 'epoch', 'iteration', 'learning_rate',
                    'nll_joint_tr', 'nll_class_tr', 'cat_ce_tr', 'acc_tr',
                    'nll_joint_val', 'nll_class_val', 'cat_ce_val',
                    'acc_val', 'delta_mu_val']

    train_loss_names = [l for l in plot_columns if l[-3:] == '_tr']
    val_loss_names   = [l for l in plot_columns if l[-4:] == '_val']

    header_fmt = '{:>15}' * len(plot_columns)
    val_header_fmt = '{:>15}' * len(val_loss_names)
    
    with open(csv, 'w') as f:
        f.write(('{:>15}' * 8).format(*plot_columns) + '\n')

    with open(csv_val, 'w') as f:
        f.write(val_header_fmt.format(*val_loss_names) + '\n')

    output_fmt =        '{:15.1f}      {:04d}/{:04d}      {:04d}/{:04d}      {:1.5f}' + '{:15.5f}' * (len(plot_columns) - 4)
    live_output_fmt =   '{:15.1f}      {:04d}/{:04d}      {:04d}/{:04d}      {:1.5f}' + '{:15.5f}' * len(train_loss_names) + '\r'
    val_output_fmt =    '{:15.5f}' * len(val_loss_names)

    if resume:
        if (not beta == "infinity") and beta < 0.5:
            file_path = join(out_folder, f'{load_name}_best_val_nll.pt')
        else:
            file_path = join(out_folder, f'{load_name}_best_val_cat_ce.pt')
        print("Loading " + file_path, flush=True)
        inn.load(file_path)

        for param_group in inn.optimizer.param_groups:
            param_group['lr'] = current_lr

    t_start = time()

    best_val_cat_ce = float("inf")
    best_val_nll = float("inf")
    try:
        for i_epoch in range(N_epochs):
            print(header_fmt.format(*plot_columns), flush=True)

            running_avg = {l: [] for l in train_loss_names}

            for i_batch, (x, y) in enumerate(data.train_loader):
                x, y = x.cuda().contiguous(), y.cuda().contiguous()

                losses = inn_parallel(x, y)
                for k, l in losses.items():
                    losses[k] = l.mean()
                if beta == 'infinity':
                    loss = losses['cat_ce_tr']
                elif beta == 0.0:
                    loss = losses['nll_joint_tr']
                else:
                    loss = beta_nll * losses['nll_joint_tr'] + beta_cat_ce * losses['cat_ce_tr']

                if not resume and i_epoch == 0 and i_batch < int(args['training_burn_in_iterations']):
                    loss *= 0.05

                loss.backward()
                torch.nn.utils.clip_grad_norm_(inn.model_parameters, 8.)
                inn.optimizer.step()
                inn.optimizer.zero_grad()

                losses_display = [(time() - t_start) / 60.,
                                  i_epoch, N_epochs,
                                  i_batch, len(data.train_loader), inn.lr]

                losses_display += [losses[l].item() for l in train_loss_names]
                
                live_output = live_output_fmt.format(*losses_display)
                print(live_output)
                with open(csv, 'a') as f:
                    f.write(live_output + '\n')

                for l_name in train_loss_names:
                    running_avg[l_name].append(losses[l_name].item())

                if not ((n_batches_per_epoch * i_epoch) + i_batch) % interval_log:

                    for l_name in train_loss_names:
                        running_avg[l_name] = np.mean(running_avg[l_name])

                    val_avg_losses = {}
                    for l_name in val_loss_names:
                        val_avg_losses[l_name] = []

                    for val_batch, (x, y) in enumerate(data.val_loader_fast):
                        with torch.no_grad():
                            x = x.cuda()
                            y = y.cuda()
                            val_losses = inn.validate(x, y)

                            for l_name in val_loss_names:
                                val_avg_losses[l_name].append(val_losses[l_name].item())

                    for l_name in val_loss_names:
                        val_avg_losses[l_name] = np.mean(val_avg_losses[l_name])

                    for l_name in val_loss_names:
                        running_avg[l_name] = val_avg_losses[l_name]

                    losses_display = [(time() - t_start) / 60.,
                                      i_epoch, N_epochs,
                                      i_batch, len(data.train_loader), inn.lr]

                    losses_display += [running_avg[l] for l in plot_columns[4:]]
                    print(output_fmt.format(*losses_display), flush=True)

                    running_avg = {l: [] for l in train_loss_names}

            print(val_header_fmt.format(*val_loss_names), flush=True)

            val_avg_losses = {}
            for l_name in val_loss_names:
                val_avg_losses[l_name] = []

            for val_batch, (x, y) in enumerate(data.val_loader):
                with torch.no_grad():
                    x = x.cuda()
                    y = y.cuda()
                    val_losses = inn.validate(x, y)

                    for l_name in val_loss_names:
                        val_avg_losses[l_name].append(val_losses[l_name].item())

            for l_name in val_loss_names:
                val_avg_losses[l_name] = np.mean(val_avg_losses[l_name])

            val_losses_display = [val_avg_losses[l] for l in plot_columns[-5:]]

            val_output = val_output_fmt.format(*val_losses_display)
            print(val_output, flush=True)
            with open(csv_val, 'a') as f:
                f.write(val_output + '\n')

            if val_avg_losses["cat_ce_val"] < best_val_cat_ce:
                inn.save(join(out_folder, f'{save_name}_best_val_cat_ce.pt'))
                best_val_cat_ce = val_avg_losses["cat_ce_val"]
            if val_avg_losses["nll_joint_val"] < best_val_nll:
                inn.save(join(out_folder, f'{save_name}_best_val_nll.pt'))
                best_val_nll = val_avg_losses["nll_joint_val"]

    except Exception as e:
        print(e, flush=True)
        if save_on_crash:
            inn.save(join(out_folder, f'{save_name}_ABORT.pt'))
        raise
    
    print('out_folder', out_folder)
    inn.save(join(out_folder, f'{save_name}.pt'))


if __name__ == '__main__':
    sys.exit(train())
