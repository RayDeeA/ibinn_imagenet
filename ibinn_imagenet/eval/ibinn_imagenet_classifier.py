import click
import sys

from ..data.imagenet import Imagenet
from . import construct_inn, construct_feed_forward
from . import accuracy

def process_evaluation(args, model, data, evaluation):
    model.eval()

    if 'accuracy' in evaluation:

        print("ACC 1 CROP")
        acc = accuracy.accuracy(model, data)
        print(acc)

        print("ACC 10 CROP")
        acc = accuracy.accuracy_10_crop(model, data, ff="feed_forward" in evaluation)
        print(acc)

@click.command()
@click.option('--model_file_path', default='', help='Path to the trained ibinn model')
@click.option('--evaluation', default='accuracy', help='Evaluation method (accuracy, feed_forward_accuracy)')
@click.option('--data_root_folder_train', default='../imagenet', help='Root folder for training dataset')
@click.option('--data_root_folder_val', default='../imagenet', help='Root folder for validation dataset')
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
def eval(**args):

    data = Imagenet(args['data_root_folder_train'], args['data_root_folder_val'], int(args['data_batch_size']))

    #args['data_dims'] = repr((3, data.img_crop_size[0], data.img_crop_size[1]))
    #args['data_n_classes'] = repr(data.n_classes)
    #args['model_soft_permutaion'] = False

    if 'feed_forward' in args['evaluation']:
        print("Loading Feed Forward Model")
        model = construct_feed_forward(args)
    else:
        model = construct_inn(args)

        print("Loading " + args['model_file_path'], flush=True)
        model.load(args['model_file_path'])

    process_evaluation(args, model, data, args['evaluation'])

if __name__ == '__main__':
    sys.exit(eval())
