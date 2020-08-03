#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ml
#SBATCH --mem=240000
#SBATCH --mail-user=radek.mackowiak@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT

# Uncomment the following in order to load images into memory
#module load modenv/classic
#module load parallel/20170222

#mkdir /dev/shm/imagenet
#mkdir /dev/shm/imagenet/train
#mkdir /dev/shm/imagenet/val

#bash train_to_mem_copy.sh
#bash val_to_mem_copy.sh

source bin/activate

NAME="train_ibinn_imagenet_classifier"

output_dir=$1
mkdir -p ${output_dir}/code
echo "copying the code for backup"
cp -r ibinn_imagenet ${output_dir}/code/

python --version
echo "python -m ibinn_imagenet.train.ibinn_imagenet_classifier --checkpoints_out_folder=$@"
python -m ibinn_imagenet.train.ibinn_imagenet_classifier --checkpoints_out_folder=$@

exit 0
