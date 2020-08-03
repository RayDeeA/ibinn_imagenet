#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
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

NAME="eval_ibinn_imagenet_classifier"

python --version
echo "python -m ibinn_imagenet.eval.ibinn_imagenet_classifier --model_file_path=$1 --evaluation=$2 ${@:4}"
python -m ibinn_imagenet.eval.ibinn_imagenet_classifier --model_file_path=$1 --evaluation=$2 ${@:4}

exit 0
