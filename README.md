# ibinn_imagenet

This repository contains the code for: 
  
"Generative Classifiers as a Basis for Trustworthy Computer Vision"  
Radek Mackowiak, Lynton Ardizzone, Ullrich Köthe, Carsten Rother  
https://arxiv.org/abs/2007.15036

## Requirements
* Python 3.x
* CUDA
* cuDNN

## Installation
For the installation we recommend a python venv.
```sh
git clone git@github.com:RayDeeA/ibinn_imagenet.git
cd ibinn_imagenet/
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```
Clone and install FrEIA in editable mode:
```sh
git clone git@github.com:VLL-HD/FrEIA.git
pip install -e FrEIA
```
In this way, any changes made to FrEIA are reflected immediately without reinstallation of the package.

## Usage
Run a training by executing:
```sh
python -m ibinn_imagenet.train.ibinn_imagenet_classifier [OPTIONS]
```
All options are visible when running:
```sh
python -m ibinn_imagenet.train.ibinn_imagenet_classifier --help
```

Alternatively you can install xptl (https://github.com/titus-leistner/xptl) from Titus Leistner and use the *.ini-files provided under ibinn_imagenet/config

With xptl installed start a training by executing:
```sh
xptl-schedule ibinn_imagenet/config/beta_[VALUE].ini [QUEUEING_COMMAND] ./SLURM_train_ibinn_imagenet_classifier.sh [OUTPUT_FOLDER]
```
e.g.:
```sh
xptl-schedule ibinn_imagenet/config/beta_0_0.ini "" ./SLURM_train_ibinn_imagenet_classifier.sh /mnt/data/output
```
to run the training on a local machine, or:
```sh
xptl-schedule ibinn_imagenet/config/beta_0_0.ini sbatch ./SLURM_train_ibinn_imagenet_classifier.sh /mnt/data/output
```
in case you want to run your experiments on a cluster system utilizing a scheduler (e.g. SLURM, LSF)

Evaluation for a learned model can be performed by executing the folling command. Note however that currently only "accuracy" and "feed_forward_accuracy" are supported values for the field EVALUATION_METHOD.
```sh
xptl-schedule ibinn_imagenet/config/beta_[VALUE].ini [QUEUEING_COMMAND] "./SLURM_eval_ibinn_imagenet_classifier.sh [PATH_TO_MODEL_FILE] [EVALUATION_METHOD]"
```
e.g.:
```sh
xptl-schedule ibinn_imagenet/config/beta_inf.ini "" "./SLURM_eval_ibinn_imagenet_classifier.sh /path/to/models/directory/beta_inf.avg accuracy"
```
The other evaluation methods described in the paper, will be made available soon.

Trained models can be downloaded here:  
https://heibox.uni-heidelberg.de/d/e7b5ba0d30f24cdca416/

## Project Organization

    ├── LICENSE                                 <- The License
    │
    ├── README.md                               <- The top-level README for developers using this project.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment
    │
    ├── ibinn_imagenet                          <- Source code
    │   │
    │   ├── config                              <- Configuration files
    │   │
    │   ├── data                                <- Code defining the data
    │   │
    │   ├── model                               <- Code defining the models
    │   │
    │   ├── train                               <- Code defining the training logic
    │   │
    │   ├── utils                               <- Utilities
