# Symmetric Wasserstein Autoencoders
This is a PyTorch implementation of the following paper:
* Sun Sun and Hongyu Guo, Symmetric Wasserstein Autoencoders, [arXiv preprint](https://arxiv.org/abs/2106.13024), 2021

## Requirements
The code is compatible with:
* pytorch==0.2.0
* numpy==1.15.4
* scipy==1.1.0
* torch==1.0.0
* wrapt==1.11.2
* matplotlib==3.0.3
* torchvision==0.2.1
* cvxopt==1.2.3
* circlify==0.10.0
* Pillow==6.2.0
* scikit_learn==0.21.3

## Data
The experiments can be run on the following datasets: MNIST, Fashion-MNIST, Coil20, and CIFAR10.
* MNIST, Fashion-MNIST, and CIFAR10 can be loaded from PyTorch.
* Coil20 can be downloaded from [link](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)

## Models
The proposed model and benchmark models can be found in the following directories. The corresponding model name is shown in brackets.
* proposed symmetric Wasserstein autoencoders (conv_wae): models/conv_wae.py
* VAE (conv_vae): models/conv_vae.py
* VampPrior (convhvae_2level): models/convHVAE_2level.py
* MIM (convhvae_2level-smim): models/convHVAE_2level.py

To generate Figure 1 on a GMM dataset in the paper, we use MLP_wae (MLPs as the building blocks) instead of conv_wae (convolutional layers as the building blocks). See models/MLP_wae.py.

## Run the experiment
1. Set up your experiment in `experiment.py`.
2. Run experiment. See examples in test.sh on different datasets.


## Citation

Please cite our paper if you use this code in your research:

```
@inproceedings{sun2021symmetric,
  author    = {Sun Sun and Hongyu Guo},
  title     = {Symmetric Wasserstein Autoencoders},
  booktitle = {UAI},
  year      = {2021}
}

```

