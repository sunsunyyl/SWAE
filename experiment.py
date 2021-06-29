from __future__ import print_function
import argparse

import torch
import torch.optim as optim
import numpy as np

from utils.optimizer import AdamNormGrad

import os

import datetime

from utils.load_data import load_dataset
from utils.train_test import train_wae, test, train_vae
from utils.create_paths import create_dirNames
from utils.visual_evaluation import plot_images,plot_images_each_label

# set specific GPU to use. The actuall device will be numbered from zero.
print('__Number CUDA Devices:', torch.cuda.device_count())
os.environ['CUDA_VISIBLE_DEVICES']='0'

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #


# Training settings
# arguments for optimization
parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='BStest',
                    help='input batch size for testing (default: 100)')
parser.add_argument('-ntrain','--num_train_data', type=int, default=162770,
                    help='number of training data')
parser.add_argument('-nval','--num_val_data', type=int, default=19867,
                    help='number of validation data')
parser.add_argument('-ntest','--num_test_data', type=int, default=19962,
                    help='number of test data')
parser.add_argument('--num_fid_data', type=int, default=2000,
                    help='number of data used for FID score')

parser.add_argument('--epochs', type=int, default=100, metavar='E',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=0, metavar='WU',
                    help='number of epochs for warmu-up')
parser.add_argument('--beta', type=float, default=1., 
                    help='coefficient before recon_loss/x-loss')
parser.add_argument('--alpha', type=float, default=1., 
                    help='coefficient before MI objective')
parser.add_argument('--lam', type=float, default=1., 
                    help='coefficient before z-loss')

parser.add_argument('--beta1', type=float, default=0.9, 
                    help='beta1 in AdamNormGrad')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--mutiple_cuda', type = bool, default=False, help = 'whether to use multiple GPUs')

# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')

# model: z1 and z2 are used in vampprior
parser.add_argument('--z1_size', type=int, default=40, metavar='M1',
                   help='latent size used for VAE, vampprior')
parser.add_argument('--z2_size', type=int, default=40, metavar='M2',
                    help='latent size')

parser.add_argument('--log_step', type=int, default=100, 
                    help='number of epochs for logging each checkpoint')

#z is used in proposed method
parser.add_argument('--z_size', type=int, default=20, metavar='M1',
                    help='latent size')

parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')

parser.add_argument('--dim_h', type=int, default=128, metavar='DH',
                    help='hidden dimension size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components_input', type=int, default=100, metavar='NCX',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=1, metavar='PMX',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.2, metavar='PSX',
                    help='std for init pseudo-inputs')

parser.add_argument('--number_components_latent', type=int, default=0, metavar='NCZ',
                    help='number of pseudo-latents')
parser.add_argument('--pseudolatents_mean', type=float, default=-0.05, metavar='PMZ',
                    help='mean for init pseudo-latents')
parser.add_argument('--pseudolatents_std', type=float, default=0.01, metavar='PSZ',
                    help='std for init pseudo-latents')


parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

# model: model name, prior
parser.add_argument('--model_name', type=str, default='wae', metavar='MN',
                    help='model name: MLP_wae, conv_wae, Pixel_wae, vae, hvae_2level, convhvae_2level, pixelhvae_2level')

parser.add_argument('--prior', type=str, default='x_prior', metavar='P',
                    help='prior: standard, vampprior, x_prior, x_z_prior')
parser.add_argument('--q_x_prior', type=str, default='marginal', metavar='XP',
                    help='prior: marginal, vampprior; used in MIM')

parser.add_argument('--input_type', type=str, default='continuous', metavar='IT',
                    help='type of the input: binary, gray, continuous')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='freyfaces', metavar='DN',
                    help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes, histopathologyGray, freyfaces, cifar10, gmm_')

parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

# load model at certain checkpoint
parser.add_argument('--checkpoint', type=str, default='b', 
                    help='load model at certain checkpoint when testing: b, number of epochs (default: use the best one)')

parser.add_argument('--Train', dest= 'trainflag',action='store_true', default=False, help = 'Train Mode')
parser.add_argument('--Test',  dest= 'testflag',action='store_true', default=False, help = 'Test Mode')
parser.add_argument('--latent_test', type = bool, default=False, help = 'whether test latent representation')
parser.add_argument('--interpolation_test', type = bool, default=False, help = 'whether test interpolation using latent representation')
parser.add_argument('--MI_obj', type = str, default='None', help = 'add objective of mutual information: ze_zd, xe_zd')
parser.add_argument('--obj', type = str, default='xzrec', help = 'objective functions: xz (x-loss and z loss), xzrec (x-loss, z-loss, and recon-loss), zrec (z-loss and recon-loss)')
parser.add_argument('--rec_loss', type = str, default='l2', help = 'reconstruction loss is based on l2 norm')


parser.add_argument('--cosine_similarity', type = bool, default=False, help = 'whether use cosine similarity to find the closest neighbor')
parser.add_argument('--latent_similarity', type = bool, default=False, help = 'whether find the closest neighbor in the latent space')

parser.add_argument('--opt_sep', type = bool, default=False, help = 'whether separately optimize Wasserstain loss and MI')
parser.add_argument('--fid_score', type = bool, default=False, help = 'whether calculate FID score for two datasets')
parser.add_argument('--fid_score_rec', type = bool, default=False, help = 'whether calculate FID score for two datasets based on reconstructed samples')
parser.add_argument('--gen_batch', type = bool, default=False, help = 'whether to generate data by batches (used for generating a large number of data)')

    


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# fix random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):

    # LOAD DATA=========================================================================================================
    print('load data')

    checkpoints_dir, results_dir = create_dirNames(args)

    # loading data
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)



    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    if args.model_name == 'vae':
        from models.VAE import VAE
        model = VAE(args)
    elif args.model_name == 'conv_vae':
        from models.conv_vae import VAE
        model = VAE(args)    
    elif args.model_name == 'hvae_2level':
        from models.HVAE_2level import VAE
        model = VAE(args)
    elif args.model_name == 'convhvae_2level':
        from models.convHVAE_2level import VAE
        model = VAE(args)
    elif args.model_name == 'convhvae_2level-smim':
        from models.convHVAE_2level import SymMIM as VAE
        model = VAE(args)
    elif args.model_name == 'MLP_wae':
        from models.MLP_wae import WAE
        model = WAE(args)
    elif args.model_name == 'conv_wae':
        from models.conv_wae import WAE
        model = WAE(args)   
    else:
        raise Exception('Wrong name of the model!')

        
    #model = VAE(args)
    if args.cuda:
        model.cuda()       

    optimizer = AdamNormGrad(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # ======================================================================================================================
    print(args)

    # ======================================================================================================================
    print('perform experiment')



    if args.trainflag:
        if args.model_name == 'MLP_wae' or args.model_name == 'conv_wae' or args.model_name =='Pixel_wae' or args.model_name == 'conv_wae_2level':
            train_wae(args, train_loader, val_loader, model, optimizer, checkpoints_dir, results_dir)
        else:
            train_vae(args, train_loader, val_loader, test_loader, model, optimizer, checkpoints_dir, results_dir)
    
    if args.testflag:
        test(args, train_loader, test_loader, model, checkpoints_dir, results_dir)

   # ======================================================================================================================

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    run(args, kwargs)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
