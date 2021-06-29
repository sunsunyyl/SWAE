import os
import random
import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.init as init
import math



def create_dirNames(args):

    dataset_name = args.dataset_name
    model_name = args.model_name
    number_components_input = args.number_components_input
    z_size = args.z_size
    z1_size = args.z1_size
    beta = args.beta
    prior = args.prior
    warmup = args.warmup
    cosine_similarity = args.cosine_similarity
    MI_obj = args.MI_obj
    alpha = args.alpha
    opt_sep = args.opt_sep
    latent_sim = args.latent_similarity
    obj = args.obj
    lam = args.lam
    seed = args.seed
    recloss = args.rec_loss
    lr = args.lr
    beta1 = args.beta1


    
    if model_name == 'MLP_wae' or model_name == 'conv_wae':
        checkpoints_dir = f'checkpoints/{dataset_name}/{model_name}_K{number_components_input}_z{z_size}_beta{beta}_cos{cosine_similarity}_obj{obj}'
        results_dir =     f'results/{dataset_name}/{model_name}_K{number_components_input}_z{z_size}_beta{beta}_cos{cosine_similarity}_obj{obj}/'


    elif model_name == 'vae' or model_name == 'conv_vae':

        checkpoints_dir = f'checkpoints/{dataset_name}/{model_name}_z1{z1_size}_beta{beta}_warmup{warmup}'
        results_dir = f'results/{dataset_name}/{model_name}_z1{z1_size}_beta{beta}_warmup{warmup}/'


    elif model_name == 'hvae_2level' or model_name == 'convhvae_2level'  or model_name == 'convhvae_2level-smim':
        
        checkpoints_dir = f'checkpoints/{dataset_name}/{model_name}_K{number_components_input}_z1{z1_size}_beta{beta}_warmup{warmup}'    
        results_dir = f'results/{dataset_name}/{model_name}_K{number_components_input}_z1{z1_size}_beta{beta}_warmup{warmup}/' 


    #create dir
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    return  checkpoints_dir, results_dir 

  



