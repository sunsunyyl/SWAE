from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np
from utils.optimizer import AdamNormGrad
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
# training model for one epoch
def train_wae_1epoch(epoch, args, train_loader, model, optimizer):

    # set loss to 0
    train_loss = 0
    train_recon = 0
    train_z = 0
    train_x = 0
    train_mi = 0

    # set model in training mode
    model.train()

    #start training
    if args.warmup == 0:
        beta = args.beta
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.

    data_train = np.empty([0, np.prod(args.input_size)])
    z_e_train = np.empty([0, args.z_size])
    z_d_train = np.empty([0, args.z_size])
    x_recon_train = np.empty([0, np.prod(args.input_size)])
    # pseudo-input that are used to generate z_d
    x_pseudo_z_train = np.empty([0, np.prod(args.input_size)])
    z_d_mean_train = np.empty([0, args.z_size])
    z_d_logvar_train = np.empty([0, args.z_size])

    for batch_idx, (data, target) in enumerate(train_loader):


        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        loss, recon_loss, z_loss, x_loss, mi, z_e, z_d, x_recon, x_pseudo_z, z_d_mean, z_d_logvar = model.calculate_loss(x, beta, average=True)
        
        # reset gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_z += z_loss.item()
        train_x += x_loss.item()

        if args.z_size == 2:
            data_train = np.append(data_train, x.data.cpu().numpy(), axis = 0)
            z_d_train = np.append(z_d_train, z_d, axis = 0)
            z_e_train = np.append(z_e_train, z_e, axis = 0)
            x_recon_train = np.append(x_recon_train, x_recon, axis = 0)
            x_pseudo_z_train = np.append(x_pseudo_z_train, x_pseudo_z, axis = 0)
            z_d_mean_train = np.append(z_d_mean_train, z_d_mean, axis = 0)
            z_d_logvar_train = np.append(z_d_logvar_train, z_d_logvar, axis = 0)


    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_recon /= len(train_loader)  # re already averages over batch size
    train_z /= len(train_loader)  # kl already averages over batch size
    train_x /= len(train_loader)

    return model, train_loss, train_recon, train_z, train_x, train_mi, data_train, z_e_train, z_d_train, x_recon_train, x_pseudo_z_train, z_d_mean_train, z_d_logvar_train




def train_vae_1epoch(epoch, args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = args.beta
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        loss, RE, KL = model.calculate_loss(x, beta, average=True)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()


        train_loss += loss.item()
        train_re += -RE.item()
        train_kl += KL.item()

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl
