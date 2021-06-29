from __future__ import print_function

import torch

import math
import numpy as np
#from os.path import join

import time

from utils.training import train_wae_1epoch, train_vae_1epoch
from utils.evaluation import evaluate_wae, evaluate_vae, latent_test
from utils.visual_evaluation import plot_images

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def train_wae(args, train_loader, val_loader, model, optimizer, checkpoints_dir, results_dir):
    
    # best_model = model
    best_loss = 100000.
    e = 0
    
    train_loss_history = []
    train_recon_history = []
    train_z_history = []
    train_x_history = []
    train_mi_history = []


    val_loss_history = []
    val_recon_history = []
    val_z_history = []
    val_x_history = []
    val_mi_history = []


    time_history = []

    # model with the best validation
    best_model_dir = checkpoints_dir + '_b.pkl'
    # model at the end of training
    model_dir = checkpoints_dir + '_m.pkl'

    for epoch in range(1, args.epochs + 1):


        model_dir_epoch = checkpoints_dir + '_' + str(epoch) + '.pkl'


        time_start = time.time()
        model, train_loss_epoch, train_recon_epoch, train_z_epoch, train_x_epoch, train_mi_epoch,  data_train, z_e_train, z_d_train, x_recon_train, x_pseudo_z_train, z_d_mean_train, z_d_logvar_train = train_wae_1epoch(epoch, args, train_loader, model, optimizer)

        val_loss_epoch, val_recon_epoch, val_z_epoch, val_x_epoch, val_mi_epoch = evaluate_wae(args, model, train_loader, val_loader, epoch, results_dir, mode='validation')
        time_end = time.time()

        time_elapsed = time_end - time_start

        print('time_elapsed:', time_elapsed)

        # appending history
        train_loss_history.append(train_loss_epoch), train_recon_history.append(train_recon_epoch), train_z_history.append(
            train_z_epoch), train_x_history.append(train_x_epoch), train_mi_history.append(train_mi_epoch)
        val_loss_history.append(val_loss_epoch), val_recon_history.append(val_recon_epoch), val_z_history.append(
            val_z_epoch), val_x_history.append(val_x_epoch), val_mi_history.append(val_mi_epoch)
        time_history.append(time_elapsed)

        # early-stopping
        if val_loss_epoch < best_loss:

            e = 0
            best_loss = val_loss_epoch
            # best_model = model
            print('->best model saved at epoch <-', epoch)
            torch.save(model.state_dict(), best_model_dir)
        else:
            e += 1
            if epoch < args.warmup:
                e = 0
            if e > args.early_stopping_epochs:
                #print('No early-stopping.')
                break

        if (epoch % args.log_step == 0):
            torch.save(model.state_dict(), model_dir_epoch)

                
        # NaN
        if math.isnan(val_loss_epoch):
            break

    # save model at the end of training
    torch.save(model.state_dict(), model_dir)


    # SAVING
    if  np.prod(args.input_size) == 2:
        np.savez(results_dir + 'train_loss.npz', train_loss = train_loss_history, train_recon = train_recon_history,
            train_z = train_z_history, train_x = train_x_history, train_mi = train_mi_history, val_loss = val_loss_history, val_recon = val_recon_history, 
            val_z = val_z_history, val_x = val_x_history, time_history = time_history,
            data_train = data_train, z_e_train = z_e_train, z_d_train = z_d_train, x_recon_train = x_recon_train,
            x_pseudo_z_train = x_pseudo_z_train, z_d_mean_train = z_d_mean_train, z_d_logvar_train = z_d_logvar_train)
    else:
        np.savez(results_dir + 'train_loss.npz', train_loss = train_loss_history, train_recon = train_recon_history,
        train_z = train_z_history, train_x = train_x_history, train_mi = train_mi_history, val_loss = val_loss_history, val_recon = val_recon_history, 
        val_z = val_z_history, val_x = val_x_history, time_history = time_history)




def test(args, train_loader, test_loader, model, checkpoints_dir, results_dir):
    

    print('checkpoints_dir ', checkpoints_dir)
    # set the path of test model
    test_checkpoint = args.checkpoint
    if (test_checkpoint is not None):
        load_model_dir = checkpoints_dir + '_' + str(test_checkpoint) + '.pkl'
    # model at the last epoch
    else:
        load_model_dir = checkpoints_dir + '_m.pkl'

    # load test model
    model.load_state_dict(torch.load(load_model_dir))
    #model.eval()

    if args.model_name == 'MLP_wae' or args.model_name =='conv_wae' or args.model_name == 'Pixel_wae' or args.model_name =='conv_wae_2level':
        test_loss, test_recon, test_z, test_x, test_mi = evaluate_wae(args, model, train_loader, test_loader, 9999, results_dir, mode='test')
        

        with open(results_dir + 'test_log.txt', 'w+') as f:
            print('FINAL EVALUATION ON TEST SET\n'
              'Loss: {:.6f}\n'
              'Loss_recon: {:.6f}\n'
              'Loss_z: {:.6f}\n'
              'Loss_x: {:.6f}\n'
              'MI: {:.6f}'.format(
            test_loss,
            test_recon,
            test_z,
            test_x,
            test_mi
            ), file=f)
        f.close()

        np.savez(results_dir + 'test_loss.npz', test_loss = test_loss, test_recon = test_recon,
        test_z = test_z, test_x = test_x, test_mi = test_mi)


   
    elif args.model_name == 'vae' or args.model_name == 'hvae_2level' or args.model_name == 'convhvae_2level' or args.model_name == 'convhvae_2level-smim' or args.model_name == 'pixelhvae_2level' or args.model_name == 'conv_vae':
        test_loss, test_recon, test_kl, test_recon_l2square = evaluate_vae(args, model, train_loader, test_loader, 9999, results_dir, mode='test')

        with open(results_dir + 'test_log.txt', 'w+') as f:
            print('FINAL EVALUATION ON TEST SET\n'
              'Loss: {:.6f}\n'
              'Loss_recon: {:.6f}\n'
              'Loss_kl: {:.6f}\n'
              'Loss_recon_l2square: {:.6f}'.format(
            test_loss,
            test_recon,
            test_kl,
            test_recon_l2square
            ), file=f)
        f.close()

        np.savez(results_dir + 'test_loss.npz', test_loss = test_loss, test_recon = test_recon,
        test_kl = test_kl, test_recon_l2square = test_recon_l2square)

    else:
        raise Exception('Wrong name of the model!')

    if args.latent_test == True:
        latent_test(args, model, train_loader, test_loader, results_dir)


def train_vae(args, train_loader, val_loader, test_loader, model, optimizer, checkpoints_dir, results_dir):


    # SAVING

    # best_model = model
    best_loss = 100000.
    e = 0
    train_loss_history = []
    train_re_history = []
    train_kl_history = []

    val_loss_history = []
    val_re_history = []
    val_kl_history = []

    time_history = []

    # model with the best validation
    best_model_dir = checkpoints_dir + '_b.pkl'
    # model at the end of training
    model_dir = checkpoints_dir + '_m.pkl'

    for epoch in range(1, args.epochs + 1):

        model_dir_epoch = checkpoints_dir + '_' + str(epoch) + '.pkl'

        time_start = time.time()
        model, train_loss_epoch, train_re_epoch, train_kl_epoch = train_vae_1epoch(epoch, args, train_loader, model, optimizer)
                                                                             

        val_loss_epoch, val_re_epoch, val_kl_epoch, _ = evaluate_vae(args, model, train_loader, val_loader, epoch, results_dir, mode='validation')
        time_end = time.time()

        time_elapsed = time_end - time_start
        print('time_elapsed', time_elapsed)

        # appending history
        train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
            train_kl_epoch)
        val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
            val_kl_epoch)
        time_history.append(time_elapsed)

        # early-stopping
        if val_loss_epoch < best_loss:
            e = 0
            best_loss = val_loss_epoch
            # best_model = model
            print('->best model saved at epoch <-', epoch)
            torch.save(model.state_dict(), best_model_dir)
        else:
            e += 1
            if epoch < args.warmup:
                e = 0
            if e > args.early_stopping_epochs:
                #print('No early-stopping.')
                break

        if (epoch % args.log_step == 0):
            torch.save(model.state_dict(), model_dir_epoch)

        # NaN
        if math.isnan(val_loss_epoch):
            print('NaN break')
            break

    # FINAL EVALUATION

    # save model at the end of training
    torch.save(model.state_dict(), model_dir)
    np.savez(results_dir + 'train_loss.npz', train_loss = train_loss_history, train_recon = train_re_history, 
        train_kl = train_kl_history, time_history = time_history, val_loss = val_loss_history, val_recon = val_re_history, val_kl = val_kl_history)

