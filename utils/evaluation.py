from __future__ import print_function

import torch
from torch.autograd import Variable

from utils.visual_evaluation import plot_images,plot_fid_images, plot_images_each_label
from utils.fid_score import evaluate_fid_score


import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import scipy
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time


import os
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def latent_test(args, model, train_loader, data_loader, results_dir):

    #load all data
    test_data, test_target = [], []
    for data, lbls in data_loader:
        test_data.append(data)
        test_target.append(lbls)


    test_data, test_target = [torch.cat(test_data, dim=0), torch.cat(test_target, dim=0).squeeze()]

    # grab the train data by iterating over the loader
    # there is no standardized tensor_dataset member across pytorch datasets
    
    full_data = []
    full_target = []
    for data, lbls in train_loader:
        full_data.append(data)
        full_target.append(lbls)

    
    full_data, full_target = [torch.cat(full_data, dim=0), torch.cat(full_target, dim=0).squeeze()]


    if args.cuda:
        test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

    if args.dynamic_binarization:
        full_data = torch.bernoulli(full_data)


    def find_permutation(n_clusters, real_labels, labels):
        permutation = []
        for i in range(n_clusters):
            idx = labels == i
            new_label = scipy.stats.mode(real_labels[idx])[0][0]  # Choose the most common label among data points in the cluster
            permutation.append(new_label)
        return permutation

    # get embeded z
    def embed_z(x, batch = args.batch_size):
        with torch.no_grad():
            
            embed_z = []
            for data in torch.split(x, batch):

                if args.model_name == 'vae' or args.model_name == 'conv_vae':
                    _,_,z_latent,_,_ = model(data)
                    embed_z.append(z_latent)

                elif args.model_name == 'MLP_wae' or args.model_name == 'conv_wae' or args.model_name == 'Pixel_wae':
                    z_latent, _ = model(data)
                    embed_z.append(z_latent)

                elif args.model_name == 'hvae_2level' or args.model_name == 'convhvae_2level' or args.model_name == 'pixelhvae_2level' or args.model_name == 'convhvae_2level-smim' or args.model_name == 'conv_wae_2level':
                    _,_,z1_q,_,_,z2_q,_,_,_,_ = model(data) 
                    embed_z.append(torch.cat([z1_q, z2_q], dim=1))               

                else:
                    raise Exception('Wrong name of the model!')


            embed_z = torch.cat(embed_z, dim=0)

        return embed_z

    # get reconstructed x
    def recon_x(x, batch = args.batch_size):
        with torch.no_grad():
            
            recon_x = []
            for data in torch.split(x, batch):

                if args.model_name == 'vae' or args.model_name == 'conv_vae':
                    recon_x_batch,_,_,_,_ = model(data)
                    recon_x.append(recon_x_batch)

                elif args.model_name == 'MLP_wae' or args.model_name == 'conv_wae' or args.model_name == 'Pixel_wae':
                    _, recon_x_batch = model(data)
                    recon_x.append(recon_x_batch)

                elif args.model_name == 'hvae_2level' or args.model_name == 'convhvae_2level' or args.model_name == 'pixelhvae_2level' or args.model_name == 'convhvae_2level-smim' or args.model_name == 'conv_wae_2level':
                    recon_x_batch,_,_,_,_,_,_,_,_,_ = model(data)    
                    recon_x.append(recon_x_batch)               

                else:
                    raise Exception('Wrong name of the model!')

            recon_x = torch.cat(recon_x, dim=0)

        return recon_x  

    # collect all additional stats
    stats = {}

    # project train/test data to latent representation
    train_target = full_target.detach().cpu().numpy()
    train_z = embed_z(full_data)
    train_z = train_z.detach().cpu().numpy()

    test_target = test_target.detach().cpu().numpy()
    test_labels = np.unique(test_target)
    
    test_z = embed_z(test_data)

       
    test_z = test_z.detach().cpu().numpy()
    test_recon_x = recon_x(test_data)
    test_recon_x = test_recon_x.detach().cpu().numpy()
    test_data = test_data.detach().cpu().numpy()


    # use latent for K-NN classification
    for clf_name, clf in [
        ("KNN1", KNeighborsClassifier(n_neighbors=1)),
        ("KNN3", KNeighborsClassifier(n_neighbors=3)),
        ("KNN5", KNeighborsClassifier(n_neighbors=5)),
        ("KNN10", KNeighborsClassifier(n_neighbors=10)),
    ]:

        print(clf_name)
        
        clf.fit(train_z, train_target)
        clf_score = clf.score(test_z, test_target)
        stats["clf_acc_" + clf_name] = clf_score

        with open(results_dir + 'test_log.txt', 'a+') as f:
            print("{clf_name} = {clf_score}".format(clf_name=clf_name, clf_score=clf_score), file = f)
        f.close()

    np.savez(results_dir + 'KNN_test.npz', KNN1 = stats["clf_acc_KNN1"], KNN3 = stats["clf_acc_KNN3"],
    KNN5 = stats["clf_acc_KNN5"], KNN10 = stats["clf_acc_KNN10"])

    
    #check whether data close in x-space also close in z-space in L2 norm
    samples_x = test_data[0:args.batch_size]
    dist_x = samples_x - test_data[0]
    dist_x = np.sum(np.power(dist_x,2), axis = 1)
    sort_x = np.argsort(dist_x)
    samples_x_sort = samples_x[sort_x]

    samples_z = test_z[0:args.batch_size]
    dist_z = samples_z - test_z[0]
    dist_z = np.sum(np.power(dist_z,2), axis = 1)
    sort_z = np.argsort(dist_z)
    samples_z_sort = samples_z[sort_z]

    with open(results_dir + 'test_log.txt', 'a+') as f:
        print("sort_x = {sort_x}".format(sort_x=sort_x), file = f)
        print("sort_z = {sort_z}".format(sort_z=sort_z), file = f)
    f.close()
    np.savez(results_dir + 'l2_test.npz', sort_x = sort_x, sort_z = sort_z, samples_x = samples_x, 
        samples_z = samples_z, samples_x_sort = samples_x_sort, samples_z_sort = samples_z_sort)



    #plot figures
    if np.shape(test_z)[1] > 2:
        test_z_embed = TSNE(n_components=2).fit_transform(test_z)
    else:
        test_z_embed = test_z


    plot_images_each_label(test_labels, test_z_embed, test_target, results_dir, "test_z_embed")
    



def evaluate_wae(args, model, train_loader, data_loader, epoch, results_dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_recon = 0
    evaluate_z = 0
    evaluate_x = 0
    evaluate_mi = 0
    # set model to evaluation mode
    model.eval()

    if args.number_components_input >= 100:

        M = 10
        N = 5
    else:
        N = min(8, args.number_components_input)
        M = max(1, args.number_components_input // N)

    # no warmup assumed
    if args.warmup == 0:
        beta = args.beta    

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        x = data

        # calculate loss function
        loss, recon_loss, z_loss, x_loss, mi, _, _, _, _, _, _ = model.calculate_loss(x, beta, average=True)



        evaluate_loss += loss.item()
        evaluate_recon += recon_loss.item()
        evaluate_z += z_loss.item()
        evaluate_x += x_loss.item()

        #print N digits
        if batch_idx == 0 and mode == 'validation':

            log_step = 50

            if epoch == 1:
                if not os.path.exists(results_dir + 'validation/'):
                    os.makedirs(results_dir + 'validation/')
                # VISUALIZATION: plot real images
                plot_images(args, data.data.cpu().numpy()[0:N*M], results_dir + 'validation/', 'real_x', size_x=N, size_y=M)
            
            if epoch % log_step == 0:
                x_recon = model.reconstruct_x(x)
                plot_images(args, x_recon.data.cpu().numpy()[0:N*M], results_dir + 'validation/', 'recon_x_epoch' + str(epoch), size_x=N, size_y=M)

            if args.prior == 'x_prior' and epoch % log_step == 0:
                # VISUALIZE pseudoinputs
                pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

                plot_images(args, pseudoinputs[0:N*M], results_dir + 'validation/', 
                    'pseu_x' + '_K' + str(args.number_components_input) + '_L' + str(args.number_components_latent) + '_epoch' + str(epoch), 
                    size_x=N, size_y=M)

    if mode == 'test':
        # load all data
        # grab the test data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        test_data, test_target = [], []
        for data, lbls in data_loader:
            test_data.append(data)
            test_target.append(lbls)

        test_data, test_target = [torch.cat(test_data, dim=0), torch.cat(test_target, dim=0).squeeze()]

        #test noisy input
        for i in np.linspace(0.1,0.5,num=5):
            test_noisy_data = test_data[0:N*M] + torch.randn(test_data[0:N*M].size()) * i
            if args.cuda:
                test_noisy_data = test_noisy_data.cuda()
            plot_images(args, test_noisy_data.data.cpu().numpy(), results_dir, 'real_noisy_x_' + str(round(i,1)), size_x=N, size_y=M)

            noisy_samples = model.reconstruct_x(test_noisy_data)
            plot_images(args, noisy_samples.data.cpu().numpy(), results_dir, 'recon_noisy_x_' + str(round(i,1)), size_x=N, size_y=M)


        if args.cuda:
            test_data, test_target = test_data.cuda(), test_target.cuda()



        if args.fid_score == True or args.fid_score_rec == True:
            evaluate_fid_score(args, model, results_dir, test_data)


        # VISUALIZATION: plot real images

        plot_images(args, test_data.data.cpu().numpy()[0:N*M], results_dir, 'real_x', size_x=N, size_y=M)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:N*M])
        
        plot_images(args, samples.data.cpu().numpy(), results_dir, 'recon_x', size_x=N, size_y=M)


        # VISUALIZATION: plot generations
        samples_rand = model.generate_x(N*M)
        plot_images(args, samples_rand.data.cpu().numpy(), results_dir, 'gen_x', size_x=N, size_y=M)


        if args.interpolation_test == True:        
            samples_z, _, _, _ = model.generate_z_prior(10)

            interp_x = np.empty([0, np.prod(args.input_size)])
            print('samples_z', samples_z[0], samples_z[1])

            for i in np.arange(0,1.1, 0.1):

                interp_z = (1-i) * samples_z[0] + i * samples_z[1]


                recon_x = model.decoder(interp_z).data.cpu().numpy()
                recon_x = recon_x.reshape((1, np.prod(args.input_size)))

                interp_x = np.append(interp_x, recon_x, axis = 0)
            plot_images(args, interp_x, results_dir, 'interpolate_z', size_x=1, size_y=len(np.arange(0,1.1,0.1)))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_recon /= len(data_loader)  # recon already averages over batch size
    evaluate_z /= len(data_loader)  # z already averages over batch size
    evaluate_x /= len(data_loader) 

    return evaluate_loss, evaluate_recon, evaluate_z, evaluate_x, evaluate_mi




def evaluate_vae(args, model, train_loader, data_loader, epoch, results_dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    evaluate_re_l2square = 0
    # set model to evaluation mode
    model.eval()

    if args.number_components_input >= 100:

        #M = N = 8
        M = 10
        N = 5
    else:
        N = min(8, args.number_components_input)
        M = max(1, args.number_components_input // N)

    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        x = data

        # calculate loss function
        loss, RE, KL = model.calculate_loss(x, average=True)

        # calcualte reconstruction loss L2 norm square
        recon_loss = model.recon_loss(x)


        evaluate_loss += loss.item()
        evaluate_re += -RE.item()
        evaluate_kl += KL.item()
        evaluate_re_l2square += recon_loss.item()

        # print N digits
        if batch_idx == 0 and mode == 'validation':

            log_step = 50

            if epoch == 1:
                if not os.path.exists(results_dir + 'validation/'):
                    os.makedirs(results_dir + 'validation/')
                # VISUALIZATION: plot real images
                plot_images(args, data.data.cpu().numpy()[0:N*M], results_dir + 'validation/', 'real_x', size_x=N, size_y=M)
            
            if epoch % log_step == 0:
                x_mean = model.reconstruct_x(x)
                plot_images(args, x_mean.data.cpu().numpy()[0:N*M], results_dir + 'validation/', 'recon_x_epoch' + str(epoch), size_x=N, size_y=M)

    if mode == 'test':
        # load all data
        # grab the test data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        test_data, test_target = [], []
        for data, lbls in data_loader:
            test_data.append(data)
            test_target.append(lbls)


        test_data, test_target = [torch.cat(test_data, dim=0), torch.cat(test_target, dim=0).squeeze()]

        #test noisy input
        for i in np.linspace(0.1,0.5,num=5):
            test_noisy_data = test_data[0:N*M] + torch.randn(test_data[0:N*M].size()) * i
            if args.cuda:
                test_noisy_data = test_noisy_data.cuda()
            plot_images(args, test_noisy_data.data.cpu().numpy(), results_dir, 'real_noisy_x_' + str(round(i, 1)), size_x=N, size_y=M)

            noisy_samples = model.reconstruct_x(test_noisy_data)
            plot_images(args, noisy_samples.data.cpu().numpy(), results_dir, 'recon_noisy_x_' + str(round(i, 1)), size_x=N, size_y=M)


        if args.cuda:
            test_data, test_target = test_data.cuda(), test_target.cuda()


        if args.fid_score == True or args.fid_score_rec == True:
            evaluate_fid_score(args, model, results_dir, test_data)

        # VISUALIZATION: plot real images
        plot_images(args, test_data.data.cpu().numpy()[0:N*M], results_dir, 'real_x', size_x=N, size_y=M)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:N*M])

        plot_images(args, samples.data.cpu().numpy(), results_dir, 'recon_x', size_x=N, size_y=M)

        # VISUALIZATION: plot generations
        samples_rand = model.generate_x(N*M)

        plot_images(args, samples_rand.data.cpu().numpy(), results_dir, 'gen_x', size_x=N, size_y=M)

        if args.prior == 'vampprior':
            # VISUALIZE pseudoinputs
            pseudoinputs = model.means(model.idle_input).cpu().data.numpy()

            plot_images(args, pseudoinputs[0:N*M], results_dir, 'pseudoinputs', size_x=N, size_y=M)


    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    evaluate_re_l2square /= len(data_loader)

    return evaluate_loss, evaluate_re, evaluate_kl, evaluate_re_l2square

