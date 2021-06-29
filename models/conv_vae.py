from __future__ import print_function

import numpy as np

import math

#from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visual_evaluation import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear, \
    Conv2d, GatedConv2d, GatedResUnit, ResizeGatedConv2d, MaskedConv2d, ResUnitBN, ResizeConv2d, GatedResUnit, GatedConvTranspose2d

#from utils.nn import he_init, GatedDense, NonLinear

from models.Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        if self.args.dataset_name == 'freyfaces':
            h_size = 210
        elif self.args.dataset_name.startswith('cifar10') or self.args.dataset_name == 'coil20' or self.args.dataset_name == 'svhn':
            h_size = 384
        elif self.args.dataset_name == 'usps':
            h_size = 96
        elif args.dataset_name == 'celeba':
            h_size = 1536
        else:
            h_size = 294

        # encoder: q(z | x)
        self.q_z_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )



        self.q_z_mean = NonLinear(h_size, self.args.z1_size, activation=None)
        self.q_z_logvar = NonLinear(h_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))


        self.fc = nn.Sequential(
            GatedDense(self.args.z1_size, 300),
            GatedDense(300, np.prod(self.args.input_size))
        )


        self.p_x_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 1),
        )


        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(64, self.args.input_size[0], 1, 1, 0, activation=nn.Sigmoid())
            self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1,
                                     0, activation=nn.Hardtanh(min_val=-4.5, max_val=0.))

 
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs if VampPrior
        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

    # AUXILIARY METHODS
    def calculate_loss(self, x, beta=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        # KL
        log_p_z = self.log_p_z(z_q)
        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)

        loss = - RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=100):
        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = X[j].unsqueeze(0)

            a = []
            for r in range(0, int(R)):
                # Repeat it for all training points
                x = x_single.expand(S, x_single.size(1))

                a_tmp, _, _ = self.calculate_loss(x)

                a.append( -a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full, MB=100):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))

            loss, RE, KL = self.calculate_loss(x,average=True)

            # RE_all += RE.cpu().data[0]
            # KL_all += KL.cpu().data[0]
            # lower_bound += loss.cpu().data[0]
            RE_all += RE.item()
            KL_all += KL.item()
            lower_bound += loss.item()

        lower_bound /= I

        return lower_bound

    # ADDITIONAL METHODS
    def generate_x_batch(self, N=25):
        if self.args.prior == 'standard':
            z_sample_rand = Variable( torch.FloatTensor(N, self.args.z1_size).normal_() )
            if self.args.cuda:
                z_sample_rand = z_sample_rand.cuda()

        elif self.args.prior == 'vampprior':
            n_pseu_input = self.args.number_components_input
            # assume each component has the same probability
            coefficients = 1/n_pseu_input * np.ones((n_pseu_input,))
            coefficients /= coefficients.sum()      # in case these did not add up to 1
            random_idx = np.random.choice(np.arange(n_pseu_input), size=(N,), p=coefficients)

            #means = self.means(self.idle_input)[0:N]
            means = self.means(self.idle_input)[random_idx]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)

        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand

    def generate_x(self, N=25):  

        batch_size = 100

        # to save memory generate images by batches
        if (self.args.gen_batch == False) or (N <= batch_size):
            sample_x = self.generate_x_batch(N)
            
        else:
            sample_x = []

            for i in range(int(N/batch_size)):
                sample_x_batch = self.generate_x_batch(N)
                sample_x.append(sample_x_batch)

            sample_x = torch.cat(sample_x, dim=0)
            
        return sample_x    

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    def decoder(self, z):
        samples_gen, _ = self.p_x(z)
        return samples_gen   

    def recon_loss(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        x_dif = x_mean - x

        #L2 norm square
        recon_loss = x_dif.pow(2).sum() / x.size(0)   
        return recon_loss

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        h = self.q_z_layers(x)
        h = h.view(x.size(0),-1)

        z_q_mean = self.q_z_mean(h)
        z_q_logvar = self.q_z_logvar(h)
        return z_q_mean, z_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):

        h = self.fc(z)
        h = h.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        h = self.p_x_layers(h)

        x_mean = self.p_x_mean(h).view(-1,np.prod(self.args.input_size))

        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(h).view(-1,np.prod(self.args.input_size))
        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)

        elif self.args.prior == 'vampprior':
            # z - MB x M
            C = self.args.number_components_input

            # calculate params
            X = self.means(self.idle_input)

            # calculate params for given data
            z_p_mean, z_p_logvar = self.q_z(X)  # C x M

            # expand z
            z_expand = z.unsqueeze(1)
            means = z_p_mean.unsqueeze(0)
            logvars = z_p_logvar.unsqueeze(0)

            a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
            a_max, _ = torch.max(a, 1)  # MB x 1

            # calculte log-sum-exp
            log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

        else:
            raise Exception('Wrong name of the prior!')

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])

        # z ~ q(z | x)

        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar
