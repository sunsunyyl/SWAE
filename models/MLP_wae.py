from __future__ import print_function

import numpy as np
from numpy import linalg as LA

import math

#from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visual_evaluation import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear

from models.Model import Model
from utils.MINE import MINE
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#assume deterministic encoder and decoder
#=======================================================================================================================
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.input_size = args.input_size

        self.n_z = args.z_size

        self.main = nn.Sequential(
            GatedDense(np.prod(self.input_size), 300),
            GatedDense(300, 300)
        )

        self.fc = NonLinear(300, self.n_z, activation=None)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    def forward(self, x):
        x = self.main(x)
        x = self.fc(x)
        return x


#p(z_d|x_e): for deterministic encoder and decoder 
class Encoder_z_prior(nn.Module):
    def __init__(self, args):
        super(Encoder_z_prior, self).__init__()

        self.input_size = args.input_size

        self.n_z = args.z_size

        self.main = nn.Sequential(
            GatedDense(np.prod(self.input_size), 300),
            GatedDense(300, 300)
        )

        #assume Gaussian distribution for p(z_d|x_e)
        self.p_z_mean = NonLinear(300, self.n_z, activation=None)
        self.p_z_logvar = NonLinear(300, self.n_z, activation=nn.Hardtanh(min_val=-9., max_val=-3.2))


        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    def forward(self, x):
        x = self.main(x)

        return self.p_z_mean(x), self.p_z_logvar(x)



class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.input_size = args.input_size

        self.n_z = args.z_size

        self.main = nn.Sequential(
            GatedDense(self.n_z, 300),
            GatedDense(300, 300)
        )

        self.fc = NonLinear(300, np.prod(self.input_size), activation=None)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    def forward(self, x):
        x = self.main(x)
        x = self.fc(x)
        return x        


class WAE(Model):
    def __init__(self, args):
        super(WAE, self).__init__(args)
       
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        #mean and variance of prior z
        self.encoder_z_prior = Encoder_z_prior(args)
       
        #add psuedo inputs
        self.add_pseudoinputs()

        if args.MI_obj != 'None':
            self.MI = MINE(args)

    def calculate_loss(self, x, beta=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        
        z_e = self.encoder(x)
        x_recon = self.decoder(z_e)
             
        x_dif = x_recon - x
        recon_loss = x_dif.pow(2).sum() / x.size(0)

        #get samples of z_d prior
        z_d, x_pseudo, z_sample_gen_mean, z_sample_gen_logvar = self.generate_z_prior_on_x(x)
        z_dif = z_e - z_d
        z_loss = z_dif.pow(2).sum() / x.size(0)

        x_d_recon = self.decoder(z_d)
        x_d_dif = x_d_recon - x
        x_loss = x_d_dif.pow(2).sum() / x.size(0)

        # designed for mutual information  
        mi = torch.zeros(1)
        loss = beta * x_loss +  z_loss

        return loss, recon_loss, z_loss, x_loss, mi, z_e.data.cpu().numpy(), z_d.data.cpu().numpy(), x_recon.data.cpu().numpy(), x_pseudo.data.cpu().numpy(), z_sample_gen_mean.data.cpu().numpy(), z_sample_gen_logvar.data.cpu().numpy()
                    

    # generate zd from p(zd|xe)
    def generate_z_prior_on_x(self, x):

        #get pseudo-inputs
        means = self.means(self.idle_input)
        #find the closest psedu-input for each xe
        psedu_idx = np.empty([0,1])

        for i in range(x.size(0)):
            data = x[i,:].view(1,-1)
            
            if self.args.cosine_similarity == True:                
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                output = cos(data, means)
                _,ind_max = torch.max(output, dim = 0)
                psedu_idx = np.append(psedu_idx, ind_max.data.cpu().numpy())
            else:                
                dist = torch.norm(means - data, dim = 1)
                knn = dist.topk(1, largest = False)      
                psedu_idx = np.append(psedu_idx, knn.indices.data.cpu().numpy())



        means_closest = means[psedu_idx]

        z_sample_gen_mean, z_sample_gen_logvar = self.encoder_z_prior(means_closest)

        z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)

        return z_sample_rand, means_closest, z_sample_gen_mean, z_sample_gen_logvar   


    def generate_z_prior(self, N=25):

        n_pseu_input = self.args.number_components_input

        # assume each component has the same probability
        coefficients = 1/n_pseu_input * np.ones((n_pseu_input,))
        coefficients /= coefficients.sum()      # in case these did not add up to 1
        random_idx = np.random.choice(np.arange(n_pseu_input), size=(N,), p=coefficients)

        #print('random_idx', random_idx)

        if N <= n_pseu_input:
            # size of means: #number of pseudo-inputs * dimension of input x
            #means = self.means(self.idle_input)[0:N]
            # pseudo-inputs that are used to generate z prior
            means = self.means(self.idle_input)[random_idx]
        else:
            print('Not enough number of pseudo-inputs.')
        #print('means', means)
        #print('self.means.linear.weight.data', self.means.linear.weight.data)
  
        z_sample_gen_mean, z_sample_gen_logvar = self.encoder_z_prior(means)

        z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
        
        return z_sample_rand, means, z_sample_gen_mean, z_sample_gen_logvar
    

    #generate new samples x from prior z
    def generate_x(self, N=25):  

        sample_z, _, _, _ = self.generate_z_prior(N)
        sample_x = self.decoder(sample_z) 

        return  sample_x

    def reconstruct_x(self, x):

        x_recon = self.decoder(self.encoder(x))

        return x_recon 

    def forward(self, x):
        
        z_latent = self.encoder(x)
        x_recon = self.decoder(z_latent)

        return z_latent, x_recon




    
       

#=======================================================================================================================

