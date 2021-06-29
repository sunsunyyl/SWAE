from __future__ import print_function

import numpy as np

import math
import time
#from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visual_evaluation import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear,normal_init,xavier_init, \
	Conv2d, GatedConv2d, GatedResUnit, ResizeGatedConv2d, MaskedConv2d, ResUnitBN, ResizeConv2d, GatedResUnit, GatedConvTranspose2d

from models.Model import Model
from utils.MINE import MINE
from utils.create_paths import create_dirNames
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()

		self.input_size = args.input_size

		if args.dataset_name == 'freyfaces':
			h_size = 210
		elif args.dataset_name.startswith('cifar10') or args.dataset_name == 'coil20' or args.dataset_name == 'svhn':
			h_size = 384
		elif args.dataset_name == 'usps':
			h_size = 96
		elif args.dataset_name == 'celeba':
			h_size = 1536
		else:
			h_size = 294

		self.n_z = args.z_size


		self.main = nn.Sequential(
			GatedConv2d(self.input_size[0], 32, 7, 1, 3),
			GatedConv2d(32, 32, 3, 2, 1),
			GatedConv2d(32, 64, 5, 1, 2),
			GatedConv2d(64, 64, 3, 2, 1),
			GatedConv2d(64, 6, 3, 1, 1)
		)

		# linear layers
		self.fc = NonLinear(h_size, self.n_z, activation=None)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Linear):
				he_init(m)

	def forward(self, x):

		x = x.view(-1, self.input_size[0], self.input_size[1], self.input_size[2])
		h = self.main(x)

		h = h.view(x.size(0),-1)

		z = self.fc(h)
		return z


class Encoder_z_prior(nn.Module):
	def __init__(self, args):
		super(Encoder_z_prior, self).__init__()

		self.input_size = args.input_size

		self.n_z = args.z_size

		if args.dataset_name == 'freyfaces':
			h_size = 210
		elif args.dataset_name.startswith('cifar10') or args.dataset_name == 'coil20' or args.dataset_name == 'svhn':
			h_size = 384
		elif args.dataset_name == 'usps':
			h_size = 96
		elif args.dataset_name == 'celeba':
			h_size = 1536
		else:
			h_size = 294


		self.main = nn.Sequential(
			GatedConv2d(self.input_size[0], 32, 7, 1, 3),
			GatedConv2d(32, 32, 3, 2, 1),
			GatedConv2d(32, 64, 5, 1, 2),
			GatedConv2d(64, 64, 3, 2, 1),
			GatedConv2d(64, 6, 3, 1, 1)
		)

		# linear layers
		self.p_z_mean = NonLinear(h_size, self.n_z, activation=None)
		self.p_z_logvar = NonLinear(h_size, self.n_z, activation=nn.Hardtanh(min_val=-6., max_val=2.))

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Linear):
				he_init(m)

	def forward(self, x):

		x = x.view(-1, self.input_size[0], self.input_size[1], self.input_size[2])
		h = self.main(x)
		h = h.view(x.size(0),-1)

		return self.p_z_mean(h), self.p_z_logvar(h)


class Decoder(nn.Module):
	def __init__(self, args):
		super(Decoder, self).__init__()

		self.input_size = args.input_size
		self.input_type = args.input_type

		self.n_z = args.z_size


		self.fc = nn.Sequential(
			GatedDense(self.n_z, 300),
			GatedDense(300, np.prod(self.input_size))
		)


		self.main = nn.Sequential(
			GatedConv2d(self.input_size[0], 64, 3, 1, 1),
			GatedConv2d(64, 64, 3, 1, 1),
			GatedConv2d(64, 64, 3, 1, 1),
			GatedConv2d(64, 64, 3, 1, 1)
		)


		if self.input_type == 'binary':
			self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
		elif self.input_type == 'gray' or self.input_type == 'continuous':
			self.p_x_mean = Conv2d(64, self.input_size[0], 1, 1, 0, activation=nn.Sigmoid())


		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Linear):
				he_init(m)



	def forward(self, x):
	
		h = self.fc(x)
		h = h.view(-1, self.input_size[0], self.input_size[1], self.input_size[2])

		x = self.main(h)

		x_mean = self.p_x_mean(x).view(-1,np.prod(self.input_size))


		if self.input_type == 'gray' or self.input_type == 'continuous':
			x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)     

		return x_mean



class WAE(Model):
	def __init__(self, args):
		super(WAE, self).__init__(args)
	   
		self.encoder = Encoder(args)
		self.decoder = Decoder(args)
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

		if self.args.obj == 'xz':
			if self.args.MI_obj == 'None' or self.args.opt_sep == True:
				loss = self.args.beta * x_loss +  self.args.lam * z_loss
			else:
				loss = self.args.beta * x_loss + self.args.lam * z_loss - self.args.alpha * mi
		elif self.args.obj == 'xzrec':
			loss = self.args.beta * x_loss  +  (1-self.args.beta) * recon_loss + self.args.lam * z_loss
		elif self.args.obj == 'zrec':
			loss = self.args.beta * recon_loss + self.args.lam * z_loss
		else:
			print('Wrong args.obj!')

		return loss, recon_loss, z_loss, x_loss, mi, z_e.data.cpu().numpy(), z_d.data.cpu().numpy(), x_recon.data.cpu().numpy(), x_pseudo.data.cpu().numpy(), z_sample_gen_mean.data.cpu().numpy(), z_sample_gen_logvar.data.cpu().numpy()
		
			

	# generate zd from p(zd|xe)
	def generate_z_prior_on_x(self, x):

		#get pseudo-inputs
		means = self.means(self.idle_input)
		#find the closest psedu-input for each xe
		psedu_idx = np.empty([0,1])

		#time_start = time.time()

		# use latent representation for comparison
		means_data = means
		if self.args.latent_similarity == True:
			x = self.encoder(x)
			means_data = self.encoder(means)

		for i in range(x.size(0)):
			data = x[i,:].view(1,-1)
			if self.args.cosine_similarity == True:                
				cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
				output = cos(data, means_data)
				_,ind_max = torch.max(output, dim = 0)
				psedu_idx = np.append(psedu_idx, ind_max.data.cpu().numpy())
			else:                
				dist = torch.norm(means_data - data, dim = 1)
				knn = dist.topk(1, largest = False)      
				psedu_idx = np.append(psedu_idx, knn.indices.data.cpu().numpy())

		#time_end = time.time()


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
		means = self.means(self.idle_input)[random_idx]
  
		z_sample_gen_mean, z_sample_gen_logvar = self.encoder_z_prior(means)

		z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
		
		return z_sample_rand, means, z_sample_gen_mean, z_sample_gen_logvar
	

	#generate new samples x from prior z
	def generate_x(self, N=25):  

		batch_size = 100

		# to save memory generate images by batches
		if (self.args.gen_batch == False) or (N <= batch_size):
			sample_z, _, _, _ = self.generate_z_prior(N)
			sample_x = self.decoder(sample_z) 
			
		else:
			n_pseu_input = self.args.number_components_input
			if n_pseu_input < batch_size:
				batch_size = n_pseu_input
			# N should be  * times of batch_size without remainder
			sample_x = []

			for i in range(int(N/batch_size)):
				sample_z, _, _, _ = self.generate_z_prior(batch_size)
				sample_x_batch = self.decoder(sample_z) 
				sample_x.append(sample_x_batch)


			sample_x = torch.cat(sample_x, dim=0)
			
		return sample_x
		

	def reconstruct_x(self, x):

		x_recon = self.decoder(self.encoder(x))

		return x_recon 

	def forward(self, x):
		
		z_latent = self.encoder(x)
		x_recon = self.decoder(z_latent)

		return z_latent, x_recon




	
	   

#=======================================================================================================================


