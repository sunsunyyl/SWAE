import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.nn import he_init, GatedDense, NonLinear, Conv2d, GatedConv2d, GatedConvTranspose2d

#from __future__ import print_function


# neural network for calucating mutual information between p and q
class MINE(nn.Module):
	
	def __init__(self, args):
		super().__init__()

		self.input_size = args.input_size
		self.n_z = args.z_size
		self.MI_obj = args.MI_obj

		if args.dataset_name == 'freyfaces':
			h_size = 210
		elif args.dataset_name == 'cifar10':
			h_size = 384
		else:
			h_size = 294

		self.main_x = nn.Sequential(
			Conv2d(self.input_size[0], 32, 7, 1, 3, activation = nn.ELU()),
			Conv2d(32, 32, 3, 2, 1, activation = nn.ELU()),
			Conv2d(32, 64, 5, 1, 2, activation = nn.ELU()),
			Conv2d(64, 64, 3, 2, 1, activation = nn.ELU()),
			Conv2d(64, 6, 3, 1, 1, activation = nn.ELU())
		)


		self.main_z = NonLinear(self.n_z, h_size, activation=None)    
		self.joint = NonLinear(2 * h_size, 1, activation=None) 
        
		
		self.ma_et = None
		
        # weights initialization
		for m in self.modules():
			if isinstance(m, nn.Linear):
				he_init(m)
				
	def forward(self, p, q):

		if self.MI_obj == 'ze_zd': 
			print('Not allowed!')     
		elif self.MI_obj == 'xe_zd' or self.MI_obj == 'xd_ze':
			x = p.view(-1, self.input_size[0], self.input_size[1], self.input_size[2])
			x = self.main_x(x)
			x = x.view(x.size(0),-1)
			z = self.main_z(q)
			h = torch.cat((x,z), dim = 1)
			h = self.joint(h)
		else:
			raise Exception('Wrong name of MI_obj!')

		h = h.double()

		return h



