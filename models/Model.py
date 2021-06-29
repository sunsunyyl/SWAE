from __future__ import print_function

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from utils.nn import normal_init, NonLinear
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

    # AUXILIARY METHODS
    def add_pseudoinputs(self):

        if self.args.dataset_name.startswith('gmm'):
            nonlinearity = None
        else:            
            nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)

        self.means = NonLinear(self.args.number_components_input, np.prod(self.args.input_size), bias=False, activation=nonlinearity)

        # init pseudo-inputs
        if self.args.use_training_data_init:
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = torch.eye(self.args.number_components_input, self.args.number_components_input)

        if self.args.cuda:
            self.idle_input = self.idle_input.cuda()


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


