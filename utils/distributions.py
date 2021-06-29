from __future__ import print_function
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

min_epsilon = 1e-5
max_epsilon = 1.-1e-5
#=======================================================================================================================
def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Bernoulli(x, mean, average=False, dim=None):
    probs = torch.clamp( mean, min=min_epsilon, max=max_epsilon )
    log_bernoulli = x * torch.log( probs ) + (1. - x ) * torch.log( 1. - probs )
    if average:
        return torch.mean( log_bernoulli, dim )
    else:
        return torch.sum( log_bernoulli, dim )

def logisticCDF(x, u, s):
    return 1. / ( 1. + torch.exp( -(x-u) / s ) )

def sigmoid(x):
    return 1. / ( 1. + torch.exp( -x ) )

def log_Logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, dim)
        else:
            return torch.sum(log_logist_256, dim)
    else:
        return log_logist_256



#generate samples from GMM in 2D
def sample_GMM_d2(n): 

    n_samples = n
    #dimension of data
    n_dim = 2

    # mean of each Gaussian compoment
    mu_normal = np.array([[0,0],[1,1],[2,2],[3,3],[4,4]])
    # standard deviation: assume diagonal
    std_normal = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    sigma_normal = np.array([[std_normal[0],std_normal[0]], [std_normal[1],std_normal[1]], [std_normal[2],std_normal[2]], 
        [std_normal[3],std_normal[3]],[std_normal[4],std_normal[4]]])
    # number of Gaussian components
    k_modes = len(mu_normal)

    distributions = []
    for i in range(k_modes):
        distributions.append({"type": np.random.normal, "kwargs": {"loc": mu_normal[i], "scale": sigma_normal[i]}})
    
    # distributions = [
    # {"type": np.random.normal, "kwargs": {"loc": mu_normal[0], "scale": sigma_normal[0]}},
    # {"type": np.random.normal, "kwargs": {"loc": mu_normal[1], "scale": sigma_normal[1]}},
    # ]

    coefficients = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    coefficients /= coefficients.sum()      # in case these did not add up to 1
   
    data = np.zeros((n_samples, n_dim, k_modes))
    for idx, distr in enumerate(distributions):
        data[:,:,idx] = distr["type"](size=(n_samples,n_dim,), **distr["kwargs"])
    random_idx = np.random.choice(np.arange(k_modes), size=(n_samples,), p=coefficients)
    # sample data from mixture model
    #samp_gmm = data[np.arange(n_samples), n_dim, random_idx]
    samp_gmm = np.zeros((n_samples, n_dim))
    samp_labels = np.zeros((n_samples,1))
    for i in np.arange(n_samples):
        samp_gmm[i,:] = data[i,:,random_idx[i]]
        samp_labels[i] = random_idx[i]
        #print('data[i,:,random_idx[i]]', data[i,:,random_idx[i]].shape)

    #samp_gmm = data[np.arange(n_samples), np.arange(n_dim), random_idx]

    return samp_gmm, samp_labels


#generate samples from GMM 
def sample_GMM(n_samples, n_dim, n_modes): 

    sigma_normal = 0.2 * np.ones(n_dim)
    sigma_normal = np.diag(sigma_normal)
    mu_normal = np.empty([n_modes, n_dim])

    coefficients = (1/n_modes) * np.ones(n_modes) 
    data = np.zeros((n_samples, n_dim, n_modes))
 
    for i in range(n_modes):
        mu_normal[i,:] = i * np.ones([1,n_dim])
        data[:,:,i] = np.random.multivariate_normal(mu_normal[i], sigma_normal, size=n_samples)

    random_idx = np.random.choice(np.arange(n_modes), size=(n_samples,), p=coefficients)
    # sample data from mixture model
    #samp_gmm = data[np.arange(n_samples), n_dim, random_idx]
    samp_gmm = np.zeros((n_samples, n_dim))
    samp_labels = np.zeros((n_samples,1))
    for i in np.arange(n_samples):
        samp_gmm[i,:] = data[i,:,random_idx[i]]
        samp_labels[i] = random_idx[i]
        #print('data[i,:,random_idx[i]]', data[i,:,random_idx[i]].shape)


    return samp_gmm, samp_labels


