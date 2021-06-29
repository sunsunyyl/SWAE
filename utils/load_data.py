from __future__ import print_function

import torch
import torch.utils.data as data_utils

import numpy as np

from scipy.io import loadmat
import os

import pickle

#from utils.auxiliary import GMMDist
#from utils.datasets import DistDataset
from utils.distributions import sample_GMM
from utils.visual_evaluation import plot_images
import matplotlib.pyplot as plt
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def load_static_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_dynamic_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True
    #args.dynamic_binarization = False

    # if args.dataset_name == 'dynamic_mnist_static':
    #     args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=False,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                               batch_size=args.batch_size, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.train_data.float().numpy() / 255.

    

    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )
    y_train = np.array( train_loader.dataset.train_labels.float().numpy(), dtype=int)

    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

    y_test = np.array( test_loader.dataset.test_labels.float().numpy(), dtype=int)

    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================

def load_dynamic_fashion_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(os.path.join('datasets', "fashion-mnist"), train=True, download=True,
                                                                     transform=transforms.Compose([
                                                                         transforms.ToTensor()
                                                                     ])),
                                               batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(os.path.join('datasets', "fashion-mnist"), train=False,
                                                                    transform=transforms.Compose([transforms.ToTensor()
                                                                                                  ])),
                                              batch_size=args.batch_size, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.train_data.float().numpy() / 255.
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

    y_train = np.array(train_loader.dataset.train_labels.float().numpy(), dtype=int)

    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    y_test = np.array(test_loader.dataset.test_labels.float().numpy(), dtype=int)


    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy(
            init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input)).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args 

# ======================================================================================================================
def load_omniglot(args, n_validation=1345, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')
    omni_raw = loadmat(os.path.join('datasets', 'omniglot', 'chardata.mat'))

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data: train data 23000
    x_train = train_data[:-n_validation]
    x_val = train_data[-n_validation:]

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    y_train = omni_raw['targetchar'].reshape((-1, 1))[:-n_validation]
    y_val = omni_raw['targetchar'].reshape((-1, 1))[-n_validation:]
    y_test = omni_raw['testtargetchar'].reshape((-1, 1))

    # shuffle test data once to get variety of characters
    I_test = np.arange(y_test.shape[0])
    np.random.shuffle(I_test)
    x_test = x_test[I_test]
    y_test = y_test[I_test]

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy(
            init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input)).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args
    
# ======================================================================================================================
def load_caltech101silhouettes(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='F')
    caltech_raw = loadmat(os.path.join('datasets', 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels'] #4100 training data
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input) ).float()
    else:
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.02

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_histopathologyGray(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    # start processing
    with open('datasets/HistopathologyGray/histopathology.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    x_train = np.asarray(data['training']).reshape(-1, 28 * 28)
    x_val = np.asarray(data['validation']).reshape(-1, 28 * 28)
    x_test = np.asarray(data['test']).reshape(-1, 28 * 28)

    x_train = np.clip(x_train, 1./512., 1. - 1./512.)
    x_val = np.clip(x_val, 1./512., 1. - 1./512.)
    x_test = np.clip(x_test, 1./512., 1. - 1./512.)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) ) #6800 training data
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    
    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input) ).float()
    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_freyfaces(args, TRAIN = 1565, VAL = 200, TEST = 200, **kwargs):
    # set args
    args.input_size = [1, 28, 20]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    # start processing
    with open('datasets/Freyfaces/freyfaces.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    data = (data[0] + 0.5) / 256.

    # shuffle data:
    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN].reshape(-1, 28*20)
    # validation images
    x_val = data[TRAIN:(TRAIN + VAL)].reshape(-1, 28*20)
    # test images
    x_test = data[(TRAIN + VAL):(TRAIN + VAL + TEST)].reshape(-1, 28*20)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input)).float()
    else:
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.02

    return train_loader, val_loader, test_loader, args


# ======================================================================================================================
def load_cifar10(args, **kwargs):
    # set args
    args.input_size = [3, 32, 32]
    args.input_type = 'continuous'
    args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load main train dataset
    train_loader = data_utils.DataLoader(datasets.CIFAR10('datasets/Cifar10/', train=True, download=True, 
        transform=transform), batch_size=args.batch_size, shuffle=True)

    test_loader = data_utils.DataLoader(datasets.CIFAR10('datasets/Cifar10/', train=False, 
        transform=transform), batch_size=args.batch_size, shuffle=True)

    
    # preparing data
    x_train = np.clip((train_loader.dataset.data + 0.5) /256., 0., 1.)
    x_train = np.swapaxes(np.swapaxes(x_train,1,2), 1, 3)
    x_train = np.reshape(x_train, (-1, np.prod(args.input_size)))
    x_train = x_train.astype('float32')
    y_train = np.array(train_loader.dataset.targets, dtype=int)

    x_test = np.clip((test_loader.dataset.data + 0.5) /256., 0., 1.)
    x_test = np.swapaxes(np.swapaxes(x_test,1,2), 1, 3)
    x_test = np.reshape(x_test, (-1, np.prod(args.input_size)))
    x_test = x_test.astype('float32')
    y_test = np.array(test_loader.dataset.targets, dtype=int)

    #validation set
    x_val = x_train[40000:50000]
    x_train = x_train[0:40000]

    y_val = y_train[40000:50000]    
    y_train = y_train[0:40000]

    #only pick samples with specific classes: 
    # airplane(0) automobile(1) bird(2) cat(3) deer(4) dog(5) frog(6) horse(7) ship(8) truck(9)
    if args.dataset_name.startswith('cifar10sub'):
        labels = [2,3,8]

        y_test_index = np.concatenate((np.reshape(np.where(y_test == labels[0]),-1),
                         np.reshape(np.where(y_test == labels[1]),-1), np.reshape(np.where(y_test == labels[2]),-1)))

        np.random.shuffle(y_test_index)
        y_test = y_test[y_test_index]
        x_test = x_test[y_test_index]

        y_train_index = np.concatenate((np.reshape(np.where(y_train == labels[0]),-1),
                         np.reshape(np.where(y_train == labels[1]),-1), np.reshape(np.where(y_train == labels[2]),-1)))

        np.random.shuffle(y_train_index)
        y_train = y_train[y_train_index]
        x_train = x_train[y_train_index]

        y_val_index = np.concatenate((np.reshape(np.where(y_val == labels[0]),-1),
                         np.reshape(np.where(y_val == labels[1]),-1), np.reshape(np.where(y_val == labels[2]),-1)))

        np.random.shuffle(y_val_index)
        y_val = y_val[y_val_index]
        x_val = x_val[y_val_index]

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

 
    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input) ).float()
    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args


# ======================================================================================================================
def load_celeba(args, **kwargs):
    # set args
    args.input_size = [3, 64, 64]
    args.input_type = 'continuous'
    args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        #transforms.CenterCrop((140, 140)),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    #train/validation/test: 162770 + 19867 + 19962 = 202599
    num_train_data = args.num_train_data
    num_val_data = args.num_val_data
    num_test_data = args.num_test_data

    num_data = num_train_data + num_val_data + num_test_data

    # load data: 202,599 samples in [0,1]
    data_loader = data_utils.DataLoader(datasets.ImageFolder('datasets/celeba', transform), 
        batch_size=args.batch_size, shuffle=True, **kwargs)


    x_data = []
    for batch_idx, data in enumerate(data_loader, 1):

        if args.cuda:
            data = data[0]
            
        #data[0] torch.Size([100, 3, 64, 64])
        data = torch.reshape(data, (-1, np.prod(args.input_size)))
        x_data.append(data)

        if batch_idx == num_data/args.batch_size:
            break

    x_data = torch.cat(x_data, dim=0)
    print(x_data.size())
    # add fake labels
    y_labels = torch.zeros(num_data)

    x_train = x_data[:num_train_data]
    y_train = y_labels[:num_train_data]

    x_val = x_data[num_train_data:(num_train_data+num_val_data)]
    y_val = y_labels[num_train_data:(num_train_data+num_val_data)]

    x_test = x_data[(num_train_data+num_val_data):]
    y_test = y_labels[(num_train_data+num_val_data):]


    # pytorch data loader
    train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=False, **kwargs)

    validation = data_utils.TensorDataset(x_val, y_val)
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

 
    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = init + args.pseudoinputs_std * torch.randn(np.prod(args.input_size), args.number_components_input) 

    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args

# ======================================================================================================================
def load_svhn(args, **kwargs):
    # set args
    args.input_size = [3, 32, 32]
    args.input_type = 'continuous'
    args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load main train dataset
    train_loader = data_utils.DataLoader(datasets.SVHN('datasets/svhn/', split='train', download=True, 
        transform=transform), batch_size=args.batch_size, shuffle=True)

    test_loader = data_utils.DataLoader(datasets.SVHN('datasets/svhn/', split='test', download = True,
        transform=transform), batch_size=args.batch_size, shuffle=True)

    
    # preparing data
    x_train = np.clip((train_loader.dataset.data + 0.5) /256., 0., 1.)
    x_train = np.reshape(x_train, (-1, np.prod(args.input_size)))
    x_train = x_train.astype('float32')
    y_train = np.array(train_loader.dataset.labels, dtype=int)

    x_test = np.clip((test_loader.dataset.data + 0.5) /256., 0., 1.)
    x_test = np.reshape(x_test, (-1, np.prod(args.input_size)))
    x_test = x_test.astype('float32')
    y_test = np.array(test_loader.dataset.labels, dtype=int)

    #validation set; 73257 digits for training, 26032 digits for testing,
    x_val = x_train[60000:]
    x_train = x_train[0:60000]

    y_val = y_train[60000:]    
    y_train = y_train[0:60000]


    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

 
    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input) ).float()
    else:
        args.pseudoinputs_mean = 0.4
        args.pseudoinputs_std = 0.05

    return train_loader, val_loader, test_loader, args



# ======================================================================================================================

def load_usps(args, **kwargs):
    # set args
    args.input_size = [1, 16, 16]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader(datasets.USPS(os.path.join('datasets', "usps"), train=True, download=True,
                                                                     transform=transforms.Compose([
                                                                         transforms.ToTensor()
                                                                     ])),
                                               batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.USPS(os.path.join('datasets', "usps"), train=False, download = True,
                                                                    transform=transforms.Compose([transforms.ToTensor()
                                                                                                  ])),
                                              batch_size=args.batch_size, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.data.astype('float32')
    x_train = x_train / 255.
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

    y_train = np.array(train_loader.dataset.targets, dtype=int)


    x_test = test_loader.dataset.data.astype('float32')
    x_test = x_test / 255.
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    y_test = np.array(test_loader.dataset.targets, dtype=int)

    # validation set; 7291 data points in training dataset; 2007 in test dataset
    x_val = x_train[6000:]
    y_val = np.array(y_train[6000:], dtype=int)
    x_train = x_train[0:6000]
    y_train = np.array(y_train[0:6000], dtype=int)


    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy(
            init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input)).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args 


# ======================================================================================================================
def load_coil20(args, TRAIN = 1040, VAL = 200, TEST = 200, **kwargs):
    # set args
    args.input_size = [1, 32, 32]
    args.input_type = 'gray'
    args.dynamic_binarization = False

    coil20_raw = loadmat(os.path.join('datasets', 'coil20', 'coil20.mat'))
    x_data = np.reshape(coil20_raw['fea'].astype('float32'), (-1, 32*32))
    y_labels = coil20_raw['gnd'].reshape((-1,1))

    Index = np.arange(y_labels.shape[0])
    np.random.shuffle(Index)
    x_data = x_data[Index]
    y_labels = y_labels[Index]


    x_train = x_data[0:TRAIN]
    x_val = x_data[TRAIN: (TRAIN + VAL)]
    x_test = x_data[(TRAIN + VAL): (TRAIN + VAL + TEST)]

    y_train = y_labels[0:TRAIN]
    y_val = y_labels[TRAIN: (TRAIN + VAL)]
    y_test = y_labels[(TRAIN + VAL): (TRAIN + VAL + TEST)]



    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components_input) ).float()
    else:
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.02

    return train_loader, val_loader, test_loader, args


# ======================================================================================================================
def load_gmm(args, **kwargs):

    #dataset_name = 'gmm_mode_dimension', e.g, gmm_m4_d20; 1 digit for mode 
    n_modes = int(args.dataset_name[5])
    x_dim = int(args.dataset_name[8:])

    print('n_modes=', n_modes, 'x_dim=', x_dim)
    
    args.input_size = [1, 1, x_dim]
    args.input_type = 'continuous'
    args.dynamic_binarization = False


    x_train, y_train = sample_GMM(args.num_train_data, x_dim, n_modes)
    x_val, y_val = sample_GMM(args.num_val_data, x_dim, n_modes)
    x_test, y_test = sample_GMM(args.num_test_data, x_dim, n_modes)

    # train loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    # validation loader
    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # test loader
    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)


   
    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components_input].T
        args.pseudoinputs_mean = torch.from_numpy(init).float()

    else:
        args.pseudoinputs_mean = 1
        args.pseudoinputs_std = 0.2

    return train_loader, val_loader, test_loader, args


# ======================================================================================================================
def load_dataset(args, **kwargs):
    if args.dataset_name == 'static_mnist':
        train_loader, val_loader, test_loader, args = load_static_mnist(args, **kwargs)
    #elif args.dataset_name == 'dynamic_mnist':
    elif args.dataset_name.startswith('dynamic_mnist'):
        train_loader, val_loader, test_loader, args = load_dynamic_mnist(args, **kwargs)
    elif args.dataset_name == 'dynamic_fashion_mnist':
        train_loader, val_loader, test_loader, args = load_dynamic_fashion_mnist(args, **kwargs) 
    elif args.dataset_name == 'omniglot':
        train_loader, val_loader, test_loader, args = load_omniglot(args, **kwargs)
    elif args.dataset_name == 'caltech101silhouettes':
        train_loader, val_loader, test_loader, args = load_caltech101silhouettes(args, **kwargs)
    elif args.dataset_name == 'histopathologyGray':
        train_loader, val_loader, test_loader, args = load_histopathologyGray(args, **kwargs)
    elif args.dataset_name == 'freyfaces':
        train_loader, val_loader, test_loader, args = load_freyfaces(args, **kwargs)
    elif args.dataset_name == 'coil20':
        train_loader, val_loader, test_loader, args = load_coil20(args, **kwargs)
    elif args.dataset_name.startswith('cifar10'):
        train_loader, val_loader, test_loader, args = load_cifar10(args, **kwargs)
    elif args.dataset_name == 'svhn':
        train_loader, val_loader, test_loader, args = load_svhn(args, **kwargs)
    elif args.dataset_name == 'usps':
        train_loader, val_loader, test_loader, args = load_usps(args, **kwargs)
    elif args.dataset_name == 'celeba':
        train_loader, val_loader, test_loader, args = load_celeba(args, **kwargs)
    elif args.dataset_name.startswith('gmm'):
        train_loader, val_loader, test_loader, args = load_gmm(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args
