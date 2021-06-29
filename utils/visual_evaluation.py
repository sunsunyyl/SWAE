import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import numpy as np
#=======================================================================================================================
def plot_histogram( x, dir, mode ):

    fig = plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(x, 100, normed=True, facecolor='blue', alpha=0.5)

    plt.xlabel('Log-likelihood value')
    plt.ylabel('Probability')
    plt.grid(True)

    plt.savefig(dir + 'histogram_' + mode + '.png', bbox_inches='tight')
    plt.close(fig)

#=======================================================================================================================
def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3):

    if np.prod(args.input_size) == 2:      
        plt.scatter(x_sample[:,0], x_sample[:,1], color='navy', marker='.')
        plt.xlim(-0.5,5)
        plt.ylim(-0.5,5)
    elif args.dataset_name.startswith('gmm') and (np.prod(args.input_size) > 2):
        print('No images.') 
    else:

        fig = plt.figure(figsize=(size_y, size_x))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        for i, sample in enumerate(x_sample):
            ax = fig.add_subplot(size_x, size_y, i+1)

            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
            sample = sample.swapaxes(0, 2)
            sample = sample.swapaxes(0, 1)
            if args.input_type == 'binary' or args.input_type == 'gray':
                sample = sample[:, :, 0]
                plt.imshow(sample, cmap='gray')
            else:
                plt.imshow(sample)

    plt.savefig(dir + file_name + '.png', bbox_inches='tight', dpi = 300)
    plt.close()


def plot_images_each_label(test_labels, test_data, test_target, dir, file_name):

    fig = plt.figure()
    ax = plt.gca()
    i = 0
    j = 1
    cmap = plt.cm.tab20    

    for ind, l in enumerate(test_labels):
        I = (test_target == l)
        c = cmap(ind / len(test_labels))
        ax.scatter(test_data[I, i], test_data[I, j], c=c, label="{l}".format(l=int(l)), s=2)
    plt.legend()    
    plt.tight_layout()

    plt.savefig(dir + file_name + '.png', bbox_inches='tight', dpi = 300)
    plt.close(fig)


def plot_fid_images(args, x_sample, dir):

    if np.prod(args.input_size) == 2:      
        plt.scatter(x_sample[:,0], x_sample[:,1], color='navy', marker='.')
        plt.xlim(-0.5,5)
        plt.ylim(-0.5,5)
    elif args.dataset_name.startswith('gmm') and (np.prod(args.input_size) > 2):
        print('No images.') 
    else:

        for i, sample in enumerate(x_sample):

            sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
            sample = sample.swapaxes(0, 2)
            sample = sample.swapaxes(0, 1)
            if args.input_type == 'binary' or args.input_type == 'gray':
                sample = sample[:, :, 0]
                plt.imshow(sample, cmap='gray')
            else:
                plt.imshow(sample)

            plt.savefig(dir + str(i) + '.png', bbox_inches='tight')
            plt.close()



  
