import pickle
import time
import os

import torch
from torch.utils.data import Dataset
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy

# import netpyne

from models import *

sns.set(font_scale=1.2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalise_data(unnormalised_data, mean, std):
    '''
    Normalises data: subtract mean and divide by std.
    '''
    normalised_data = torch.nan_to_num((unnormalised_data - mean)/ std, nan=0.0, posinf=0.0, neginf=0.0)
    assert normalised_data.shape == unnormalised_data.shape
    return normalised_data

def unnormalise_data(normalised_data, mean, std):
    '''
    Unnormalise data: multiply std and add the mean.
    ''' 
    unnormalised_data = torch.nan_to_num(torch.multiply(normalised_data, std) + mean, nan=0.0, posinf=0.0, neginf=0.0)
    assert unnormalised_data.shape == normalised_data.shape
    return unnormalised_data

def save_data(X, Y, path, filename):
    '''
    Saves synthetic neuron firing data to pickle file
    '''
    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + filename, 'wb') as f:
        pickle.dump((X, Y), f)
    
    f.close()


def load_data(path, filename):
    '''
    Loads synthetic neuron firing data from pickle file
    '''
    with open(path + filename, 'rb') as f:
        X, Y = pickle.load(f)
    f.close()

    X = X.to(device)
    Y = Y.to(device)

    return X, Y


def save_non_linearities(_dict, filepath):
    '''
    Saves the non-linearities of a biological neural network
    '''
    with open(filepath, 'wb') as f:
        pickle.dump(_dict, f)

    f.close()


def load_non_linearities(filepath):
    with open(filepath, 'rb') as f:
        _dict = pickle.load(f)
    f.close()
    return _dict

def elapsed_time(start, end):
    '''
    Helper function to compute elapsed time
    '''
    secs = end - start
    mins = int(secs / 60)
    hrs = int(mins / 60)
    return hrs, mins % 60, int(secs % 60)


class BNN_Dataset(Dataset):
    '''
    Dataset class for creating iterable dataloader
    '''
    def __init__(self, X, Y):
        self.inputs = X
        self.labels = Y

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx, :]
        label = self.labels[idx, :]
        return input, label


def visualise_prediction(y, y_hat, reshape='square', cb=False, fname=False):
    '''
    Visualise and compare the prediction of a neuronal firing pattern in a colour map.
    :param y:                   true label
    :param y_hat:               prediction
    :param reshape:             tuple (w, h), how the colour map is shown. By default show in a square
    '''
    if reshape == 'square':
        dim = len(y)
        w, l = int(np.sqrt(dim)), int(np.sqrt(dim))
    else:
        w, l = reshape
    
    try:
        y_r = y.reshape((w,l)).cpu()
        y_hat_r = y_hat.reshape(w,l).cpu().detach().numpy()
    except:
        raise ValueError('Reshape dimension mismatch')

    plt.figure(figsize=(12,6))
    
    plt.subplot(121)
    plt.imshow(y_r)
    plt.axis('off')
    plt.title('True firing pattern')
    if cb:
        plt.colorbar()

    plt.subplot(122)
    plt.imshow(y_hat_r)
    plt.axis('off')
    plt.title('Predicted firing pattern')
    if cb:
        plt.colorbar()

    if fname is not False:
        plt.savefig('./figures/' + fname, dpi=350, bbox_inches='tight')
    plt.show()


def moving_average(time_series, lag):
    time_series = torch.tensor(time_series).view(1,1,-1)
    conv = nn.Conv1d(1,1,lag, bias=False)
    conv.weight.data = torch.ones(conv.weight.data.shape)/lag
    return conv(time_series).flatten().tolist()


def plot_loss_curves(train_losses, eval_losses, smoothen=25, loss_func='MSE loss', fname=False):
    '''
    Plots the loss history per epoch.
    '''
    n_epochs = len(train_losses)

    if smoothen is not False:
        train_losses = moving_average(train_losses, smoothen)
        eval_losses = moving_average(eval_losses, smoothen)
    
    plt.figure(figsize=(12,4))
    plt.plot(train_losses)
    plt.plot(eval_losses)

    plt.legend(['train', 'eval'])
    
    plt.xlabel('Epochs')
    plt.ylabel(loss_func)

    plt.title(f'Training and evaluation {loss_func} curve over {n_epochs} epochs')

    if fname is not False:
        plt.savefig(fname, dpi=350, bbox_inches='tight')
    plt.show()


def find_argmin_in_matrix(mat):
    '''
    Find the row and coloumn of the smallest element in a matrix
    '''
    nr, nc = mat.shape
    return int(np.argmin(mat)/nc), np.argmin(mat) - int(np.argmin(mat)/nc) * nc


def plot_3d_scatter(x, y, z, x_label, y_label, z_label, colorbar=True, fname=False, title=False, figsize=(12,10)):
    '''
    Produces 3d scatter plot
    '''
    xyz = np.zeros([len(x)*len(y), 3])
    for i in range(len(x)):
        for j in range(len(y)):
            xyz[i*len(x)+j,:] = np.array([x[i], y[j], z[i,j]])
    
    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt3d = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=xyz[:,2])
    if colorbar:
        cbar = plt.colorbar(plt3d)
        cbar.set_label(z_label)
    if title is not False:
        plt.title(title)
    if fname is not False:
        plt.savefig('./figures/' + fname, dpi=350, bbox_inches='tight')
    plt.show()


def generate_locations(length, width, n):
    '''
    Generates multiple random location
    '''
    xs = dist.Uniform(0, length).sample(sample_shape=torch.Size([n]))
    ys = dist.Uniform(0, width).sample(sample_shape=torch.Size([n]))

    return xs, ys

def calc_entropy(x:torch.Tensor):
    '''
    Computes the (Shannon) entropy of a tensor based on its empirical distribution
    '''
    x = x.flatten()
    freq = x.unique(return_counts=True)[1]
    probs = freq/torch.sum(freq)
    return -torch.multiply(probs, torch.log(probs)).sum()


def diffusion_process(xt, dt, t, drift=0.0, diffusion=1.0):
    '''
    Generalised, continous-time Brownian motion (Ornstein-Uhlenbeck process): 
        dx_t = a(x_t, t)dt + b(x_t, t)dB_t
    params:
        drift:      drift function a(x_t, t) function, lambda function or constant
        diffusion:  diffusion coefficient function, lambda function or constant
    returns:
        dxt:        change in x in dt
    '''
    try:
        mu = drift(xt, t)
    except:
        mu = drift * torch.ones_like(xt)
        
    try:
        sigma = diffusion(xt, t)
    except:
        sigma = diffusion * torch.ones_like(xt)
    
    assert mu.shape == xt.shape
    assert sigma.shape == xt.shape
    
    dxt = mu * dt + sigma * dist.Normal(0, dt).sample(xt.shape)
    return dxt

def read_NEURON_data(path):
    
    with open(path, 'rb') as f:
        dataSave = pickle.load(f)
        f.close()

    spkt = dataSave['simData']['spkt']
    spkid = dataSave['simData']['spkid']
    V_soma = dataSave['simData']['V_soma']
    LFP = dataSave['simData']['LFP']
    popRates = dataSave['simData']['popRates']
    avgRates = dataSave['simData']['avgRate']

    # extract cell types
    cellDetails = dataSave['net']['cells']
    cellTypesBygid = {}
    gidBycellTypes = {}
    for gid in range(len(cellDetails)):
        cellTypesBygid[gid] = cellDetails[gid]['tags']['pop']
        if cellDetails[gid]['tags']['pop'] not in gidBycellTypes:
            gidBycellTypes[cellDetails[gid]['tags']['pop']] = [gid]
        else:
            gidBycellTypes[cellDetails[gid]['tags']['pop']].append(gid)

    print('Total number of spikes recorded: ', len(spkt))
    print('Total number of neurons which spiked:', len(np.unique(spkid)))
    print('Total number of neurons:', len(cellTypesBygid))
    
    return spkt, spkid, V_soma, LFP, popRates, avgRates, cellDetails, cellTypesBygid, gidBycellTypes

def rbf_rate_convolution_1d(spikes, sigma, dt=0.01):
    # Convolution with Gaussian kernel
    time_window = np.arange(-5*sigma, 5*sigma, dt)
    gaussian_kernel = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (time_window / sigma)**2)
    smooth_rate = np.convolve(spikes, gaussian_kernel, mode='same')
    exclusion_steps = int(5*sigma/dt)
    
    return smooth_rate[exclusion_steps:-exclusion_steps]

def rbf_rate_convolution_2d(spikes, sigma, rt='tensor', dt=0.01):
    '''
    Normalised Gaussian kernel convolution.
    Implemented using a for loop
    Allows smoothing multiple neurons at the same time.
    
    spikes: number of spikes per neuron per time bin. Array of integers shape (number of neurons, number of time bins)
    
    returns:
    smooth_rate: continuous smooth firing rate, shape (number of neurons, number of time bins)
    '''
    n_neurons = spikes.shape[1]
    smooth_rates = []
    
    for i in range(n_neurons):
        smooth_rate = rbf_rate_convolution_1d(spikes[:,i], sigma, dt=dt)
        smooth_rates.append(smooth_rate)
    
    if rt == 'tensor':
        return torch.Tensor(np.vstack(smooth_rates))
    elif rt == 'numpy':
        return np.vstack(smooth_rates)
    else:
        raise ValueError('Specify return type')


def randomise_dataset_split(X, Y, split_ratio=(0.8, 0.1, 0.1), seed=123):
    '''
    Randomises a dataset then split into train, test, valid sets according to the ratio specified.
    '''
    assert sum(split_ratio) == 1.0
    if seed is not False:
        torch.manual_seed(seed)
    idx = torch.randperm(X.size(0))
    n_vecs = len(idx)
    
    X = X[idx, :]
    Y = Y[idx, :]
    
    train_ratio, test_ratio, valid_ratio = split_ratio
    
    X_train, X_test, X_valid = X[:int(n_vecs*train_ratio), :, :], X[int(n_vecs*train_ratio): int(n_vecs*(train_ratio+test_ratio)), :, :], X[int(n_vecs*(train_ratio+test_ratio)):, :, :]
    Y_train, Y_test, Y_valid = Y[:int(n_vecs*train_ratio), :, :], Y[int(n_vecs*train_ratio): int(n_vecs*(train_ratio+test_ratio)), :, :], Y[int(n_vecs*(train_ratio+test_ratio)):, :, :]

    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid

class RateWeightedMSE(nn.Module):
    '''
    Custom loss function where MSE is weighted by the firing rate.
    '''
    def __init__(self):    
        super(RateWeightedMSE, self).__init__()
        
    def forward(self, pred: torch.Tensor, target:torch.Tensor):
        assert pred.shape == target.shape
        assert len(pred.shape) == 3
        batch_size, output_dim = pred.shape[1], pred.shape[2]
        # (1, batch_size, output_dim)
        
        softmax = nn.Softmax(dim=2)
        weights = softmax(target)
        return torch.sum(torch.multiply(weights, torch.pow(pred - target, 2))) / batch_size
    
def process_spike_bins(spkt, spkid, cellTypesBygid, spike_bins, cutoff=80000, use_full_dataset=True):
    # chop off unstable data
    scatter_xs, scatter_ys = [], []
    for i in range(len(spkt)):
        # cut off time before reaching steady states
        if spkt[i] < cutoff:
            continue
        elif not use_full_dataset:
            if spkt[i] > 50000:
                break
        else:
            scatter_xs.append(spkt[i]) # time at which spike occurred
            scatter_ys.append(spkid[i]) # id at which spike occurred
    scatter_xs = [i - cutoff for i in scatter_xs]
    print('Loading complete.')
    
    # spike binning
    total_duration = max(scatter_xs)

    n_bins = math.ceil(total_duration/spike_bins)
    print('Total number of spike bins', n_bins)
    spike_data = torch.zeros(n_bins, len(cellTypesBygid)) # number of spikes per neuron every 10 ms bins

    for i in tqdm(range(len(scatter_xs))):
        _gid = int(scatter_ys[i])
        spike_data[int(scatter_xs[i]/spike_bins), _gid] += 1
    
    n_bins = spike_data.shape[0]
    
    return spike_data, total_duration, n_bins

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def bin_spikes(spks_1ms, bin_size):
    assert spks_1ms.shape[1] > spks_1ms.shape[0]
    n_bins = int(spks_1ms.shape[1] / bin_size)
    binned_spks = np.zeros((spks_1ms.shape[0], n_bins))
    for b in range(n_bins):
        binned_spks[:,b] = np.sum(spks_1ms[:, int(bin_size * b): int(bin_size * (b+1))], axis=1)
    return binned_spks