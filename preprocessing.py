'''
preprocessing.py

'''
from utils import rbf_rate_convolution_2d
import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger
import pathlib

def convolve_spikes(spks, sigma=0.1, dt=0.01, rt='tensor'):
    '''
    :params:
    spks:   shape(input_dim, bins)
    '''

    # dim check
    transposed = False
    assert isinstance(spks, np.ndarray)
    if spks.shape[0] < spks.shape[1]:
        transposed = True
        spks = spks.T   # converts to (n_bins, input_dim)
    
    ndim, nbins = spks.shape[1], spks.shape[0]
    logger.info(f'Detected Ephys file with {ndim} neurons and {nbins} bins.')

    # convolve with gaussian kernel
    rates = rbf_rate_convolution_2d(spks, sigma=sigma, dt=dt, rt=rt)

    # pad rates such that its centered
    padding_steps = int((nbins - rates.shape[1]) / 2)
    # throw in a check
    if sigma == 0.1 and dt == 0.01: assert padding_steps == 50
    
    if rt == 'tensor':
        rates = torch.hstack(
            [torch.zeros((ndim, padding_steps)), rates, torch.zeros((ndim, padding_steps))]
            )
    elif rt == 'numpy':
        rates = np.hstack(
            [np.zeros((ndim, padding_steps)), rates, np.zeros((ndim, padding_steps))]
            )
    else:
        raise ValueError('Invalid return type, choose tensor or numpy')
    
    assert np.array(rates.shape).all() == np.array(rates.shape).all()

    if transposed: rates = rates.T

    logger.info(f'Produced convolved rates at sigma={sigma} with {ndim} neurons and {nbins} bins.')

    return rates, ndim, nbins

def plot_convolution_example(rates, spks, _range=200, seed=None, save_path=False):
    if  rates.shape == spks.shape:
        spks == spks.T
    assert rates.shape == spks.T.shape
    nbins, ndim = rates.shape[0], rates.shape[1]
    _range = 200
    
    if seed is not None:
        np.random.seed(seed)
    
    _random_start = np.random.randint(0, nbins-_range)
    _random_neuron_idx = np.random.randint(0, ndim)

    plt.figure(figsize=(12,6))
    plt.plot(rates[_random_start:_random_start+_range, _random_neuron_idx], label='smoothened rate')
    plt.bar(np.arange(200), 
            height = spks[_random_neuron_idx, _random_start:_random_start+_range] * rates[_random_start:_random_start+_range, _random_neuron_idx].mean().numpy(), color='r', edgecolor='r', label='spikes')
    plt.ylabel('Firing rate / Hz')
    plt.xlabel('Time / ms')
    plt.legend()
    if save_path:
        assert isinstance(save_path, pathlib.Path)
        plt.savefig(save_path, bbox_inches='tight', dpi=250)
    plt.show()

def preprocess_dff_rate_pairs(dff, rates, has_stim=True, stim_time=None, stim_ID=None, cutoff_size=1, offset=-3, tsteps=15):
    if not dff.shape[0] == rates.shape[0]:
        dff = dff.T
    assert dff.shape[0] == rates.shape[0]

    # basic quantities
    n_vecs = int(rates.shape[0] / tsteps)
    input_dim = dff.shape[1]
    output_dim = rates.shape[1]

    # construct empty arrays
    inputs, targets = torch.zeros(n_vecs, tsteps, input_dim), torch.zeros(n_vecs, 1, output_dim)
    ephys_rates = torch.as_tensor(rates)
    img_array = torch.as_tensor(dff)
    for n in range(1, n_vecs-1):
        inputs[n,:,:] = img_array[n*tsteps:(n+1)*tsteps,:]
        targets[n,:,:] = ephys_rates[n*tsteps+offset:n*tsteps+offset+1,:]

    # note that the first vector would be 0, but will keep this to keep the stim/non-stim indexing correct.

    if has_stim:
        non_stim_vecs_idx, stim_vecs_idx = get_non_stim_vecs_idx(stim_time, stim_ID, n_vecs, cutoff_size)
        return inputs, targets, non_stim_vecs_idx, stim_vecs_idx
    else:
        return inputs, targets


def get_non_stim_vecs_idx(stim_time, stim_ID, n_vecs, cutoff_size):
    # this function processes gcamp stim data.
    n_stims = len(stim_ID)
    stim_vecs_idx = np.array([int(stim_time[i]/15) for i in range(n_stims)])
    
    # first label each vector as either stim or non-stim: they are cut off if within cutoff_size range from a stim instance
    non_stim_vecs_idx = []

    for i in range(cutoff_size, n_vecs-cutoff_size):
        flag = True
        for j in range(i-cutoff_size, i+cutoff_size):
            if j in stim_vecs_idx:
                flag = False
        if flag:
            non_stim_vecs_idx.append(i)
    return non_stim_vecs_idx, stim_vecs_idx

def sample_non_stim_vecs(non_stim_vecs_idx, cutoff_size=1, n=200, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # sample 200 endogenous data, chop off ends and add to test set
    endogenous_idxs = []
    c = 0
    while c < 200:
        endogenous_idx = np.random.choice(non_stim_vecs_idx)
        
        if np.all([x in non_stim_vecs_idx for x in range(endogenous_idx-cutoff_size, endogenous_idx+cutoff_size)]):
            c += 1
            for i in range(endogenous_idx-cutoff_size, endogenous_idx+cutoff_size):
                non_stim_vecs_idx.remove(i)
            endogenous_idxs.append(endogenous_idx)
    return endogenous_idxs