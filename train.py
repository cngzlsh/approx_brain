import torch
import torch.nn as nn
import numpy as np
from models import *
from utils import *
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import copy
import wandb


seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_mlp(model, train_loader, test_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False, return_init_eval_loss=False):
    '''
    Main training function. Iterates through training set in mini batches, updates gradients and compute loss.
    '''
    
    start = time.time()

    eval_losses, train_losses = [], []

    init_eval_loss = eval_mlp(model, test_loader, criterion)
    if verbose:
        print(f'Initial eval loss: {init_eval_loss}')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        n_batches = len(train_loader)
        
        for i, (X, Y) in enumerate(iter(train_loader)):
            
            optimiser.zero_grad()    
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

            if force_stop and i == 20:
                break

        eval_loss = eval_mlp(model, test_loader, criterion)

        train_losses.append(epoch_loss / n_batches)
        eval_losses.append(eval_loss)

        if verbose:
            epoch_end = time.time()
            if num_epochs < 50:
                hrs, mins, secs = elapsed_time(start, epoch_end)
                print(f'Epoch {epoch+1}: training loss {epoch_loss / n_batches}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
            else:
                if epoch % 50 == 0:
                    hrs, mins, secs = elapsed_time(start, epoch_end)
                    print(f'Epoch {epoch+1}: training loss {epoch_loss/ n_batches}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        
        if force_stop and i == 20:
            break
    
    hrs, mins, secs = elapsed_time(start, time.time())

    if verbose:
        print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
    
    if return_init_eval_loss:
        return train_losses, eval_losses, init_eval_loss
    else:
        return train_losses, eval_losses


def eval_mlp(model, test_loader, criterion):
    '''
    Evaluation function. Iterates through test set and compute loss.
    '''
    model.eval()
    
    n_batches = len(test_loader)

    with torch.no_grad():

        eval_loss = 0

        for _, (X, Y) in enumerate(iter(test_loader)):
            
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            
            eval_loss += loss.item()
    
    return eval_loss / n_batches



def train_rnn(model, train_loader, test_loader, optimiser, criterion, num_epochs, verbose=True, force_stop=False, scheduler=None):
    '''
    Main training function. Iterates through training set in mini batches, updates gradients and compute loss.
    '''
    start = time.time()
    eval_losses, train_losses, best_eval_epoch, best_eval_params = [], [], -1, -1
    init_eval_loss, best_eval_Y_hats = eval_rnn(model, test_loader, criterion, save_Y_hat=True)
    batch_first = model.rnn.batch_first
    directions = int(model.rnn.bidirectional) + 1

    if verbose:
        print(f'Initial eval loss: {init_eval_loss}')

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0
        n_batches = len(train_loader)
        
        for i, (X, Y) in enumerate(iter(train_loader)): # X: [batch_size, seq_len, input_dim]    
            X = X.to(device)
            Y = Y.to(device)
            
            # if model._type == 'lstm':
            #     h_prev = torch.zeros([model.n_rec_layers * directions, X.shape[0], model.rnn_output_dim]).to(device)
            #     c_prev = torch.zeros([model.n_rec_layers * directions, X.shape[0], model.rnn_output_dim]).to(device)
            #     rec_prev = (h_prev, c_prev)
            # elif model._type == 'rnn':
            #     rec_prev = torch.zeros([model.n_rec_layers * directions, X.shape[0], model.rnn_output_dim]).to(device) 
            # else:
            #     raise ValueError('Model type incorrect')
            optimiser.zero_grad()
            
            # # detach gradients from hidden tensors
            # if isinstance(rec_prev, torch.Tensor):
            #     rec_prev.detach()
            #     # pass
            # else:
            #     rec_prev = tuple(t.detach() for t in rec_prev)
            
            Y_hat, _ = model(X)
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

            if force_stop and i == 20:
                break

        eval_loss, Y_hats_temp = eval_rnn(model, test_loader, criterion, save_Y_hat=True)
        
        # save best params
        if len(eval_losses) > 1 and eval_loss < min(eval_losses):
            # assert False
            best_eval_Y_hats = Y_hats_temp
            best_eval_epoch = epoch
            best_eval_params = copy.deepcopy(model.state_dict())

        train_losses.append(epoch_loss/n_batches)
        eval_losses.append(eval_loss)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(eval_loss)
            else:
                scheduler.step()
        
        if verbose:
            epoch_end = time.time()
            if num_epochs < 50:
                hrs, mins, secs = elapsed_time(start, epoch_end)
                print(f'Epoch {epoch+1}: training loss {epoch_loss/n_batches}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
            else:
                if epoch % int(num_epochs/20) == 0:
                    hrs, mins, secs = elapsed_time(start, epoch_end)
                    print(f'Epoch {epoch+1}: training loss {epoch_loss/n_batches}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        
        if force_stop and i == 20:
            break
        
        if scheduler is not None and scheduler.get_last_lr()[0] < 1e-6:
            if verbose:
                hrs, mins, secs = elapsed_time(start, time.time())
                print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
            
            return {'train_losses': train_losses,
            'eval_losses': eval_losses,
            'best_eval_epoch': best_eval_epoch,
            'init_eval_loss': init_eval_loss,
            'best_eval_params': best_eval_params,
            'best_eval_Y_hats': best_eval_Y_hats}
    
    hrs, mins, secs = elapsed_time(start, time.time())

    if verbose:
        print(f'Training completed with final epoch loss {epoch_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        
    return {'train_losses': train_losses,
            'eval_losses': eval_losses,
            'best_eval_epoch': best_eval_epoch,
            'init_eval_loss': init_eval_loss,
            'best_eval_params': best_eval_params,
            'best_eval_Y_hats': best_eval_Y_hats}
    

def eval_rnn(model, test_loader, criterion, save_Y_hat=False):
    '''
    Evaluation function. Iterates through test set and compute loss.
    '''
    model.eval()
    directions = int(model.rnn.bidirectional) + 1
    if save_Y_hat:
        Y_hats = []
    

    with torch.no_grad():

        eval_loss = 0
        n_batches = len(test_loader)

        for _, (X, Y) in enumerate(iter(test_loader)):
            
            # if model._type == 'lstm':
            #     h_prev = torch.zeros([model.n_rec_layers * directions, X.shape[0], model.rnn_output_dim]).to(device)
            #     c_prev = torch.zeros([model.n_rec_layers * directions, X.shape[0], model.rnn_output_dim]).to(device)
            #     rec_prev = (h_prev, c_prev)
            # elif model._type == 'rnn':
            #         rec_prev = torch.zeros([model.n_rec_layers * directions, X.shape[0], model.rnn_output_dim]).to(device) 
            #     # raise NotImplementedError('Implement training for RNN please')
            # else:
            #     raise ValueError('Model type incorrect')
            
            X = X.to(device)
            Y = Y.to(device)
            Y_hat, _ = model(X)
            
            if save_Y_hat:
                Y_hats.append(Y_hat)
                
            loss = criterion(Y_hat, Y)
            
            eval_loss += loss.item()
    
    if save_Y_hat:
        return eval_loss/n_batches, torch.vstack(Y_hats)
    else:
        return eval_loss/n_batches
    
def train_transformer(model,
                      train_loader,
                      test_loader,
                      optimiser,
                      criterion,
                      num_epochs,
                      verbose=True,
                      force_stop=False,
                      batch_first=True,
                      scheduler=None,
                      use_wandb=False,
                      stim_type_indices=False,
                      prev_return_dict=None):
    
    start = time.time()

    if prev_return_dict is None:
        eval_losses, train_losses, best_eval_epoch, best_eval_params = [], [], -1, -1
        min_eval_loss = np.inf
        if stim_type_indices is not False:
            eval_losses_by_type = [[] for _ in range(len(stim_type_indices))]
    else:
        eval_losses = prev_return_dict['eval_losses']
        train_losses = prev_return_dict['train_losses']
        best_eval_epoch = prev_return_dict['best_eval_epoch']
        best_eval_params = prev_return_dict['best_eval_params']
        eval_losses_by_type = prev_return_dict['eval_losses_by_type']
        min_eval_loss = np.min(prev_return_dict['eval_losses'])

    
    init_eval_loss = eval_transformer(model, test_loader, criterion, batch_first=batch_first)
    if verbose:
        print(f'Initial eval loss: {init_eval_loss.mean()}')
        
    for epoch in tqdm(range(num_epochs)):
        
        model.train()
        epoch_loss = 0
        n_batches = len(train_loader)
        
        for _, (X, Y) in enumerate(iter(train_loader)):
            
            if batch_first:
                # convert to (seq_len, bs, input_dim)
                X = X.permute(1,0,2).to(device)
                Y = Y.permute(1,0,2).to(device)
            else:
                X = X.to(device)
                Y = Y.to(device)
            
            optimiser.zero_grad()
            
            Y_hat = model(X)
            
            loss = criterion(Y_hat, Y)
            normaliser = np.prod(loss.shape)
            loss = loss.sum() / normaliser
            loss.backward()
            optimiser.step()
            
            epoch_loss += loss.detach().item()
            
            if force_stop:
                break

        eval_loss = eval_transformer(model, test_loader, criterion, batch_first=batch_first)
        train_losses.append(epoch_loss / n_batches)
        
        if stim_type_indices is not False: # if test set is made of multiple stim types
            for i, type_idx in enumerate(stim_type_indices):
                eval_losses_by_type[i].append(eval_loss[0, type_idx].sum()/Y.shape[-1]/len(type_idx))

        if criterion.reduction == 'none':
            eval_loss = eval_loss.sum() / np.prod(eval_loss.shape)

        if  eval_loss < min_eval_loss:
            best_eval_epoch = epoch
            min_eval_loss = eval_loss
            best_eval_params = copy.deepcopy(model.state_dict())
        eval_losses.append(eval_loss)
        
        # learning rate scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(eval_loss)
            else:
                scheduler.step()
                
        
        if verbose:
            epoch_end = time.time()
            if num_epochs < 100:
                hrs, mins, secs = elapsed_time(start, epoch_end)
                print(f'Epoch {epoch+1}: training loss {epoch_loss/n_batches}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
            else:
                if epoch == 0 or epoch % int(num_epochs/20) == 0:
                    hrs, mins, secs = elapsed_time(start, epoch_end)
                    print(f'Epoch {epoch+1}: training loss {epoch_loss/n_batches}, eval loss {eval_loss}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')
        if force_stop:
            break
        
        if use_wandb:
            if stim_type_indices is False:
                wandb.log({'train_loss':epoch_loss / n_batches,
                            'eval_loss':eval_loss.sum() / np.prod(eval_loss.shape),
                            'learning_rate':scheduler._last_lr[-1],})
            else:
                if len(stim_type_indices) == 4:
                    wandb.log({'train_loss':epoch_loss / n_batches,
                                'eval_loss':eval_loss.sum() / np.prod(eval_loss.shape),
                                'learning_rate':scheduler._last_lr[-1],
                                'forward_loss': eval_losses_by_type[0][-1],
                                'backward_loss': eval_losses_by_type[1][-1],
                                'random_loss': eval_losses_by_type[2][-1],
                                'nonstim_loss':eval_losses_by_type[3][-1]})
        
        if scheduler is not None and scheduler.get_last_lr()[0] < 1e-6:
            if stim_type_indices is not False:

                return {'train_losses': train_losses,
                        'eval_losses': eval_losses,
                        'eval_losses_by_type':eval_losses_by_type,
                        'best_eval_epoch': best_eval_epoch,
                        'init_eval_loss': init_eval_loss,
                        'best_eval_params': best_eval_params}
    
            else:
                return {'train_losses': train_losses,
                        'eval_losses': eval_losses,
                        'best_eval_epoch': best_eval_epoch,
                        'init_eval_loss': init_eval_loss,
                        'best_eval_params': best_eval_params}
        
    hrs, mins, secs = elapsed_time(start, time.time())

    if verbose:
        print(f'Training completed with final epoch loss {epoch_loss / n_batches}. Time elapsed: {int(hrs)} h {int(mins)} m {int(secs)} s.')

    if stim_type_indices is not False:

        return {'train_losses': train_losses,
                'eval_losses': eval_losses,
                'eval_losses_by_type':eval_losses_by_type,
                'best_eval_epoch': best_eval_epoch,
                'init_eval_loss': init_eval_loss,
                'best_eval_params': best_eval_params}
    
    else:
        return {'train_losses': train_losses,
                'eval_losses': eval_losses,
                'best_eval_epoch': best_eval_epoch,
                'init_eval_loss': init_eval_loss,
                'best_eval_params': best_eval_params}

def eval_transformer(model, test_loader, criterion, batch_first=True):
    model.eval()
    
    with torch.no_grad():
        
        if criterion.reduction == 'none':
            sample_Y = next(iter(test_loader))[1]
            eval_loss = np.zeros((sample_Y.shape[1], 1, sample_Y.shape[2]))
            
        else:
            eval_loss = 0
        
        n_batches = len(test_loader)
        
        for _, (X, Y) in enumerate(iter(test_loader)):
            if batch_first:
                # convert to (seq_len, bs, input_dim)
                X = X.permute(1,0,2).to(device)
                Y = Y.permute(1,0,2).to(device)
            else:
                X = X.to(device)
                Y = Y.to(device)
            Y_hat = model(X)
            loss = criterion(Y_hat, Y)
            
            if criterion.reduction == 'none':
                eval_loss = np.concatenate((eval_loss, loss.detach().cpu().numpy()), axis=1) # [seq_len, batch_size, output_dim,]
            else:
                eval_loss += loss.item()
    
    if criterion.reduction == 'none':
        return eval_loss[:,1:,:]
    else:
        return eval_loss / n_batches