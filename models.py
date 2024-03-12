import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import OrderedDict

class FeedForwardDNN(nn.Module):
    '''
    Feed-forward deep neural network
    '''
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, transfer_function=nn.ReLU(), pos_output=False):
        super().__init__()
        
        self.layers = nn.Sequential()

        # input layer
        self.layers.add_module('input', nn.Linear(input_dim, hidden_dim))
        self.layers.add_module('relu_1', transfer_function)
        
        # hidden layers
        for hidden_idx in range(1, n_layers+1):
            self.layers.add_module(f'hidden_{hidden_idx}', nn.Linear(hidden_dim, hidden_dim))
            self.layers.add_module(f'relu_{hidden_idx+1}', transfer_function)

        # output layer
        self.layers.add_module('output', nn.Linear(hidden_dim, output_dim))
        if pos_output:
            self.layers.add_module('relu_output', transfer_function)
        
    def forward(self, x):
        return self.layers(x)
    

class RecurrentDNN(nn.Module):
    '''
    Deep neural network with LSTM units followed by a Dense output layer. Optional input layer.
    '''
    def __init__(self,
                input_dim,
                hidden_dim,
                n_rec_layers, 
                output_dim, 
                n_input_layers=0,
                n_output_layers=1,
                bidirectional=False,
                _type='lstm',
                pos_output=False,
                output_step=1,
                batch_first=True):
        
        super(RecurrentDNN, self).__init__()

        self.input_dim = input_dim
        self.n_rec_layers = n_rec_layers
        self.hidden_dim = hidden_dim
        self._type = _type
        self.pos_output = pos_output
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.directions = int(bidirectional) + 1
        self.n_input_layers = n_input_layers
        self.n_output_layers = n_output_layers
        self.output_step = output_step
        
        # input layers
        if n_input_layers > 0:
            self.input_layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
            for _ in range(n_input_layers-1):
                self.input_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.input_layers.append(nn.ReLU())
            self.rnn_input_dim = self.hidden_dim
        else:
            self.rnn_input_dim = self.input_dim

        # output layers
        if n_output_layers == 0:
            print('No output linear layers. Setting hidden_dim to output_dim')
            self.rnn_output_dim = self.output_dim
        elif n_output_layers == 1:
            self.rnn_output_dim = self.hidden_dim
            self.output_layers = nn.ModuleList([nn.Linear(self.rnn_output_dim * self.directions, self.output_dim)])
        else:
            self.rnn_output_dim = self.hidden_dim
            self.output_layers = nn.ModuleList([nn.Linear(self.rnn_output_dim  * self.directions, self.hidden_dim)])
            for _ in range(n_output_layers-1):
                self.output_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.output_layers.append(nn.ReLU())
            self.output_layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        # rnn layers
        if _type == 'lstm':
            self.rnn = nn.LSTM(self.rnn_input_dim, self.rnn_output_dim, num_layers=self.n_rec_layers, bias=True, batch_first=batch_first, bidirectional=self.bidirectional)
        elif _type == 'rnn':
            self.rnn = nn.RNN(self.rnn_input_dim, self.rnn_output_dim, num_layers=self.n_rec_layers, bias=True, batch_first=batch_first, bidirectional=self.bidirectional)
        else:
            raise ValueError('Incorrect RNN type')
        
    def forward(self, x, rec_prev=None):
        
        bs, seq_len, input_dim = x.shape
        
        assert input_dim == self.input_dim
        
        if self.n_input_layers > 0:
            for m in self.input_layers:
                x = m(x)

        if rec_prev is None:
            if self._type == 'rnn':
                rec_prev = torch.zeros(self.n_rec_layers * self.directions, bs, self.rnn_output_dim).to(x.device)
            elif self._type == 'lstm':
                rec_prev = (torch.zeros(self.n_rec_layers * self.directions, x.size(0), self.rnn_output_dim, device=x.device),
                          torch.zeros(self.n_rec_layers * self.directions, x.size(0), self.rnn_output_dim, device=x.device))

        out, rec_prev = self.rnn(x, rec_prev)

        if self.n_output_layers > 0:
            for m in self.output_layers:
                out = m(out)
        
        if self.pos_output:
            out = nn.ReLU()(out)
        
        assert out.shape == torch.Size([bs, seq_len, self.output_dim])
        if self.output_step == 1:
            out = out[:,-1,:][:,None,:] # keep dimensionality consistant
        return out, rec_prev


class TransformerOneStep(nn.Module):
    '''
    Transformer model with multi-head attention encoder and a linear decoder.
    '''
    def __init__(self,
                input_dim,
                d_model,
                num_heads,
                hidden_dim,
                output_dim,
                n_encoder_layers,
                device,
                max_len=30,
                dropout=0.1,
                decoder='linear',
                decoder_hidden_dim=None,
                use_mask=True,
                pos_output=False,
                bin_output=False,
                softmax_output=False):
        
        super().__init__()

        self.input_dim = input_dim          # input dimension, usually the number of input neurons
        self.d_model = d_model              # dimensionality of encoder embedding
        self.num_heads = num_heads          # number of attention heads
        self.n_layers = n_encoder_layers    # number of encoder layers
        self.hidden_dim = hidden_dim        # number of hidden layers in the encoder
        self.output_dim = output_dim        # output dimension, usually the number of output neurons
        self.dropout = nn.Dropout(dropout)

        self.use_future_mask = use_mask
        self.device = device

        self.pos_output = pos_output
        self.bin_output = bin_output
        self.softmax_output = softmax_output
        assert int(self.pos_output) + int(self.bin_output) + int(self.softmax_output) <= 1
        
        # input layer
        self.embedding_layer = nn.Linear(input_dim, d_model)
    
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout), 
            num_layers=n_encoder_layers)
        
        if decoder == 'linear':
            self.decoder = nn.Linear(d_model, output_dim)
            if decoder_hidden_dim is not None:
                print('Using a linear decoder, decoder hidden dim will be reset to 0.')
        elif decoder == 'mlp':
            assert decoder_hidden_dim is not None
            self.decoder = nn.Sequential(
                nn.Linear(d_model, decoder_hidden_dim),
                nn.ReLU(),
                self.dropout,
                nn.Linear(decoder_hidden_dim, output_dim)
                )
        else:
            raise NotImplementedError
    
    def _create_future_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        seq_len, bs, input_dim = x.shape
        assert input_dim == self.input_dim

        # x = self.dropout(self.embedding_layer(x))
        # x = self.pe(x)
        mask = self._create_future_mask(seq_len).to(self.device)
        x = self.embedding_layer(x) * math.sqrt(self.d_model)
        
        x = self.pe(x) # shape [seq_len, bs, d_model]

        if self.use_future_mask:   
            h = self.encoder(x, mask=mask)
        else:
            h = self.encoder(x)
        
        y = self.decoder(h)
        
        if self.pos_output:
            y = nn.ReLU()(y[-1,:,:][None,:,:] )
        elif self.bin_output:
            y = nn.Sigmoid()(y[-1,:,:][None,:,:])
        elif self.softmax_output:
            y = nn.Softmax()(y[-1,:,:][None,:,:])
        else:
            y = y[-1,:,:][None,:,:]
        
        assert y.shape == torch.Size([1, bs, self.output_dim])
            
        return y
    
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True
        print('All Weights are unfrozen.')
    
    def prepare_for_transfer_learning(self, new_input_dim, new_output_dim, freeze_weights=True):
        # re-initialises the input and output layers, optionally freezes the params of the transformer encoder layer
        if freeze_weights:
            print('Encoder Weights are frozen.')
            self.freeze_weights()
        else:
            self.unfreeze_weights()
        
        self.embedding_layer = nn.Linear(new_input_dim, self.d_model).to(self.device)
        if isinstance(self.decoder, nn.Linear):
            self.decoder = nn.Linear(self.d_model, new_output_dim).to(self.device)
            print('Using a linear decoder, decoder hidden dim will be reset to 0.')
        else:
            assert self.decoder_hidden_dim is not None
            self.decoder = nn.Sequential(
                nn.Linear(self.d_model, self.decoder_hidden_dim),
                nn.ReLU(),
                self.dropout,
                nn.Linear(self.decoder_hidden_dim, new_output_dim)
                ).to(self.device)
        
        self.input_dim = new_input_dim
        self.output_dim = new_output_dim
        print(f'Reinitialised input and output layers with {self.input_dim} input units and {self.output_dim} output units.')
    
    def get_separate_params(self):
        # returns parameters for the encoder as well as the input/output layers separately.
        encoder_params_dict = OrderedDict()
        non_encoder_params_dict = OrderedDict()
        for key in self.state_dict().keys():
            if 'encoder' in key:
                encoder_params_dict[key] = self.state_dict()[key]
            else:
                non_encoder_params_dict[key] = self.state_dict()[key]
        return encoder_params_dict, non_encoder_params_dict
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=120):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        
        # pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe = torch.zeros(max_len, 1, d_model, dtype=torch.float)

        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # pe = pe.unsqueeze(1).expand(-1,-1, d_model)
        # shape (max_len, 1, d_model)
    
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # (seq_length, batch_size, input_dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerMultiSteps(nn.Module):
    '''
    Transformer model with multi-head attention encoder and a multi-head attention decoder.
    Handles variable length.
    Decoder need not have the same hyperparameters.
    '''
    def __init__(self,
                 input_dim,
                 d_model,
                 num_heads,
                 hidden_dim,
                 n_layers,
                 output_dim, 
                 device, 
                 pos_output=False,#
                 max_len=30, 
                 dropout=0.1, 
                 use_mask=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.pos_output = pos_output
        self.dropout = nn.Dropout(dropout)
        self.use_future_mask = use_mask
        self.device = device
        
        if isinstance(d_model, tuple): # different settings for encoder and decoder
            self.enc_d_model, self.dec_d_model = d_model
        else:
            self.enc_d_model = d_model
            self.dec_d_model = d_model

        if isinstance(num_heads, tuple):
            self.enc_num_heads, self.dec_num_heads = num_heads
        else:
            self.enc_num_heads = num_heads
            self.dec_num_heads = num_heads
        if isinstance(n_layers, tuple):
            self.enc_n_layers, self.dec_n_layers = n_layers
        else:
            self.enc_n_layers = n_layers
            self.dec_n_layers = n_layers
        
        self.embedding_layer = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU()
        
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.enc_d_model,
                                       nhead=self.enc_num_heads,
                                       dim_feedforward=hidden_dim,
                                       dropout=dropout), 
            num_layers=self.enc_n_layers)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.dec_d_model,
                                       nhead=self.dec_num_heads,
                                       dim_feedforward=hidden_dim,
                                       dropout=dropout),
            num_layers=self.dec_n_layers)
        
    def _create_future_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        seq_len, bs, input_dim = src.shape
        # (seq_len, batch_size, input_dim)
        src_emb = self.dropout(self.embedding_layer(src) * math.sqrt(self.hidden_dim))
        
        src_emb = self.pe(src_emb)

        if self.use_future_mask_for_encoder:
            mask = self._create_future_mask(seq_len).to(self.device)
            enc_output = self.encoder(src_emb, mask=mask)
        else:
            enc_output = self.encoder(src_emb)
            
        decoder_mask = self._create_future_mask(seq_len).to(self.device)
        out = self.decoder(tgt=enc_output, memory=enc_output, tgt_mask=decoder_mask)
        
        if self.pos_output:
            out = self.relu(out)
        
        return out
    
    
if __name__ == '__main__':
    assert False