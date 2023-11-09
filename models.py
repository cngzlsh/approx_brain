import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, input_dim, hidden_dim, n_linear_layers, n_rec_layers, output_dim, transfer_function=nn.ReLU(), _type='lstm', has_input_layer=True, pos_output=False):
        super().__init__()

        self.n_rec_layers = n_rec_layers
        self.n_linear_layers = n_linear_layers
        self.hidden_dim = hidden_dim
        self._type = _type
        self.has_input_layer = has_input_layer
        self.pos_output = pos_output
        self.output_dim = output_dim
        self.transfer_function = transfer_function
        
        # input layer
        if has_input_layer:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            rnn_input_dim = hidden_dim
        else:
            rnn_input_dim = input_dim

        # hidden layers
        if _type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim, hidden_dim, num_layers=self.n_rec_layers, bias=True, batch_first=True, bidirectional=False)
        elif _type == 'rnn':
            self.rnn = nn.RNN(rnn_input_dim, hidden_dim, num_layers=self.n_rec_layers, bias=True, batch_first=True, bidirectional=False)
        else:
            raise ValueError('Incorrect RNN type')

        self.linears = nn.Sequential()
        if self.n_linear_layers > 0:
            for i in range(n_linear_layers):
                self.linears.add_module(f'hidden_linear_{i+1}', nn.Linear(hidden_dim, hidden_dim))
                self.linears.add_module(f'relu_{i+1}', self.transfer_function)
        
        # output layer
        self.linears.add_module('output', nn.Linear(hidden_dim, output_dim))
        if pos_output:
            self.linears.add_module('output_relu', self.transfer_function)
        
    
    def forward(self, x, rec_prev):
        
        if self.has_input_layer:
            x = self.input_layer(x)
            x = self.transfer_function(x)
        
        out, rec_prev = self.rnn(x, rec_prev)
        
        out = self.linears(out)
        
        return out, rec_prev


class TransformerOneStep(nn.Module):
    '''
    Transformer model with multi-head attention encoder and a linear decoder.
    '''
    def __init__(self, input_dim, d_model, num_heads, hidden_dim, output_dim, n_layers, device, 
                 pos_output=True, max_len=120, dropout=0.1, use_mask=False, bin_output=False, softmax_output=False):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.use_future_mask = use_mask
        self.device = device
        self.pos_output = pos_output
        self.bin_output = bin_output
        self.softmax_output = softmax_output
        assert int(self.pos_output) + int(self.bin_output) + int(self.softmax_output) <= 1
        
        self.embedding_layer = nn.Linear(input_dim, d_model)
    
        # self.ffn = nn.Sequential(nn.Linear(d_model, hidden_dim),
        #                           nn.ReLU(),
        #                           nn.Linear(hidden_dim, d_model))
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout), 
            num_layers=n_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
    
    def _create_future_mask(self, x):
        seq_len = x.size(0)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask

    def forward(self, x):
        # (seq_len, batch_size, input_dim)
        x = self.dropout(self.embedding_layer(x))
        # x = self.ffn(x)
        x = self.pe(x)
        if self.use_future_mask:
            mask = self._create_future_mask(x).to(self.device)
            x = self.encoder(x, mask=mask)
        else:
            x = self.encoder(x)
        x = self.decoder(x)
        
        if self.pos_output:
            x = self.relu(x[-1,:,:][None,:,:] )
        elif self.bin_output:
            x = nn.Sigmoid()(x[-1,:,:][None,:,:])
        elif self.softmax_output:
            x = nn.Softmax()(x[-1,:,:][None,:,:])
        else:
            x = x[-1,:,:][None,:,:] # (1, batch_size, output_dim)
            
        return x
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=120):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(1).expand(-1,-1, d_model)
        # shape (max_len, 1, d_model)
    
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # (seq_length, batch_size, input_dim)
        x = x + self.pe[:x.size(0), ...]
        return self.dropout(x)
    
    
class TransformerMultiSteps(nn.Module):
    '''
    Transformer model with multi-head attention encoder and a multi-head attention decoder.
    Handles variable length.
    Decoder need not have the same hyperparameters.
    '''
    def __init__(self, input_dim, d_model, num_heads, hidden_dim, output_dim, n_layers, device, 
                 pos_output=True, max_len=120, dropout=0.1, use_mask=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.pos_output = pos_output
        self.dropout = nn.Dropout(dropout)
        self.use_future_mask_for_encoder = use_mask
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
        
    def _create_future_mask(self, x):
        batch_size, seq_len = x.size(1), x.size(0)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask

    def forward(self, src, tgt):
        # (seq_len, batch_size, input_dim)
        src_emb = self.dropout(self.embedding_layer(src) * math.sqrt(self.hidden_dim))
        tgt_emb = self.dropout(self.embedding_layer(tgt) * math.sqrt(self.hidden_dim)) 
        
        src_emb = self.pe(src_emb)
        tgt_emb = self.pe(tgt_emb)
        if self.use_future_mask_for_encoder:
            mask = self._create_future_mask(src_emb).to(self.device)
            enc_output = self.encoder(src_emb, mask=mask)
        else:
            enc_output = self.encoder(src_emb)
            
        decoder_mask = self._create_future_mask(tgt_emb).to(self.device)
        out = self.decoder(tgt=tgt_emb, memory=enc_output, tgt_mask=decoder_mask)
        
        if self.pos_output:
            out = self.relu(out)
        
        return out
    
    
if __name__ == '__main__':
    assert False