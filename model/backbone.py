'''

A Transformer in Encoder-only mode.

The base Transformer code is loosely adapted from the PyTorch implementation (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py)
and partially from the Annotated Transformer tutorial code (https://nlp.seas.harvard.edu/2018/04/03/attention.html).
    
Also implemented are several innovations from OpenAI's Sparse Transformer paper (https://arxiv.org/abs/1904.10509):
- A modified residual block architecture (as described in Section 5.2)
- Special initialization schemes (as described in Sections 5.2 and 6)
- GeLU instead of ReLU activation (as described in Section 5.2)
though not sparse attention itself, hence the name FancyEncoder instead of SparseEncoder.

FancyEncoder begins with an embedding layer and ends with an averaging operation over the sequence length (in order to obtain outputs compatible with SimSiam).

'''


import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from copy import deepcopy
import math


class Embedding(nn.Module):
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self._init_parameters()

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    def _init_parameters(self):
        # Special initialization (as described in Section 6 of arxiv:1904.10509):
        nn.init.normal_(self.lut.weight, std = 0.125 / math.sqrt(self.d_model))    

        
class PositionalEncoding(nn.Module):

    def __init__(self, d_emb, max_seq_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        timesteps = torch.arange(0, max_seq_len, 1).view(-1,1)
        frequencies = 1. / (10000. ** (torch.arange(0, d_emb, 2) / d_emb))
        pmat = torch.zeros(max_seq_len, d_emb)
        pmat[:, 0::2] = torch.sin(frequencies * timesteps)
        pmat[:, 1::2] = torch.cos(frequencies * timesteps)
        pmat = pmat.unsqueeze(0)
        self.pmat = pmat
        
    def forward(self, x):
        x = x + Variable(self.pmat[:, :x.size(1)], requires_grad=False).to(x.device)
        return self.dropout(x)
    
        
def _clone_and_stack(layer, N):
    clones = [deepcopy(layer) for _ in range(N)]
    stack = nn.ModuleList(clones)
    return stack


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise RuntimeError('Activation {} not implemented.'.format(activation))
    
    
def attention(Q, K, V, mask=None, dropout=None):
    ''' Q, K, V have dimensions batch_size, num_heads, max_seq_len, d_k 
        Output has dimensions batch_size, num_heads, max_seq_len, d_v (= d_k, by popular assumption) '''
    d_k = K.size(-1)
    output = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        output = output + (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -10000.0
    output = F.softmax(output, dim=-1)
    if dropout is not None:
        output = dropout(output)
    output = torch.matmul(output, V)
    return output


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.W_QKV = _clone_and_stack(nn.Linear(d_model, d_model, bias=True), 3) 
        self.W_O = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y, mask=None):
        ''' Input: X (Y) is attender (attendee) and has dimensions batch_size, max_seq_len, d_model
            Output: Z is attention and has dimensions batch_size, max_seq_len, d_model '''
        batch_size, max_seq_len, d_model = X.shape
        X_WQ, Y_WK, Y_WV = [W(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) for W,v in zip(self.W_QKV, (X, Y, Y))]
        Z_concat = attention(X_WQ, Y_WK, Y_WV, mask=mask, dropout=self.dropout)
        Z_concat = Z_concat.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        Z = self.W_O(Z_concat)
        return Z
    
    
class FFLayer(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(self.norm(x)))))
    
    
class AttnLayer(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1, activation='gelu'):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm(x)
        return self.dropout(self.self_attn(x, x, mask=mask))
    
    
class EncoderLayer(nn.Module):

    def __init__(self, num_heads, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.d_model = d_model
        self.attn_block = AttnLayer(d_model, num_heads, dropout=dropout)
        self.ffnn_block = FFLayer(d_model, d_ff, dropout=dropout, activation=activation)
        self._init_parameters()

    def forward(self, x, src_mask=None):
        ax = self.attn_block(x, mask=src_mask)
        bx = self.ffnn_block(x + ax)
        return x + ax + bx
        
    def _init_parameters(self):
        # Special initialization (as described in Section 6 of arxiv:1904.10509):
        nn.init.normal_(self.ffnn_block.linear1.weight, std = 0.125 / math.sqrt(self.d_model))
        nn.init.normal_(self.ffnn_block.linear2.weight, std = 0.125 / math.sqrt(self.d_model))
        nn.init.zeros_(self.ffnn_block.linear1.bias)
        nn.init.zeros_(self.ffnn_block.linear2.bias)
        W_QKVO = self.attn_block.self_attn.W_QKV.append(self.attn_block.self_attn.W_O)
        for w in W_QKVO:
            nn.init.normal_(w.weight, std = 0.125 / math.sqrt(self.d_model))
            nn.init.zeros_(w.bias)

            
class EncoderStack(nn.Module):

    def __init__(self, encoder_template, num_encoders):
        super().__init__()
        d_model = encoder_template.d_model
        self.layers = _clone_and_stack(encoder_template, num_encoders)
        self.num_layers = num_encoders
        self.norm = nn.LayerNorm(d_model)
        self._init_parameters()

    def forward(self, src_tensor, src_mask=None):
        output = src_tensor
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)
        return self.norm(output)
    
    def _init_parameters(self):
        # Adjust certain initializations based on number of layers (see Section 5.2 of arxiv:1904.10509)
        for layer in self.layers:
            layer.ffnn_block.linear2.weight.data /= math.sqrt(2 * self.num_layers)
            layer.attn_block.self_attn.W_O.weight.data /= math.sqrt(2 * self.num_layers)        


class FancyEncoder(nn.Module):

    def __init__(self, vocab_size, max_seq_len=50, num_heads=8, num_encoders=6, d_model=512, d_ff=2048, dropout=0.1, activation='gelu', **args):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Sequential(Embedding(vocab_size, d_model), PositionalEncoding(d_model, max_seq_len=max_seq_len, dropout=dropout))        
        encoder = EncoderLayer(num_heads, d_model, d_ff, dropout=dropout, activation=activation)
        self.encoders = EncoderStack(encoder, num_encoders)

    def forward(self, src_tensor, src_mask=None):
        src_tensor = self.embedding(src_tensor)
        output = self.encoders(src_tensor, src_mask=src_mask)
        return output
