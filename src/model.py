import torch
import torch.nn as nn
import math
from torch.nn.modules import dropout

class EmbeddingInput(nn.Module):

    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def forward( self, x ):
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoder(nn.Module):

    def __init__(self, d_model : int, context_length : int, drop : float):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = nn.Dropout(drop)
        self.pe = torch.zeros([context_length, d_model])

        position = torch.arange(0, context_length, dtype = torch.float)
        divide = torch.exp( torch.arange(0, d_model,2) * ( - math.log(10000) / d_model) )

        # Applying The Encoder For Each Word But For Different Dims
        self.pe[:, 1::2] = torch.cos(position * divide) 
        self.pe[:, 0::2] = torch.sin(position * divide)
        
        self.pe.unsqueeze_(0) # Adding one more dimension for batching.
        self.register_buffer('pe', self.pe)
    
    def forward(self, x):

        x = x + self.pe[:, :x.shape, :] # Batch X InputLen X Embeddings
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps : float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std  = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / ( std + self.eps ) + self.bias

class FeedForwardNN(nn.Module):

    def __init__(self, d_model : int, d_ff : int, dropout : float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, h:int, d_model:int, dropout:int):
        super().__init__()
        assert d_model % h != 0, "Can't create d_k"
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout : nn.Dropout, mask):
        d_k = query.shape[-1]
        qk = (query @ key.transpose(-1,-2)) / math.sqrt(d_k) # B x H x Seq x Seq
        qkv = qk @ value # B x H x Seq x Embed

        if dropout is not None:
            qkv = dropout(qkv)

        if mask is not None:
            qkv.masked_fill_(mask == 0, -1e9)

        return qk, qkv.softmax(dim=-1)



    def forward(self, k, q, v, mask):
        key = self.w_k(k)
        value = self.w_v(v)
        query = self.w_q(q)

        key = key.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1,2)
        query = query.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query=query, key=key, value=value, dropout=self.dropout, mask=mask)

        # B x H x Seq x Embed --> B x Seq X Embed
        x = x.transpose(1,2).contiguos().view(x.shape[0], -1, self.d_k * self.h )
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout : float):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + x.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_head : MultiHeadAttention, feed_forward : FeedForwardNN, dropout:float):
        self.self_attention_head = self_attention_head
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_head(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, blocks : nn.ModuleList, n : int):
        super().__init__()
        self.blocks = blocks
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, feed_forward: FeedForwardNN, self_attention : MultiHeadAttention, cross_attention : MultiHeadAttention , dropout : float):
        super().__init__()
        self.feed_forward = feed_forward
        self.self_attention = self_attention 
        self.cross_attention = cross_attention 
        self.dropout = dropout
        self.norm = LayerNormalization()
        self.residual_connections = [ ResidualConnection(dropout) for _ in range(3) ]
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):

        x = self.residual_connections[0](x, lambda x: self.self_attention(x,x,x, src_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention(encoder_output,x,x, tgt_mask))
        x = self.residual_connections[2](x, self.feed_forward(encoder_output,x,x, tgt_mask))

        return x

class Decoder(nn.Module):
    def __init__(self, blocks : nn.ModuleList):
        super().__init__()
        self.blocks = blocks
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)






