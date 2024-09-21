# @title model.py
import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, vocab_sz:int, d_model:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, d_model)
    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pe = torch.empty(seq_len, d_model).to(device)
        self.dropout = nn.Dropout(dropout)

        # ( seq_len, 1 )
        pos = torch.arange(0,seq_len).unsqueeze(1)

        # ( d_model/2 )
        i   = torch.arange(0,d_model,2)

        # ( d_model/2 )
        denom = torch.pow(10000, i / d_model)

        self.pe[:, 0::2] = torch.sin(pos * denom)
        self.pe[:, 1::2] = torch.cos(pos * denom)

        # pe -> (1, seq_len, d_model)
        self.pe.unsqueeze_(0)

    def forward(self, x):
        # x -> (batch, seq_len, d_model)
        return x + self.pe[:, x.shape[1]-1, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, head:int, d_model:int, dropout:float):
        super().__init__()

        assert d_model % head == 0, f"Can't create heads. d_model:{d_model}, head:{head}"
        self.d_k  = d_model // head
        self.head = head
        self.w_q  = nn.Linear(d_model, d_model)
        self.w_k  = nn.Linear(d_model, d_model)
        self.w_v  = nn.Linear(d_model, d_model)
        self.w_o  = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, padding_mask):
        # query -> ( batch, seq_len, d_model )
        # -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        batch_sz, seq_len = query.shape[0], query.shape[1]
        # print("Query Shape:", query.shape)
        q = self.w_q(query).view(batch_sz, seq_len, self.head, self.d_k).transpose(1,2)
        k = self.w_k(key).view(batch_sz, seq_len, self.head, self.d_k).transpose(1,2)
        v = self.w_v(value).view(batch_sz, seq_len, self.head, self.d_k).transpose(1,2)

        # print("\n * self.head:", self.head)
        # print("\n * self.d_k:", self.d_k)
        # print("\n * q.shape:", q.shape)
        # print("\n * k.shape:", k.shape)

        # (batch, h, seq_len, d_model) X (batch, h, d_model, seq_len)
        # -> (batch, h, seq_len, seq_len)
        attention = q @ k.transpose(-1,-2) / math.sqrt(self.d_k)
        # print("\n * qk.shape:", attention.shape)

        if padding_mask is not None:
            causal_mask = torch.triu(attention, diagonal = 1).bool()
            attention.masked_fill_(causal_mask , 1e-10)
            attention.masked_fill_(padding_mask , 1e-10)

        attention = attention.softmax(dim=-1)

        # (batch, h, seq_len, seq_len) X (batch, h, seq_len, d_k)
        # -> (batch, h, seq_len, d_k)
        attention = attention @ v
        # print(" * qkv.shape:", attention.shape)

        # (batch, seq_len, d_model)
        return self.w_o(attention.transpose(1,2).contiguous().view(batch_sz, -1, self.d_k * self.head))

class ResidualConnection(nn.Module):
    def __init__(self, d_model:int):
        super().__init__()
        self.normalize = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_x):
        x += sublayer_x
        return self.normalize(x)

class FFN(nn.Module):
    def __init__(self, d_model:int, ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff)
        self.linear2 = nn.Linear(ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.relu(self.linear1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, multi_head_attn:MultiHeadAttention,
                 residual_1:ResidualConnection,
                 ffn:FFN):
        super().__init__()
        self.multi_head_attn = multi_head_attn
        self.residual_1 = residual_1
        self.ffn = ffn

    def forward(self, x, padding_source_mask=None):

        before_resid = x
        x = self.multi_head_attn(x,x,x, padding_source_mask)
        # print("Multi-head Attn Shape:", x.shape)

        x = self.residual_1(x, before_resid)
        # print("Normalized:", x.shape)

        x = self.ffn(x)
        # print("ffn:", x.shape)
        return x


def build_encoder_block(encoder_config):

    d_model = encoder_config['d_model']
    head = encoder_config['head_num']
    dropout = encoder_config['dropout_value']
    ff = encoder_config['ffn']

    multi_head_attn = MultiHeadAttention(head, d_model, dropout)
    residul_1 = ResidualConnection(d_model)
    ffn = FFN(d_model, ff, dropout)
    return EncoderBlock(multi_head_attn, residul_1, ffn)

class Encoder(nn.Module):
    def __init__(self, blocks:nn.ModuleList):
        super().__init__()
        self.blocks = blocks
    def forward(self, x, padding_source_mask):
        for block in self.blocks:
            x = block(x, padding_source_mask)
        return x


def build_encoder(encoder_config):
    blocks_num = encoder_config['blocks_num']
    blocks = nn.ModuleList([build_encoder_block(encoder_config) for _ in range(blocks_num)])
    return Encoder(blocks)


class DecoderBlock(nn.Module):
    def __init__(self, multi_head_attn:MultiHeadAttention,
                 cross_attn:MultiHeadAttention,
                 residual_1:ResidualConnection,
                 residual_2:ResidualConnection,
                 residual_3:ResidualConnection,
                 ffn:FFN):
        super().__init__()
        self.multi_head_attn = multi_head_attn
        self.cross_attn = cross_attn
        self.residual_1 = residual_1
        self.residual_2 = residual_2
        self.residual_3 = residual_3
        self.ffn = ffn

    def forward(self, encoder_output, x, padding_target_mask=None):

        before_resid = x
        x = self.multi_head_attn(x,x,x, padding_target_mask)
        x = self.residual_1(x, before_resid)

        before_resid = x
        x = self.cross_attn(x, encoder_output, encoder_output, padding_target_mask)
        x = self.residual_2(x, before_resid)

        before_resid = x
        x = self.ffn(x)
        x = self.residual_3(x, before_resid)

        return x


class Decoder(nn.Module):
    def __init__(self, blocks:nn.ModuleList):
        super().__init__()
        self.blocks = blocks
    def forward(self, encoder_output, x, padding_target_mask=None):
        for block in self.blocks:
            x = block(encoder_output, x, padding_target_mask)
        return x


def build_decoder_block(model_config):

    d_model = model_config['d_model']
    head = model_config['head_num']
    dropout = model_config['dropout_value']
    ff = model_config['ffn']

    multi_head_attn = MultiHeadAttention(head, d_model, dropout)
    cross_attention = MultiHeadAttention(head, d_model, dropout)
    residul_1 = ResidualConnection(d_model)
    residul_2 = ResidualConnection(d_model)
    residul_3 = ResidualConnection(d_model)
    ffn = FFN(d_model, ff, dropout)
    return DecoderBlock(multi_head_attn, cross_attention, residul_1,
                        residul_2, residul_3, ffn)

def build_decoder(model_config):
    blocks_num = model_config['blocks_num']
    blocks = nn.ModuleList([build_decoder_block(model_config) for _ in range(blocks_num)])
    return Decoder(blocks)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_sz : int):
        super().__init__()
        self.d_model  = d_model
        self.vocab_sz = vocab_sz
        self.linear   = nn.Linear(d_model, vocab_sz)

    # (B, seq_len, d_model) -> (B, seq_len, vocab_sz)
    def forward(self, x):
        x = self.linear(x)
        # print("X_Shape:", x.shape)
        return x

class Transformer(nn.Module):
    def __init__(self, input_embeddings:InputEmbedding,
                 positional_encoding:PositionalEncoding,
                 encoder_blocks:Encoder,
                 decoder_blocks:Decoder,
                 projection_layer:ProjectionLayer):
        super().__init__()
        self.input_embeddings = input_embeddings
        self.positional_encoding = positional_encoding
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.projection_layer = projection_layer

    def forward(self, encoder_input, decoder_input, padding_source_mask, padding_target_mask):

        encoder_input  = self.input_embeddings(encoder_input)
        encoder_input  = self.positional_encoding(encoder_input)

        decoder_input  = self.input_embeddings(decoder_input)
        decoder_input  = self.positional_encoding(decoder_input)

        encoder_output = self.encoder_blocks(encoder_input, padding_source_mask)
        decoder_output = self.decoder_blocks(encoder_output, decoder_input, padding_target_mask)

        return self.projection_layer(decoder_output)

def build_transformer(model_config):
    vocab_sz = model_config['vocab_size']
    d_model = model_config['d_model']
    seq_len = model_config['seq_len']
    dropout = model_config['dropout_value']

    input_embeddings = InputEmbedding(vocab_sz, d_model)
    positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
    projection_layer = ProjectionLayer(model_config['d_model'], model_config['vocab_size'])
    encoder = build_encoder(model_config)
    decoder = build_decoder(model_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Model Created And Moved To ", device)
    return Transformer(input_embeddings, positional_encoding, encoder, decoder, projection_layer).to(device)
