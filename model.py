import torch
import torch.nn as nn
import math
from typing import Union

class InpEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embdLayer = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        # print(x.shape)
        x = self.embdLayer(x)
        return  x * math.sqrt(self.d_model) # for numerical stability

class PositionalEmbd(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        # create positional embedding vector for each token in sequence with shape (self.seq_length, self.d_model)
        self.pos_embd = torch.zeros((seq_length, d_model))
        pos = torch.arange(0, self.seq_length).view((self.seq_length, 1))
        two_i_term = torch.arange(0, self.d_model, 2)
        div_term = (10_000) ** (two_i_term/self.d_model)
        self.pos_embd[:, 0::2] = torch.sin(pos/div_term)
        self.pos_embd[:, 1::2] = torch.cos(pos/div_term)
        # our input has shape of (batch_size, seq_length, d_model)
        self.pos_embd = self.pos_embd.unsqueeze(0) # (1, seq_length, d_model)
        # dropout
        self.dpout = nn.Dropout(dropout)
        # it is better to register this embedding because it is already learned and i want to look at it later
        self.register_buffer('PositionalEmbedding', self.pos_embd)
    def forward(self, x):
        assert x.shape[1] <= self.seq_length, f"Your sequence length is bigger than defaults {x.shape[1]} > {self.seq_length}"
        x = x + self.pos_embd[:, :x.shape[1], :].requires_grad_(False).to('cpu')
        return self.dpout(x)

class LayerNorm(nn.Module):
    def __init__(self, eps:float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros((1,))) # additive parameter
        self.beta = nn.Parameter(torch.ones((1,))) # multiply parameter (bias)
    def forward(self, x):
        # x has shape of (batch_size, seq_length, d_model)
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        # by keeping dim  true we preserve shape of input (batch_size, seq_length, d_model)
        norm_x = (x - mean) / torch.sqrt(std + self.eps)
        return norm_x * self.beta + self.alpha

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dpout = nn.Dropout(p=dropout)
        # output = (batch_size, seq_length, d_ff)
        self.LinearLayer_1 = nn.Linear(d_model, d_ff, bias=True)
        # output = (batch_size, seq_length, d_model)
        self.LinearLayer_2 = nn.Linear(d_ff, d_model, bias=True)
    def forward(self, x):
        return self.LinearLayer_2(self.dpout(self.LinearLayer_1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_head:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        # d_model have to be divisible by number of heads
        assert d_model % n_head == 0, "embedding dimension[{d_model}] have to be divisible by number of heads[{n_head}]"
        self.n_head = n_head
        self.d_k = self.d_model // self.n_head
        self.LinearLayer_q = nn.Linear(self.d_model, self.d_model) # q`
        self.LinearLayer_k = nn.Linear(self.d_model, self.d_model) # k`
        self.LinearLayer_v = nn.Linear(self.d_model, self.d_model) # v`
        self.LinearLayer_Final = nn.Linear(self.d_k * self.n_head, self.d_model)
        self.dpout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(key, query, value, dpout:Union[nn.Dropout, None]=None, mask=None):
        d_k = query.shape[-1]

        # transpose(-1,-2) change it to (d_k, seq_length)
        #(batch_size, n_head, seq_length, seq_length)
        attention_score_logits = (query @ key.transpose(-1,-2)) / math.sqrt(d_k)
        if mask is not None:
            # we replace all places we want to hide their interaction with - infinity so their softmax will be zero
            # which indicate no interation between those tokens (query and keys)
            attention_score_logits.masked_fill_(mask == 0, -float('inf'))
            #(batch_size, n_head, seq_length, seq_length)
        # we apply softmax on last
        attention_score = torch.softmax(attention_score_logits, dim=-1)
        if dpout is not None:
            attention_score = dpout(attention_score)
         #(batch_size, n_head, seq_length, seq_length) @ v = (batch_size, n_head, seq_length, d_k)
        return (attention_score @ value), attention_score # we use attention_score for visualization


    def forward(self,q, k, v, mask=None):
        query = self.LinearLayer_q(q)
        key = self.LinearLayer_k(k)
        value = self.LinearLayer_v(v)
        # shape of query key value is (batch_size, seq_length, d_model)
        # we should reshaped them to (batch_size, seq_length, n_head, d_k)
        # and also we want to have (seq_length, d_k) shape in each head
        # we fllowing transpose we can get to shape (batch_size, n_head, seq_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.n_head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.n_head, self.d_k).transpose(1,2)
        # calculate attention over heads
        x, att_score = MultiHeadAttention.attention(key, query, value, self.dpout)
        # concatnate all heads
        # (batch_size, n_head, seq_length, d_k) --> (batch_size, seq_length, d_k * n_head)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_k * self.n_head)
        # (batch_size, seq_length, d_model)
        return self.LinearLayer_Final(x)

class ResConnection(nn.Module):
    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dpout = nn.Dropout(dropout)
        self.normLayer =  LayerNorm()
    def forward(self, x, sublayer): # sublayer is output of previous layer
        x = x + self.normLayer(sublayer(x))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, self_Attention:MultiHeadAttention, ResidualConn:ResConnection,
                 FFN:FeedForward, dropout:float) -> None:
        super().__init__()
        self.self_Attention = self_Attention
        # wee need two Residual Connection in Encoder block
        self.ResidualConns = nn.ModuleList([ResidualConn for _ in range(2)])
        self.FFN = FFN
    def forward(self, x, src_mask):
        # we need mask for encoder to prevent padding token to interact with others
        # now we create residual connection in encoder with combine of other layers
        x = self.ResidualConns[0](x, lambda x: self.self_Attention(q=x,k=x,v=x,mask=src_mask))
        x = self.ResidualConns[0](x, self.FFN)
        return x

# we want to stack EncoderBlock
class Encoder(nn.Module):
    def __init__(self,n_layers:nn.ModuleList) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.normLayer = LayerNorm()
    def forward(self, x, src_mask):
        for layer in self.n_layers:
            x = layer(x, src_mask)
        # we stability we apply a normalization to output of final layer of stacked layers
        return self.normLayer(x)


# generally we built all parts of our transformer
# specifically we should put all of them correctly with correct setup
# lets build decoder block
class DecoderBlock(nn.Module):
    def __init__(self, self_attention:MultiHeadAttention, cross_attention:MultiHeadAttention,
                 ResidualConn:ResConnection, FFN:FeedForward, dropout:float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ResidualConns = nn.ModuleList([ResidualConn for _ in range(3)])
        self.FFN = FFN
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.ResidualConns[0](x, lambda x: self.self_attention(q=x,k=x,v=x,mask=target_mask))
        x = self.ResidualConns[1](x, lambda x: self.cross_attention(q=x,k=encoder_output,
                                                                v=encoder_output, mask=src_mask))
        return self.ResidualConns[2](x, self.FFN)

# now we build Decoder
class Decoder(nn.Module):
    def __init__(self, n_layers:nn.ModuleList) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.normLayer = LayerNorm()
    def forward(self,x, encoder_output, src_mask, target_mask):
        for layer in self.n_layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        # we stability we apply a normalization to output of final layer of stacked layers
        return self.normLayer(x)

# now we should project output of decoder which is (Batch_size, seq_length, d_model) to (Batch_size, seq_length, token_space)
class LinearProjection(nn.Module):
    def __init__(self, d_model:int, token_space:int) -> None: #token_space is Vocab_size
        super().__init__()
        self.LinearProj = nn.Linear(d_model, token_space)
    def forward(self, x):
        # we also apply softmax as final operation in our transformer
        return torch.softmax(self.LinearProj(x), dim=-1) # apply on token_space dimension


# now we put all things together to create TRANSFORMER FOR TRANSLATION
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embedding:InpEmbedding,
                 target_embedding:InpEmbedding, src_positionalENC:PositionalEmbd,
                 target_positionalENC:PositionalEmbd, proj_layer:LinearProjection)-> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_positionalENC = src_positionalENC
        self.target_positionalENC = target_positionalENC
        self.proj_layer = proj_layer

    # we define three function to implement each part of transformer
    def encoder_p(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_positionalENC(src)
        src = self.encoder(src, src_mask)
        return src

    def decoder_p(self, target_data, encoder_output, src_mask, target_mask):
        target_data = self.target_embedding(target_data)
        target_data = self.target_positionalENC(target_data)
        target_data = self.decoder(target_data, encoder_output, src_mask, target_mask)
        return target_data

    def proj(self, x):
        return self.proj_layer(x)

# now we go to define our transformer
def transformer_builder(src_token_space:int, target_token_space:int,
                        src_seq_len:int, target_seq_len:int, N_layer:int=1,
                        n_head:int=4, dff:int=200, d_model:int=100, dropout:float=0.1) -> Transformer:
    # instantiate Embedding layers
    src_embd = InpEmbedding(d_model=d_model, vocab_size=src_token_space)
    target_embd = InpEmbedding(d_model=d_model, vocab_size=target_token_space)
    # instantiate Positional Encoding layer
    src_pos = PositionalEmbd(d_model=d_model, seq_length=src_seq_len, dropout=dropout)
    target_pos = PositionalEmbd(d_model=d_model, seq_length=target_seq_len, dropout=dropout)
    # create stacked encoder blocks
    encoder_stacked = []
    for _ in range(N_layer):
        enc = EncoderBlock(self_Attention=MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout),
                           ResidualConn=ResConnection(dropout=dropout),
                 FFN=FeedForward(d_model=d_model, d_ff=dff, dropout=dropout),dropout=dropout)
        encoder_stacked.append(enc)
    # create stacked decoder blocks
    decoder_stacked = []
    for _ in range(N_layer):
        dec = DecoderBlock(self_attention=MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout),
                            cross_attention=MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout),
                            ResidualConn=ResConnection(dropout=dropout),
                    FFN=FeedForward(d_model=d_model, d_ff=dff, dropout=dropout),dropout=dropout)
        decoder_stacked.append(dec)

    # instantiate EncoderPart
    encoder = Encoder(nn.ModuleList(encoder_stacked))
    # instantiate DecoderPart
    decoder = Decoder(nn.ModuleList(decoder_stacked))
    # instantiate Projection part
    projection = LinearProjection(d_model=d_model, token_space=target_token_space)
    # instantiate Transformer
    transformer = Transformer(encoder=encoder, decoder=decoder,
                              src_embedding=src_embd, target_embedding=target_embd,
                              src_positionalENC=src_pos, target_positionalENC=target_pos,
                              proj_layer=projection)

    # initialize parameters of model from uniform distirbution for better start
    for p in transformer.parameters():
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)
    return transformer

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = transformer_builder(src_token_space=vocab_src_len, target_token_space=vocab_tgt_len, src_seq_len=config["seq_len_src"],
                                target_seq_len=config['seq_len_tgt'], d_model=config['d_model'])
    return model