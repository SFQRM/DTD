from .attention import *
from .layers import *
from .embedding import *
from .functions import *
import torch as th
import dgl.function as fn


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = th.nn.LayerNorm(layer.size)

    """
    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)                              # 规范化节点表示
            return layer.self_attn.get(norm_x, fields=fields)
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func
    """


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = th.nn.LayerNorm(layer.size)

    """
    def pre_func(self, i, fields='qkv', l=0):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[l].norm(x) if fields.startswith('q') else x
            if fields != 'qkv':
                return layer.src_attn.get(norm_x, fields)
            else:
                return layer.self_attn.get(norm_x, fields)
        return func

    def post_func(self, i, l=0):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[l].dropout(o)
            if l == 1:
                x = layer.sublayer[2](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func
    """


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, pos_enc, h, d_k):
        super(Transformer, self).__init__()
        self.encoder,  self.decoder = encoder, decoder
        # self.src_embed, self.tgt_embed = src_embed, tgt_embed
        self.pos_enc = pos_enc
        # self.generator = generator
        self.h, self.d_k = h, d_k
        self.att_weight_map = None

    def propagate_attention(self, g, eids):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)), eids)
        # Send weighted values to target nodes
        g.send_and_recv(eids, fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))


def make_mode(h, dim_model, dim_ff, dropout, N):
    '注意力计算'
    attn = MultiHeadAttention(h, dim_model)
    '前馈网络'
    ff = PositionwiseFeedForward(dim_model, dim_ff)
    '位置编码'
    pos_enc = PositionalEncoding(dim_model, dropout)
    '编码器'
    c = copy.deepcopy
    encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
    '解码器'
    decoder = Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff), dropout), N)

    # model = Transformer(encoder, decoder, src_embed, tgt_embed, pos_enc, generator, h, dim_model // h)
    model = Transformer(encoder, decoder, pos_enc, h, dim_model // h)