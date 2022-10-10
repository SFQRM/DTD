from .attention import *
from .layers import *
from .embedding import *
from .functions import *
import torch as th
import dgl.function as fn
import torch.nn.init as INIT


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = th.nn.LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            "规范化节点表示"
            norm_x = layer.sublayer[0].norm(x)                              # x←LayerNorm(x)
            "使用自我注意力机制将规范化后的节点表示映射到一组“查询”，“键”和“值”"
            "??? 是怎么映射到get上的 ???"
            "换句话说，layer.self_attn.映射的结果是什么"
            return layer.self_attn.get(norm_x, fields=fields)               # [Q,K,V]←[W_Q,W_K,W_V ]∙x
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            "规范化wv得到多头注意力层的输出o"
            o = layer.self_attn.get_o(wv / z)                               # o←W_o∙(wv/z)+b_o
            "残差连接"
            x = x + layer.sublayer[0].dropout(o)                            # x←x+o
            "在x上应用两层位置前馈层，然后添加残差连接"
            x = layer.sublayer[1](x, layer.feed_forward)                    # x←x+LayerNorm(FNN(x))
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = th.nn.LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv', l=0):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            "规范化节点表示: x←LayerNorm(x)"
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


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, pos_enc, h, d_k, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder,  self.decoder = encoder, decoder
        self.src_embed, self.tgt_embed = src_embed, tgt_embed
        self.pos_enc = pos_enc
        self.generator = generator
        self.h, self.d_k = h, d_k
        self.att_weight_map = None

    def propagate_attention(self, g, eids):
        # Compute attention score
        # 计算注意力得分
        g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)             # score = W_q·x dot W_k·x
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)), eids)     # score = exp(score/sqrt(d_k))
        # Send weighted values to target nodes
        # 获取相关节点上的 "值"V，并和每个节点入边上的注意力得分加权
        g.send_and_recv(eids, fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))    # wv_j = Σscore_ij·V_i
        # 对每个节点入边上的注意力得分求和，用于归一化
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))     # Z_j = Σscore_ij

    def update_graph(self, g, eids, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."
        "更新图的节点状态和边缘状态。"

        # Pre-compute queries and key-value pairs.
        for pre_func, nids in pre_pairs:
            g.apply_nodes(pre_func, nids)
        self.propagate_attention(g, eids)
        # Further calculation after attention mechanism
        for post_func, nids in post_pairs:
            g.apply_nodes(post_func, nids)

    def forward(self, graph):
        g = graph.g
        nids, eids = graph.nids, graph.eids

        # Word Embedding and Position Embedding
        src_embed, src_pos = self.src_embed(graph.src[0]), self.pos_enc(graph.src[1])
        tgt_embed, tgt_pos = self.tgt_embed(graph.tgt[0]), self.pos_enc(graph.tgt[1])
        g.nodes[nids['enc']].data['x'] = self.pos_enc.dropout(src_embed + src_pos)
        g.nodes[nids['dec']].data['x'] = self.pos_enc.dropout(tgt_embed + tgt_pos)

        for i in range(self.encoder.N):
            # Step 1: Encoder Self-attention
            pre_func = self.encoder.pre_func(i, 'qkv')
            post_func = self.encoder.post_func(i)
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])

        for i in range(self.decoder.N):
            # Step 2: Dncoder Self-attention
            pre_func = self.decoder.pre_func(i, 'qkv')
            post_func = self.decoder.post_func(i)
            nodes, edges = nids['dec'], eids['dd']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])
            # Step 3: Encoder-Decoder attention
            "编码器和解码器之间的交叉注意力"
            pre_q = self.decoder.pre_func(i, 'q', 1)
            pre_kv = self.decoder.pre_func(i, 'kv', 1)
            post_func = self.decoder.post_func(i, 1)
            nodes_e, nodes_d, edges = nids['enc'], nids['dec'], eids['ed']
            self.update_graph(g, edges, [(pre_q, nodes_d), (pre_kv, nodes_e)], [(post_func, nodes_d)])

        return self.generator(g.ndata['x'][nids['dec']])


def make_model(h, dim_model, dim_ff, dropout, N, src_vocab, tgt_vocab):
    "注意力计算"
    attn = MultiHeadAttention(h, dim_model)
    "前馈网络"
    ff = PositionwiseFeedForward(dim_model, dim_ff)
    "位置编码"
    pos_enc = PositionalEncoding(dim_model, dropout)
    "编码器"
    c = copy.deepcopy
    encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
    "解码器"
    decoder = Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff), dropout), N)
    "源点嵌入"
    src_embed = Embeddings(src_vocab, dim_model)
    "目标点嵌入"
    tgt_embed = Embeddings(tgt_vocab, dim_model)

    generator = Generator(dim_model, tgt_vocab)

    # model = Transformer(encoder, decoder, src_embed, tgt_embed, pos_enc, generator, h, dim_model // h)
    model = Transformer(encoder, decoder, pos_enc, h, dim_model // h, src_embed, tgt_embed, generator)

    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model