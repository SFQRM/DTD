graph_pool = GraphPool()

data_iter = dataset(graph_pool, mode='train', batch_size=1, devices=devices)
for graph in data_iter:
    print(graph.nids['enc']) # encoder node ids
    print(graph.nids['dec']) # decoder node ids
    print(graph.eids['ee']) # encoder-encoder edge ids
    print(graph.eids['ed']) # encoder-decoder edge ids
    print(graph.eids['dd']) # decoder-decoder edge ids
    print(graph.src[0]) # Input word index list
    print(graph.src[1]) # Input positions
    print(graph.tgt[0]) # Output word index list
    print(graph.tgt[1]) # Ouptut positions
    break