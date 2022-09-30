"""
    # Dynamic transformer with DGL
"""
import torch
import argparse
import dgl
from modules.models import make_model
from modules.noamopt import NoamOpt


def main(dev_id):
    if dev_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(dev_id))

    # Set current device
    torch.cuda.set_device(device)

    # Prepare dataset
    dataset = dgl.data.KarateClubDataset()
    V = dataset.num_classes

    print(dataset)
    # print(num_classes)
    # print(dataset[0].ndata['label'])

    dim_model = 4               # 128
    dim_ff = 4                  # 128
    dropout = 0.1
    N = 1

    # Dataset
    # dataset = get_dataset("copy")

    # Create model
    model = make_model(h=1, dim_model=dim_model, dim_ff=dim_ff, dropout=dropout, N=N, src_vocab=V, tgt_vocab=V)

    # Sharing weights between Encoder & Decoder
    model.src_embed.lut.weight = model.tgt_embed.lut.weight
    # model.generator.proj.weight = model.tgt_embed.lut.weight

    # Move model to corresponding device
    model = model.to(device)

    # Optimizer
    model_opt = NoamOpt(dim_model, 1, 400, torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))

    # loss_compute = SimpleLossCompute



if __name__ == '__main__':
    argparser = argparse.ArgumentParser('training translation model')
    args = argparser.parse_args()
    # print(args)
    main(dev_id=0)