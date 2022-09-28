"""
    # Dynamic transformer with DGL
"""
import torch
import argparse
import dgl
from modules.models import make_mode


def main(dev_id):
    if dev_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(dev_id))

    # Set current device
    torch.cuda.set_device(device)

    # Prepare dataset
    dataset = dgl.data.KarateClubDataset()
    num_classes = dataset.num_classes

    # print(num_classes)
    # print(dataset[0].ndata['label'])

    dim_model = 4               # 128
    dim_ff = 4                  # 128
    dropout = 0.1
    N = 1

    # Create model
    model = make_mode(h=1, dim_model=dim_model, dim_ff=dim_ff, dropout=dropout, N=N)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser('training translation model')
    args = argparser.parse_args()
    # print(args)
    main(dev_id=0)