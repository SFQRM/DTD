"""
    # Dynamic transformer with DGL
"""
import torch
import argparse
import dgl
import time
from tqdm import *
from modules.models import make_model
from modules.noamopt import NoamOpt
from modules.loss import SimpleLossCompute
from modules.get_dataset import get_dataset


"""
def run_epoch(epoch, data_iter, dev_rank, ndev, model, loss_compute, is_train=True):
    universal = isinstance(model, UTransformer)
    with loss_compute:
        for i, g in enumerate(data_iter):
            with T.set_grad_enabled(is_train):
                if universal:
                    output, loss_act = model(g)
                    if is_train: loss_act.backward(retain_graph=True)
                else:
                    output = model(g)
                tgt_y = g.tgt_y
                n_tokens = g.n_tokens
                loss = loss_compute(output, tgt_y, n_tokens)
"""


def run_epoch(data_iter, model, loss_compute, is_train=True):
    for i, g in tqdm(enumerate(data_iter)):
        with torch.set_grad_enabled(is_train):
            output = model(g)
            loss = loss_compute(output, g.tgt_y, g.n_tokens)
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))


def main(dev_id):
# def main():
    if dev_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(dev_id))

    # Set current device
    torch.cuda.set_device(device)
    # device = ['cuda' if torch.cuda.is_available() else 'cpu']

    # Prepare dataset
    # dataset = dgl.data.KarateClubDataset()
    # dataset = get_dataset("copy")
    dataset = get_dataset(11, 30, 30)
    # V = dataset.vocab_size
    V = 11;

    # print(dataset)
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
    # model.src_embed.lut.weight = model.tgt_embed.lut.weight
    # model.generator.proj.weight = model.tgt_embed.lut.weight

    # Move model to corresponding device
    # model = model.to(device)

    # Optimizer
    model_opt = NoamOpt(dim_model, 1, 400, torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
    # model_opt = NoamOpt(dim_model, 1, 400, torch.optim.Adam(lr=1e-3, betas=(0.9, 0.98), eps=1e-9))

    # Loss function
    loss_compute = SimpleLossCompute

    # Train & evaluate
    for epoch in range(4):
        start = time.time()
        # train_iter = dataset(graph_pool, mode='train', batch_size=args.batch, device=device, dev_rank=dev_rank, ndev=ndev)
        train_iter = dataset
        valid_iter = dataset
        print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        # run_epoch(epoch, train_iter, dev_rank, ndev, model,loss_compute(opt=model_opt), is_train=True)
        run_epoch(train_iter, model, loss_compute(opt=model_opt), is_train=True)
        print('Epoch: {} Evaluating...'.format(epoch))
        model.att_weight_map = None
        model.eval()
        # run_epoch(epoch, valid_iter, dev_rank, 1, model, loss_compute(opt=None), is_train=False)
        run_epoch(valid_iter, model, loss_compute(opt=None), is_train=False)
        end = time.time()
        print("epoch time: {}".format(end - start))

    """
        if dev_rank == 0:
            model.att_weight_map = None
            model.eval()
            valid_iter = dataset(graph_pool, mode='valid', batch_size=args.batch,device=device, dev_rank=dev_rank, ndev=1)
            run_epoch(epoch, valid_iter, dev_rank, 1, model,
                      loss_compute(opt=None), is_train=False)
            end = time.time()
            print("epoch time: {}".format(end - start))
    """


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('training translation model')
    args = argparser.parse_args()
    # print(args)
    main(dev_id=0)