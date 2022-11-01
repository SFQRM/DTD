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
from dataset import *
from loss import *
from modules.viz import *


Loss_list_train = []    # 存储每次训练epoch的损失值
Loss_list_valid = []    # 存储每次验证epoch的损失值


# """
def run_epoch(data_iter, model, loss_compute, is_train=True):
    for i, g in tqdm(enumerate(data_iter)):
        with torch.set_grad_enabled(is_train):
            output = model(g)
            loss = loss_compute(output, g.tgt_y, g.n_tokens)
    if is_train:
        Loss_list_train.append(loss_compute.avg_loss)
    else:
        Loss_list_valid.append(loss_compute.avg_loss)
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))
# """


def draw_loss():
    plt.plot(np.arange(len(Loss_list_train)), Loss_list_train, label="train loss")

    # plt.plot(np.arange(len(train_acces)), train_acces, label="train acc")

    plt.plot(np.arange(len(Loss_list_valid)), Loss_list_valid, label="valid loss")

    # plt.plot(np.arange(len(eval_acces)), eval_acces, label="valid acc")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    # plt.ylabel("epoch")
    plt.title('Model accuracy&loss')
    plt.show()


def main(dev_id):               # dev_id = 0
# def main():
    if dev_id == -1:
        devices = torch.device('cpu')
    else:
        devices = torch.device('cuda:{}'.format(dev_id))

    # Set current device
    torch.cuda.set_device(devices)
    # device = ['cuda' if torch.cuda.is_available() else 'cpu']
    # print(torch.cuda.current_device())          # 判断是否使用GPU

    N = 1
    batch_size = 128

    # Prepare dataset
    dataset = get_dataset("copy")
    # print(dataset)

    # Build graph pool
    graph_pool = GraphPool()

    """
    data_iter = dataset(graph_pool, mode='train', batch_size=1, device=devices)
    # data_iter = dataset(graph_pool, mode='train', batch_size=1)
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
    """

    V = dataset.vocab_size
    # V = 11;

    criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    # criterion = LabelSmoothing(V, smoothing=0.1)

    # """
    dim_model = 128               # 128
    dim_ff = 128                  # 128
    dropout = 0.1
    # N = 1
    # """

    # Create model
    model = make_model(h=1, dim_model=dim_model, dim_ff=dim_ff, dropout=dropout, N=N, src_vocab=V, tgt_vocab=V)
    # model = make_model(V, V, N=N, dim_model=128, dim_ff=128, h=1)

    # Sharing weights between Encoder & Decoder
    model.src_embed.lut.weight = model.tgt_embed.lut.weight
    model.generator.proj.weight = model.tgt_embed.lut.weight

    # Move model to corresponding device
    model, criterion = model.to(devices), criterion.to(devices)

    # """
    # Optimizer
    model_opt = NoamOpt(dim_model, 1, 400, torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
    # model_opt = NoamOpt(dim_model, 1, 400, torch.optim.Adam(lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
    # """

    # """
    # Loss function
    loss_compute = SimpleLossCompute
    # """

    VIZ_IDX = 3
    att_maps = []

    epoch_num = 1500

    # """
    # Train & evaluate
    for epoch in range(epoch_num):
        start = time.time()
        # train_iter = dataset(graph_pool, mode='train', batch_size=args.batch, device=device, dev_rank=dev_rank, ndev=ndev)
        train_iter = dataset(graph_pool, mode='train', batch_size=batch_size, device=devices)
        # print(train_iter)
        valid_iter = dataset(graph_pool, mode='valid', batch_size=batch_size, device=devices)
        print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        # run_epoch(epoch, train_iter, dev_rank, ndev, model,loss_compute(opt=model_opt), is_train=True)
        run_epoch(train_iter, model, loss_compute(criterion, model_opt), is_train=True)
        print('Epoch: {} Evaluating...'.format(epoch))
        model.att_weight_map = None
        model.eval()
        # run_epoch(epoch, valid_iter, dev_rank, 1, model, loss_compute(opt=None), is_train=False)
        run_epoch(valid_iter, model, loss_compute(criterion, None), is_train=False)
        # att_maps.append(model.att_weight_map)
        end = time.time()
        print("epoch time: {}".format(end - start))
    # """

        """
            # Visualize attention
        exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
        # args_filter = ['batch', 'gpus', 'viz', 'master_ip', 'master_port', 'grad_accum', 'ngpu']
        # exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
        src_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='src')
        tgt_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='tgt')[:-1]
        # visualize head 0 of encoder-decoder attention
        att_animation(att_maps, 'e2d', src_seq, tgt_seq, 0)
        # draw_atts(model.att_weight_map, src_seq, tgt_seq, exp_setting, 'epoch_{}'.format(epoch))
        # with open('checkpoints/{}-{}.pkl'.format(exp_setting, epoch), 'wb') as f:
            # torch.save(model.state_dict(), f)
        """


        """
        if args.viz:
            src_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='src')
            tgt_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='tgt')[:-1]
            draw_atts(model.att_weight_map, src_seq, tgt_seq, exp_setting, 'epoch_{}'.format(epoch))
        args_filter = ['batch', 'gpus', 'viz', 'master_ip', 'master_port', 'grad_accum', 'ngpu']
        exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
        with open('checkpoints/{}-{}.pkl'.format(exp_setting, epoch), 'wb') as f:
            torch.save(model.state_dict(), f)
        """

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

    draw_loss()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('training translation model')
    args = argparser.parse_args()
    # print(args)
    main(dev_id=0)