import torch
import utils
import time
import numpy as np
import torch.nn.functional as F
import torch_geometric.utils as gutils

from model import NLGNN



if __name__ == '__main__':
    dataset_name = 'cornell'
    # dataset_name = 'Cora'

    heter_dataset = ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']
    homo_dataset = ['Cora', 'Citeseer', 'Pubmed']

    lr = 0.01
    weight_decay = 5e-5
    max_epoch = 500
    patience = 200
    num_hidden = [96, 48, 16, 8]
    dropout = 0
    le_list = ['mlp', 'gcn', 'gat']
    le = le_list[2]

    # Currently, the window_size should be an odd number, otherwise we need to modify the setting of 'padding' in Conv1d
    window_size = 5

    re_generate_train_val_test = True

    split_by_label_flag = True
    if dataset_name in ['chameleon', 'cornell', 'texas']:
        split_by_label_flag = False



    if dataset_name in heter_dataset:
        data, num_features, num_classes = utils.load_heter_data(dataset_name)
    elif dataset_name in homo_dataset:
        dataset = utils.load_homo_data(dataset_name)
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    else:
        print("We do not have {} dataset right now.".format(dataset_name))




    utils.set_seed(15)

    if re_generate_train_val_test:
        idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.6, 0.2, 0.2, 15, split_by_label_flag)
        # idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.48, 0.32, 0.20, 15, split_by_label_flag)
    else:
        if dataset_name in heter_dataset:
            idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.6, 0.2, 0.2, 15, split_by_label_flag)
            # idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.48, 0.32, 0.20, 15, split_by_label_flag)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    data = data.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    data.edge_index = gutils.remove_self_loops(data.edge_index)[0]





    net = NLGNN(data, le, window_size, num_features, num_hidden, num_classes, dropout)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)



    dur = []
    los = []
    loc = []
    counter = 0
    min_loss = 100.0
    max_acc = 0.0



    for epoch in range(max_epoch):
        if epoch >= 3:
            t0 = time.time()
        net.train()
        logp = net()


        cla_loss = F.nll_loss(logp[idx_train], data.y[idx_train])
        loss = cla_loss
        train_acc = utils.accuracy(logp[idx_train], data.y[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp = net()
        test_acc = utils.accuracy(logp[idx_test], data.y[idx_test])
        loss_val = F.nll_loss(logp[idx_val], data.y[idx_val]).item()
        val_acc = utils.accuracy(logp[idx_val], data.y[idx_val])
        los.append([epoch, loss_val, val_acc, test_acc])

        if loss_val < min_loss and max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience and dataset_name in homo_dataset:
            print('early stop')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
            epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))

    if dataset_name in homo_dataset or 'syn' in dataset_name:
        los.sort(key=lambda x: x[1])
        print(los)
        acc = los[0][-1]
        print(acc)
    else:
        los.sort(key=lambda x: -x[2])
        print(los)
        acc = los[0][-1]
        print(acc)
