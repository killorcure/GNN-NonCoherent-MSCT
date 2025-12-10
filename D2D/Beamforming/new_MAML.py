import time
from copy import deepcopy
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import higher

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # import numpy
import matplotlib.pyplot as plt  # import matplotlib.pyplot for figure plotting
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import wireless_networks_generator as wg
import wmmse

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())  # , BN(channels[i])
        for i in range(1, len(channels))
    ])


class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, Nt, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.Nt = Nt
        # self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)

    def init_IGConv_parameters(self, train_K, users):
        self.train_K = train_K
        self.users = users

    # def update(self, aggr_out, x):
    #     print('update===aggr_out:{}'.format(aggr_out))
    #     print('update===x:{}'.format(x))
    #     tmp = torch.cat([x, aggr_out], dim=1)
    #     comb_all = self.mlp2(tmp)
    #     print('update===comb_all:{}'.format(comb_all))
    #     comb = comb_all[:, 1:2 * Nt + 1]
    #     links_res = comb_all[:, 0:1]
    #     add_delta_delay = comb_all[:, 1:2]
    #     add_delta_delay = torch.sigmoid(add_delta_delay)
    #     # links_onehot = update_links(links_res)
    #     links_onehot = links_res
    #     print('update===tmp:{}'.format(tmp))
    #     print('update===comb:{}'.format(comb))
    #     print('update===add_delta_delay:{}'.format(add_delta_delay))
    #     nor = torch.sqrt(torch.sum(torch.mul(comb, comb), axis=1))
    #     print('update===nor:{}'.format(nor))
    #     nor = nor.unsqueeze(axis=-1)
    #     print('update===unsqueeze nor:{}'.format(nor))
    #     comp1 = torch.ones(comb.size(), device=device)
    #     print('update===comp1:{}'.format(comp1))
    #     comb = torch.div(comb, torch.max(comp1, nor))
    #     print('update===comb:{}'.format(comb))
    #     comb_res = torch.cat([links_onehot, add_delta_delay, comb], dim=1)
    #     print('update===comb_res:{}'.format(comb_res))
    #     return torch.cat([comb_res, x[:, :2 * Nt - 2]], dim=1)

    def update(self, aggr_out, x):
        print('update===aggr_out:{}'.format(aggr_out))
        print('update===x:{}'.format(x))
        tmp = torch.cat([x, aggr_out], dim=1)
        comb_all = self.mlp2(tmp)
        print('update===comb_all:{}'.format(comb_all))
        comb = comb_all[:, 2:2 * self.Nt + 2]
        links_res = comb_all[:, 0:1]
        add_delta_delay = comb_all[:, 1:2]
        # add_delta_delay = torch.sigmoid(add_delta_delay/0.5)
        # links_onehot = update_links(links_res)
        links_onehot = links_res
        print('update===tmp:{}'.format(tmp))
        print('update===comb:{}'.format(comb))
        print('update===add_delta_delay:{}'.format(add_delta_delay))
        # shape_length = comb.shape[0]
        comb_group = comb.view(-1,self.users,2*self.Nt)
        square_sum = (comb_group ** 2).sum(dim=(1,2), keepdim=True)
        square_sum = square_sum + 1e-6
        comb_normalized = comb_group / torch.sqrt(square_sum)
        comb = comb_normalized.view(comb.shape[0],2*self.Nt)
        print('update===normalized comb:{}'.format(comb))
        # nor = torch.sqrt(torch.sum(torch.mul(comb, comb), axis=1))
        # print('update===nor:{}'.format(nor))
        # nor = nor.unsqueeze(axis=-1)
        # print('update===unsqueeze nor:{}'.format(nor))
        # comp1 = torch.ones(comb.size(), device=device)
        # print('update===comp1:{}'.format(comp1))
        # comb = torch.div(comb, torch.max(comp1, nor))
        # print('update===comb:{}'.format(comb))
        comb_res = torch.cat([links_onehot, add_delta_delay, comb], dim=1)
        print('update===comb_res:{}'.format(comb_res))
        return torch.cat([comb_res, x[:, :2 * self.Nt - 2]], dim=1)

    def forward(self, x, edge_index, edge_attr):
        print('forward===x:{}, edge_index:{}, edge_attr:{}'.format(x, edge_index, edge_attr))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        print('unsqueeze x:{}'.format(x))
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        print('unsqueeze edge_attr:{}'.format(edge_attr))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        print('message===x_i:{}, x_j:{}'.format(x_i, x_j)) ## x_i为当前节点，x_j为邻居节点
        print('message edge_attr:{}'.format(edge_attr))
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        print('message tmp:{}'.format(tmp))
        print('message agg:{}'.format(agg))
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)

class IGCNet(torch.nn.Module):
    def __init__(self, num_layers, hard, Nt):
        super(IGCNet, self).__init__()
        self.num_layers = num_layers
        self.hard = hard
        self.Nt = Nt
        self.mlp1 = MLP([6 * Nt+1, 64, 64])
        self.mlp2 = MLP([64 + 4 * Nt, 32])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(32, 2 + 2 * Nt))])
        self.conv = IGConv(self.mlp1, self.mlp2, self.Nt)

        # self.network_layer = 3

    def init_IGCNet_parameters(self, train_K, users):
        # self.Nt = Nt
        self.train_K = train_K
        self.users = users
        self.conv.init_IGConv_parameters(train_K, users)
        # self.mlp1 = MLP([6 * Nt+1, 64, 64])
        # self.mlp2 = MLP([64 + 4 * Nt, 32])
        # self.mlp2 = Seq(*[self.mlp2, Seq(Lin(32, 2 + 2 * Nt))])
        # self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, data, weights):
        # weights = gumbel_softmax_sample(self.layer_logits, tau=self.tau, hard=self.hard)
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        num_layers = 0
        for i in range(self.num_layers):
            if weights[i] > 0:
                num_layers += 1
                out = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
                x0 = out
        print('====forward====valid num_layers: {}, weights:{}'.format(num_layers, weights))
        # x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        # # new_data1 = update_edge_attr(data, x1[:,0:1])
        # x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        # # new_data2 = update_edge_attr(new_data1, x2[:,0:1])
        # out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        return x0

# 自定义 GNN 网络模型（你可以替换为自己的设计）
class YourGNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=5):
        super(YourGNNModel, self).__init__()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = IGCNet().to(device)
        # model = IGCNet()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList([
            nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)
        ])
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        h = x
        for layer in self.gcn_layers:
            h = F.relu(layer(torch.matmul(adj, h)))  # GCN 传播
        return self.out_layer(h)



# Gumbel-Softmax 采样函数
def gumbel_softmax_sample(logits, tau=1.0, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    y = logits + gumbel_noise
    y = F.softmax(y / tau, dim=-1)
    if hard:
        index = y.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
        y = (y_hard - y).detach() + y  # 使用 straight-through trick
    return y


# 定义 GumbelControlledGNN
class GumbelControlledGNN(nn.Module):
    def __init__(self, model, num_layers=5, tau=1.0):
        super(GumbelControlledGNN, self).__init__()
        self.model = model  # 你的自定义 GNN 模型
        self.num_layers = num_layers  # GNN 最大层数
        self.tau = tau  # 控制采样的温度
        self.layer_logits = nn.Parameter(torch.ones(num_layers))  # 每一层的控制参数（学习的 logits）

    def set_parameters(self, SNR_dB, train_S, users, train_K, Nt):
        self.SNR_dB = SNR_dB
        self.train_S = train_S
        self.users = users
        self.train_K = train_K
        self.Nt = Nt
        self.model.init_IGCNet_parameters(train_K, users)

    def forward(self, data, hard=False):
        # 使用 Gumbel-Softmax 采样，得到每一层是否被激活（1表示使用该层，0表示跳过该层）
        weights = gumbel_softmax_sample(self.layer_logits, tau=self.tau, hard=True)
        print('====GumbelControlledGNN=====forward layer_logits: {}, weights: {}'.format(self.layer_logits, weights))
        return self.model(data, weights)
        # # 获取每一层的输出，并决定是否使用该层
        # h = x
        # for i in range(self.num_layers):
        #     # 判断当前层是否激活（即是否使用该层）
        #     if weights[i] > 0:  # 如果该层被激活，才进行传播
        #         h = self.model.gcn_layers[i](h, adj)  # 只有激活的层才执行计算
        #     # 如果该层被跳过，当前层的输出将不会被传递给下一层
        #
        # return self.model.out_layer(h), weights

def build_graph_MSCT_asyn(CSI, dist, delays, norm_csi_real, norm_csi_imag, K, users, threshold):
    print('CSI:{}'.format(CSI))
    print('dist:{}'.format(dist))
    print('delays:{}'.format(delays))
    print('norm_csi_real:{}'.format(norm_csi_real))
    print('norm_csi_imag:{}'.format(norm_csi_imag))
    print('K:{}'.format(K))
    n = CSI.shape[0]
    Nt = CSI.shape[2]
    x1 = np.array([CSI[ii, ii, :] for ii in range(K)])
    x2 = np.imag(x1)
    x1 = np.real(x1)
    x3 = 1 / np.sqrt(Nt) * np.ones((n, 2 * Nt))
    print('x1:{}'.format(x1))
    print('x2:{}'.format(x2))
    print('x3:{}'.format(x3))
    x = np.concatenate((x3, x1, x2), axis=1)
    print('x:{}'.format(x))
    x = torch.tensor(x, dtype=torch.float)

    dist2 = np.copy(dist)
    for i in range(dist2.shape[0]):
        for j in range(dist2.shape[1]):
            # if (i % users) == (j % users):
            #     dist2[i][j] = 0
            # ## 卫星间波束干扰约束
            # elif (i / (users * beams)) == (j / (users * beams)) and (i % users) == (j % users):
            #     dist2[i][j] = 0
            # else:
            #     dist2[i][j] = 1
            if i == j:
                dist2[i][j] = 0
            elif (i % users) == (j % users):
                dist2[i][j] = -1
            else:
                dist2[i][j] = 1
    print('dist2:{}'.format(dist2))
    # init_links = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    # s = len(init_links)
    # for i in range(s):
    #     for j in range(s):
    #         ## 波束干扰约束
    #         if i != j and init_links[i] == 1 and init_links[j] == 1:
    #             dist2[i][j] = 10
    #         else:
    #             dist2[i][j] = 0
    # if (i / users) == (j / users):
    #     dist2[i][j] = 0
    # elif (i / (users*beams)) == (j / (users*beams)) and (i % users) == (j % users):
    #     dist2[i][j] = 0
    # ## 卫星间波束干扰约束
    # elif (i / (users*beams)) == (j / (users*beams)) and (i % users) == (j % users):
    #     dist2[i][j] = 0

    # mask = np.eye(K)
    # diag_dist = np.multiply(mask, dist2)
    # print('dist2:{}'.format(dist2))
    # print('mask:{}'.format(mask))
    # print('diag_dist1:{}'.format(diag_dist))
    # dist2 = dist2 + 1000 * diag_dist
    # print('dist2_1:{}'.format(dist2))
    # # dist2[dist2 > threshold] = 0
    # dist2[dist2 = 10] = 0  # 这里每个连接对之间都有干扰
    print('dist2_1:{}'.format(dist2))
    attr_ind = np.nonzero(dist2)
    print('attr_ind:{}'.format(attr_ind))
    edge_attr_real = norm_csi_real[attr_ind]
    edge_attr_imag = norm_csi_imag[attr_ind]
    delta_delay = delays[attr_ind]
    delta_delay = np.expand_dims(delta_delay, axis=-1)
    print('delta_delay reshape:{}'.format(delta_delay))
    attr_ind = np.array(attr_ind)
    print('attr_ind array:{}'.format(attr_ind))
    flag = np.zeros(delta_delay.shape)
    for i in range(len(attr_ind[0, :])):
        flag[i][0] = dist2[attr_ind[0, i], attr_ind[1, i]]
    print('flag:{}, size:{}'.format(flag, flag.shape))
    # edge_attr = np.concatenate((edge_attr_real, edge_attr_imag, flag, delta_delay), axis=1)
    edge_attr = np.concatenate((edge_attr_real, edge_attr_imag, flag), axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    edge_index = torch.tensor(adj, dtype=torch.long)

    H1 = np.expand_dims(np.real(CSI), axis=-1)
    H2 = np.expand_dims(np.imag(CSI), axis=-1)
    HH = np.concatenate((H1, H2), axis=-1)
    y = torch.tensor(np.expand_dims(HH, axis=0), dtype=torch.float)
    initial_delay = torch.tensor(np.expand_dims(delta_delay, axis=0), dtype=torch.float)
    print('H1:{}'.format(H1))
    print('H2:{}'.format(H2))
    print('HH:{}'.format(HH)) # [实部 虚部] [实部 虚部]
    print('y:{}'.format(np.expand_dims(HH, axis=0)))
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y, initial_delay=torch.tensor(delays, dtype=torch.float),
                norm_csi_real=norm_csi_real, norm_csi_imag=norm_csi_imag, dist=dist2,
                CSI=torch.tensor(CSI, dtype=torch.float))
    return data

def proc_data_asyn(HH, dists, delays, norm_csi_real, norm_csi_imag, K, users):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
        # data = build_graph(HH[i,:,:,:],dists[i,:,:], norm_csi_real[i,:,:,:], norm_csi_imag[i,:,:,:], K,500)
        data = build_graph_MSCT_asyn(HH[i, :, :, :], dists[i, :, :], delays[i, :, :], norm_csi_real[i, :, :, :],
                                     norm_csi_imag[i, :, :, :], K, users, 500)
        data_list.append(data)
    return data_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetaGNNTrainer:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, inner_steps=1):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)  # 优化 α（层数控制参数）
        self.inner_steps = inner_steps
        self.device = device
        # self.SNR = 50

    def set_parameters(self, SNR_dB, train_S, users, train_K, Nt):
        self.model.set_parameters(SNR_dB, train_S, users, train_K, Nt)

    def train_step(self, task_batch):
        meta_loss = 0.0
        self.meta_optimizer.zero_grad()  # 清除外层梯度

        for task in task_batch:
            # print('=====train_step=====task:{}'.format(task_batch))
            # x_train, adj_train, y_train, x_val, adj_val, y_val = [t.to(self.device) for t in task]
            SNR_dB, train_S, users, train_K, Nt, train_layouts, test_layouts, train_dists, train_csis, train_delays, norm_train_real, norm_train_imag, \
            test_dists, test_csis, test_delays, norm_test_real, norm_test_imag = task
            train_data_list = proc_data_asyn(train_csis, train_dists, train_delays, norm_train_real, norm_train_imag,
                                             train_K, users)
            # print('train_data_list:{}'.format(train_data_list))
            # train_layout = train_csis.shape[0]
            test_data_list = proc_data_asyn(test_csis, test_dists, test_delays, norm_test_real, norm_test_imag, train_K, users)

            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # model = IGCNet().to(device)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

            # train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True, num_workers=1)
            # test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
            train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

            # 1. Clone the model for inner loop optimization (任务副本)
            model_clone = deepcopy(self.model)  # 内层任务副本
            model_clone.set_parameters(SNR_dB, train_S, users, train_K, Nt)
            inner_optimizer = torch.optim.SGD(model_clone.model.parameters(), lr=0.01)
            inner_scheduler = torch.optim.lr_scheduler.StepLR(inner_optimizer, step_size=1, gamma=0.9)
            # fnet.model.init_IGCNet_parameters(train_K, users)

            # with higher.innerloop_ctx(model_clone, inner_optimizer) as (fnet, diffopt):
            #     # 2. Inner loop: train GNN for the specific task
            #     for epoch in range(100):
            #         print('Start to train...')
            #         loss1, avr_rate_syn, avr_rate_asyn_no_add, avr_rate_asyn_add, total_avr_sync_wmmse, \
            #         total_avr_async_noadd_wmmse, total_avr_async_wmmse = train(fnet, diffopt, train_loader, epoch, SNR_dB, train_S, users, train_K, Nt, train_layouts)
            #         # print('Start to test...')
            #         # loss2, test_average_wmmse = test()
            #         # print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Train Loss: {:.4f}, avr_rate_syn: {:.4f}, avr_rate_asyn_no_add: {:.4f}, '
            #         #       'avr_rate_asyn_add: {:.4f}, Average syncWMMSE Rate: {:.4f}, Average asyncWMMSE_Noadd Rate: {:.4f}, '
            #         #       'Average asyncWMMSE Rate: {:.4f}'.format(SNR_dB, imperfect_channel_condition, epoch, loss1, avr_rate_syn, avr_rate_asyn_no_add,
            #         #                                                avr_rate_asyn_add, total_avr_sync_wmmse, total_avr_async_noadd_wmmse,
            #         #                                                total_avr_async_wmmse))
            #
            #         inner_scheduler.step()
            #         #
            #         # pred, _ = fnet(x_train, adj_train, hard=True)
            #         # loss = sum_rate_loss(pred, y_train)  # 最小化负的和速率损失
            #         # diffopt.step()  # 更新任务特定的参数 θ
            #         # diffopt.step(loss1)
            #         # diffopt.step(loss1)
            #     # 3. Outer loop: meta optimization for structure parameter (α)
            # with higher.innerloop_ctx(model_clone, inner_optimizer) as (fnet, diffopt):
                # 2. Inner loop: train GNN for the specific task
            for epoch in range(100):
                print('Start to train...')
                inner_optimizer.zero_grad()
                loss1, avr_rate_syn, avr_rate_asyn_no_add, avr_rate_asyn_add, total_avr_sync_wmmse, \
                total_avr_async_noadd_wmmse, total_avr_async_wmmse = train(model_clone, inner_optimizer, train_loader, epoch, SNR_dB, train_S, users, train_K, Nt, train_layouts)
                loss1.backward()
                inner_optimizer.step()
                inner_scheduler.step()
                # print('Start to test...')
                # loss2, test_average_wmmse = test()
                # print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Train Loss: {:.4f}, avr_rate_syn: {:.4f}, avr_rate_asyn_no_add: {:.4f}, '
                #       'avr_rate_asyn_add: {:.4f}, Average syncWMMSE Rate: {:.4f}, Average asyncWMMSE_Noadd Rate: {:.4f}, '
                #       'Average asyncWMMSE Rate: {:.4f}'.format(SNR_dB, imperfect_channel_condition, epoch, loss1, avr_rate_syn, avr_rate_asyn_no_add,
                #                                                avr_rate_asyn_add, total_avr_sync_wmmse, total_avr_async_noadd_wmmse,
                #                                                total_avr_async_wmmse))
                # inner_scheduler.step()
            print('Start to test...')
            tloss1, tavr_rate_syn, tavr_rate_asyn_no_add, tavr_rate_asyn_add, ttotal_avr_sync_wmmse, \
            ttotal_avr_async_noadd_wmmse, ttotal_avr_async_wmmse = test(model_clone, inner_optimizer, train_loader, train_S, train_K, users, Nt, epoch, SNR_dB,
                                                                       train_layouts)
                # pred_val, _ = fnet(x_val, adj_val, hard=True)  # 用验证集计算任务损失
                # val_loss = sum_rate_loss(pred_val, y_val)
                # meta_loss += tloss1
            meta_loss += tloss1

        meta_loss /= len(task_batch)
        meta_loss.backward()  # 反向传播计算元梯度
        self.meta_optimizer.step()  # 更新 α（结构超参数）

        return meta_loss.item()

def test(model, optimizer, test_loader, train_S, train_K, users, Nt, epoch, SNR_dB, train_layouts):
    model.eval()

    total_loss = 0.0
    total_avr_rate_syn = 0
    total_avr_rate_asyn_no_add = 0
    total_avr_rate_asyn_add = 0
    total_avr_async_wmmse = 0
    total_avr_async_noadd_wmmse = 0
    total_avr_sync_wmmse = 0
    total_loss_imperfect = 0
    total_avr_rate_syn_imperfect = 0
    total_avr_rate_asyn_no_add_imperfect = 0
    total_avr_rate_asyn_add_imperfect = 0
    total_avr_async_wmmse_imperfect = 0
    total_avr_async_noadd_wmmse_imperfect = 0
    total_avr_sync_wmmse_imperfect = 0
    total_avr_gnn_time = 0
    total_avr_wmmse_time = []
    times = 0
    num = 0
    for data in test_loader:
        data = data.to(device)
        num += 1
        # with torch.no_grad():
        gnn_start_time = time.time()
        out = model(data)
        gnn_end_time = time.time()
        print('train====out:{}'.format(out))
        loss, avr_rate_syn, avr_rate_asyn_no_add, avr_rate_asyn_add, sync_wmmse_rate, asyn_wmmse_rate_noadd, async_wmmse_rate, wmmse_time = sr_loss_all_test(data, out, SNR_dB, train_S, users, train_K, Nt, epoch, 0)
        # loss_imperfect, avr_rate_syn_imperfect, avr_rate_asyn_no_add_imperfect, avr_rate_asyn_add_imperfect, sync_wmmse_rate_imperfect, \
        # asyn_wmmse_rate_noadd_imperfect, async_wmmse_rate_imperfect, wmmse_time_imperfect = sr_loss_all_test(data, out, SNR_dB, train_S, users, train_K, Nt, epoch, 1)
        # loss.backward()
        total_loss += loss * data.num_graphs
        total_avr_gnn_time += gnn_end_time - gnn_start_time
        times += 1
        if wmmse_time > 0:
            total_avr_wmmse_time.append(wmmse_time)
        # if wmmse_time_imperfect > 0:
        #     total_avr_wmmse_time.append(wmmse_time_imperfect)
        total_avr_rate_syn += avr_rate_syn.item() * data.num_graphs
        total_avr_rate_asyn_no_add += avr_rate_asyn_no_add.item() * data.num_graphs
        total_avr_rate_asyn_add += avr_rate_asyn_add.item() * data.num_graphs
        total_avr_async_wmmse += async_wmmse_rate * data.num_graphs
        total_avr_async_noadd_wmmse +=  asyn_wmmse_rate_noadd * data.num_graphs
        total_avr_sync_wmmse += sync_wmmse_rate * data.num_graphs
        # total_loss_imperfect += loss_imperfect.item() * data.num_graphs
        # total_avr_rate_syn_imperfect += avr_rate_syn_imperfect.item() * data.num_graphs
        # total_avr_rate_asyn_no_add_imperfect += avr_rate_asyn_no_add_imperfect.item() * data.num_graphs
        # total_avr_rate_asyn_add_imperfect += avr_rate_asyn_add_imperfect.item() * data.num_graphs
        # total_avr_async_wmmse_imperfect += async_wmmse_rate_imperfect * data.num_graphs
        # total_avr_async_noadd_wmmse_imperfect +=  asyn_wmmse_rate_noadd_imperfect * data.num_graphs
        # total_avr_sync_wmmse_imperfect += sync_wmmse_rate_imperfect * data.num_graphs
        print('train====data.num_graphs:{}'.format(data.num_graphs))
            # optimizer.step()
        print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Test Loss: {:.4f}, avr_rate_syn: {:.4f}, avr_rate_asyn_no_add: {:.4f}, '
              'avr_rate_asyn_add: {:.4f}, Average syncWMMSE Rate: {:.4f}, Average asyncWMMSE_Noadd Rate: {:.4f}, '
              'Average asyncWMMSE Rate: {:.4f}, GNN_Time: {}, WMMSE_Time: {}'.format(SNR_dB, 0, epoch, total_loss / train_layouts, total_avr_rate_syn / train_layouts, total_avr_rate_asyn_no_add / train_layouts, \
               total_avr_rate_asyn_add / train_layouts, total_avr_sync_wmmse / train_layouts, total_avr_async_noadd_wmmse / train_layouts, total_avr_async_wmmse / train_layouts, total_avr_gnn_time / times, total_avr_wmmse_time))
        # print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Test Loss: {:.4f}, avr_rate_syn_imperfect: {:.4f}, avr_rate_asyn_no_add_imperfect: {:.4f}, '
        #       'avr_rate_asyn_add_imperfect: {:.4f}, Average syncWMMSE_imperfect Rate: {:.4f}, Average asyncWMMSE_Noadd_imperfect Rate: {:.4f}, '
        #       'Average asyncWMMSE_imperfect Rate: {:.4f}'.format(SNR_dB, 1, epoch, total_loss_imperfect / train_layouts, total_avr_rate_syn_imperfect / train_layouts, total_avr_rate_asyn_no_add_imperfect / train_layouts, \
        #        total_avr_rate_asyn_add_imperfect / train_layouts, total_avr_sync_wmmse_imperfect / train_layouts, \
        #        total_avr_async_noadd_wmmse / train_layouts, total_avr_async_wmmse / train_layouts))
        return total_loss / train_layouts, total_avr_rate_syn / train_layouts, total_avr_rate_asyn_no_add / train_layouts, \
               total_avr_rate_asyn_add / train_layouts, total_avr_sync_wmmse / train_layouts, \
               total_avr_async_noadd_wmmse / train_layouts, total_avr_async_wmmse / train_layouts


def train(model, optimizer, train_loader, epoch, SNR_dB, train_S, users, train_K, Nt, train_layouts):
    # model.train()

    total_loss = 0
    total_avr_rate_syn = 0
    total_avr_rate_asyn_no_add = 0
    total_avr_rate_asyn_add = 0
    total_avr_async_wmmse = 0
    total_avr_async_noadd_wmmse = 0
    total_avr_sync_wmmse = 0
    total_loss_imperfect = 0
    total_avr_rate_syn_imperfect = 0
    total_avr_rate_asyn_no_add_imperfect = 0
    total_avr_rate_asyn_add_imperfect = 0
    total_avr_async_wmmse_imperfect = 0
    total_avr_async_noadd_wmmse_imperfect = 0
    total_avr_sync_wmmse_imperfect = 0
    total_avr_gnn_time = 0
    total_avr_wmmse_time = []
    times = 0
    imperfect_channel = 0
    # before_loss = 0
    start = time.time()
    # optimizer.zero_grad()
    for data in train_loader:
        data = data.to(device)
        # optimizer.zero_grad()
        gnn_start_time = time.time()
        out = model(data)
        gnn_end_time = time.time()
        print('train====out:{}'.format(out))
        loss, avr_rate_syn, avr_rate_asyn_no_add, avr_rate_asyn_add, sync_wmmse_rate, asyn_wmmse_rate_noadd, async_wmmse_rate, wmmse_time = sr_loss_all_test(data, out, SNR_dB, train_S, users, train_K, Nt, epoch, 0)
        # loss_imperfect, avr_rate_syn_imperfect, avr_rate_asyn_no_add_imperfect, avr_rate_asyn_add_imperfect, sync_wmmse_rate_imperfect, \
        # asyn_wmmse_rate_noadd_imperfect, async_wmmse_rate_imperfect, wmmse_time_imperfect = sr_loss_all_test(data, out, SNR_dB, train_S, users, train_K, Nt, epoch, 1)
        # # loss.backward()
        total_loss += loss * data.num_graphs
        total_avr_gnn_time += gnn_end_time - gnn_start_time
        times += 1
        if wmmse_time > 0:
            total_avr_wmmse_time.append(wmmse_time)
        # if wmmse_time_imperfect > 0:
        #     total_avr_wmmse_time.append(wmmse_time_imperfect)
        total_avr_rate_syn += avr_rate_syn.item() * data.num_graphs
        total_avr_rate_asyn_no_add += avr_rate_asyn_no_add.item() * data.num_graphs
        total_avr_rate_asyn_add += avr_rate_asyn_add.item() * data.num_graphs
        total_avr_async_wmmse += async_wmmse_rate * data.num_graphs
        total_avr_async_noadd_wmmse +=  asyn_wmmse_rate_noadd * data.num_graphs
        total_avr_sync_wmmse += sync_wmmse_rate * data.num_graphs
        # total_loss_imperfect += loss_imperfect.item() * data.num_graphs
        # total_avr_rate_syn_imperfect += avr_rate_syn_imperfect.item() * data.num_graphs
        # total_avr_rate_asyn_no_add_imperfect += avr_rate_asyn_no_add_imperfect.item() * data.num_graphs
        # total_avr_rate_asyn_add_imperfect += avr_rate_asyn_add_imperfect.item() * data.num_graphs
        # total_avr_async_wmmse_imperfect += async_wmmse_rate_imperfect * data.num_graphs
        # total_avr_async_noadd_wmmse_imperfect +=  asyn_wmmse_rate_noadd_imperfect * data.num_graphs
        # total_avr_sync_wmmse_imperfect += sync_wmmse_rate_imperfect * data.num_graphs
        print('train====data.num_graphs:{}'.format(data.num_graphs))
        # loss.backward()
        # optimizer.step()
    print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Train Loss: {:.4f}, avr_rate_syn: {:.4f}, avr_rate_asyn_no_add: {:.4f}, '
          'avr_rate_asyn_add: {:.4f}, Average syncWMMSE Rate: {:.4f}, Average asyncWMMSE_Noadd Rate: {:.4f}, '
          'Average asyncWMMSE Rate: {:.4f}, GNN_Time: {}, WMMSE_Time: {}'.format(SNR_dB, 0, epoch, total_loss / train_layouts, total_avr_rate_syn / train_layouts, total_avr_rate_asyn_no_add / train_layouts, \
           total_avr_rate_asyn_add / train_layouts, total_avr_sync_wmmse / train_layouts, total_avr_async_noadd_wmmse / train_layouts, total_avr_async_wmmse / train_layouts, total_avr_gnn_time / times, total_avr_wmmse_time))
    # print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Train Loss: {:.4f}, avr_rate_syn_imperfect: {:.4f}, avr_rate_asyn_no_add_imperfect: {:.4f}, '
    #       'avr_rate_asyn_add_imperfect: {:.4f}, Average syncWMMSE_imperfect Rate: {:.4f}, Average asyncWMMSE_Noadd_imperfect Rate: {:.4f}, '
    #       'Average asyncWMMSE_imperfect Rate: {:.4f}'.format(SNR_dB, 1, epoch, total_loss_imperfect / train_layouts, total_avr_rate_syn_imperfect / train_layouts, total_avr_rate_asyn_no_add_imperfect / train_layouts, \
    #        total_avr_rate_asyn_add_imperfect / train_layouts, total_avr_sync_wmmse_imperfect / train_layouts, \
    #        total_avr_async_noadd_wmmse_imperfect / train_layouts, total_avr_async_wmmse_imperfect / train_layouts))
    return total_loss / train_layouts, total_avr_rate_syn / train_layouts, total_avr_rate_asyn_no_add / train_layouts, \
           total_avr_rate_asyn_add / train_layouts, total_avr_sync_wmmse / train_layouts, \
           total_avr_async_noadd_wmmse / train_layouts, total_avr_async_wmmse / train_layouts


def sr_loss_all_test(data, p, SNR_dB, train_S, users, K, N, epoch, imperfect_channel):
    # H1 K*K*N
    # p1 K*N
    eta_enable = 1
    eta_add = 1
    print('sr_loss===data:{}'.format(data))
    print('sr_loss===p:{}'.format(p))
    H1 = data.y[:, :, :, :, 0]  ##实部
    H2 = data.y[:, :, :, :, 1]  ##虚部
    print('sr_loss===H1:{}, size:{}'.format(H1, H1.shape))
    print('sr_loss===H2:{}, size:{}'.format(H2, H2.shape))
    H3 = torch.complex(H1,H2)
    print('sr_loss===H3:{}, size:{}'.format(H3, H3.shape))
    links = p[:, 0:1]
    add_delta_delay = p[:, 1:2]
    add_delta_delay = torch.sigmoid(add_delta_delay/0.5)
    print('sr_loss===add_delta_delay sigmoid: {} sizez:{}'.format(add_delta_delay, add_delta_delay.shape))
    add_delta_delay = torch.reshape(add_delta_delay, (-1, K, 1, 1))
    # links = update_links(links)
    p1 = p[:, 2:N + 2]
    p2 = p[:, N + 2:2 * N + 2]
    print('sr_loss===p1:{}'.format(p1))
    print('sr_loss===p2:{}'.format(p2))
    # p1 = torch.mul(links, p1)
    # p2 = torch.mul(links, p2)
    print('sr_loss===mul links p1:{}'.format(p1))
    print('sr_loss===mul links p2:{}'.format(p2))
    p1 = torch.reshape(p1, (-1, K, 1, N))
    p2 = torch.reshape(p2, (-1, K, 1, N))
    print('sr_loss===reshape p1:{}, size:{}'.format(p1, p1.shape))
    print('sr_loss===reshape p2:{}, size:{}'.format(p2, p2.shape))
    p3 = torch.complex(p1, p2)
    print('sr_loss===complex p3:{}, size:{}'.format(p3, p3.shape))
    H4 = H3.conj()
    print('sr_loss===H4:{}, size:{}'.format(H4, H4.shape))
    H_new = torch.clone(H3)
    # H_new_conj = H_new.conj()
    if imperfect_channel == 1:
        mod = torch.abs(H_new)
        phase = torch.angle(H_new)
        sigma = 0.1
        phase_error = torch.normal(0, sigma, size=phase.shape).cuda()
        new_phase = phase_error + phase
        H_new = mod * torch.exp(1j*new_phase)
        H_new_conj = H_new.conj()
        rx_all_power = torch.mul(H_new_conj, p3)
    else:
        rx_all_power = torch.mul(H4, p3)
    print('sr_loss===rx_all_power:{}, size:{}'.format(rx_all_power, rx_all_power.shape))
    rx_all_power = torch.sum(rx_all_power, axis=-1)
    print('sr_loss===sum rx_all_power:{}, size:{}'.format(rx_all_power, rx_all_power.shape))
    initial_delay = data.initial_delay
    print('sr_loss===initial_delay:{}, size:{}'.format(initial_delay, initial_delay.shape))
    initial_delay = torch.reshape(initial_delay, (-1, K, K, 1))
    print('sr_loss===reshape initial_delay:{}, size:{}'.format(initial_delay, initial_delay.shape))
    print('sr_loss===add_delta_delay: {} sizez:{}'.format(add_delta_delay, add_delta_delay.shape))
    asyn_wmmse_rate = 0
    asyn_wmmse_rate_noadd = 0
    syn_wmmse_rate = 0
    wmmse_exec_time = 0
    if epoch == 1 or (epoch % 100 == 0):
        start_wmmse = time.time()
        print('sr_loss===start to compute async_WMMSE and sync_WMMSE')
        asyn_wmmse_rate = compute_asyn_wmmse(H3, H_new, p3, initial_delay, add_delta_delay, SNR_dB, train_S, users, K, N, 1, 1)
        asyn_wmmse_rate_noadd = compute_asyn_wmmse(H3, H_new, p3, initial_delay, add_delta_delay, SNR_dB, train_S, users, K, N, 1, 0)
        syn_wmmse_rate = compute_asyn_wmmse(H3, H_new, p3, initial_delay, add_delta_delay, SNR_dB, train_S, users, K, N, 0, 0)
        end_wmmse = time.time()
        wmmse_exec_time = (end_wmmse - start_wmmse) / 3
        print('sr_loss===end to compute async_WMMSE and sync_WMMSE with time: {}'.format(wmmse_exec_time))
    # rx_all_power_1 = torch.abs(rx_all_power)**2
    rate_iter_syn = torch.empty(0).cuda()
    rate_iter_asyn_no_add = torch.empty(0).cuda()
    rate_iter_asyn_add = torch.empty(0).cuda()
    for iter in range(rx_all_power.shape[0]):
        rate_syn = torch.zeros(1).cuda()
        rate_asyn_no_add = torch.zeros(1).cuda()
        rate_asyn_add = torch.zeros(1).cuda()
        for user_index in range(users):
            valid_signal = torch.empty(0).cuda()
            signal = H_new[iter, user_index, user_index,:].abs()**2
            signal_power = signal.mean()
            noise_power = signal_power / (10**(SNR_dB/10))
            print('sr_loss===H3[iter, user_index, user_index,:]: {} signal:{} signal_power: {} noise_power: {}'.format(
                H_new[iter, user_index, user_index,:], signal, signal_power, noise_power))
            interference_sum_power_syn = torch.empty(0).cuda()
            interference_sum_power_asyn_no_add = torch.empty(0).cuda()
            interference_sum_power_asyn_add = torch.empty(0).cuda()
            for link in range(rx_all_power.shape[1]):
                # satellite_index = int(link/users)
                if int(link % users) == user_index:
                    valid_signal = torch.cat(
                        (valid_signal, rx_all_power[iter, link, link].view(1)))
                # else:
                #     valid_link = satellite_index*users+user_index
                #     interference_signal = torch.cat(
                #         (interference_signal, rx_all_power[iter, link, user_index].view(1)))
                #     Delta_tau = torch.cat(
                #         (Delta_tau, initial_delay[iter, link, valid_link].view(1)))
            for other_user_index in range(users):
                interference_signal = torch.empty(0).cuda()
                Delta_tau_asyn = torch.empty(0).cuda()
                Delta_tau_syn = torch.empty(0).cuda()
                Add_delta_tau = torch.empty(0).cuda()
                if other_user_index != user_index:
                    print('before calculate interference_signal:{}'.format(interference_signal))
                    for other_link in range(rx_all_power.shape[1]):
                        if int(other_link % users) == other_user_index:
                            satellite_index = int(other_link / users)
                            valid_link = satellite_index * users + user_index
                            interference_signal = torch.cat(
                                (interference_signal, rx_all_power[iter, other_link, valid_link].view(1)))
                            Delta_tau_asyn = torch.cat(
                                        (Delta_tau_asyn, initial_delay[iter, other_link, valid_link].view(1)))
                            if eta_add == 1:
                                Add_delta_tau = torch.cat((Add_delta_tau, add_delta_delay[iter, valid_link, 0].view(1)
                                                           - add_delta_delay[iter, other_link, 0].view(1)))
                    interference_signal_conj = interference_signal.conj()
                    print('sr_loss===Add_delta_tau:{} valid_signal: {} interference_signal:{} interference_signal_conj:{} Delta_tau:{}'.format(
                        Add_delta_tau, valid_signal, interference_signal, interference_signal_conj, Delta_tau_asyn))
                    # if eta_enable == 0:
                        # interference_power = torch.outer(interference_signal, interference_signal_conj)
                    interference_conj_matric = torch.outer(interference_signal, interference_signal_conj)
                    interference_conj_matric_transpose = interference_conj_matric.conj().T
                    print('sr_loss===interference_conj_matric:{}, whether a gongezhuanzhi:{}'.format(
                        interference_conj_matric, torch.allclose(interference_conj_matric, interference_conj_matric_transpose)))
                    # interference_power_syn = torch.sum(torch.outer(interference_signal, interference_signal_conj))
                    # interference_sum_power_syn = torch.cat((interference_sum_power_syn, interference_power_syn))
                    # else:
                    interference_power_asyn = torch.outer(interference_signal, interference_signal_conj)
                    print('sr_loss===outer interference_power: {} size:{}'.format(interference_power_asyn, interference_power_asyn.shape))
                    eta_value_no_add = torch.outer(Delta_tau_asyn, Delta_tau_asyn)
                    eta_value_add = torch.outer(Delta_tau_asyn, Delta_tau_asyn)
                    print('sr_loss===init eta_value: {} sizez:{}'.format(eta_value_no_add, eta_value_no_add.shape))
                    for i in range(Delta_tau_asyn.shape[0]):
                        for j in range(Delta_tau_asyn.shape[0]):
                            # if eta_add == 0:
                            eta_value_no_add[i, j] = calculate_eta(Delta_tau_asyn[i], Delta_tau_asyn[j])
                            # elif eta_add == 1:
                            # eta_value_add[i, j] = calculate_eta(Delta_tau_asyn[i]+Add_delta_tau[i], Delta_tau_asyn[j]+Add_delta_tau[j])
                            eta_value_add[i, j] = calculate_eta(Add_delta_tau[i], Add_delta_tau[j])
                    print('sr_loss===config eta_value_no_add: {} eta_value_add:{}'.format(eta_value_no_add, eta_value_add))
                    interference_power_asyn_no_add = torch.mul(interference_power_asyn, eta_value_no_add)
                    interference_power_asyn_add = torch.mul(interference_power_asyn, eta_value_add)
                    print('sr_loss===mul interference_power_asyn_no_add: {} interference_power_asyn_add:{}'.format(interference_power_asyn_no_add, interference_power_asyn_add))
                    # interference_power_syn = torch.sum(interference_power_syn)
                    interference_power_syn = torch.sum(interference_power_asyn)
                    interference_power_asyn_no_add = torch.sum(interference_power_asyn_no_add)
                    interference_power_asyn_add = torch.sum(interference_power_asyn_add)
                    print('sr_loss===interference_power_syn:{} interference_power_asyn_no_add:{} interference_power_asyn_add:{}'.format(
                        interference_power_syn, interference_power_asyn_no_add,interference_power_asyn_add))
                    interference_sum_power_syn = torch.cat((interference_sum_power_syn, interference_power_syn.view(1)))
                    interference_sum_power_asyn_no_add = torch.cat((interference_sum_power_asyn_no_add, interference_power_asyn_no_add.view(1)))
                    interference_sum_power_asyn_add = torch.cat((interference_sum_power_asyn_add, interference_power_asyn_add.view(1)))
                    # interference_signal_outer_async = torch.mul(interference_signal_outer, eta_value)
                    print('sr_loss===interference_sum_power_syn:{} interference_sum_power_asyn_no_add: {} interference_sum_power_asyn_add:{}'.format(
                        interference_sum_power_syn, interference_sum_power_asyn_no_add,interference_sum_power_asyn_add))
                        # interference_power = torch.sum(interference_signal_outer_async)
            interference_sum_power_syn = torch.sum(interference_sum_power_syn)
            interference_sum_power_asyn_no_add = torch.sum(interference_sum_power_asyn_no_add)
            interference_sum_power_asyn_add = torch.sum(interference_sum_power_asyn_add)
            if torch.abs(interference_sum_power_syn).item() < torch.abs(interference_sum_power_asyn_no_add).item() or \
                    torch.abs(interference_sum_power_syn).item() < torch.abs(interference_sum_power_asyn_add).item():
                print('invalid syn and asyn results...')
            # print('sr_loss===Delta_tau: {} sizez:{}'.format(Delta_tau, Delta_tau.shape))
            valid_signal_conj = valid_signal.conj()
            valid_power = torch.sum(torch.outer(valid_signal, valid_signal_conj))
            valid_power_another = torch.abs(torch.sum(valid_signal))**2
            print('sr_loss===valid_power: {} valid_power_another:{} interference_sum_power_syn:{} interference_sum_power_asyn_no_add:{} interference_sum_power_asyn_add:{} noise_power:{}'.format(
                valid_power, valid_power_another, interference_sum_power_syn, interference_sum_power_asyn_no_add, interference_sum_power_asyn_add, noise_power))
            rate_syn += torch.log2(1 + torch.div(torch.abs(valid_power), torch.abs(interference_sum_power_syn)+noise_power))
            rate_asyn_no_add += torch.log2(1 + torch.div(torch.abs(valid_power), torch.abs(interference_sum_power_asyn_no_add)+noise_power))
            rate_asyn_add += torch.log2(1 + torch.div(torch.abs(valid_power), torch.abs(interference_sum_power_asyn_add)+noise_power))
            print('sr_loss===user: {} rate_syn:{}, rate_asyn_no_add:{} rate_asyn_add:{}'.format(user_index, rate_syn, rate_asyn_no_add, rate_asyn_add))
            ## 计算WMMSE
                # for satellite in range(int(rx_all_power.shape[1]/users)):
                #     valid_signal = torch.cat((valid_signal, rx_all_power[iter,satellite*users+user_index, user_index]))
                #     valid_signal += rx_all_power[iter,satellite*users+user_index]
        rate_iter_syn = torch.cat((rate_iter_syn, rate_syn.view(1)))
        rate_iter_asyn_no_add = torch.cat((rate_iter_asyn_no_add, rate_asyn_no_add.view(1)))
        rate_iter_asyn_add = torch.cat((rate_iter_asyn_add, rate_asyn_add.view(1)))
        print('sr_loss===iter: {} rate_iter_syn:{}, rate_iter_asyn_no_add:{}, rate_iter_asyn_add:{}'.format(iter, rate_iter_syn, rate_iter_asyn_no_add, rate_iter_asyn_add))
    avr_rate_syn = torch.mean(rate_iter_syn)
    avr_rate_asyn_no_add = torch.mean(rate_iter_asyn_no_add)
    avr_rate_asyn_add = torch.mean(rate_iter_asyn_add)
    if avr_rate_syn.item() > avr_rate_asyn_no_add or avr_rate_syn.item() > avr_rate_asyn_add:
        print('invalid results...')
    print('sr_loss===avr_rate_syn: {}, avr_rate_asyn_no_add:{} avr_rate_asyn_add:{}'.format(avr_rate_syn, avr_rate_asyn_no_add, avr_rate_asyn_add))
    loss = torch.neg(avr_rate_asyn_add)
    print('sr_loss===loss:{}'.format(loss))
    return loss, avr_rate_syn, avr_rate_asyn_no_add, avr_rate_asyn_add, syn_wmmse_rate, asyn_wmmse_rate_noadd, asyn_wmmse_rate, wmmse_exec_time

def compute_asyn_wmmse(H3, H_new, p3, initial_delay, add_delta_delay, SNR_dB, train_S, users, K, Nt, eta_enable, eta_add):
    H6 = torch.clone(H3)
    H7 = torch.clone(H_new)
    print('compute_asyn_wmmse===H6:{}, size:{}, H7:{}, size:{}'.format(H6, H6.shape, H7, H7.shape))
    print('compute_asyn_wmmse===initial_delay:{}, size:{}, add_delta_delay:{}, size:{}, p3:{}, size:{}'.format(
        initial_delay, initial_delay.shape, add_delta_delay, add_delta_delay.shape, p3, p3.shape))
    # rx_all_power = torch.mul(H4, p3)
    # p_update = p3
    # print('sr_loss===rx_all_power:{}, size:{}'.format(rx_all_power, rx_all_power.shape))
    # rx_all_power = torch.sum(rx_all_power, axis=-1)
    # H = torch.clone(H3)
    another_H = torch.zeros(H6.shape[0], users, train_S*H6.shape[-1], dtype=torch.complex64).cuda() ## hk
    print('compute_asyn_wmmse===another_H:{}, size:{}'.format(another_H, another_H.shape))
    rate = np.zeros(H6.shape[0])
    for iter in range(H6.shape[0]):
        a_k = torch.zeros(users, dtype=torch.complex64).cuda()
        u_k = torch.zeros(users).cuda()
        v_k = torch.zeros(users).cuda()
        H_k = torch.zeros(users, train_S*Nt, train_S*Nt, dtype=torch.complex64).cuda()
        for user_index in range(users):
            for satellite_index in range(train_S):
                start_index = satellite_index*H6.shape[-1]
                end_index = (satellite_index+1)*H6.shape[-1]
                another_H[iter, user_index, start_index:end_index] = H6[iter, satellite_index*users+user_index, satellite_index*users+user_index, :]
                for another_satellite_index in range(train_S):
                    for another_user in range(users):
                        if eta_enable == 1 and eta_add == 0:
                            Delta_tau_i = initial_delay[iter, satellite_index*users+another_user, satellite_index*users+user_index, 0].view(1)
                            Delta_tau_j = initial_delay[iter, another_satellite_index*users+another_user, another_satellite_index*users+user_index, 0].view(1)
                            eta_value = calculate_eta(Delta_tau_i, Delta_tau_j)
                        elif eta_enable == 1 and eta_add == 1:
                            Delta_tau_i = initial_delay[
                                iter, satellite_index * users + another_user, satellite_index * users + user_index, 0].view(
                                1) + (add_delta_delay[iter, satellite_index * users + user_index, 0].view(1)
                                      - add_delta_delay[iter, satellite_index * users + another_user, 0].view(1))
                            Delta_tau_j = initial_delay[
                                iter, another_satellite_index * users + another_user, another_satellite_index * users + user_index, 0].view(
                                1) + (add_delta_delay[iter, another_satellite_index * users + user_index, 0].view(1)
                                      - add_delta_delay[iter, another_satellite_index * users + another_user, 0].view(1))
                            eta_value = calculate_eta(Delta_tau_i, Delta_tau_j)
                        elif eta_enable == 0:
                            eta_value = 1
                        X_another_user = torch.outer(H6[iter, satellite_index*users+another_user, satellite_index*users+another_user,:],
                                                     H6[iter, another_satellite_index*users+another_user, another_satellite_index * users + another_user, :].conj())
                        H_k[user_index, satellite_index*Nt:satellite_index*Nt+Nt, another_satellite_index*Nt:another_satellite_index*Nt+Nt] += eta_value * X_another_user
        print('compute_asyn_wmmse===iter: {}, another_H:{}, size:{}, H_k:{}, size:{}'.format(
            iter, another_H, another_H.shape, H_k, H_k.shape))
        # = torch.zeros(H.shape[0], users, train_S*H.shape[-1])
        # W_k = 1 / np.sqrt(Nt*train_S) * np.ones((users, train_S*H6.shape[-1]), dtype=complex)
        W_k = torch.complex(torch.randn(users, train_S*Nt), torch.randn(users, train_S*Nt)).cuda()
        noise_powers = np.zeros(users)
        for i in range(users):
            submatrix = W_k[i,:]
            mode_square_sum = torch.sum(torch.abs(submatrix) ** 2)
            W_k[i, :] = submatrix / torch.sqrt(mode_square_sum)
            signal = H6[iter, i, i, :].abs() ** 2
            signal_power = signal.mean()
            noise_powers[i] = signal_power / (10 ** (SNR_dB / 10))
        print('compute_asyn_wmmse===iter: {}, W_k:{}, size:{}, noise_powers:{}'.format(
            iter, W_k, W_k.shape, noise_powers))
        # for i in range(train_S):
        #     submatrix = W_k[:,i*Nt:(i+1)*Nt]
        #     mode_square_sum = torch.sum(torch.abs(submatrix)**2)
        #     W_k[:,i*Nt:(i+1)*Nt] = submatrix / torch.sqrt(mode_square_sum)
        ## 开始迭代计算
        t = 0
        t_max = 50
        sinrs = torch.empty(0).cuda()
        p_update = get_p_from_W(W_k, p3[iter, :, :, :], users, Nt)
        rx_all_power = torch.mul(H6[iter, :, :, :], p_update)
        rx_all_power = torch.sum(rx_all_power, axis=-1)
        sinr_iters = np.zeros((users, t_max))
        print('compute_asyn_wmmse===iter:{} init p_update: {}, size:{}, rx_all_power:{}, size:{}'.format(
            iter, p_update, p_update.shape, rx_all_power, rx_all_power.shape))
        while t < t_max:
            ## 轮询用户
            for u_index in range(users):
                valid_signal, inference, noise_power, sinr_k = calculate_SINR(
                    iter, rx_all_power, initial_delay, add_delta_delay, u_index, noise_powers[u_index], users, eta_enable, eta_add)
                sinr_iters[u_index, t] = sinr_k.item()
                ## 计算a_k
                a_k[u_index] = torch.sum(torch.mul(W_k[u_index,:].conj(), another_H[iter, u_index,:])) * (
                        1/(valid_signal+inference+noise_power))
                u_k[u_index] = 1 + sinr_k
                I_k = torch.eye(H_k.shape[1]).cuda()
                v_k[u_index] = u_k[u_index] * (torch.abs(a_k[u_index]) ** 2) / 1
                print('compute_asyn_wmmse===iter:{}-t:{} sinr_iters: {}, a_k:{}, u_k:{}, v_k:{}, I_k:{}'.format(
                    iter, t, sinr_iters, a_k, u_k, v_k, I_k))
                tmp = u_k[u_index] * (torch.abs(a_k[u_index])**2) * H_k[u_index,:,:] + v_k[u_index] * I_k
                print('compute_asyn_wmmse===iter:{}-t:{} tmp: {}'.format(
                    iter, t, tmp))
                W_k[u_index, :] = (torch.linalg.inv(tmp) @ (another_H[iter, u_index,:]) * a_k[u_index].conj() * u_k[u_index])
                print('compute_asyn_wmmse===iter:{}-t:{} W_k: {}, p_update:{}'.format(
                    iter, t, W_k, p_update))
                submatrix = W_k[u_index, :]
                mode_square_sum = torch.sum(torch.abs(submatrix) ** 2)
                W_k[u_index, :] = submatrix / torch.sqrt(mode_square_sum)
            p_update = get_p_from_W(W_k, p3[iter, :, :, :], users, Nt)
            rx_all_power = torch.mul(H6[iter, :, :, :], p_update)
            rx_all_power = torch.sum(rx_all_power, axis=-1)
            t += 1
                # sinr_k_next = calculate_SINR(rx_all_power, initial_delay, add_delta_delay, iter, u_index)
                # sinrs = torch.cat(
                #     (sinrs, sinr_k_next.view(1)))
        # p_update = get_p_from_W(W_k, p3[iter, :, :, :])
        # p3[iter, :, :, :] = p_update
        for s in range(train_S):
            submatrix = W_k[:,s*Nt:(s+1)*Nt]
            mode_square_sum = torch.sum(torch.abs(submatrix)**2)
            W_k[:,s*Nt:(s+1)*Nt] = submatrix / torch.sqrt(mode_square_sum)
        p_update = get_p_from_W(W_k, p3[iter, :, :, :], users, Nt)
        rx_all_power = torch.mul(H7[iter, :, :, :], p_update)
        rx_all_power = torch.sum(rx_all_power, axis=-1)
        print('compute_asyn_wmmse===iter: {}, sinr_iters: {}'.format(iter, sinr_iters))
        rate[iter] = 0
        for j in range(users):
            _, _, _, SINR_k = calculate_SINR(
                iter, rx_all_power, initial_delay, add_delta_delay, j, noise_powers[j], users, eta_enable, eta_add)
            rate[iter] += np.log2(1 + SINR_k.item())
    avr_rate = np.mean(rate)
    if eta_enable == 1:
        print('compute_asyn_wmmse===async WMMSE rate: {}, rate iters:{}'.format(avr_rate, rate))
    else:
        print('compute_syn_wmmse===sync WMMSE rate: {}, rate iters:{}'.format(avr_rate, rate))
    return avr_rate

def get_p_from_W(W, p3, users, Nt):
    p_update = torch.zeros_like(p3).cuda()
    for i in range(p3.shape[0]):
        user_index = int(i % users)
        satellite_index = int(i / users)
        p_update[i, 0, :] = W[user_index, satellite_index*Nt:(satellite_index+1)*Nt]
    return p_update

def calculate_SINR(iter, rx_all_power, initial_delay, add_delta_delay, user_index, noise_power, users, eta_enable, eta_add):
    # H4 = H3.conj()
    # print('sr_loss===H4:{}, size:{}'.format(H4, H4.shape))
    # rx_all_power = torch.mul(H4, p3)
    # eta_enable = eta_enable
    # eta_add = 0
    rate = torch.zeros(1).cuda()
    # for user_index in range(users):
    valid_signal = torch.empty(0).cuda()
    interference_sum_power = torch.empty(0).cuda()
    for link in range(rx_all_power.shape[0]):
        # satellite_index = int(link / users)
        if int(link % users) == user_index:
            valid_signal = torch.cat(
                (valid_signal, rx_all_power[link, link].view(1)))
        # else:
        #     valid_link = satellite_index*users+user_index
        #     interference_signal = torch.cat(
        #         (interference_signal, rx_all_power[iter, link, user_index].view(1)))
        #     Delta_tau = torch.cat(
        #         (Delta_tau, initial_delay[iter, link, valid_link].view(1)))
    for other_user_index in range(users):
        interference_signal = torch.empty(0).cuda()
        Delta_tau = torch.empty(0).cuda()
        Add_delta_tau = torch.empty(0).cuda()
        if other_user_index != user_index:
            for other_link in range(rx_all_power.shape[0]):
                if int(other_link % users) == other_user_index:
                    satellite_index = int(other_link / users)
                    valid_link = satellite_index * users + user_index
                    interference_signal = torch.cat(
                        (interference_signal, rx_all_power[other_link, valid_link].view(1)))
                    Delta_tau = torch.cat(
                        (Delta_tau, initial_delay[iter, other_link, valid_link].view(1)))
                    Add_delta_tau = torch.cat((Add_delta_tau, add_delta_delay[iter, valid_link, 0].view(1)
                                                   - add_delta_delay[iter, other_link, 0].view(1)))
            interference_signal_conj = interference_signal.conj()
            print(
                'calculate_SINR===valid_signal: {} interference_signal:{} interference_signal_conj:{} Delta_tau:{}'.format(
                    valid_signal, interference_signal, interference_signal_conj, Delta_tau))
            if eta_enable == 0:
                # interference_power = torch.outer(interference_signal, interference_signal_conj)
                interference_power = torch.sum(torch.outer(interference_signal, interference_signal_conj))
                interference_sum_power = torch.cat((interference_sum_power, interference_power.view(1)))
            else:
                interference_power = torch.outer(interference_signal, interference_signal_conj)
                print('calculate_SINR===outer interference_power: {} size:{}'.format(interference_power,
                                                                              interference_power.shape))
                eta_value = torch.outer(Delta_tau, Delta_tau)
                print('calculate_SINR===init eta_value: {} sizez:{}'.format(eta_value, eta_value.shape))
                for i in range(Delta_tau.shape[0]):
                    for j in range(Delta_tau.shape[0]):
                        if eta_add == 0:
                            eta_value[i, j] = calculate_eta(Delta_tau[i], Delta_tau[j])
                        elif eta_add == 1:
                            # eta_value[i, j] = calculate_eta(Delta_tau[i] + Add_delta_tau[i],
                            #                                 Delta_tau[j] + Add_delta_tau[j])
                            eta_value[i, j] = calculate_eta(Add_delta_tau[i],
                                                            Add_delta_tau[j])
                print('calculate_SINR===config eta_value: {} sizez:{}'.format(eta_value, eta_value.shape))
                interference_power = torch.mul(interference_power, eta_value)
                print('calculate_SINR===mul interference_power: {} size:{}'.format(interference_power,
                                                                            interference_power.shape))
                interference_power = torch.sum(interference_power)
                print(
                    'calculate_SINR===interference_power:{} size:{}'.format(interference_power, interference_power.shape))
                interference_sum_power = torch.cat((interference_sum_power, interference_power.view(1)))
                # interference_signal_outer_async = torch.mul(interference_signal_outer, eta_value)
                print('calculate_SINR===interference_power:{} interference_sum_power: {} size:{}'.format(
                    interference_power, interference_sum_power, interference_sum_power.shape))
                # interference_power = torch.sum(interference_signal_outer_async)
    interference_sum_power = torch.sum(interference_sum_power)
    # print('sr_loss===Delta_tau: {} sizez:{}'.format(Delta_tau, Delta_tau.shape))
    valid_signal_conj = valid_signal.conj()
    valid_power = torch.abs(torch.sum(torch.outer(valid_signal, valid_signal_conj)))
    print('calculate_SINR===valid_power: {} interference_power:{}'.format(valid_power, interference_sum_power))
    sinr = torch.div(torch.abs(valid_power), torch.abs(interference_sum_power)+noise_power)
    # rate += torch.log2(1 + torch.div(torch.abs(valid_power), torch.abs(interference_sum_power)))
    print('calculate_SINR===user: {} sinr:{}'.format(user_index, sinr))
    return valid_power, interference_sum_power, noise_power, sinr

def calculate_eta(Delta_1, Delta_2):
    beta = 0.2
    if Delta_1 == Delta_2:
        return 1-0.5*beta*torch.sin(torch.pi*Delta_1)*torch.sin(torch.pi*Delta_2)
    else:
        return torch.div(torch.cos(torch.pi*(Delta_2-Delta_1)*beta),
                         1-4*beta*beta*(Delta_2-Delta_1)*(Delta_2-Delta_1))*torch.sinc(Delta_2-Delta_1) + torch.div(
            beta*torch.sin(torch.pi*Delta_1)*torch.sin(torch.pi*Delta_2),
            2*(beta*beta*(Delta_2-Delta_1)*(Delta_2-Delta_1)-1))*torch.sinc((Delta_2-Delta_1)*beta)


import networkx as nx
import numpy as np

class init_parameters():
    N_antennas: int

    def __init__(self, train_S, users, Nt):
        # wireless network settings
        train_K = train_S * users
        self.n_links = train_K
        # self.beams = beams
        self.n_receiver = train_K
        self.satellite_num = train_S
        self.user_num = users
        self.field_length = 1000
        self.shortest_directLink_length = 2
        self.longest_directLink_length = 65
        self.shortest_crossLink_length = 1
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = 1  # np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel - 30) / 10)) * self.bandwidth
        self.output_noise_power = 1  # self.input_noise_power
        self.SNR_gap_dB = 6
        self.SNR_gap = 1  # np.power(10, self.SNR_gap_dB/10)
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.n_links, self.field_length, self.field_length,
                                                                self.shortest_directLink_length,
                                                                self.longest_directLink_length)
        # 2D occupancy grid setting
        self.cell_length = 5
        self.N_antennas = Nt
        self.maxrx = 2
        self.minrx = 1
        self.n_grids = np.round(self.field_length / self.cell_length).astype(int)

def generate_synthetic_task(train_S, users, Nt, SNR_dB):
    # train_S = 2  ## 卫星数量
    beams = 2  ## 每个卫星配置的波束数量
    # users = 2  ## 用户数量
    # train_K = train_S * beams * users  ## 可能的收发匹配数量
    train_K = train_S * users  ## 可能的收发匹配数量
    test_K = train_K
    train_layouts = 10  ## 训练次数
    test_layouts = 5  ## 测试次数
    # SNR_dB = 50
    # Nt = 32  ## 每个波束的发射天线数
    # imperfect_channel_condition = 1
    train_config = init_parameters(train_S, users, Nt)
    var = 1
    train_dists, train_csis, train_delays = wg.sample_generate_all(train_config, train_layouts)
    test_config = init_parameters(train_S, users, Nt)
    test_dists, test_csis, test_delays = wg.sample_generate_all(test_config, test_layouts)

    train_csi_real, train_csi_imag = np.real(train_csis), np.imag(train_csis)
    test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
    norm_train_real, norm_test_real = normalize_data(train_csi_real, test_csi_real, train_config, train_K, train_layouts, test_K)
    norm_train_imag, norm_test_imag = normalize_data(train_csi_imag, test_csi_imag, train_config, train_K, train_layouts, test_K)
    return (SNR_dB, train_S, users, train_K, Nt, train_layouts, test_layouts, train_dists, train_csis, train_delays, norm_train_real, norm_train_imag,
            test_dists, test_csis, test_delays, norm_test_real, norm_test_imag)
    # return (torch.tensor(x_train), torch.tensor(adj_train), torch.tensor(y_train),
    #         torch.tensor(x_val), torch.tensor(adj_val), torch.tensor(y_val))

def normalize_data(train_data, test_data, general_para, train_K, train_layouts, test_K):
    Nt = general_para.N_antennas
    print('train_data:{}'.format(train_data))  # 200*3*3*2
    tmp_mask = np.expand_dims(np.eye(train_K), axis=-1)  # 3*3*3
    print('tmp_mask1:{}'.format(tmp_mask))
    tmp_mask = [tmp_mask for i in range(Nt)]
    print('tmp_mask2:{}'.format(tmp_mask))  # 2*3*3*3
    mask = np.concatenate(tmp_mask, axis=-1)
    print('mask1:{}'.format(mask))  # 3*3*3*2
    mask = np.expand_dims(mask, axis=0)  # 1*3*3*2
    print('mask2:{}'.format(mask))

    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask, train_copy)
    diag_mean = np.sum(diag_H / Nt) / train_layouts / train_K
    diag_var = np.sqrt(np.sum(np.square(diag_H)) / train_layouts / train_K / Nt)
    tmp_diag = (diag_H - diag_mean) / diag_var
    print('diag_H:{}'.format(diag_H))
    print('diag_mean:{}'.format(diag_mean))
    print('diag_var:{}'.format(diag_var))
    print('tmp_diag:{}'.format(tmp_diag))

    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag / Nt) / train_layouts / train_K / (train_K - 1)
    off_diag_var = np.sqrt(np.sum(np.square(off_diag)) / Nt / train_layouts / train_K / (train_K - 1))
    tmp_off = (off_diag - off_diag_mean) / off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off, mask)

    norm_train = np.multiply(tmp_diag, mask) + tmp_off_diag

    # normlize test
    tmp_mask = np.expand_dims(np.eye(test_K), axis=-1)
    tmp_mask = [tmp_mask for i in range(Nt)]
    mask = np.concatenate(tmp_mask, axis=-1)
    mask = np.expand_dims(mask, axis=0)

    test_copy = np.copy(test_data)
    diag_H = np.multiply(mask, test_copy)
    tmp_diag = (diag_H - diag_mean) / diag_var

    off_diag = test_copy - diag_H
    tmp_off = (off_diag - off_diag_mean) / off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off, mask)

    norm_test = np.multiply(tmp_diag, mask) + tmp_off_diag
    print(diag_mean, diag_var, off_diag_mean, off_diag_var)
    return norm_train, norm_test

def run_experiment():
    # input_dim = 10  # 输入特征维度
    # hidden_dim = 32  # 隐层大小
    # output_dim = 3  # 用户数目
    num_layers = 5  # 最大层数
    # num_tasks = 8  # 批量任务数量
    # epochs = 100  # 训练轮次
    Nt = 2
    # 初始化你的 GNN 模型
    # base_model = YourGNNModel(input_dim, hidden_dim, output_dim, num_layers)
    base_model = IGCNet(num_layers, 0, Nt)
    model = GumbelControlledGNN(base_model, num_layers=num_layers)
    trainer = MetaGNNTrainer(model)

    for epoch in range(100):
        task_batch = []
        task_batch.append(generate_synthetic_task(train_S=2, users=4, Nt=2, SNR_dB=50))
        task_batch.append(generate_synthetic_task(train_S=2, users=3, Nt=2, SNR_dB=50))
        # task_batch.append(generate_synthetic_task(train_S=3, users=3, Nt=2, SNR_dB=50))
        # task_batch.append(generate_synthetic_task(train_S=3, users=6, Nt=2, SNR_dB=50))
        # task_batch.append(generate_synthetic_task(train_S=6, users=6, Nt=2, SNR_dB=50))

        # task_batch = [generate_synthetic_task(num_nodes=random.randint(15, 30)train_S=random.randint(2, 5), users=random.randint(2, 30), Nt=2) for _ in range(num_tasks)]
        print('=====run_experiment=====task_batch:{}'.format(task_batch))
        loss = trainer.train_step(task_batch)

        if epoch % 1 == 0:
            gate_values = torch.sigmoid(model.layer_logits).detach().cpu().numpy()
            print(f"[MEpoch {epoch}] Meta Loss: {loss:.4f} | Layer Gates: {gate_values}")


if __name__ == "__main__":
    run_experiment()
