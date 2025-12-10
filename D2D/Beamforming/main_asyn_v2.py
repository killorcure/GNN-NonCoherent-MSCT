import math

import scipy.io as sio  # import scipy.io for .mat file I/O
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

#### 该版本和main_asyn.py的区别在与:没有伪随机delta_delay

class init_parameters():
    N_antennas: int

    def __init__(self):
        # wireless network settings
        self.n_links = train_K
        self.beams = beams
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


def normalize_data(train_data, test_data, general_para):
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


def batch_wmmse(csis, var_noise):
    print('batch_wmmse====csis:{}, size:{}'.format(csis, csis.shape))
    Nt = test_config.N_antennas
    K = test_config.n_receiver
    n = csis.shape[0]
    Y = np.zeros((n, K, Nt), dtype=complex)
    print('batch_wmmse====Y:{}, size:{}'.format(Y, Y.shape))
    Pini = 1 / np.sqrt(Nt) * np.ones((K, Nt), dtype=complex)
    print('batch_wmmse====Pini:{}, size:{}'.format(Pini, Pini.shape))
    for ii in range(n):
        Y[ii, :, :] = wmmse.np_WMMSE_vector(np.copy(Pini), csis[ii, :, :, :], 1, var_noise)
    print('batch_wmmse====Y:{}, size:{}'.format(Y, Y.shape))
    return Y


def build_graph(CSI, dist, norm_csi_real, norm_csi_imag, K, threshold):
    print('CSI:{}'.format(CSI))
    print('dist:{}'.format(dist))
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
    init_links = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    s = len(init_links)
    for i in range(s):
        for j in range(s):
            ## 波束干扰约束
            if i != j and init_links[i] == 1 and init_links[j] == 1:
                dist2[i][j] = 10
                dist2[j][i] = 10
            # if (i / users) == (j / users):
            #     dist2[i][j] = 0
            # elif (i / (users*beams)) == (j / (users*beams)) and (i % users) == (j % users):
            #     dist2[i][j] = 0
            # ## 卫星间波束干扰约束
            # elif (i / (users*beams)) == (j / (users*beams)) and (i % users) == (j % users):
            #     dist2[i][j] = 0

    mask = np.eye(K)
    diag_dist = np.multiply(mask, dist2)
    print('dist2:{}'.format(dist2))
    print('mask:{}'.format(mask))
    print('diag_dist1:{}'.format(diag_dist))
    dist2 = dist2 + 1000 * diag_dist
    print('dist2_1:{}'.format(dist2))
    # dist2[dist2 > threshold] = 0
    dist2[dist2 > 1000000] = 0  # 这里每个连接对之间都有干扰
    attr_ind = np.nonzero(dist2)
    print('attr_ind:{}'.format(attr_ind))
    edge_attr_real = norm_csi_real[attr_ind]
    edge_attr_imag = norm_csi_imag[attr_ind]

    edge_attr = np.concatenate((edge_attr_real, edge_attr_imag), axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    edge_index = torch.tensor(adj, dtype=torch.long)

    H1 = np.expand_dims(np.real(CSI), axis=-1)
    H2 = np.expand_dims(np.imag(CSI), axis=-1)
    HH = np.concatenate((H1, H2), axis=-1)
    y = torch.tensor(np.expand_dims(HH, axis=0), dtype=torch.float)
    print('H1:{}'.format(H1))
    print('H2:{}'.format(H2))
    print('HH:{}'.format(HH))
    print('y:{}'.format(np.expand_dims(HH, axis=0)))
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y)
    return data


def build_graph_MSCT(CSI, dist, norm_csi_real, norm_csi_imag, K, threshold):
    print('CSI:{}'.format(CSI))
    print('dist:{}'.format(dist))
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
            if (i / (users * beams)) == (j / (users * beams)) and (i % users) == (j % users):
                dist2[i][j] = 0
            ## 卫星间波束干扰约束
            elif (i / (users * beams)) == (j / (users * beams)) and (i % users) == (j % users):
                dist2[i][j] = 0
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

    edge_attr = np.concatenate((edge_attr_real, edge_attr_imag), axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    edge_index = torch.tensor(adj, dtype=torch.long)

    H1 = np.expand_dims(np.real(CSI), axis=-1)
    H2 = np.expand_dims(np.imag(CSI), axis=-1)
    HH = np.concatenate((H1, H2), axis=-1)
    y = torch.tensor(np.expand_dims(HH, axis=0), dtype=torch.float)
    print('H1:{}'.format(H1))
    print('H2:{}'.format(H2))
    print('HH:{}'.format(HH))
    print('y:{}'.format(np.expand_dims(HH, axis=0)))
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y,
                norm_csi_real=norm_csi_real, norm_csi_imag=norm_csi_imag, dist=dist2,
                CSI=torch.tensor(CSI, dtype=torch.float))
    return data

def build_graph_MSCT_asyn(CSI, dist, delays, norm_csi_real, norm_csi_imag, K, threshold):
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

# def update_edge_attr(old_data, links):
#     old_edge_attr = old_data.edge_attr
#     old_edge_index = old_data.edge_index
#     norm_csi_imag = old_data.norm_csi_imag
#     norm_csi_real = old_data.norm_csi_real
#     new_edge_attr = 0
#     new_edge_index = 0
#     new_links = torch.tensor()
#     i=0
#     before = 0
#     threshold = 0.5
#     satellite_beam_action = []
#     while before < len(links):
#     # for i in range(links.shape[0]):
#         choose_users_index = set()
#         for i in range(beams*train_S):
#             tmp_link = links[before:before+users]
#             tmp_link = np.abs(tmp_link)/sum(np.abs(tmp_link))
#             tmp_link[tmp_link < threshold] = 0
#             max_index = np.argmax(tmp_link)
#             while max_index in choose_users_index:
#                 tmp_link[max_index] = -1
#                 max_index = np.argmax(tmp_link)
#             tmp_link1 = np.zeros_like(tmp_link)
#             if max_index != -1 and tmp_link[max_index] > 0:
#                 choose_users_index.add(max_index)
#                 tmp_link1[max_index] = 1
#             satellite_beam_action += tmp_link1
#             before = before + users
#         # before += i*beams*users
#     return new_data

# def update_links(oldlinks):
#     print('start to update links...')
#     print('updatelinks===oldlinks:{}'.format(oldlinks))
#     before = 0
#     # threshold = 0.5
#     links1 = oldlinks.clone()
#     links = links1.numpy()
#     satellite_beam_action = torch.zeros(links.size(), device=device)
#     update_links = torch.zeros(links.size(), device=device)
#     print('updatelinks===links:{}'.format(links))
#     while before < len(links):
#     # for i in range(links.shape[0]):
#         choose_users_index = set()
#         for i in range(beams):
#             tmp_link = links[before:before+users]
#             tmp_link = np.abs(tmp_link)/sum(np.abs(tmp_link))
#             update_links[before:before+users] = links[before:before+users]
#             # tmp_link[tmp_link < threshold] = 0
#             max_index = np.argmax(tmp_link)
#             while max_index in choose_users_index:
#                 tmp_link[max_index] = -1
#                 max_index = np.argmax(tmp_link)
#             tmp_link1 = np.zeros_like(tmp_link)
#             if max_index != -1 and tmp_link[max_index] > 0:
#                 choose_users_index.add(max_index)
#                 tmp_link1[max_index] = 1
#             oldlinks[before:before+users] = torch.from_numpy(tmp_link1)
#             # satellite_beam_action += tmp_link1
#             before = before + users
#     print('updatelinks===update_links:{}'.format(update_links))
#     print('updatelinks===update oldlinks:{}'.format(oldlinks))
#     print('end to update links...')
#     return oldlinks

def update_links(oldlinks):
    print('start to update links...')
    print('updatelinks===oldlinks:{}'.format(oldlinks))
    before = 0
    # threshold = 0.5
    links = oldlinks.clone()
    # links = links1.numpy()
    satellite_beam_action = torch.zeros(links.size(), device=device)
    # update_links = torch.zeros(links.size(), device=device)
    temperature = 0.5
    threshold = 0
    # print('updatelinks===links:{}'.format(links))
    while before < len(links):
        # for i in range(links.shape[0]):
        choose_users_index = []
        for i in range(beams):
            tmp_link = links[before:before + users]
            print('choose===tmp_link:{}'.format(tmp_link))
            # print('updatelinks===tmp_link:{}'.format(tmp_link))
            exp_tmp_link = torch.exp(torch.abs(tmp_link) / temperature - torch.max(torch.abs(tmp_link)))
            print('choose===exp_tmp_link:{}'.format(exp_tmp_link))
            # exp_tmp_link = np.exp(tmp_link / temperature - np.max(tmp_link))
            tmp_link = exp_tmp_link / torch.sum(exp_tmp_link)
            print('choose===tmp_link1:{}'.format(tmp_link))
            # tmp_link = torch.div(torch.abs(tmp_link), torch.sum(torch.abs(tmp_link)))
            # print('updatelinks===tmp_link2:{}'.format(tmp_link))
            # update_links[before:before+users] = links[before:before+users]
            # tmp_link[tmp_link < threshold] = 0
            max_index = torch.argmax(tmp_link)
            # print('updatelinks===max_index:{}'.format(max_index))
            # print('updatelinks===choose_users_index:{}'.format(choose_users_index))
            while max_index in choose_users_index:
                # print('updatelinks===max_index in choose_users_index:{}'.format('True'))
                tmp_link[max_index] = -1
                max_index = torch.argmax(tmp_link)
            tmp_link1 = np.zeros_like(tmp_link.tolist())
            # print('updatelinks===tmp_link3:{}'.format(tmp_link1))
            if tmp_link[max_index] > threshold:
                choose_users_index.append(max_index)
                # print('updatelinks===after append {}, choose_users_index :{}'.format(max_index, choose_users_index))
                tmp_link1[max_index] = 1
            oldlinks[before:before + users] = torch.from_numpy(tmp_link1)
            # print('updatelinks===oldlinks:{}'.format(oldlinks))
            # satellite_beam_action += tmp_link1
            before = before + users
    # print('updatelinks===update_links:{}'.format(update_links))
    print('updatelinks===update final links:{}'.format(oldlinks))
    print('end to update links...')
    return oldlinks


def proc_data(HH, dists, norm_csi_real, norm_csi_imag, K):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
        # data = build_graph(HH[i,:,:,:],dists[i,:,:], norm_csi_real[i,:,:,:], norm_csi_imag[i,:,:,:], K,500)
        data = build_graph_MSCT(HH[i, :, :, :], dists[i, :, :], norm_csi_real[i, :, :, :], norm_csi_imag[i, :, :, :], K,
                                500)
        data_list.append(data)
    return data_list

def proc_data_asyn(HH, dists, delays, norm_csi_real, norm_csi_imag, K):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
        # data = build_graph(HH[i,:,:,:],dists[i,:,:], norm_csi_real[i,:,:,:], norm_csi_imag[i,:,:,:], K,500)
        data = build_graph_MSCT_asyn(HH[i, :, :, :], dists[i, :, :], delays[i, :, :], norm_csi_real[i, :, :, :],
                                     norm_csi_imag[i, :, :, :], K, 500)
        data_list.append(data)
    return data_list

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())  # , BN(channels[i])
        for i in range(1, len(channels))
    ])


class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        # self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)

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
        comb = comb_all[:, 1:2 * Nt + 1]
        links_res = comb_all[:, 0:1]
        # add_delta_delay = comb_all[:, 1:2]
        # add_delta_delay = torch.sigmoid(add_delta_delay/0.5)
        # links_onehot = update_links(links_res)
        links_onehot = links_res
        print('update===tmp:{}'.format(tmp))
        print('update===comb:{}'.format(comb))
        # print('update===add_delta_delay:{}'.format(add_delta_delay))
        # shape_length = comb.shape[0]
        comb_group = comb.view(-1,users,2*Nt)
        square_sum = (comb_group ** 2).sum(dim=(1,2), keepdim=True)
        square_sum = square_sum + 1e-6
        comb_normalized = comb_group / torch.sqrt(square_sum)
        comb = comb_normalized.view(comb.shape[0],2*Nt)
        print('update===normalized comb:{}'.format(comb))
        # nor = torch.sqrt(torch.sum(torch.mul(comb, comb), axis=1))
        # print('update===nor:{}'.format(nor))
        # nor = nor.unsqueeze(axis=-1)
        # print('update===unsqueeze nor:{}'.format(nor))
        # comp1 = torch.ones(comb.size(), device=device)
        # print('update===comp1:{}'.format(comp1))
        # comb = torch.div(comb, torch.max(comp1, nor))
        # print('update===comb:{}'.format(comb))
        comb_res = torch.cat([links_onehot, comb], dim=1)
        print('update===comb_res:{}'.format(comb_res))
        return torch.cat([comb_res, x[:, :2 * Nt - 1]], dim=1)

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


# class IGCNet(torch.nn.Module):
#     def __init__(self):
#         super(IGCNet, self).__init__()
#
#         self.mlp1 = MLP([6*Nt, 64, 64])
#         self.mlp2 = MLP([64+4*Nt, 32])
#         self.mlp2 = Seq(*[self.mlp2,Seq(Lin(32, 1+2*Nt))])
#         self.conv = IGConv(self.mlp1,self.mlp2)
#
#     def forward(self, data):
#         x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
#         x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
#         new_data1 = update_edge_attr(data, x1[:,0:1])
#         x2 = self.conv(x = x1, edge_index = new_data1.edge_index, edge_attr = new_data1.edge_attr)
#         new_data2 = update_edge_attr(new_data1, x2[:,0:1])
#         out = self.conv(x = x2, edge_index = new_data2.edge_index, edge_attr = new_data2.edge_attr)
#         return out

class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()

        self.mlp1 = MLP([6 * Nt+1, 64, 64])
        self.mlp2 = MLP([64 + 4 * Nt, 32])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(32, 1 + 2 * Nt))])
        self.conv = IGConv(self.mlp1, self.mlp2)
        self.network_layer = 3

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        for i in range(self.network_layer):
            out = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
            x0 = out
        # x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        # # new_data1 = update_edge_attr(data, x1[:,0:1])
        # x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        # # new_data2 = update_edge_attr(new_data1, x2[:,0:1])
        # out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        return x0


def power_check(p):
    n = p.shape[0]
    pp = np.sum(np.square(p), axis=1)
    print(np.sum(pp > 1.1))



def compute_asyn_wmmse(H3, H_new, p3, initial_delay, eta_enable):
    H6 = torch.clone(H3)
    H7 = torch.clone(H_new)
    print('compute_asyn_wmmse===H6:{}, size:{}, H7:{}, size:{}'.format(H6, H6.shape, H7, H7.shape))
    print('compute_asyn_wmmse===initial_delay:{}, size:{}, p3:{}, size:{}'.format(
        initial_delay, initial_delay.shape, p3, p3.shape))
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
                        if eta_enable == 1:
                            Delta_tau_i = initial_delay[iter, satellite_index*users+another_user, satellite_index*users+user_index, 0].view(1)
                            Delta_tau_j = initial_delay[iter, another_satellite_index*users+another_user, another_satellite_index*users+user_index, 0].view(1)
                            eta_value = calculate_eta(Delta_tau_i, Delta_tau_j)
                        # elif eta_enable == 1 and eta_add == 1:
                        #     Delta_tau_i = initial_delay[
                        #         iter, satellite_index * users + another_user, satellite_index * users + user_index, 0].view(
                        #         1) + (add_delta_delay[iter, satellite_index * users + user_index, 0].view(1)
                        #               - add_delta_delay[iter, satellite_index * users + another_user, 0].view(1))
                        #     Delta_tau_j = initial_delay[
                        #         iter, another_satellite_index * users + another_user, another_satellite_index * users + user_index, 0].view(
                        #         1) + (add_delta_delay[iter, another_satellite_index * users + user_index, 0].view(1)
                        #               - add_delta_delay[iter, another_satellite_index * users + another_user, 0].view(1))
                        #     eta_value = calculate_eta(Delta_tau_i, Delta_tau_j)
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
        p_update = get_p_from_W(W_k, p3[iter, :, :, :])
        rx_all_power = torch.mul(H6[iter, :, :, :], p_update)
        rx_all_power = torch.sum(rx_all_power, axis=-1)
        sinr_iters = np.zeros((users, t_max))
        print('compute_asyn_wmmse===iter:{} init p_update: {}, size:{}, rx_all_power:{}, size:{}'.format(
            iter, p_update, p_update.shape, rx_all_power, rx_all_power.shape))
        while t < t_max:
            ## 轮询用户
            for u_index in range(users):
                valid_signal, inference, noise_power, sinr_k = calculate_SINR(
                    iter, rx_all_power, initial_delay, u_index, noise_powers[u_index], eta_enable)
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
            p_update = get_p_from_W(W_k, p3[iter, :, :, :])
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
        p_update = get_p_from_W(W_k, p3[iter, :, :, :])
        rx_all_power = torch.mul(H7[iter, :, :, :], p_update)
        rx_all_power = torch.sum(rx_all_power, axis=-1)
        print('compute_asyn_wmmse===iter: {}, sinr_iters: {}'.format(iter, sinr_iters))
        rate[iter] = 0
        for j in range(users):
            _, _, _, SINR_k = calculate_SINR(
                iter, rx_all_power, initial_delay, j, noise_powers[j], eta_enable)
            rate[iter] += np.log2(1 + SINR_k.item())
    avr_rate = np.mean(rate)
    if eta_enable == 1:
        print('compute_asyn_wmmse===async WMMSE rate: {}, rate iters:{}'.format(avr_rate, rate))
    else:
        print('compute_syn_wmmse===sync WMMSE rate: {}, rate iters:{}'.format(avr_rate, rate))
    return avr_rate

# def compute_syn_wmmse(H3, p3, initial_delay, add_delta_delay):
#     H8 = torch.clone(H3)
#     H9 = H3.conj()
#     print('compute_asyn_wmmse===H8:{}, size:{}, H9:{}, size:{}'.format(H8, H8.shape, H9, H9.shape))
#     print('compute_asyn_wmmse===initial_delay:{}, size:{}, add_delta_delay:{}, size:{}, p3:{}, size:{}'.format(
#         initial_delay, initial_delay.shape, add_delta_delay, add_delta_delay.shape, p3, p3.shape))
#     # rx_all_power = torch.mul(H4, p3)
#     # p_update = p3
#     # print('sr_loss===rx_all_power:{}, size:{}'.format(rx_all_power, rx_all_power.shape))
#     # rx_all_power = torch.sum(rx_all_power, axis=-1)
#     # H = torch.clone(H3)
#     another_H = torch.zeros(H8.shape[0], users, train_S*H8.shape[-1], dtype=torch.complex64).cuda() ## hk
#     print('compute_asyn_wmmse===another_H:{}, size:{}'.format(another_H, another_H.shape))
#     rate = np.zeros(H8.shape[0])
#     for iter in range(H8.shape[0]):
#         a_k = torch.zeros(users, dtype=torch.complex64).cuda()
#         u_k = torch.zeros(users).cuda()
#         v_k = torch.zeros(users).cuda()
#         H_k = torch.zeros(users, train_S*Nt, train_S*Nt, dtype=torch.complex64).cuda()
#         for user_index in range(users):
#             for satellite_index in range(train_S):
#                 start_index = satellite_index*H8.shape[-1]
#                 end_index = (satellite_index+1)*H8.shape[-1]
#                 another_H[iter, user_index, start_index:end_index] = H8[iter, satellite_index*users+user_index, satellite_index*users+user_index, :]
#                 for another_satellite_index in range(train_S):
#                     for another_user in range(users):
#                         # Delta_tau_i = initial_delay[iter, satellite_index*users+another_user, satellite_index*users+user_index, 0].view(1)
#                         # Delta_tau_j = initial_delay[iter, another_satellite_index*users+another_user, another_satellite_index*users+user_index, 0].view(1)
#                         eta_value = 1
#                         X_another_user = torch.outer(H8[iter, satellite_index*users+another_user, satellite_index*users+another_user,:],
#                                                      H8[iter, another_satellite_index*users+another_user, another_satellite_index * users + another_user, :].conj())
#                         H_k[user_index, satellite_index*Nt:satellite_index*Nt+Nt, another_satellite_index*Nt:another_satellite_index*Nt+Nt] += eta_value * X_another_user
#         print('compute_asyn_wmmse===iter: {}, another_H:{}, size:{}, H_k:{}, size:{}'.format(
#             iter, another_H, another_H.shape, H_k, H_k.shape))
#         # = torch.zeros(H.shape[0], users, train_S*H.shape[-1])
#         # W_k = 1 / np.sqrt(Nt*train_S) * np.ones((users, train_S*H6.shape[-1]), dtype=complex)
#         W_k = torch.complex(torch.randn(users, train_S*Nt), torch.randn(users, train_S*Nt)).cuda()
#         noise_powers = np.zeros(users)
#         for i in range(users):
#             submatrix = W_k[i,:]
#             mode_square_sum = torch.sum(torch.abs(submatrix) ** 2)
#             W_k[i, :] = submatrix / torch.sqrt(mode_square_sum)
#             signal = H8[iter, i, i, :].abs() ** 2
#             signal_power = signal.mean()
#             noise_powers[i] = signal_power / (10 ** (SNR_dB / 10))
#         print('compute_asyn_wmmse===iter: {}, W_k:{}, size:{}, noise_powers:{}'.format(
#             iter, W_k, W_k.shape, noise_powers))
#         # for i in range(train_S):
#         #     submatrix = W_k[:,i*Nt:(i+1)*Nt]
#         #     mode_square_sum = torch.sum(torch.abs(submatrix)**2)
#         #     W_k[:,i*Nt:(i+1)*Nt] = submatrix / torch.sqrt(mode_square_sum)
#         ## 开始迭代计算
#         t = 0
#         t_max = 50
#         sinrs = torch.empty(0).cuda()
#         p_update = get_p_from_W(W_k, p3[iter, :, :, :])
#         rx_all_power = torch.mul(H8[iter, :, :, :], p_update)
#         rx_all_power = torch.sum(rx_all_power, axis=-1)
#         sinr_iters = np.zeros((users, t_max))
#         print('compute_asyn_wmmse===iter:{} init p_update: {}, size:{}, rx_all_power:{}, size:{}'.format(
#             iter, p_update, p_update.shape, rx_all_power, rx_all_power.shape))
#         while t < t_max:
#             ## 轮询用户
#             for u_index in range(users):
#                 valid_signal, inference, noise_power, sinr_k = calculate_SINR(iter, rx_all_power, initial_delay, add_delta_delay, u_index, noise_powers[u_index])
#                 sinr_iters[u_index, t] = sinr_k.item()
#                 ## 计算a_k
#                 a_k[u_index] = torch.sum(torch.mul(W_k[u_index,:].conj(), another_H[iter, u_index,:])) * (
#                         1/(valid_signal+inference+noise_power))
#                 u_k[u_index] = 1 + sinr_k
#                 I_k = torch.eye(H_k.shape[1]).cuda()
#                 v_k[u_index] = u_k[u_index] * (torch.abs(a_k[u_index]) ** 2) / 1
#                 print('compute_asyn_wmmse===iter:{}-t:{} sinr_iters: {}, a_k:{}, u_k:{}, v_k:{}, I_k:{}'.format(
#                     iter, t, sinr_iters, a_k, u_k, v_k, I_k))
#                 tmp = u_k[u_index] * (torch.abs(a_k[u_index])**2) * H_k[u_index,:,:] + v_k[u_index] * I_k
#                 print('compute_asyn_wmmse===iter:{}-t:{} tmp: {}'.format(
#                     iter, t, tmp))
#                 W_k[u_index, :] = (torch.linalg.inv(tmp) @ (another_H[iter, u_index,:]) * a_k[u_index].conj() * u_k[u_index])
#                 print('compute_asyn_wmmse===iter:{}-t:{} W_k: {}, p_update:{}'.format(
#                     iter, t, W_k, p_update))
#                 submatrix = W_k[u_index, :]
#                 mode_square_sum = torch.sum(torch.abs(submatrix) ** 2)
#                 W_k[u_index, :] = submatrix / torch.sqrt(mode_square_sum)
#             p_update = get_p_from_W(W_k, p3[iter, :, :, :])
#             rx_all_power = torch.mul(H8[iter, :, :, :], p_update)
#             rx_all_power = torch.sum(rx_all_power, axis=-1)
#             t += 1
#                 # sinr_k_next = calculate_SINR(rx_all_power, initial_delay, add_delta_delay, iter, u_index)
#                 # sinrs = torch.cat(
#                 #     (sinrs, sinr_k_next.view(1)))
#         # p_update = get_p_from_W(W_k, p3[iter, :, :, :])
#         # p3[iter, :, :, :] = p_update
#         for s in range(train_S):
#             submatrix = W_k[:,s*Nt:(s+1)*Nt]
#             mode_square_sum = torch.sum(torch.abs(submatrix)**2)
#             W_k[:,s*Nt:(s+1)*Nt] = submatrix / torch.sqrt(mode_square_sum)
#         p_update = get_p_from_W(W_k, p3[iter, :, :, :])
#         rx_all_power = torch.mul(H8[iter, :, :, :], p_update)
#         rx_all_power = torch.sum(rx_all_power, axis=-1)
#         print('compute_asyn_wmmse===iter: {}, sinr_iters: {}'.format(iter, sinr_iters))
#         rate[iter] = 0
#         for j in range(users):
#             _, _, _, SINR_k = calculate_SINR(iter, rx_all_power, initial_delay, add_delta_delay, j, noise_powers[j])
#             rate[iter] += np.log2(1 + SINR_k.item())
#     avr_rate = np.mean(rate)
#     print('compute_asyn_wmmse===async WMMSE rate: {}, rate iters:{}'.format(avr_rate, rate))
#     return avr_rate

def get_p_from_W(W, p3):
    p_update = torch.zeros_like(p3).cuda()
    for i in range(p3.shape[0]):
        user_index = int(i % users)
        satellite_index = int(i / users)
        p_update[i, 0, :] = W[user_index, satellite_index*Nt:(satellite_index+1)*Nt]
    return p_update

def calculate_SINR(iter, rx_all_power, initial_delay, user_index, noise_power, eta_enable):
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
        # Add_delta_tau = torch.empty(0).cuda()
        if other_user_index != user_index:
            for other_link in range(rx_all_power.shape[0]):
                if int(other_link % users) == other_user_index:
                    satellite_index = int(other_link / users)
                    valid_link = satellite_index * users + user_index
                    interference_signal = torch.cat(
                        (interference_signal, rx_all_power[other_link, valid_link].view(1)))
                    Delta_tau = torch.cat(
                        (Delta_tau, initial_delay[iter, other_link, valid_link].view(1)))
                    # Add_delta_tau = torch.cat((Add_delta_tau, add_delta_delay[iter, valid_link, 0].view(1)
                    #                                - add_delta_delay[iter, other_link, 0].view(1)))
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
                        eta_value[i, j] = calculate_eta(Delta_tau[i], Delta_tau[j])
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


def sr_loss_all(data, p, K, N, epoch, imperfect_channel):
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
    # add_delta_delay = p[:, 1:2]
    # add_delta_delay = torch.reshape(add_delta_delay, (-1, K, 1, 1))
    # links = update_links(links)
    p1 = p[:, 1:N + 1]
    p2 = p[:, N + 1:2 * N + 1]
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
    # mod = torch.abs(H3)
    # phase = torch.angle(H3)
    # sigma = 0.5
    # phase_error = torch.normal(0, sigma, size=phase.shape)
    # new_phase = phase_error + phase
    # H_new = mod * torch.exp(1j*new_phase)
    print('sr_loss===H4:{}, size:{}'.format(H4, H4.shape))
    rx_all_power = torch.mul(H4, p3)
    print('sr_loss===rx_all_power:{}, size:{}'.format(rx_all_power, rx_all_power.shape))
    rx_all_power = torch.sum(rx_all_power, axis=-1)
    print('sr_loss===sum rx_all_power:{}, size:{}'.format(rx_all_power, rx_all_power.shape))
    initial_delay = data.initial_delay
    print('sr_loss===initial_delay:{}, size:{}'.format(initial_delay, initial_delay.shape))
    initial_delay = torch.reshape(initial_delay, (-1, K, K, 1))
    print('sr_loss===reshape initial_delay:{}, size:{}'.format(initial_delay, initial_delay.shape))
    # print('sr_loss===add_delta_delay: {} sizez:{}'.format(add_delta_delay, add_delta_delay.shape))
    asyn_wmmse_rate = 0
    syn_wmmse_rate = 0
    if epoch == 1:
        print('sr_loss===start to compute async_WMMSE and sync_WMMSE')
        asyn_wmmse_rate = compute_asyn_wmmse(H3, p3, initial_delay, 1)
        syn_wmmse_rate = compute_asyn_wmmse(H3, p3, initial_delay, 0)
        print('sr_loss===end to compute async_WMMSE and sync_WMMSE')
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
            signal = H3[iter, user_index, user_index,:].abs()**2
            signal_power = signal.mean()
            noise_power = signal_power / (10**(SNR_dB/10))
            print('sr_loss===H3[iter, user_index, user_index,:]: {} signal:{} signal_power: {} noise_power: {}'.format(
                H3[iter, user_index, user_index,:], signal, signal_power, noise_power))
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
                            eta_value_add[i, j] = calculate_eta(Delta_tau_asyn[i]+Add_delta_tau[i], Delta_tau_asyn[j]+Add_delta_tau[j])
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
    return loss, avr_rate_syn, avr_rate_asyn_no_add, avr_rate_asyn_add, syn_wmmse_rate, asyn_wmmse_rate

def sr_loss_all_test(data, p, K, N, epoch, imperfect_channel):
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
    # add_delta_delay = p[:, 1:2]
    # add_delta_delay = torch.sigmoid(add_delta_delay/0.5)
    # print('sr_loss===add_delta_delay sigmoid: {} sizez:{}'.format(add_delta_delay, add_delta_delay.shape))
    # add_delta_delay = torch.reshape(add_delta_delay, (-1, K, 1, 1))
    # links = update_links(links)
    p1 = p[:, 1:N + 1]
    p2 = p[:, N + 1:2 * N + 1]
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
    # print('sr_loss===add_delta_delay: {} sizez:{}'.format(add_delta_delay, add_delta_delay.shape))
    asyn_wmmse_rate = 0
    asyn_wmmse_rate_noadd = 0
    syn_wmmse_rate = 0
    if epoch == 1 or (epoch % 100 == 0):
        print('sr_loss===start to compute async_WMMSE and sync_WMMSE')
        # asyn_wmmse_rate = compute_asyn_wmmse(H3, H_new, p3, initial_delay, 1)
        asyn_wmmse_rate_noadd = compute_asyn_wmmse(H3, H_new, p3, initial_delay, 1)
        syn_wmmse_rate = compute_asyn_wmmse(H3, H_new, p3, initial_delay, 0)
        print('sr_loss===end to compute async_WMMSE and sync_WMMSE')
    # rx_all_power_1 = torch.abs(rx_all_power)**2
    rate_iter_syn = torch.empty(0).cuda()
    rate_iter_asyn_no_add = torch.empty(0).cuda()
    rate_iter_asyn_add = torch.empty(0).cuda()
    for iter in range(rx_all_power.shape[0]):
        rate_syn = torch.zeros(1).cuda()
        rate_asyn_no_add = torch.zeros(1).cuda()
        # rate_asyn_add = torch.zeros(1).cuda()
        for user_index in range(users):
            valid_signal = torch.empty(0).cuda()
            signal = H_new[iter, user_index, user_index,:].abs()**2
            signal_power = signal.mean()
            noise_power = signal_power / (10**(SNR_dB/10))
            print('sr_loss===H3[iter, user_index, user_index,:]: {} signal:{} signal_power: {} noise_power: {}'.format(
                H_new[iter, user_index, user_index,:], signal, signal_power, noise_power))
            interference_sum_power_syn = torch.empty(0).cuda()
            interference_sum_power_asyn_no_add = torch.empty(0).cuda()
            # interference_sum_power_asyn_add = torch.empty(0).cuda()
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
                # Add_delta_tau = torch.empty(0).cuda()
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
                            # if eta_add == 1:
                            #     Add_delta_tau = torch.cat((Add_delta_tau, add_delta_delay[iter, valid_link, 0].view(1)
                            #                                - add_delta_delay[iter, other_link, 0].view(1)))
                    interference_signal_conj = interference_signal.conj()
                    print('sr_loss===valid_signal: {} interference_signal:{} interference_signal_conj:{} Delta_tau:{}'.format(
                        valid_signal, interference_signal, interference_signal_conj, Delta_tau_asyn))
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
                    # eta_value_add = torch.outer(Delta_tau_asyn, Delta_tau_asyn)
                    print('sr_loss===init eta_value: {} sizez:{}'.format(eta_value_no_add, eta_value_no_add.shape))
                    for i in range(Delta_tau_asyn.shape[0]):
                        for j in range(Delta_tau_asyn.shape[0]):
                            # if eta_add == 0:
                            eta_value_no_add[i, j] = calculate_eta(Delta_tau_asyn[i], Delta_tau_asyn[j])
                            # elif eta_add == 1:
                            # eta_value_add[i, j] = calculate_eta(Delta_tau_asyn[i]+Add_delta_tau[i], Delta_tau_asyn[j]+Add_delta_tau[j])
                            # eta_value_add[i, j] = calculate_eta(Add_delta_tau[i], Add_delta_tau[j])
                    # print('sr_loss===config eta_value_no_add: {} eta_value_add:{}'.format(eta_value_no_add, eta_value_add))
                    interference_power_asyn_no_add = torch.mul(interference_power_asyn, eta_value_no_add)
                    # interference_power_asyn_add = torch.mul(interference_power_asyn, eta_value_add)
                    print('sr_loss===mul interference_power_asyn_no_add: {}'.format(interference_power_asyn_no_add))
                    # interference_power_syn = torch.sum(interference_power_syn)
                    interference_power_syn = torch.sum(interference_power_asyn)
                    interference_power_asyn_no_add = torch.sum(interference_power_asyn_no_add)
                    # interference_power_asyn_add = torch.sum(interference_power_asyn_add)
                    # print('sr_loss===interference_power_syn:{} interference_power_asyn_no_add:{} interference_power_asyn_add:{}'.format(
                    #     interference_power_syn, interference_power_asyn_no_add,interference_power_asyn_add))
                    interference_sum_power_syn = torch.cat((interference_sum_power_syn, interference_power_syn.view(1)))
                    interference_sum_power_asyn_no_add = torch.cat((interference_sum_power_asyn_no_add, interference_power_asyn_no_add.view(1)))
                    # interference_sum_power_asyn_add = torch.cat((interference_sum_power_asyn_add, interference_power_asyn_add.view(1)))
                    # interference_signal_outer_async = torch.mul(interference_signal_outer, eta_value)
                    # print('sr_loss===interference_sum_power_syn:{} interference_sum_power_asyn_no_add: {} interference_sum_power_asyn_add:{}'.format(
                    #     interference_sum_power_syn, interference_sum_power_asyn_no_add,interference_sum_power_asyn_add))
                        # interference_power = torch.sum(interference_signal_outer_async)
            interference_sum_power_syn = torch.sum(interference_sum_power_syn)
            interference_sum_power_asyn_no_add = torch.sum(interference_sum_power_asyn_no_add)
            # interference_sum_power_asyn_add = torch.sum(interference_sum_power_asyn_add)
            if torch.abs(interference_sum_power_syn).item() < torch.abs(interference_sum_power_asyn_no_add).item():
                print('invalid syn and asyn results...')
            # print('sr_loss===Delta_tau: {} sizez:{}'.format(Delta_tau, Delta_tau.shape))
            valid_signal_conj = valid_signal.conj()
            valid_power = torch.sum(torch.outer(valid_signal, valid_signal_conj))
            valid_power_another = torch.abs(torch.sum(valid_signal))**2
            # print('sr_loss===valid_power: {} valid_power_another:{} interference_sum_power_syn:{} interference_sum_power_asyn_no_add:{} interference_sum_power_asyn_add:{} noise_power:{}'.format(
            #     valid_power, valid_power_another, interference_sum_power_syn, interference_sum_power_asyn_no_add, interference_sum_power_asyn_add, noise_power))
            rate_syn += torch.log2(1 + torch.div(torch.abs(valid_power), torch.abs(interference_sum_power_syn)+noise_power))
            rate_asyn_no_add += torch.log2(1 + torch.div(torch.abs(valid_power), torch.abs(interference_sum_power_asyn_no_add)+noise_power))
            # rate_asyn_add += torch.log2(1 + torch.div(torch.abs(valid_power), torch.abs(interference_sum_power_asyn_add)+noise_power))
            print('sr_loss===user: {} rate_syn:{}, rate_asyn_no_add:{}'.format(user_index, rate_syn, rate_asyn_no_add))
            ## 计算WMMSE
                # for satellite in range(int(rx_all_power.shape[1]/users)):
                #     valid_signal = torch.cat((valid_signal, rx_all_power[iter,satellite*users+user_index, user_index]))
                #     valid_signal += rx_all_power[iter,satellite*users+user_index]
        rate_iter_syn = torch.cat((rate_iter_syn, rate_syn.view(1)))
        rate_iter_asyn_no_add = torch.cat((rate_iter_asyn_no_add, rate_asyn_no_add.view(1)))
        # rate_iter_asyn_add = torch.cat((rate_iter_asyn_add, rate_asyn_add.view(1)))
        print('sr_loss===iter: {} rate_iter_syn:{}, rate_iter_asyn_no_add:{}, rate_iter_asyn_add:{}'.format(iter, rate_iter_syn, rate_iter_asyn_no_add, rate_iter_asyn_add))
    avr_rate_syn = torch.mean(rate_iter_syn)
    avr_rate_asyn_no_add = torch.mean(rate_iter_asyn_no_add)
    # avr_rate_asyn_add = torch.mean(rate_iter_asyn_add)
    if avr_rate_syn.item() > avr_rate_asyn_no_add:
        print('invalid results...')
    print('sr_loss===avr_rate_syn: {}, avr_rate_asyn_no_add:{}'.format(avr_rate_syn, avr_rate_asyn_no_add))
    loss = torch.neg(avr_rate_asyn_no_add)
    print('sr_loss===loss:{}'.format(loss))
    return loss, avr_rate_syn, avr_rate_asyn_no_add, syn_wmmse_rate, asyn_wmmse_rate_noadd


def calculate_eta(Delta_1, Delta_2):
    beta = 0.2
    if Delta_1 == Delta_2:
        return 1-0.5*beta*torch.sin(torch.pi*Delta_1)*torch.sin(torch.pi*Delta_2)
    else:
        return torch.div(torch.cos(torch.pi*(Delta_2-Delta_1)*beta),
                         1-4*beta*beta*(Delta_2-Delta_1)*(Delta_2-Delta_1))*torch.sinc(Delta_2-Delta_1) + torch.div(
            beta*torch.sin(torch.pi*Delta_1)*torch.sin(torch.pi*Delta_2),
            2*(beta*beta*(Delta_2-Delta_1)*(Delta_2-Delta_1)-1))*torch.sinc((Delta_2-Delta_1)*beta)

def train(epoch):
    model.train()

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
    # before_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        print('train====out:{}'.format(out))
        loss, avr_rate_syn, avr_rate_asyn_no_add, sync_wmmse_rate, asyn_wmmse_rate_noadd = sr_loss_all_test(data, out, train_K, Nt, epoch, 0)
        loss_imperfect, avr_rate_syn_imperfect, avr_rate_asyn_no_add_imperfect, sync_wmmse_rate_imperfect, \
        asyn_wmmse_rate_noadd_imperfect = sr_loss_all_test(data, out, train_K, Nt, epoch, 1)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        total_avr_rate_syn += avr_rate_syn.item() * data.num_graphs
        total_avr_rate_asyn_no_add += avr_rate_asyn_no_add.item() * data.num_graphs
        # total_avr_rate_asyn_add += avr_rate_asyn_add.item() * data.num_graphs
        # total_avr_async_wmmse += async_wmmse_rate * data.num_graphs
        total_avr_async_noadd_wmmse +=  asyn_wmmse_rate_noadd * data.num_graphs
        total_avr_sync_wmmse += sync_wmmse_rate * data.num_graphs
        total_loss_imperfect += loss_imperfect.item() * data.num_graphs
        total_avr_rate_syn_imperfect += avr_rate_syn_imperfect.item() * data.num_graphs
        total_avr_rate_asyn_no_add_imperfect += avr_rate_asyn_no_add_imperfect.item() * data.num_graphs
        # total_avr_rate_asyn_add_imperfect += avr_rate_asyn_add_imperfect.item() * data.num_graphs
        # total_avr_async_wmmse_imperfect += async_wmmse_rate_imperfect * data.num_graphs
        total_avr_async_noadd_wmmse_imperfect +=  asyn_wmmse_rate_noadd_imperfect * data.num_graphs
        total_avr_sync_wmmse_imperfect += sync_wmmse_rate_imperfect * data.num_graphs
        print('train====data.num_graphs:{}'.format(data.num_graphs))
        optimizer.step()
    print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Train Loss: {:.4f}, avr_rate_syn: {:.4f}, avr_rate_asyn_no_add: {:.4f}, '
          'Average syncWMMSE Rate: {:.4f}, Average asyncWMMSE_Noadd Rate: {:.4f}'.format(SNR_dB, 0, epoch, total_loss / train_layouts, total_avr_rate_syn / train_layouts, total_avr_rate_asyn_no_add / train_layouts, \
           total_avr_sync_wmmse / train_layouts, total_avr_async_noadd_wmmse / train_layouts))
    print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Train Loss: {:.4f}, avr_rate_syn_imperfect: {:.4f}, avr_rate_asyn_no_add_imperfect: {:.4f}, '
          'Average syncWMMSE_imperfect Rate: {:.4f}, Average asyncWMMSE_Noadd_imperfect Rate: {:.4f}'.format(SNR_dB, 1, epoch, total_loss_imperfect / train_layouts, total_avr_rate_syn_imperfect / train_layouts, total_avr_rate_asyn_no_add_imperfect / train_layouts, \
           total_avr_sync_wmmse_imperfect / train_layouts, total_avr_async_noadd_wmmse / train_layouts))
    return total_loss / train_layouts, total_avr_rate_syn / train_layouts, total_avr_rate_asyn_no_add / train_layouts, \
           total_avr_sync_wmmse / train_layouts, total_avr_async_noadd_wmmse / train_layouts


def test(epoch):
    model.eval()

    total_loss = 0
    sum_rate = []
    num = 0
    for data in test_loader:
        data = data.to(device)
        num += 1
        with torch.no_grad():
            start = time.time()
            out = model(data)
            end = time.time()
            print('CGCNet time:', end - start)
            loss, _, _, _, _, async_wmmse_rate = sr_loss_all(data, out, test_K, Nt, epoch, 0)
            sum_rate.append(async_wmmse_rate)
            total_loss += loss.item() * data.num_graphs
            # Y = batch_wmmse(data.CSI.transpose(0, 2, 1, 3), var)
            # power = out[:,:2*Nt]
            # Y = power.numpy()
            # power_check(Y)
        #     links = out[:, 0:1]
        # data_CSI = data.CSI
        # p1 = torch.mul(links, p1)
        # data_CSI = data_CSI.transpose(0, 2, 1, 3)
        # Y = batch_wmmse(data.CSI.transpose(0, 2, 1, 3), var)
        # for i in range(Y.shape[0]):
        #     Y[i,:,:] = np.multiply(Y[i,:,:], )
        # end = time.time()
        # print('WMMSE time:', end - start)
        # sr = wmmse.IC_sum_rate(test_csis, Y, var)
        # print('WMMSE rate:', sr)
    # average_wmmse_rate = sum_rate / num
    return total_loss / test_layouts, np.mean(sum_rate)


train_S = 2  ## 卫星数量
beams = 2  ## 每个卫星配置的波束数量
users = 2  ## 用户数量
# train_K = train_S * beams * users  ## 可能的收发匹配数量
train_K = train_S * users  ## 可能的收发匹配数量
test_K = train_K
train_layouts = 10  ## 训练次数
test_layouts = 1  ## 测试次数
SNR_dB = 50
Nt = 32  ## 每个波束的发射天线数
imperfect_channel_condition = 1
train_config = init_parameters()
var = 1
train_dists, train_csis, train_delays = wg.sample_generate_all(train_config, train_layouts)
test_config = init_parameters()
test_dists, test_csis, test_delays = wg.sample_generate_all(test_config, test_layouts)

train_csi_real, train_csi_imag = np.real(train_csis), np.imag(train_csis)
test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
norm_train_real, norm_test_real = normalize_data(train_csi_real, test_csi_real, train_config)
norm_train_imag, norm_test_imag = normalize_data(train_csi_imag, test_csi_imag, train_config)

import time

start = time.time()
print('train_csis:{}'.format(train_csis))
print('train_dists:{}'.format(train_dists))
print('train_delays:{}'.format(train_delays))
print('norm_train_real:{}'.format(norm_train_real))
print('norm_train_imag:{}'.format(norm_train_imag))
print('train_K:{}'.format(train_K))
# vectors = [np.zeros(users, dtype=int)]
# for i in range(users):
#     vector = np.zeros(users, dtype=int)
#     vector[i] = 1
#     vectors.append(vector)
#
# for j in range(beams):
#     ## 遍历取可行的选择，判断是否合理
# basic_optional_links = [[]]
# for
# links =
Y = batch_wmmse(test_csis.transpose(0, 2, 1, 3), var)
end = time.time()
print('WMMSE time:', end - start)
sr = wmmse.IC_sum_rate(test_csis, Y, var)
print('WMMSE rate:', sr)

train_data_list = proc_data_asyn(train_csis, train_dists, train_delays, norm_train_real, norm_train_imag, train_K)
print('train_data_list:{}'.format(train_data_list))
test_data_list = proc_data_asyn(test_csis, test_dists, test_delays, norm_test_real, norm_test_imag, test_K)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IGCNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True, num_workers=1)
# test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False)
before_train_loss = 0
# for epoch in range(1, 2000):
#     print('Start to train...')
#     loss1, avr_rate_syn, avr_rate_asyn_no_add,  avr_rate_asyn_add, total_avr_async_wmmse= train()
#     print('Start to test...')
#     loss2, test_average_wmmse = test()
#     print('SNR {:03d}dB, Epoch {:03d}, Train Loss: {:.4f}, avr_rate_syn: {:.4f}, avr_rate_asyn_no_add: {:.4f}, '
#           'avr_rate_asyn_add: {:.4f}, Average asyncWMMSE Rate: {}, Test Loss: {:.4f}, Test Average asyncWMMSE Rate: {}'.format(
#         SNR_dB, epoch, loss1, avr_rate_syn, avr_rate_asyn_no_add,  avr_rate_asyn_add, total_avr_async_wmmse, loss2, test_average_wmmse))
#
#     scheduler.step()

for epoch in range(1, 500):
    print('Start to train...')
    loss1, avr_rate_syn, avr_rate_asyn_no_add, total_avr_sync_wmmse, \
    total_avr_async_noadd_wmmse = train(epoch)
    # print('Start to test...')
    # loss2, test_average_wmmse = test()
    # print('SNR {:03d}dB, IPC {:03d},  Epoch {:03d}, Train Loss: {:.4f}, avr_rate_syn: {:.4f}, avr_rate_asyn_no_add: {:.4f}, '
    #       'avr_rate_asyn_add: {:.4f}, Average syncWMMSE Rate: {:.4f}, Average asyncWMMSE_Noadd Rate: {:.4f}, '
    #       'Average asyncWMMSE Rate: {:.4f}'.format(SNR_dB, imperfect_channel_condition, epoch, loss1, avr_rate_syn, avr_rate_asyn_no_add,
    #                                                avr_rate_asyn_add, total_avr_sync_wmmse, total_avr_async_noadd_wmmse,
    #                                                total_avr_async_wmmse))

    scheduler.step()

# density = test_config.field_length ** 2 / test_K
# ## 40:
# # gen_tests = [40, 80, 160]
# gen_tests = []
# for test_K in gen_tests:
#     print('More test Start...')
#     test_layouts = 50
#     test_config = init_parameters()
#     test_config.n_links = test_K
#     test_config.n_receiver = test_K
#     field_length = int(np.sqrt(density * test_K))
#     test_config.field_length = field_length
#     test_dists, test_csis = wg.sample_generate(test_config, test_layouts)
#     print('test size', test_csis.shape, field_length)
#
#     start = time.time()
#     start = time.time()
#     Y = batch_wmmse(test_csis.transpose(0, 2, 1, 3), var)
#     end = time.time()
#     print('WMMSE time:', end - start)
#     sr = wmmse.IC_sum_rate(test_csis, Y, var)
#     print('WMMSE rate:', sr)
#
#     test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
#     _, norm_test_real = normalize_data(train_csi_real, test_csi_real, train_config)
#     _, norm_test_imag = normalize_data(train_csi_imag, test_csi_imag, test_config)
#
#     test_data_list = proc_data(test_csis, test_dists, norm_test_real, norm_test_imag, test_K)
#     test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
#     loss2, average_wmmse = test()
#     print('CGCNet rate:', loss2)