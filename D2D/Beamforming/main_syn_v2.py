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

#### 该版本和main_syn.py的区别在与，将边特征区分了+1,-1,0属性
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
    for i in range(len(attr_ind[0,:])):
        flag[i][0] = dist2[attr_ind[0, i], attr_ind[1,i]]
    print('flag:{}, size:{}'.format(flag, flag.shape))
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
        data = build_graph_MSCT_asyn(HH[i, :, :, :], dists[i, :, :], delays[i, :, :], norm_csi_real[i, :, :, :], norm_csi_imag[i, :, :, :], K,
                                500)
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
        # add_delta_delay = torch.sigmoid(add_delta_delay)
        # links_onehot = update_links(links_res)
        links_onehot = links_res
        print('update===tmp:{}'.format(tmp))
        print('update===comb:{}'.format(comb))
        # print('update===add_delta_delay:{}'.format(add_delta_delay))
        nor = torch.sqrt(torch.sum(torch.mul(comb, comb), axis=1))
        print('update===nor:{}'.format(nor))
        nor = nor.unsqueeze(axis=-1)
        print('update===unsqueeze nor:{}'.format(nor))
        comp1 = torch.ones(comb.size(), device=device)
        print('update===comp1:{}'.format(comp1))
        comb = torch.div(comb, torch.max(comp1, nor))
        print('update===comb:{}'.format(comb))
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
        self.network_layer = 8

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
        return out


def power_check(p):
    n = p.shape[0]
    pp = np.sum(np.square(p), axis=1)
    print(np.sum(pp > 1.1))


def sr_loss(data, p, K, N):
    # H1 K*K*N
    # p1 K*N
    print('sr_loss===data:{}'.format(data))
    print('sr_loss===p:{}'.format(p))
    H1 = data.y[:, :, :, :, 0]  ##实部
    H2 = data.y[:, :, :, :, 1]  ##虚部
    print('sr_loss===H1:{}, size:{}'.format(H1, H1.shape))
    print('sr_loss===H2:{}, size:{}'.format(H2, H2.shape))
    links = p[:, 0:1]
    add_delta_delay = p[:, 1:2]
    links = update_links(links)
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
    rx_power1 = torch.mul(H1, p1)
    print('sr_loss===mul rx_power1:{}, size:{}'.format(rx_power1, rx_power1.shape))
    rx_power1 = torch.sum(rx_power1, axis=-1)
    print('sr_loss===rx_power1:{}, size:{}'.format(rx_power1, rx_power1.shape))

    rx_power2 = torch.mul(H2, p2)
    rx_power2 = torch.sum(rx_power2, axis=-1)
    print('sr_loss===rx_power2:{}'.format(rx_power2))

    rx_power3 = torch.mul(H1, p2)
    rx_power3 = torch.sum(rx_power3, axis=-1)
    print('sr_loss===rx_power3:{}'.format(rx_power3))

    rx_power4 = torch.mul(H2, p1)
    rx_power4 = torch.sum(rx_power4, axis=-1)
    print('sr_loss===rx_power4:{}'.format(rx_power4))

    rx_power = torch.mul(rx_power1 - rx_power2, rx_power1 - rx_power2) + torch.mul(rx_power3 + rx_power4,
                                                                                   rx_power3 + rx_power4)
    print('sr_loss===rx_power:{}, size:{}'.format(rx_power, rx_power.shape))
    mask = torch.eye(K, device=device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), axis=1)
    new_mask = 1 - mask
    # for i in range(K):
    #     for j in range(K):
    #         if (i % users) == (j % users):
    #             new_mask[i][j] = 0
    interference = torch.sum(torch.mul(rx_power, new_mask), axis=1) + 1
    print('sr_loss===valid_rx_power:{}, size:{}'.format(valid_rx_power, valid_rx_power.shape))
    print('sr_loss===interference:{}, size:{}'.format(interference, interference.shape))
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    print('sr_loss===rate:{}, size:{}'.format(rate, rate.shape))
    sum_rate = torch.mean(torch.sum(rate, axis=1))
    print('sr_loss===sum_rate:{}'.format(sum_rate))
    loss = torch.neg(sum_rate)
    print('sr_loss===loss:{}'.format(loss))
    ## 计算WMMSE
    # print('sr_loss===data.CSI:{}, size:{}'.format(data.CSI, data.CSI.shape))
    # links_a = links.detach().numpy()
    # CSIs = []
    # for i in range(test_csis.shape[0]):
    #     CSIs.append(np.multiply(links_a, test_csis[i,:,:]))
    # Y = batch_wmmse(CSIs.transpose(0, 2, 1, 3), var)
    print('sr_loss===data.CSI:{}, size:{}'.format(data.CSI, data.CSI.shape))
    # print('sr_loss===wmmse_CSI:{}, size:{}'.format(wmmse_CSI, wmmse_CSI.shape))
    num = int(data.CSI.shape[0] / K)
    wmmse_CSI = torch.reshape(data.CSI, (num, K, K, N))  # 2*18*18*2
    print('sr_loss===wmmse_CSI reshape:{}, size:{}'.format(wmmse_CSI, wmmse_CSI.shape))
    print('sr_loss===links:{}, size:{}'.format(links, links.shape))
    links = torch.reshape(links, (int(links.shape[0] / K), K, 1))  # 2*18*1
    new_links = links.unsqueeze(2)  # 2*18*1*1
    print('sr_loss===new_links:{}, size:{}'.format(new_links, new_links.shape))
    wmmse_CSI = torch.mul(wmmse_CSI, new_links)
    print('sr_loss===wmmse_CSI mul:{}, size:{}'.format(wmmse_CSI, wmmse_CSI.shape))
    CSI = wmmse_CSI.detach().cpu()
    print('sr_loss===CSI mul:{}, size:{}'.format(CSI, CSI.shape))
    Y = batch_wmmse(CSI.numpy().transpose(0, 2, 1, 3), var)
    # end = time.time()
    # print('WMMSE time:', end - start)
    sr = wmmse.IC_sum_rate(CSI.numpy(), Y, var)
    print('WMMSE rate at sr_loss:', sr)
    return loss, sr

def sr_loss_asyn(data, p, K, N):
    # H1 K*K*N
    # p1 K*N
    eta_enable = 0
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
    links = update_links(links)
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
    rx_all_power = torch.mul(H4, p3)
    print('sr_loss===rx_all_power:{}, size:{}'.format(rx_all_power, rx_all_power.shape))
    rx_all_power = torch.sum(rx_all_power, axis=-1)
    print('sr_loss===sum rx_all_power:{}, size:{}'.format(rx_all_power, rx_all_power.shape))
    initial_delay = data.initial_delay
    print('sr_loss===initial_delay:{}, size:{}'.format(initial_delay, initial_delay.shape))
    initial_delay = torch.reshape(initial_delay, (-1, K, K, 1))
    print('sr_loss===reshape initial_delay:{}, size:{}'.format(initial_delay, initial_delay.shape))
    # rx_all_power_1 = torch.abs(rx_all_power)**2
    rate_iter = torch.empty(0).cuda()
    for iter in range(rx_all_power.shape[0]):
        rate = torch.zeros(1).cuda()
        for user_index in range(users):
            valid_signal = torch.empty(0).cuda()
            signal = H3[iter, user_index, user_index, :].abs() ** 2
            signal_power = signal.mean()
            noise_power = signal_power / (10 ** (SNR_dB / 10))
            print('sr_loss===H3[iter, user_index, user_index,:]: {} signal:{} signal_power: {} noise_power: {}'.format(
                H3[iter, user_index, user_index, :], signal, signal_power, noise_power))
            interference_sum_power = torch.empty(0).cuda()
            for link in range(rx_all_power.shape[1]):
                satellite_index = int(link/users)
                if int(link % users) == user_index:
                    valid_signal = torch.cat(
                        (valid_signal, rx_all_power[iter, link, user_index].view(1)))
                # else:
                #     valid_link = satellite_index*users+user_index
                #     interference_signal = torch.cat(
                #         (interference_signal, rx_all_power[iter, link, user_index].view(1)))
                #     Delta_tau = torch.cat(
                #         (Delta_tau, initial_delay[iter, link, valid_link].view(1)))
            for other_user_index in range(users):
                interference_signal = torch.empty(0).cuda()
                Delta_tau = torch.empty(0).cuda()
                if other_user_index != user_index:
                    for other_link in range(rx_all_power.shape[1]):
                        if int(other_link % users) == other_user_index:
                            satellite_index = int(other_link / users)
                            valid_link = satellite_index * users + user_index
                            interference_signal = torch.cat(
                                (interference_signal, rx_all_power[iter, other_link, user_index].view(1)))
                            Delta_tau = torch.cat(
                                        (Delta_tau, initial_delay[iter, other_link, valid_link].view(1)))
                    interference_signal_conj = interference_signal.conj()
                    print('sr_loss===valid_signal: {} interference_signal:{} interference_signal_conj:{} Delta_tau:{}'.format(
                        valid_signal, interference_signal, interference_signal_conj, Delta_tau))
                    if eta_enable == 0:
                        # interference_power = torch.outer(interference_signal, interference_signal_conj)
                        interference_power = torch.sum(torch.outer(interference_signal, interference_signal_conj))
                        interference_sum_power = torch.cat((interference_sum_power, interference_power.view(1)))
                    else:
                        interference_power = torch.outer(interference_signal, interference_signal_conj)
                        print('sr_loss===outer interference_power: {} size:{}'.format(interference_power, interference_power.shape))
                        eta_value = torch.outer(Delta_tau, Delta_tau)
                        print('sr_loss===init eta_value: {} sizez:{}'.format(eta_value, eta_value.shape))
                        for i in range(Delta_tau.shape[0]):
                            for j in range(Delta_tau.shape[0]):
                                eta_value[i, j] = calculate_eta(Delta_tau[i], Delta_tau[j])
                        print('sr_loss===config eta_value: {} sizez:{}'.format(eta_value, eta_value.shape))
                        interference_power = torch.mul(interference_power, eta_value)
                        print('sr_loss===mul interference_power: {} size:{}'.format(interference_power, interference_power.shape))
                        interference_power = torch.sum(interference_power)
                        print('sr_loss===interference_power:{} size:{}'.format(interference_power, interference_power.shape))
                        interference_sum_power = torch.cat((interference_sum_power, interference_power.view(1)))
                        # interference_signal_outer_async = torch.mul(interference_signal_outer, eta_value)
                        print('sr_loss===interference_power:{} interference_sum_power: {} size:{}'.format(
                            interference_power, interference_sum_power,interference_sum_power.shape))
                        # interference_power = torch.sum(interference_signal_outer_async)
            interference_sum_power = torch.sum(torch.abs(interference_sum_power)) + noise_power
            # print('sr_loss===Delta_tau: {} sizez:{}'.format(Delta_tau, Delta_tau.shape))
            valid_signal_conj = valid_signal.conj()
            valid_power = torch.sum(torch.outer(valid_signal, valid_signal_conj))
            # if eta_enable == 0:
            #     interference_signal_conj = interference_signal.conj()
            #     interference_power = torch.sum(torch.outer(interference_signal, interference_signal_conj))
            # else:
            #     interference_signal_conj = interference_signal.conj()
            #     interference_signal_outer = torch.outer(interference_signal, interference_signal_conj)
            #     print('sr_loss===interference_signal_outer: {} sizez:{}'.format(interference_signal_outer, interference_signal_outer.shape))
            #     # interference_power = torch.sum(torch.outer(interference_signal, interference_signal_conj))
            #     # Delta_tau_2 = Delta_tau
            #     eta_value = torch.sum(torch.outer(Delta_tau, Delta_tau))
            #     print('sr_loss===init eta_value: {} sizez:{}'.format(eta_value, eta_value.shape))
            #     for i in range(Delta_tau.shape[0]):
            #         for j in range(Delta_tau.shape[0]):
            #             eta_value[i,j] = calculate_eta(Delta_tau[i], Delta_tau[j])
            #     print('sr_loss===config eta_value: {} sizez:{}'.format(eta_value, eta_value.shape))
            #     interference_signal_outer_async = torch.mul(interference_signal_outer, eta_value)
            #     print('sr_loss===interference_signal_outer_async: {} sizez:{}'.format(interference_signal_outer_async, interference_signal_outer_async.shape))
            #     interference_power = torch.sum(interference_signal_outer_async)
            print('sr_loss===valid_power: {} interference_sum_power:{}'.format(valid_power, interference_sum_power))
            rate += torch.log2(1 + torch.div(torch.abs(valid_power), torch.abs(interference_sum_power)))
            print('sr_loss===user: {} rate:{}, size:{}'.format(user_index, rate, rate.shape))
            ## 计算WMMSE
                # for satellite in range(int(rx_all_power.shape[1]/users)):
                #     valid_signal = torch.cat((valid_signal, rx_all_power[iter,satellite*users+user_index, user_index]))
                #     valid_signal += rx_all_power[iter,satellite*users+user_index]
        rate_iter = torch.cat((rate_iter, rate.view(1)))
        print('sr_loss===iter: {} rate:{}, size:{}'.format(iter, rate_iter, rate_iter.shape))
    avr_rate = torch.mean(rate_iter)
    print('sr_loss===average rate: {}, size:{}'.format(avr_rate, avr_rate.shape))
    loss = torch.neg(avr_rate)
    print('sr_loss===loss:{}'.format(loss))
    print('sr_loss===data.CSI:{}, size:{}'.format(data.CSI, data.CSI.shape))
    # print('sr_loss===wmmse_CSI:{}, size:{}'.format(wmmse_CSI, wmmse_CSI.shape))
    num = int(data.CSI.shape[0] / K)
    wmmse_CSI = torch.reshape(data.CSI, (num, K, K, N))  # 2*18*18*2
    print('sr_loss===wmmse_CSI reshape:{}, size:{}'.format(wmmse_CSI, wmmse_CSI.shape))
    print('sr_loss===links:{}, size:{}'.format(links, links.shape))
    links = torch.reshape(links, (int(links.shape[0] / K), K, 1))  # 2*18*1
    new_links = links.unsqueeze(2)  # 2*18*1*1
    print('sr_loss===new_links:{}, size:{}'.format(new_links, new_links.shape))
    wmmse_CSI = torch.mul(wmmse_CSI, new_links)
    print('sr_loss===wmmse_CSI mul:{}, size:{}'.format(wmmse_CSI, wmmse_CSI.shape))
    CSI = wmmse_CSI.detach().cpu()
    print('sr_loss===CSI mul:{}, size:{}'.format(CSI, CSI.shape))
    Y = batch_wmmse(CSI.numpy().transpose(0, 2, 1, 3), var)
    # end = time.time()
    # print('WMMSE time:', end - start)
    sr = wmmse.IC_sum_rate(CSI.numpy(), Y, var)
    print('WMMSE rate at sr_loss:', sr)
    return loss, sr

def calculate_eta(Delta_1, Delta_2):
    beta = 0.5
    if Delta_1 == Delta_2:
        return 1-0.5*beta*torch.sin(torch.pi*Delta_1)*torch.sin(torch.pi*Delta_2)
    else:
        return torch.div(torch.cos(torch.pi*(Delta_2-Delta_1))*beta,
                         1-4*beta*beta*(Delta_2-Delta_1)*(Delta_2-Delta_1))*torch.sinc(Delta_2-Delta_1) + torch.div(
            beta*torch.sin(torch.pi*Delta_1)*torch.sin(torch.pi*Delta_2),
            2*(beta*beta*(Delta_2-Delta_1)*(Delta_2-Delta_1)-1))*torch.sinc((Delta_2-Delta_1)*beta)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        print('train====out:{}'.format(out))
        loss, wmmse_rate = sr_loss_asyn(data, out, train_K, Nt)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        print('train====data.num_graphs:{}'.format(data.num_graphs))
        optimizer.step()
    return total_loss / train_layouts


def test():
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
            loss, wmmse_rate = sr_loss_asyn(data, out, test_K, Nt)
            sum_rate.append(wmmse_rate)
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


train_S = 3  ## 卫星数量
beams = 2  ## 每个卫星配置的波束数量
users = 3  ## 用户数量
# train_K = train_S * beams * users  ## 可能的收发匹配数量
train_K = train_S * users  ## 可能的收发匹配数量
test_K = train_K
train_layouts = 2  ## 训练次数
test_layouts = 10  ## 测试次数
SNR_dB = 10
Nt = 2  ## 每个波束的发射天线数
train_config = init_parameters()
var = 1
train_dists, train_csis, train_delays = wg.sample_generate_asyn(train_config, train_layouts)
test_config = init_parameters()
test_dists, test_csis, test_delays = wg.sample_generate_asyn(test_config, test_layouts)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

train_loader = DataLoader(train_data_list, batch_size=128, shuffle=True, num_workers=5)
test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
for epoch in range(1, 5000):
    print('Start to train...')
    loss1 = train()
    print('Start to test...')
    loss2, average_wmmse = test()
    print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Average WMMSE Rate: {}'.format(
        epoch, loss1, loss2, average_wmmse))

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