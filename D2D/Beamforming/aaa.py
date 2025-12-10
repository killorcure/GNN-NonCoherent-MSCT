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

    def update(self, aggr_out, x):
        print('update===aggr_out:{}'.format(aggr_out))
        print('update===x:{}'.format(x))
        tmp = torch.cat([x, aggr_out], dim=1)
        comb_all = self.mlp2(tmp)
        print('update===comb_all:{}'.format(comb_all))
        comb = comb_all[:, 1:2 * Nt + 1] # w
        add_delta_delay = comb_all[:, 0:1] # tau_c
        add_delta_delay = torch.sigmoid(add_delta_delay)
        print('update===tmp:{}'.format(tmp))
        print('update===comb:{}, shape:{}'.format(comb, comb.shape))
        print('update===add_delta_delay:{}'.format(add_delta_delay))
        # shape_length = comb.shape[0]
        ## normalize
        comb_group = comb.view(-1, users, 2*Nt)
        square_sum = (comb_group ** 2).sum(dim=(1,2), keepdim=True)
        # square_sum = square_sum
        square_sum[square_sum <= 1] = 1
        # square_sum[square_sum > 100] = 100
        comb_normalized = 0.4*comb_group / torch.sqrt(square_sum)
        comb = comb_normalized.view(comb.shape[0], 2*Nt)
        comb_res = torch.cat([add_delta_delay, comb], dim=1)
        return torch.cat([comb_res, x[:, :2 * Nt - 1]], dim=1)

    def forward(self, x, edge_index, edge_attr):
        # print('forward===x:{}, edge_index:{}, edge_attr:{}'.format(x, edge_index, edge_attr))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # print('unsqueeze x:{}'.format(x))
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        # print('unsqueeze edge_attr:{}, size:{}'.format(edge_attr, edge_attr.shape))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        print('message===x_i:{}, x_j:{}'.format(x_i, x_j)) ## x_i为当前节点，x_j为邻居节点
        # print('message edge_attr:{}'.format(edge_attr))
        tmp = torch.cat([x_j, edge_attr], dim=1)
        # print('tmp:{}, size:{}'.format(tmp, tmp.shape))
        agg = self.mlp1(tmp)
        # print('message tmp:{}'.format(tmp))
        # print('message agg:{}'.format(agg))
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)

