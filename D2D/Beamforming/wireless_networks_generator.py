# This script contains the generator code for producing wireless network layouts and channel losses
# for the work "Spatial Deep Learning for Wireless Scheduling",
# available at https://ieeexplore.ieee.org/document/8664604.

# For any reproduce, further research or development, please kindly cite our JSAC journal paper:
# @Article{spatial_learn,
#    author = "W. Cui and K. Shen and W. Yu",
#    title = "Spatial Deep Learning for Wireless Scheduling",
#    journal = "{\it IEEE J. Sel. Areas Commun.}",
#    year = 2019,
#    volume = 37,
#    issue = 6,
#    pages = "1248-1261",
#    month = "June",
# }

import numpy as np
def layout_generate(general_para):
    N = general_para.n_links
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    layout_rx = []
    # generate rx one by one rather than N together to ensure checking validity one by one
    rx_xs = []; rx_ys = []
    tot_links = 0
    n_re = general_para.n_receiver
    for i in range(N):
        n_links = i
        rx_i = []

        num_rx = np.random.randint(general_para.minrx, general_para.maxrx)
        num_rx = min(num_rx,  n_re - tot_links)
        tot_links += num_rx
        for j in range(num_rx): 
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directLink_length, high=general_para.longest_directLink_length)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                    got_valid_rx = True
            rx_i.append([rx_x[0], rx_y[0]])
        layout_rx.append(rx_i)
        if(tot_links >= n_re):
            break

    # For now, assuming equal weights and equal power, so not generating them
    layout_tx = np.concatenate((tx_xs, tx_ys), axis=1)
    
    return layout_tx, layout_rx

def distance_generate(general_para,layout_tx,layout_rx):
    distances = np.zeros((general_para.n_receiver,general_para.n_receiver))
    N = len(layout_rx)
    sum_tx = 0
    for tx_index in range(N):
        num_loops = len(layout_rx[tx_index])
        tx_coor = layout_tx[tx_index]
        for tx_i in range(num_loops):
            sum_rx = 0
            for rx_index in range(N):
                for rx_i in layout_rx[rx_index]:
                    rx_coor = rx_i
                    # distances[sum_rx][sum_tx] = np.linalg.norm(tx_coor - rx_coor)
                    distances[sum_rx][sum_tx] = 1 ## 这里初始化每个收发端之间距离都为1
                    sum_rx += 1
            sum_tx += 1
    return distances

def distance_generate_new(general_para, layout_tx, layout_rx):
    distances = np.zeros((general_para.n_receiver, general_para.n_receiver))
    satellite_num = general_para.satellite_num
    user_num = general_para.user_num
    L = general_para.n_receiver
    distances_new = 500+np.random.rand(satellite_num, user_num)*500
    # final_distance_new = np.random.randn(L, L)
    # for i in range(L):
    #     for j in range(L):
    #         satellite_index = int(i / user_num)
    #         user_index = int(j % user_num)
    #         final_distance_new[i,j] = distances_new[satellite_index, user_index]
    #         # final_CSI_new[i,j,:] = small_scale_CSI_new[satellite_index, user_index,:]
    return distances_new

def CSI_generate(general_para, distances): ##单天线 这里要改成对应的星地信道
    Nt = general_para.N_antennas
    L = general_para.n_receiver
    satellite_num = general_para
    dists = np.expand_dims(distances,axis=-1)
    shadowing = np.random.randn(L,L,Nt)
    large_scale_CSI = 4.4*10**5/((dists**1.88)*(10**(shadowing*6.3/20)))
    small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))*np.sqrt(large_scale_CSI)
    print('distance:{}'.format(distances))
    print('dists:{}'.format(dists))
    print('small_scale_CSI:{}'.format(small_scale_CSI))
    return small_scale_CSI

def CSI_generate_single_connection(general_para, distances): ##单天线 这里要改成对应的星地信道
    Nt = general_para.N_antennas
    L = general_para.n_receiver
    satellite_num = general_para
    dists = np.expand_dims(distances,axis=-1)
    # shadowing = np.random.randn(L,L,Nt)
    # large_scale_CSI = 4.4*10**5/((dists**1.88)*(10**(shadowing*6.3/20)))
    # small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))*np.sqrt(large_scale_CSI)
    small_scale_CSI = 1 / np.sqrt(2) * (
                np.random.randn(L, L, Nt) + 1j * np.random.randn(L, L, Nt))
    final_CSI = np.random.randn(L, L, Nt) + 1j * np.random.randn(L, L, Nt)
    for i in range(L):
        for j in range(L):
            for k in range(Nt):
                final_CSI[i,j,k] = TFD_NTN_TDL_channel()
    print('distance:{}'.format(distances))
    print('dists:{}'.format(dists))
    print('small_scale_CSI:{}'.format(small_scale_CSI))
    print('final_CSI:{}'.format(small_scale_CSI))
    return small_scale_CSI

def TFD_NTN_TDL_channel(satellite_params=None, channel_params=None, timefreq_params=None, doppler_model='random'):
    """
    NTN-TDL-D Channel model based on 3GPP TR 38.811

    Inputs:
        satellite_params: dict containing satellite parameters
        channel_params: dict containing channel tap parameters
        timefreq_params: dict containing time-frequency grid parameters
        doppler_model: 'fixed' or 'random'

    Outputs:
        H: Time-frequency channel response matrix [Nt x Nf]
        t: Time vector [Nt]
        f: Frequency vector [Nf]
        params: dict containing full configuration
    """

    # Default parameters
    if satellite_params is None:
        satellite_params = {
            'v_sat': 7500,   # m/s
            'h': 500e3,      # m
            'alpha_model': 50, # degrees
            'f_c': 2.1e9     # Hz
        }

    if channel_params is None:
        channel_params = {
            'tapParams': np.array([
                [0, -0.284, 1],
                [0, -11.991, 0],
                [0.5596, -9.887, 0],
                [7.334, -16.771, 0]
            ]),
            'K_factor_dB': 11.707
        }

    if timefreq_params is None:
        timefreq_params = {
            'T': 1e-6,   # s
            'F': 2e9,    # Hz
            'Nt': 1,
            'Nf': 1
        }

    # Constants
    c = 3e8          # speed of light (m/s)
    R = 6371e3       # earth radius (m)

    # Calculate Doppler shift due to satellite motion
    f_d_shift = (satellite_params['v_sat'] / c) * (R / (R + satellite_params['h'])) * \
                np.cos(np.deg2rad(satellite_params['alpha_model'])) * satellite_params['f_c']

    # Time and frequency grids
    t = np.linspace(0, timefreq_params['T'], timefreq_params['Nt'])
    f = np.linspace(0, timefreq_params['F'], timefreq_params['Nf'])

    # Initialize channel matrix
    num_taps = channel_params['tapParams'].shape[0]
    H = np.zeros((timefreq_params['Nt'], timefreq_params['Nf']), dtype=complex)

    for tap_idx in range(num_taps):
        # Extract tap parameters
        tau_i = channel_params['tapParams'][tap_idx, 0] * 1e-6  # s
        pow_dB = channel_params['tapParams'][tap_idx, 1]
        fading_type = channel_params['tapParams'][tap_idx, 2]

        # Convert power to linear scale
        pow_linear = 10 ** (pow_dB / 10)

        # Fading sampling
        if fading_type == 1:  # Rician
            K_linear = 10 ** (channel_params['K_factor_dB'] / 10)
            A = np.sqrt(pow_linear * K_linear / (K_linear + 1))
            n = np.sqrt(pow_linear / (K_linear + 1) / 2) * (np.random.randn(timefreq_params['Nt']) + \
                                                            1j * np.random.randn(timefreq_params['Nt']))
            chi_i = A + n
        else:  # Rayleigh
            chi_i = np.sqrt(pow_linear / 2) * (np.random.randn(timefreq_params['Nt']) + \
                                               1j * np.random.randn(timefreq_params['Nt']))

        # Doppler shift
        if doppler_model.lower() == 'fixed':
            nu_i = f_d_shift
        else:
            nu_i = f_d_shift + 10 * np.random.randn()

        # Time-frequency response
        for tt in range(timefreq_params['Nt']):
            phase = 2 * np.pi * (t[tt] * nu_i - f * tau_i)  # vectorized over frequency
            H[tt, :] += chi_i[tt] * np.exp(1j * phase)

    # Output parameters
    params = {
        'SatelliteParams': satellite_params,
        'ChannelParams': channel_params,
        'TimeFreqParams': timefreq_params,
        'DopplerModel': doppler_model,
        'f_d_shift': f_d_shift
    }

    return H[0,0]

def CSI_generate_new(general_para, distances):
    Nt = general_para.N_antennas
    L = general_para.n_receiver
    satellite_num = general_para.satellite_num
    user_num = general_para.user_num
    dists = np.expand_dims(distances,axis=-1)
    shadowing = np.random.randn(satellite_num,user_num,Nt)
    # print('dists:{}, shadowing:{}'.format(dists, shadowing))
    large_scale_CSI = 4.4*10**5/((1**1.88)*(10**(shadowing*6.3/20)))
    # small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(satellite_num,user_num,Nt)+1j*np.random.randn(satellite_num,user_num,Nt))*np.sqrt(large_scale_CSI)
    small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(satellite_num,user_num,Nt)+1j*np.random.randn(satellite_num,user_num,Nt))
    small_scale_CSI_new = np.random.randn(satellite_num,user_num,Nt)+1j*np.random.randn(satellite_num,user_num,Nt)
    # final_CSI = np.random.randn(L, L, Nt) + 1j * np.random.randn(L, L, Nt)
    final_CSI_new = np.random.randn(L, L, Nt) + 1j * np.random.randn(L, L, Nt)
    for i in range(satellite_num):
        for j in range(user_num):
            for k in range(Nt):
                small_scale_CSI_new[i,j,k] = TFD_NTN_TDL_channel()
    for i in range(L):
        for j in range(L):
            satellite_index = int(j / user_num)
            user_index = int(j % user_num)
            large_scale_CSI = 4.4 * 10 ** 5 / ((dists[satellite_index, user_index,:] ** 1.88) * (10 ** (shadowing[satellite_index, user_index,:] * 6.3 / 20)))
            # print('large_scale_CSI:{}'.format(large_scale_CSI))
            # large_scale_CSI = 1
            # final_CSI_new[i,j,:] = small_scale_CSI_new[satellite_index, user_index,:]*np.sqrt(large_scale_CSI)
            final_CSI_new[i,j,:] = small_scale_CSI_new[satellite_index, user_index,:]
    print('distance:{}'.format(distances))
    print('dists:{}'.format(dists))
    print('small_scale_CSI_new:{}'.format(small_scale_CSI_new))
    print('small_scale_CSI:{}'.format(small_scale_CSI))
    # print('final_CSI:{}'.format(final_CSI))
    print('final_CSI_new:{}'.format(final_CSI_new))
    return final_CSI_new

def Delay_generate(general_para, distances): ##单天线 这里要改成对应的星地信道
    # Nt = general_para.N_antennas
    L = general_para.n_receiver
    satellite_num = general_para.satellite_num
    user_num = general_para.user_num
    delay = np.random.rand(satellite_num,user_num)
    # for m in range(satellite_num):
    #     for n in range(user_num):
    #         delay[m,n] = distances[m,n] / 1000
    delays_sample = np.random.rand(L,L)*2-1
    eta_sample = np.random.rand(L,L)*2-1
    for i in range(L):
        for j in range(L):
            satellite_from_index = int(i / user_num)
            satellite_to_index = int(j / user_num)
            user_from_index = int(i % user_num)
            user_to_index = int(j % user_num)
            delays_sample[i,j] = delay[satellite_to_index, user_to_index] - delay[satellite_from_index, user_from_index]
    # delays_sample = np.triu(delays_sample, k=1)
    # delays = delays_sample - delays_sample.T
    # large_scale_CSI = 4.4*10**5/((dists**1.88)*(10**(shadowing*6.3/20)))
    # small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))*np.sqrt(large_scale_CSI)
    # print('distance:{}'.format(distances))
    print('delays:{}'.format(delays_sample))
    for i in range(L):
        for j in range(L):
            satellite_from_index = int(i / user_num)
            satellite_to_index = int(j / user_num)
            user_from_index = int(i % user_num)
            user_to_index = int(j % user_num)
            # delays_sample[i,j] = delay[satellite_to_index, user_to_index] - delay[satellite_from_index, user_from_index]
            left_delta_delay = delay[satellite_to_index, user_to_index] - delay[satellite_to_index, user_from_index]
            right_delta_delay = delay[satellite_from_index, user_to_index] - delay[satellite_from_index, user_from_index]
            if i == j:
                eta_sample[i, j] = 0
            else:
                eta_sample[i, j] = calculate_eta(left_delta_delay, right_delta_delay)
    # print('small_scale_CSI:{}'.format(small_scale_CSI))
    return delays_sample, eta_sample

def calculate_eta(Delta_1, Delta_2):
    beta = 0.2
    if Delta_1 == Delta_2:
        return 1-0.5*beta*np.sin(np.pi*Delta_1)*np.sin(np.pi*Delta_2)
    else:
        return np.divide(np.cos(np.pi*(Delta_2-Delta_1)*beta),
                         1-4*beta*beta*(Delta_2-Delta_1)*(Delta_2-Delta_1))*np.sinc(Delta_2-Delta_1) + np.divide(
            beta*np.sin(np.pi*Delta_1)*np.sin(np.pi*Delta_2),
            2*(beta*beta*(Delta_2-Delta_1)*(Delta_2-Delta_1)-1))*np.sinc((Delta_2-Delta_1)*beta)

def Delay_generate_single_connection(general_para, distances): ##单天线 这里要改成对应的星地信道
    # Nt = general_para.N_antennas
    L = general_para.n_receiver
    satellite_num = general_para.satellite_num
    user_num = general_para.user_num
    delay = np.random.rand(satellite_num,user_num)
    # for m in range(satellite_num):
    #     for n in range(user_num):
    #         delay[m,n] = distances[m,n] / 1000
    delays_sample = np.random.rand(L,L)*2-1
    for i in range(L):
        for j in range(L):
            satellite_from_index = int(i / user_num)
            satellite_to_index = int(j / user_num)
            user_from_index = int(i % user_num)
            user_to_index = int(j % user_num)
            delays_sample[i,j] = delay[satellite_to_index, user_to_index] - delay[satellite_from_index, user_from_index]
    # delays_sample = np.triu(delays_sample, k=1)
    # delays = delays_sample - delays_sample.T
    # large_scale_CSI = 4.4*10**5/((dists**1.88)*(10**(shadowing*6.3/20)))
    # small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))*np.sqrt(large_scale_CSI)
    # print('distance:{}'.format(distances))
    print('delays:{}'.format(delays_sample))
    # print('small_scale_CSI:{}'.format(small_scale_CSI))
    return delays_sample

def sample_generate(general_para, number_of_layouts, norm = None):
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    layouts = []
    dists = []
    CSIs = []
    for i in range(number_of_layouts):
        # generate layouts
        layout_tx, layout_rx = layout_generate(general_para)
        n_re = general_para.n_receiver
        dis = distance_generate(general_para,layout_tx,layout_rx)
        csis = CSI_generate_single_connection(general_para, dis)
        
        #data collection
        dists.append(dis)
        CSIs.append(csis)
            
    dists = np.array(dists)
    CSIs = np.array(CSIs)
    return dists, CSIs


def sample_generate_asyn(general_para, number_of_layouts, norm=None):
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    layouts = []
    dists = []
    CSIs = []
    delays = []
    for i in range(number_of_layouts):
        # generate layouts
        layout_tx, layout_rx = layout_generate(general_para)
        n_re = general_para.n_receiver
        dis = distance_generate_new(general_para, layout_tx, layout_rx)
        csis = CSI_generate_new(general_para, dis)
        delay = Delay_generate_single_connection(general_para, dis)
        # data collection
        dists.append(dis)
        CSIs.append(csis)
        delays.append(delay)
    dists = np.array(dists)
    CSIs = np.array(CSIs)
    delays = np.array(delays)
    return dists, CSIs, delays

def sample_generate_all(general_para, number_of_layouts, norm=None):
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    layouts = []
    dists = []
    CSIs = []
    delays = []
    etas = []
    for i in range(number_of_layouts):
        # generate layouts
        layout_tx, layout_rx = layout_generate(general_para)
        n_re = general_para.n_receiver
        dis = distance_generate_new(general_para, layout_tx, layout_rx)
        csis = CSI_generate_new(general_para, dis)
        delay, eta = Delay_generate(general_para, dis)
        # data collection
        dists.append(dis)
        CSIs.append(csis)
        delays.append(delay)
        etas.append(eta)
    dists = np.array(dists)
    CSIs = np.array(CSIs)
    delays = np.array(delays)
    etas = np.array(etas)
    return dists, CSIs, delays, etas
