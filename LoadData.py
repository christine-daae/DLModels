# load data

import numpy as np
import scipy.io

data_path = '/data/Datasets/EEG_01-29_MATLAB/'
subject_list = ['VP001-EEG', 'VP002-EEG', 'VP003-EEG', 'VP004-EEG', 'VP005-EEG', 'VP006-EEG', 'VP007-EEG', 'VP008-EEG', 'VP009-EEG', 'VP010-EEG', 'VP011-EEG',
                'VP012-EEG', 'VP013-EEG', 'VP014-EEG', 'VP015-EEG', 'VP016-EEG', 'VP017-EEG', 'VP018-EEG', 'VP019-EEG', 'VP020-EEG', 'VP021-EEG', 'VP022-EEG',
                'VP023-EEG', 'VP024-EEG', 'VP025-EEG', 'VP026-EEG', 'VP027-EEG', 'VP028-EEG', 'VP029-EEG']

# 初始化数据集和标签
dataset = []
labels = []

# 定义时间窗口大小（秒）
window_size = 2
b = 1
fs = 200

def Reshape(data):
    re_data = np.array(data)
    re_data = np.transpose(re_data)
    re_data = re_data.tolist()
    zero = np.zeros(400).tolist()
    out = [[zero, zero, zero, re_data[3], zero, re_data[4], zero, zero, zero],
           [zero, zero, re_data[1], zero, re_data[9], zero, re_data[5], zero, zero],
           [zero, zero, zero, re_data[8], zero, re_data[9], zero, zero, zero],
           [zero, re_data[12], zero, re_data[13], zero, re_data[23], zero, re_data[25], zero],
           [re_data[16], zero, re_data[6], zero, re_data[7], zero, re_data[24], zero, re_data[29]],
           [zero, re_data[14], zero, re_data[15], zero, re_data[26], zero, re_data[27], zero],
           [re_data[17], zero, re_data[18], zero, re_data[11], zero, re_data[23], zero, re_data[28]],
           [zero, zero, zero, zero, re_data[21], zero, zero, zero, zero],
           [zero, zero, zero, re_data[20], zero, re_data[21], zero, zero, zero]]
    return out

def LoadData(d_path):
    print("-----------------load data from", d_path, "-----------------------------")
    # 读取cnt.mat文件，这是EEG数据
    cnt_data = scipy.io.loadmat(d_path + 'cnt_dsr.mat')

    # 读取mrk.mat文件，这是标签数据
    mrk_data = scipy.io.loadmat(d_path + 'mrk_dsr.mat')

    # 如果需要，你还可以读取mnt.mat文件，包含电极位置cond信息

    # 提取EEG数据
    eeg_data = cnt_data['cnt_dsr']['x'][0][0]  # 这是EEG信号的多维数组 [T, #channels]

    print("eeg data shape:", eeg_data.shape)
    print("sampling rate:", fs)  # sampling rate is 200Hz, which means it has already been preprocessed

    # 提取标签数据
    time_info = mrk_data['mrk_dsr']['time'][0][0][0]  # 这是标签数据的数组
    return eeg_data, time_info

# 2s with 1s overlap
def epo_cut(eeg_data, time_info):
    for i in range(1, len(time_info)):
        time_interval = (time_info[i] - time_info[i - 1]) / 1000

        if time_interval > 10:
            end_idx = int((time_info[i] * 200) / 1000) - int(b * fs)
            for _ in range(1, 21):
                start_idx = end_idx - int(window_size * fs)
                non_att_data = eeg_data[start_idx:end_idx, :]
                non_att_data = Reshape(non_att_data)
                dataset.append(non_att_data)
                labels.append(0)
                # 1s overlap
                end_idx = end_idx - int(window_size / 2 * fs)

        if i > 1 and time_interval > 10:
            end_idx = int((time_info[i - 1] * 200) / 1000) - int(b * 5 * fs)
            for _ in range(1, 11):
                start_idx = end_idx - int(window_size * fs)
                att_data = eeg_data[start_idx:end_idx, :]
                att_data = Reshape(att_data)
                dataset.append(att_data)
                labels.append(1)
                # 1s overlap
                end_idx = end_idx - int(window_size / 2 * fs)

    return dataset, labels

def Get_data(ii):
    EEGdata, T_info = LoadData(data_path + subject_list[ii] + '/')
    dataset, labels = epo_cut(EEGdata, T_info)

    # 将数据集和标签转换为 NumPy 数组
    dataset = np.array(dataset)
    labels = np.array(labels)
    dataset = np.expand_dims(dataset, 4)
    print(dataset.shape)
    return dataset, labels
