import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io

def generate_random_dataset(num_samples=400, test_size=0.2, random_state=42):
    # 生成随机特征数据
    X = np.random.rand(num_samples, 128, 64)

    # 生成随机标签数据（假设是二分类任务）
    y = np.random.randint(2, size=num_samples)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


data_path = '/data/Datasets/KULeuven_Dataset/'

subject_list = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
                "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]

def Reshape(data):
    re_data = np.array(data)
    re_data = np.transpose(re_data)
    re_data = re_data.tolist()
    zero = np.zeros(286).tolist()
    out = [[zero, zero, zero, re_data[0], re_data[32], re_data[33], zero, zero, zero],
           [zero, zero, re_data[1], re_data[2], re_data[36], re_data[35], re_data[34], zero, zero],
           [re_data[6], re_data[5], re_data[4], re_data[3], re_data[37], re_data[38], re_data[39], re_data[40],
            re_data[41]],
           [re_data[7], re_data[8], re_data[9], re_data[10], re_data[46], re_data[45], re_data[44], re_data[43],
            re_data[42]],
           [re_data[14], re_data[13], re_data[12], re_data[11], re_data[47], re_data[48], re_data[49], re_data[50],
            re_data[51]],
           [re_data[15], re_data[16], re_data[17], re_data[18], re_data[31], re_data[55], re_data[54], re_data[53],
            re_data[52]],
           [re_data[22], re_data[21], re_data[20], re_data[19], re_data[30], re_data[56], re_data[57], re_data[58],
            re_data[59]],
           [zero, zero, re_data[24], re_data[25], re_data[29], re_data[62], re_data[61], zero, zero],
           [zero, zero, re_data[23], re_data[26], re_data[28], re_data[63], re_data[60], zero, zero]]
    return out

# 初始化数据集和标签
dataset = []
labels = []

# 定义时间窗口大小（秒）
window_size = 2
fs = 128

def LoadData(sub):
    sub_path = data_path + sub
    data = scipy.io.loadmat(sub_path + '.mat')
    trials = data["trials"][0]
    for trial in trials:
        t_field = trial[0, 0]
        tick = -1
        if t_field["attended_ear"][0] == 'L':
            tick = 0
        if t_field["attended_ear"][0] == 'R':
            tick = 1
        EEGdata = t_field["RawData"]["EegData"][0][0]
        ii = 0
        while (ii + fs * window_size) <= len(EEGdata):
            epo_data = EEGdata[ii: ii + fs * window_size, :]
            epo_data = Reshape(epo_data)
            dataset.append(epo_data)
            labels.append(tick)

            ii = ii + fs * window_size

    return dataset, labels


def Get_data(ii):
    dataset, labels = LoadData(subject_list[ii])

    # 将数据集和标签转换为 NumPy 数组
    dataset = np.array(dataset)
    labels = np.array(labels)
    dataset = np.expand_dims(dataset, 4)
    print(dataset.shape)
    return dataset, labels

a, b = Get_data(0)

