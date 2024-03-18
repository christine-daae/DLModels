# zhe shujuji xianghua ma ??? haiyao laozi shoudong suan sampling rate!!!!! xianghua ma???????? paper limian buxie?!!! a?????
import numpy as np

# data_path = "/data/Datasets/MW/MWEEG_Subject_"
# data = np.load(data_path)
#
# # 查看文件中的所有数组
# for key in data.files:
#     print(f'{key}: {data[key].shape}')

fs = 256
window_size = 2

def Reshape(data):
    re_data = data
    zero = np.zeros(512).tolist()
    out = [[zero, re_data[0], re_data[1], re_data[2], zero],
           [re_data[3], re_data[4], zero, re_data[5], re_data[6]],
           [re_data[7], re_data[8], zero, re_data[9], re_data[10]],
           [zero, re_data[11], zero, re_data[12], zero],
           [zero, re_data[13], re_data[14], re_data[15], zero]]
    return out

def Get_data(id):
    dataset = []
    labels = []
    time_stamp = []
    d_path = "/data/Datasets/MW/MWEEG_Subject_" + str(id) + '.npz'
    print(d_path)
    data = np.load(d_path)
    EEGdata = data["EEG"]
    trigger_value = data["TriggerValues"]
    sample_time = data["SampleTime"]
    trigger_time = data["TriggerTime"]
    print("trigger_value", trigger_value)
    print("sample time", sample_time)
    print("trigger_time", trigger_time)

    for i in range(len(trigger_time)):
        if trigger_time[i] > 0.1:
            time_stamp.append(i)

    print("len", len(time_stamp))
    for i in range(3, (len(time_stamp))):
        ii = i - 3
        low_t = time_stamp[i - 1]
        upper_t = time_stamp[i]
        tick = -1
        if trigger_value[ii] - 1 < 0.1:
            tick = 1
        elif trigger_value[ii] - 2 < 0.1:
            tick = 0
        win_len = fs * window_size
        if (tick != -1):
            while (low_t + win_len <= upper_t):
                t1 = low_t
                t2 = low_t + win_len
                epo_data = EEGdata[:, t1: t2]
                epo_data = Reshape(epo_data)
                dataset.append(epo_data)
                labels.append(tick)
                low_t = low_t + win_len

    dataset = np.array(dataset)
    labels = np.array(labels)
    print(dataset.shape)
    dataset = np.expand_dims(dataset, 4)
    print(dataset.shape)
    return dataset, labels




