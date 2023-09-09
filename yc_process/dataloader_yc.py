import os
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from scipy import signal
import scipy.io as scio
from yacs.config import CfgNode as CN
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Ninapro(Dataset):
    def __init__(self, emgs, labels) -> None:
        super().__init__()
        self.emgs = emgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.emgs[index]
        sample = torch.tensor(sample).float() #（length,channel）
        if self.exercise == 1:
            label = self.labels[index]-1
        elif self.exercise==2:
            label= self.labels[index]-18
        label = torch.tensor(label).long()
        return sample, label
    
class Ninapro_regress(Dataset):
    def __init__(self, emgs, labels) -> None:
        super().__init__()
        self.emgs = emgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.emgs[index]
        sample = torch.tensor(sample).float() #（length,channel）

        label = self.labels[index]-1

        label = torch.tensor(label)
        return sample, label
    

def low_butterworth_filter(x, fs, order=5):


    b, a = signal.butter(order, btype='low', analog=False,output='ba', fs=fs,Wn=500)
    y = signal.filtfilt(b, a, x)
    return y



def low_filter_3ch(emgs,fs):
    # init 3 array that have the same shape with emgs
    emgs_bf1 = np.zeros((emgs.shape[0], emgs.shape[1]))
    emgs_bf2 = np.zeros((emgs.shape[0], emgs.shape[1]))
    emgs_bf3 = np.zeros((emgs.shape[0], emgs.shape[1]))

    # low pass filter with 3 order

    for i in range(emgs.shape[1]):
        emgs_bf1[:, i] = low_butterworth_filter(emgs[:, i], fs, 1)
        emgs_bf2[:, i] = low_butterworth_filter(emgs[:, i], fs, 3)
        emgs_bf3[:, i] = low_butterworth_filter(emgs[:, i], fs, 5)
    # concatenate the 3 filtered emgs
    res = np.concatenate((emgs_bf1, emgs_bf2, emgs_bf3), axis=1)
    return res
def RMS(x,window_size,step):
    feature=np.zeros([(x.shape[0]-window_size)//step,x.shape[1]])
    for i in range(x.shape[1]):
        for j in range(x.shape[0]-window_size,step):
            if j+window_size>x.shape[0]:
                break
            feature[j,i]=np.sqrt(np.mean(x[j:j+window_size,i]**2))
    return feature

def get_dataloader_db2_e1(cfg):
    seq_lens = cfg.seq_lens
    step = cfg.step
    data_path1 = r'D:\Datasets\NinaproDataset\DB2\S1_E1_A1.mat'
    EMGData1 = scio.loadmat(data_path1)
    emgs = EMGData1['emg']
    labels = EMGData1['restimulus']
    rerepetition = EMGData1['rerepetition']
    emgs=low_filter_3ch(emgs,2000)

    u = 256
    for i in range(emgs.shape[1]):
        emgs[:, i] = np.sign(emgs[:, i]) * np.log(1+u*abs(emgs[:, i]))/np.log(1+u)
    length_dots = len(emgs)
    print(emgs.shape)
    print(labels.shape)
    print(np.amax(labels))

    data_train = []
    labels_train = []
    data_val = []
    labels_val = []

    for seq_len in seq_lens:
        for idx in range(0, length_dots - length_dots % seq_len, step):
            if idx + seq_len > length_dots:
                break
            if labels[idx] > 0 and labels[idx + seq_len - 1] > 0 and labels[idx] == labels[idx + seq_len - 1]:
                repetition = rerepetition[idx]
                if repetition in [2, 5]:  # val dataset
                    data_val.append(emgs[idx:idx + seq_len, :])
                    labels_val.append(labels[idx])
                else:  # train dataset #[1,3,4,6]
                    data_train.append(emgs[idx:idx + seq_len, :])
                    labels_train.append(labels[idx])
    
    trainset = Ninapro_regress(data_train, labels_train)
    valset = Ninapro_regress(data_val, labels_val)
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True,drop_last=True)
    val_loader = DataLoader(valset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False,drop_last=True)
    return train_loader, val_loader


def get_dataloader_db2(cfg,exercise):
    seq_lens = cfg.seq_lens
    step = cfg.step

    data_path1 = r'D:\Datasets\NinaproDataset\DB2\S1_A1_E1.mat'
    data_path2 = r'D:\Datasets\NinaproDataset\DB2\S1_A1_E2.mat'
    data_path3 = r'D:\Datasets\NinaproDataset\DB2\S1_A1_E3.mat'
    EMGData1 = scio.loadmat(data_path1)
    EMGData2 = scio.loadmat(data_path2)
    EMGData3 = scio.loadmat(data_path3)

    # extract emg, restimulus, and rerepetition
    # and stack together
    emg1 = EMGData1['emg']
    restimulus1 = EMGData1['restimulus']
    rerepetition1 = EMGData1['rerepetition']
    emg2 = EMGData2['emg']
    restimulus2 = EMGData2['restimulus']
    restimulus2 = restimulus2 + restimulus1.max() * (restimulus2>0).astype('int')
    rerepetition2 = EMGData2['rerepetition']
    emg3 = EMGData3['emg']
    restimulus3 = EMGData3['restimulus']
    restimulus3 = restimulus3 + restimulus2.max() * (restimulus3>0).astype('int')
    rerepetition3 = EMGData3['rerepetition']
    print(emg1.shape, emg2.shape, emg3.shape)
    print(restimulus1.shape, restimulus2.shape, restimulus3.shape)
    print(rerepetition1.shape, rerepetition2.shape, rerepetition3.shape)
    emgs = np.vstack([emg1,emg2,emg3])
    labels = np.vstack([restimulus1,restimulus2,restimulus3])
    rerepetition = np.vstack([rerepetition1,rerepetition2,rerepetition3])
    # print(emgs.shape)
    # print(labels.shape)
    # print(rerepetition.shape)
    length_dots = len(emgs)
    print(emgs.shape)
    print(labels.shape)
    data_train = []
    labels_train = []
    data_val = []
    labels_val = []
    for seq_len in seq_lens:
        for idx in range(0, length_dots - length_dots % seq_len, step):
            if idx + seq_len > length_dots:
                break
            if labels[idx] > 0 and labels[idx + seq_len - 1] > 0 and labels[idx] == labels[idx + seq_len - 1]:
                repetition = rerepetition[idx]
                if repetition in [2, 5]:  # val dataset
                    data_val.append(emgs[idx:idx + seq_len, :])
                    labels_val.append(labels[idx])
                else:  # train dataset #[1,3,4,6]
                    data_train.append(emgs[idx:idx + seq_len, :])
                    labels_train.append(labels[idx])


    trainset = Ninapro_regress(data_train, labels_train,exercise)
    valset = Ninapro_regress(data_val, labels_val,exercise)
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True,drop_last=True)
    val_loader = DataLoader(valset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False,drop_last=True)
    return train_loader, val_loader


def get_dataloader_db2_regress(cfg):
    seq_lens = cfg.seq_lens
    step = cfg.step

    data_path1 = r'D:\Datasets\NinaproDataset\DB2\S1_E1_A1.mat'
    data_path2 = r'D:\Datasets\NinaproDataset\DB2\S1_E2_A1.mat'
    data_path3 = r'D:\Datasets\NinaproDataset\DB2\S1_E3_A1.mat'
    EMGData1 = scio.loadmat(data_path1)
    EMGData2 = scio.loadmat(data_path2)
    EMGData3 = scio.loadmat(data_path3)

    # extract emg, restimulus, and rerepetition
    # and stack together
    emg1 = EMGData1['emg']
    restimulus1 = EMGData1['restimulus']
    rerepetition1 = EMGData1['rerepetition']
    emg2 = EMGData2['emg']
    restimulus2 = EMGData2['restimulus']
    restimulus2 = restimulus2 + restimulus1.max() * (restimulus2>0).astype('int')
    rerepetition2 = EMGData2['rerepetition']
    emg3 = EMGData3['emg']
    restimulus3 = EMGData3['restimulus']
    restimulus3 = restimulus3 + restimulus2.max() * (restimulus3>0).astype('int')
    rerepetition3 = EMGData3['rerepetition']
    # print(emg1.shape, emg2.shape, emg3.shape)
    # print(restimulus1.shape, restimulus2.shape, restimulus3.shape)
    # print(rerepetition1.shape, rerepetition2.shape, rerepetition3.shape)
    emgs = np.vstack([emg1,emg2,emg3])
    labels = np.vstack([restimulus1,restimulus2,restimulus3])
    rerepetition = np.vstack([rerepetition1,rerepetition2,rerepetition3])
    # emgs=low_filter_3ch(emgs,2000)

    # u = 256
    #for i in range(emgs.shape[1]):
    #    emgs[:, i] = np.sign(emgs[:, i]) * np.log(1+u*abs(emgs[:, i]))/np.log(1+u)
    length_dots = len(emgs)
    print(emgs.shape)
    print(labels.shape)
    data_train = []
    labels_train = []
    data_val = []
    labels_val = []
    for seq_len in seq_lens:
        for idx in range(0, length_dots - length_dots % seq_len, step):
            if idx + seq_len > length_dots:
                break
            if labels[idx] in [18,19,22,25,27,37] and labels[idx + seq_len - 1] > 0 and labels[idx] == labels[idx + seq_len - 1]:
                repetition = rerepetition[idx]
                if repetition in [2, 5]:  # val dataset
                    data_val.append(emgs[idx:idx + seq_len, :])
                    labels_val.append(labels[idx])
                else:  # train dataset #[1,3,4,6]
                    data_train.append(emgs[idx:idx + seq_len, :])
                    labels_train.append(labels[idx])
    trainset = Ninapro_regress(data_train, labels_train)
    valset = Ninapro_regress(data_val, labels_val)
    train_loader = DataLoaderX(trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True,drop_last=True)
    val_loader = DataLoaderX(valset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False,drop_last=True)
    return train_loader, val_loader

def main():
    with open('cfg/db2.yaml') as cfg_file:
        cfg = CN.load_cfg(cfg_file)
        print('Successfully loading the config file...')
        dataCfg = cfg['DatasetConfig']
        #paths_s = glob.glob(os.path.join(dataCfg.root_path, 'DB2_s1'))
        train_loader, val_loader = get_dataloader_db2_e1(dataCfg)
        print('Successfully get dataloader of Ninapro dataset...')
        return train_loader, val_loader

if __name__ == "__main__":
    main()


