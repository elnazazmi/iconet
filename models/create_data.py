import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch import Tensor

from scipy import interpolate

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

WS_PATH = 'data/'


def mean_std():
    mean = pd.read_csv(WS_PATH+'data_mean.csv', header=None, squeeze=True, dtype=np.float32).to_numpy()
    mean = Tensor(np.expand_dims(mean, axis=(0))) #, 1)))
    # mean = np.mean(mean)

    std = pd.read_csv(WS_PATH+'data_std.csv', header=None, squeeze=True, dtype=np.float32).to_numpy()
    std = Tensor(np.expand_dims(std, axis=(0))) #, 1)))
    # std = np.std(std)

    return mean, std


def create_sequences(input_path, cell, seq_len):
    X = xr.open_dataset(input_path)
    # X['cosmu0'] = X['cosmu0'].expand_dims(dim={'height': 90}, axis=1)

    X = xr.concat([X['temp'], X['pres'], X['cosmu0'], X['N2O_full'], X['N2O5_full'],
                   X['HO2_full'], X['H2O_full'], X['NO_full'], X['NO3_full'], X['HNO3_full'],
                   X['O3P_full'], X['NO2_full'], X['OH_full'], X['O3_full'], X['O1D_full']], 'var')

    X = X.transpose('ncells', 'height', 'time', 'var')

    if X['time'].shape[0] != 241:
        X = X.values
        X = np.reshape(X, (np.prod([X.shape[:2]]), X.shape[2], X.shape[3]))
        
    else:
        X = X[:,:,1:,:].values
        X = np.reshape(X, (np.prod([X.shape[:2]]), X.shape[2], X.shape[3]))

    # # normalization with mean and standard devieation
    # mean, std = mean_std()
    # input_data = Tensor(X[cell, :, :]).sub(mean).div(std)
    # input_data[:, 2] = Tensor(X[cell, :, 2])
    
    # normalization with mean and standard devieation
    input_data = Tensor(X[cell, :, :])
    
    # interpolation and smoothing of SZA variable in input data
    ts_len = input_data[:, 2].shape[0]
    x_slice = np.linspace(0, ts_len, num=int(ts_len/6), endpoint=True)
    y_slice = input_data[:, 2][::6]
    func = interpolate.interp1d(x_slice, y_slice)
    x_total = np.linspace(0, ts_len, num=ts_len, endpoint=True)
    y_interp = func(x_total)
    input_data[:, 2] = Tensor(y_interp)
    
    # sequences = []
    n_samples = input_data.shape[0] - seq_len
    sequences = torch.zeros((n_samples,seq_len,input_data.shape[1]), dtype=torch.float32)
    labels = torch.zeros((n_samples,input_data.shape[1]), dtype=torch.float32)

    # for cell in range(X.shape[0]):        
    #     input_data = X[cell, :, :]
    #     n_samples = input_data.shape[0] - seq_len
    for i in range(n_samples):
        sequence = input_data[i:i+seq_len, :]
        label = input_data[i+seq_len, :]
        # sequences.append((sequence, label))
        sequences[i, :, :] = torch.unsqueeze(sequence, 0)
        labels[i, :] = torch.unsqueeze(label, 0)
    
    return sequences, labels


class Dataset(torch.utils.data.Dataset):

    def __init__(self, seq_len, input_path):  # (self, sequences):
        'Initialization'       
        # self.sequences = sequences
        self.seq_len = seq_len
        self.input_path = input_path
        self.nf = 1                 # number of files 365
        self.np = 4                 # number of file parts 256
        self.nc = 80                # number of cells at each part 90 * 80
        # self.ns = (240 - seq_len)  # number of time steps - sequence length
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.nf * self.np * self.nc  # * self.ns #len(self.sequences)

    def __getitem__(self, idx):
        'Generates one sample of data'
        file_idx = idx // (self.np * self.nc)  # * self.ns)
        part_idx = (idx - (file_idx * self.np * self.nc)) // self.nc  # * self.ns)) // (self.nc * self.ns)
        cell_idx = idx % self.nc  # (idx - (((file_idx * self.np) + part_idx) * self.nc * self.ns)) // self.ns

        input_path = self.input_path + 'iconart_DOM01_ML_{:04d}_{:04d}.nc'.format((file_idx + 1), part_idx + 1)
        sequences, labels = create_sequences(input_path, cell_idx, self.seq_len)
        
        return dict(
            sequence=sequences,
            label=labels
        )


class ValDataset(torch.utils.data.Dataset):

    def __init__(self, seq_len, input_path): #(self, sequences):
        'Initialization'
        self.seq_len = seq_len
        self.input_path = input_path
        self.nf = 1                 # number of files 365
        self.np = 1                 # number of file parts 256
        self.nc = 30                # number of cells at each part 90 * 80
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.nf * self.np * self.nc

    def __getitem__(self, idx):
        'Generates one sample of data'
        file_idx = idx // (self.np * self.nc)
        part_idx = (idx - (file_idx * self.np * self.nc)) // self.nc
        cell_idx = idx % self.nc

        # input_path = self.input_path + 'iconart_DOM01_ML_{:04d}_{:04d}.nc'.format((file_idx+1)*15, part_idx+3)
        # sequences, labels = create_sequences(input_path, (cell_idx*90)+cell_idx+20, self.seq_len)
        input_path = self.input_path + 'iconart_DOM01_ML_{:04d}_{:04d}.nc'.format((file_idx + 1), part_idx + 3)
        sequences, labels = create_sequences(input_path, cell_idx, self.seq_len)
        
        return dict(
            sequence=sequences,
            label=labels
        )


def read_data(path):
    X = xr.open_dataset(path)
    X['cosmu0'] = X['cosmu0'].expand_dims(dim={'height': 90}, axis=1)

    X = xr.concat([X['temp'], X['pres'], X['cosmu0'], X['N2O_full'], X['N2O5_full'],
                   X['HO2_full'], X['H2O_full'], X['NO_full'], X['NO3_full'], X['HNO3_full'],
                   X['O3P_full'], X['NO2_full'], X['OH_full'], X['O3_full'], X['O1D_full']], 'var')

    X = X.transpose('ncells', 'height', 'time', 'var')

    if X['time'].shape[0] != 241:
        X = X.values
        X = np.reshape(X, (np.prod([X.shape[:2]]), X.shape[2], X.shape[3]))
    else:
        X = X[:,:,1:,:].values
        X = np.reshape(X, (np.prod([X.shape[:2]]), X.shape[2], X.shape[3]))

    # normalization with mean and standard devieation
    mean, std = mean_std()
    X_ = Tensor(X).sub(mean).div(std)
    X_[:, :, 2] = Tensor(X[:, :, 2])
        
    # interpolation and smoothing of SZA variable in input data
    for cell in range(X_[:, :, 2].shape[0]):
        ts_len = X_[cell, :, 2].shape[0]
        x_slice = np.linspace(0, ts_len, num=int(ts_len/6), endpoint=True)
        y_slice = X_[cell, :, 2][::6]
        func = interpolate.interp1d(x_slice, y_slice)
        x_total = np.linspace(0, ts_len, num=ts_len, endpoint=True)
        y_interp = func(x_total)
        X_[cell, :, 2] = Tensor(y_interp)

    return X_


def read_data_single_height(path, hei):
    data = xr.open_dataset(path)
    data['cosmu0'] = data['cosmu0'].expand_dims(dim={'height': 90}, axis=1)

    dat = xr.concat([data['temp'], data['pres'], data['cosmu0'], data['N2O_full'], data['N2O5_full'],
                   data['HO2_full'], data['H2O_full'], data['NO_full'], data['NO3_full'], data['HNO3_full'],
                   data['O3P_full'], data['NO2_full'], data['OH_full'], data['O3_full'], data['O1D_full']], 'var')

    dat = dat.transpose('ncells', 'height', 'time', 'var')
    dat = dat[:, hei, :, :]
    
    if dat['time'].shape[0] != 241:
        X = dat.values
    else:
        X = dat[:,1:,:].values

    # normalization with mean and standard devieation
    mean, std = mean_std()
    X_ = Tensor(X).sub(mean).div(std)
    X_[:, :, 2] = Tensor(X[:, :, 2])
        
    # interpolation and smoothing of SZA variable in input data
    for cell in range(X_[:, :, 2].shape[0]):
        ts_len = X_[cell, :, 2].shape[0]
        x_slice = np.linspace(0, ts_len, num=int(ts_len/6), endpoint=True)
        y_slice = X_[cell, :, 2][::6]
        func = interpolate.interp1d(x_slice, y_slice)
        x_total = np.linspace(0, ts_len, num=ts_len, endpoint=True)
        y_interp = func(x_total)
        X_[cell, :, 2] = Tensor(y_interp)

    return data, X_
