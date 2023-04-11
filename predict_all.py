import numpy as np
import torch
from models.create_data import *
from models.iconet import *
import time

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


def run_predict():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use all the available GPUs
    model = ICONET.load_from_checkpoint('lightning_logs/version_1/checkpoints/epoch=3500-step=1470419.ckpt').to(dev)
    
    print('model=', model)
        
    for fid in range(0, 1):
        input_pred = 'data/iconart_DOM01_ML_0366_{:04d}.nc'.format(fid+1)
        # input_pred = 'data/iconart_DOM01_ML_0030_0001.nc'
        # input_pnex = 'data/iconart_DOM01_ML_0367_0001.nc'
        # input_pmex = 'data/iconart_DOM01_ML_0368_0001.nc'
        
        star = time.time()
        X = read_data(input_pred).to(dev)
        # Y = read_data(input_pnex).to(dev)
        # W = read_data(input_pmex).to(dev)
        # X = torch.cat((Z, Y, W), 1)
        endr = time.time()
        print('----------> elapsed time in read input ', fid, ' =', endr-star, flush=True)

        # mean, std = mean_std()

        n_steps_past = 10
        X_ = torch.torch.empty_like(X[:, :n_steps_past, :]).copy_(X[:, :n_steps_past, :])  # copy tensor
        n_samples = X.shape[1] - n_steps_past

        pred = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        gron = np.zeros((X.shape[0], X.shape[1], X.shape[2]))

        for i in range(n_samples):
            X_[:, :, :3] = X[:, i:i + n_steps_past, :3]
            # X_ = X[:, i:i + n_steps_past, :]
            stal = time.time()
            out = model(X_)
            endl = time.time()
            print('elapsed time in prediction timestep ', i, ' =', endl-stal, flush=True)
            out_ = torch.unsqueeze(out[1], 1)
            X_ = torch.cat((X_[:, 1:, :], out_), 1)

            pred[:, i + n_steps_past, :] = out[1].detach().cpu().numpy()  #.mul(std).add(mean)
            gron[:, i + n_steps_past, :] = X[:, i + n_steps_past, :].detach().cpu().numpy()  #.mul(std).add(mean)

        pred[:, :n_steps_past, :] = X[:, :n_steps_past, :].detach().cpu().numpy()  #.mul(std).add(mean)
        gron[:, :n_steps_past, :] = X[:, :n_steps_past, :].detach().cpu().numpy()  #.mul(std).add(mean)
        
        
        np.save('data/pred_l1_s10_h15_d00_e3500_iconart_ml_0366_{:04d}'.format(fid+1), pred)
        np.save('data/gron_l1_s10_h15_d00_e3500_iconart_ml_0366_{:04d}'.format(fid+1), gron)
        
    return

    
def run_predict_single_height():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use all the available GPUs
    model = ICONET.load_from_checkpoint('lightning_logs/version_1/checkpoints/epoch=3500-step=1470419.ckpt').to(dev)

    print('model=', model)
        
    for fid in range(0, 1):
        input_pred = 'data/iconart_DOM01_ML_0366.nc'
        
        height = 50
        star = time.time()
        data, X = read_data_single_height(input_pred, height)
        X = X.to(dev)
        endr = time.time()
        print('----------> elapsed time in read input ', fid, ' =', endr-star, flush=True)

        mean, std = mean_std()

        n_steps_past = 10
        X_ = torch.torch.empty_like(X[:, :n_steps_past, :]).copy_(X[:, :n_steps_past, :])  # copy tensor
        n_samples = X.shape[1] - n_steps_past

        pred = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        gron = np.zeros((X.shape[0], X.shape[1], X.shape[2]))

        for i in range(n_samples):
            X_[:, :, :3] = X[:, i:i + n_steps_past, :3]
            # X_ = X[:, i:i + n_steps_past, :]
            stal = time.time()
            out = model(X_)
            endl = time.time()
            print('elapsed time in prediction timestep ', i, ' =', endl-stal, flush=True)
            out_ = torch.unsqueeze(out[1], 1)
            X_ = torch.cat((X_[:, 1:, :], out_), 1)

            pred[:, i + n_steps_past, :] = out[1].detach().cpu().mul(std).add(mean).numpy()  #.mul(std).add(mean)
            gron[:, i + n_steps_past, :] = X[:, i + n_steps_past, :].detach().cpu().mul(std).add(mean).numpy()  #.mul(std).add(mean)

        pred[:, :n_steps_past, :] = X[:, :n_steps_past, :].detach().cpu().mul(std).add(mean).numpy()  #.mul(std).add(mean)
        gron[:, :n_steps_past, :] = X[:, :n_steps_past, :].detach().cpu().mul(std).add(mean).numpy()  #.mul(std).add(mean)
        
        
        np.save('data/pred_alcels_40mix_interp_l1_s10_h15_d00_e3500_hei50_iconart_ml_0366', pred) ##.format(fid+1), pred)
        np.save('data/gron_alcels_40mix_interp_l1_s10_h15_d00_e3500_hei50_iconart_ml_0366', gron) ##.format(fid+1), gron)
        
        forecast = data.copy(deep=True)
        sel_vars = ['temp','pres','cosmu0','N2O_full','N2O5_full','HO2_full','H2O_full','NO_full','NO3_full','HNO3_full','O3P_full','NO2_full','OH_full','O3_full','O1D_full']
        j = 0
        for v in sel_vars:
            forecast[v][:, height, :] = pred[:, :, j].transpose()
            j = j+1
        
        xr.save_mfdataset([data, forecast], ['data/data.nc', 'data/forecast.nc'])
        
    return


sta = time.time()
run_predict()
# run_predict_single_height()
end = time.time()
print('---> elapsed time total =', end-sta, flush=True)
