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
                                                      
    input_pred = 'data/iconart_DOM01_ML_0520_0001.nc'
    #input_pnex = 'data/iconart_DOM01_ML_0367_0001.nc'
    #input_pmex = 'data/iconart_DOM01_ML_0368_0001.nc'
    
    X = read_data(input_pred).to(dev)
    #Y = read_data(input_pnex).to(dev)
    #W = read_data(input_pmex).to(dev)
    #X = torch.cat((Z, Y), 1)
    print(X.shape)
    mean, std = mean_std()

    n_steps_past = 30
    cell = 0
    X = X[cell, :, :]
    X_ = torch.unsqueeze(X[:n_steps_past, :], 0)
    n_samples = X.shape[0] - n_steps_past
    
    pred = np.zeros((X.shape[0], X.shape[1]))
    gron = np.zeros((X.shape[0], X.shape[1]))

    for i in range(n_samples):
        X_[:, :, :3] = torch.unsqueeze(X[i:i + n_steps_past, :3], 0)
        out = model(X_)
        out_ = torch.unsqueeze(out[1], 0)
        X_ = torch.cat((X_[:, 1:, :], out_), 1)
        
        pred[i + n_steps_past] = out[1].detach().cpu().numpy() #.mul(std).add(mean)
        gron[i + n_steps_past] = X[i + n_steps_past, :].detach().cpu().numpy() #.mul(std).add(mean)

    pred[:n_steps_past, :] = X[:n_steps_past, :].detach().cpu().numpy()
    gron[:n_steps_past, :] = X[:n_steps_past, :].detach().cpu().numpy()
    
    np.save('data/pred_iconart_ml_0520_0001', pred)
    np.save('data/gron_iconart_ml_0520_0001', gron)
        
    return


sta = time.time()
run_predict()
end = time.time()
print('elapsed time =', end-sta)

