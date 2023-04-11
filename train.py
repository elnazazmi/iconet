import torch
from pytorch_lightning import Trainer
from models.multifeature_LSTM import MultiFeatureLSTM
from models.create_data import *
from models.iconet import *
import time
import sys
import argparse

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')   #1e-3
parser.add_argument('--n_features', type=int, default=15, help='number of features')
parser.add_argument('--n_hidden', type=int, default=15, help='number of hidden')
parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')  #3501
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_nodes', type=int, default=1, help='number of GPU nodes for distributed training')
parser.add_argument('--n_gpus', type=int, default=0, help='number of GPUs')

opt = parser.parse_args()


def run_trainer():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use all the available GPUs
    lstm_model = MultiFeatureLSTM(n_features=opt.n_features, n_hidden=opt.n_hidden, n_layers=opt.n_layers).to(dev)
    model = ICONET(model=lstm_model)

    # checkpoint = 'lightning_logs/version_1676406/checkpoints/epoch=8000-step=1920239.ckpt'
    # print('checkpoint path = ', os.path.exists(checkpoint))
    
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = Trainer(max_epochs=opt.epochs,
                      num_nodes=opt.n_nodes,
                      gpus=opt.n_gpus,
                      strategy='ddp',
                      # auto_lr_find=True,
                      log_every_n_steps=1,
                      ## precision=16,  # speed up training
                      ## profile=True,
                      # callbacks=[lr_monitor],
                      # num_sanity_val_steps=0,
                      # Path/URL of the checkpoint from which training is resumed. If there is no checkpoint file at the path, start from scratch
                      # resume_from_checkpoint=None #'lightning_logs/version_1429123/checkpoints/epoch=49-step=749.ckpt' #None
                      # early_stop_callback=False,
                      # use_amp=opt.use_amp
                      )
    
#     # run learning rate finder
#     lr_finder = trainer.tuner.lr_find(model, min_lr=3e-5, max_lr=3e-1, num_training=500)
#     pd.DataFrame.from_dict(lr_finder.results).to_csv('lr_finder_results_3e-5.csv', index=False)
    
#     model.hparams.lr = lr_finder.suggestion()
#     print('model.lr=', model.lr)    
#     model.lr = lr_finder.suggestion()
#     print('model.lr=', model.lr)
    
    trainer.fit(model)
    # trainer.fit(model, ckpt_path=checkpoint)


sta = time.time()
run_trainer()
end = time.time()
print('training elapsed time: ', end-sta)

