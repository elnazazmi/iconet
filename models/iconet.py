import os
import torch
import pytorch_lightning as pl
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
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')  #1e-3
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1') #0.6
parser.add_argument('--beta_2', type=float, default=0.999, help='decay rate 2') #0.3
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--seq_len', type=int, default=10, help='sequence length')
parser.add_argument('--n_gpus', type=int, default=0, help='number of GPUs')

opt = parser.parse_args()


class ICONET(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(ICONET, self).__init__()

        # call this to save hyperparameters like model layres and learning_rate to the checkpoint
        self.save_hyperparameters()
        
        # default config
        self.path = os.getcwd() + 'data'
        self.model = model

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = opt.batch_size
        self.lr = opt.lr
        self.input_path = 'data/'
        # self.cell = 3
        self.seq_len = opt.seq_len

    def forward(self, x, labels=None):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use all the available GPUs

        #x = x.to(dev)
        #output = self.model(x)
        sequences = x['sequence'].squeeze(0).to(dev)
        labels = x['label'].squeeze(0).to(dev)
        output = self.model(sequences)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)

        return loss, output
    
    def training_step(self, batch, batch_idx):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use all the available GPUs

        # sequences = batch['sequence'].squeeze(0)
        labels = batch['label'].squeeze(0)

        # loss, outputs = self(sequences, labels)
        loss, outputs = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # # save learning_rate
        # lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        # lr_saved = torch.scalar_tensor(lr_saved).to(dev)
        # self.log('learning_rate', lr_saved, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.current_epoch == 3500:
        # if self.global_step % 2000 == 0:
            np.save('output/pd_epoch_' + str(self.current_epoch) + '_step_' + str(self.global_step), outputs.detach().cpu().numpy())  # prediction
            np.save('output/gt_epoch_' + str(self.current_epoch) + '_step_' + str(self.global_step), labels.detach().cpu().numpy())  # groundtruth

        return (loss, outputs) #{'loss': loss}
    
    def training_step_end(self, training_step_outputs):
        #print('training_step_outputs=', training_step_outputs)
        gpu_0_pred = training_step_outputs[0]
        # gpu_1_pred = training_step_outputs[1]["pred"]
        # gpu_n_pred = training_step_outputs[n]["pred"]

        # # this softmax now uses the full batch
        # loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
        return gpu_0_pred #loss

    def validation_step(self, batch, batch_idx):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use all the available GPUs
        
        # sequences = batch['sequence'].squeeze(0)
        labels = batch['label'].squeeze(0)
    
        # loss, outputs = self(sequences, labels)
        loss, outputs = self(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.current_epoch == 3500:
        # if self.global_step % 2000 == 0:
            np.save('output/pdval_epoch_' + str(self.current_epoch) + '_step_' + str(self.global_step), outputs.detach().cpu().numpy())  # prediction
            np.save('output/gtval_epoch_' + str(self.current_epoch) + '_step_' + str(self.global_step), labels.detach().cpu().numpy())  # groundtruth
    
        return loss

    def test_step(self, batch, batch_idx):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use all the available GPUs
        sequences = batch['sequence']
        labels = batch['label']

        loss, outputs = self(sequences, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(opt.beta_1, opt.beta_2))
#         lr_scheduler = {
#            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1),
#            'name': 'learning_rate'
#         }

#         return [optimizer], [lr_scheduler]
        #return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(opt.beta_1, opt.beta_2), eps=1e-8, weight_decay=0, amsgrad=True)
        
        return optimizer
    
    def train_dataloader(self):
        train_data = Dataset(self.seq_len, self.input_path)
        print('length of train_data: ', len(train_data))
        print('size of train_data: ', sys.getsizeof(train_data))
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            num_workers=(4*opt.n_gpus),  # good rule of thumb is: 4 * num_GPU
            batch_size=self.batch_size,
            shuffle=True)

        return train_loader

    def val_dataloader(self):
        val_data = ValDataset(self.seq_len, self.input_path)
    
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            num_workers=(4*opt.n_gpus),
            batch_size=self.batch_size,
            shuffle=False)
    
        return val_loader

    def test_dataloader(self):
        sequences = create_sequences(self.input_path, self.cell, self.seq_len)
        test_data = Dataset(sequences)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            num_workers=(4*opt.n_gpus),
            batch_size=self.batch_size,
            shuffle=False)

        return test_loader
 
