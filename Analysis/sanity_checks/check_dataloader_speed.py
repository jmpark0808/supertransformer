import argparse
import datetime
import os
import random
import time
from re import X
import git
import sys
sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
import pytorch_lightning as pl

# Callbacks and loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger



# Import dataset modules
from dataset.superpixel import DUTSDataModule,  SPDataModule
from dataset.superpixel_pyg import SPGDataModule
from dataset.superpixel_pyg_image import SPGIDataModule
from dataset.superpixel_fast import SPFDataModule
from dataset.superpixel_fast_cnn import SPFCDataModule
from dataset.youtube_davis import YDDataModule
from dataset.youtube_davis_swin_pyg import YDGDataModule
from dataset.superpixel_pyg_swin import SPGSWINDataModule


# 
DATALOADER_DIRECTORY = {
    'SP': SPDataModule,
    'DUTS': DUTSDataModule,
    'SPCNN': SPDataModule,
    'SPLAP': SPDataModule,
    "SPFFT": SPDataModule,
    'SPContour': SPDataModule,
    'SPGFFT': SPGDataModule,
    'SPG': SPGDataModule,
    'SPGIFFT': SPGIDataModule,
    'SPGI': SPGIDataModule,
    'SPF': SPFDataModule,
    'SPFFFT': SPFDataModule,
    'SPFC': SPFCDataModule,
    'YD': YDDataModule,
    'YDG':  YDGDataModule,
    'SPGSWIN': SPGSWINDataModule

} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)

    parser.add_argument('--dataset_tr', help='Directory of your train Dataset', required=True, default=None)
    parser.add_argument('--dataset_test', help='Directory of your test Dataset', default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpus', help="Number of gpus to use for training", default=0, type=int)
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('--num_workers', help="# of dataloader cpu process", default=0, type=int)
    parser.add_argument('--val_freq', help='How often to run validation set within a training epoch, i.e. 0.25 will run 4 validation runs in 1 training epoch', default=1.0, type=float)
    parser.add_argument('--es_patience', help='Max # of consecutive validation runs w/o improvment', default=5, type=int)
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.0001', default=0.0001, type=float)
    parser.add_argument('--num_seg', help='Approximate number of segmentations', default=600, type=int)
    parser.add_argument('--dropout', help='Dropout for Transformers', default=0., type=float)
    parser.add_argument('--dropout_edge', help='Dropout for dropout_edge', default=0., type=float)
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--compactness', help='Compactness for SLIC', type=float, default=10)
    parser.add_argument('--size', help='Image size for DUTS', type=int, default=224)
    parser.add_argument('--coeff', help='Number of coefficients for fft', type=int, default=7)
    parser.add_argument('--dilation', help='Dilation for local transformer', type=int, default=5)
    parser.add_argument('--downsample', help='Downsample resolution', type=int, default=28)
    parser.add_argument('--tag', help='Tag for differentiating runs on CC', default='', type=str)
    parser.add_argument('--tfmhp', default=[8, 16, 6, 128], 
                    nargs=4, metavar=('Heads', 'Head Dim', 'Number of Layers', 'Embed dim'),
                    type=int, help='Hyperparameters for Transformer')
    parser.add_argument('--ignore_phase', help='Whether or not to use phase of FFT'
                        , default=False, action="store_true")
    parser.add_argument('--force_aug', help='Force data augmentation on select dataloaders'
                        , default=False, action="store_true")
    parser.add_argument('--sigma', help='Sigma for feature augmentation', default=0.04, type=float)
    parser.add_argument('--debug', help='Whether or not to switch to debug mode, only runs on 100 samples'
                        , default=False, action="store_true")
    parser.add_argument('--dilation_mode', help='Dilation mode, 0 for fully connected, 1 for spotted global', default=0, type=int)
    parser.add_argument('--gunet_mode', help='GUnet mode, graclus or predefined pooling', default='graclus', type=str)
    parser.add_argument('--fully_connected', help='Use fully connected neighbourhood'
                        , default=False, action="store_true")
    parser.add_argument('--kernels', default=[32, 32, 32, 32], 
                    nargs="*", 
                    type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    parser.add_argument('--window_size', help='Window size for SWIN Transformer', type=int, default=4)
    parser.add_argument('--memory', help='Whether to put the data into memory'
                        , default=False, action="store_true")
    
    


    import torch 
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    dict_args = vars(args)
    
    
    pl.seed_everything(dict_args['seed'], True)
    # Initialize model to train
  
   
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)

    # Trainer: initialize training behaviour
   
    end = time.time()
    times = []
    for batch in data_module.train_dataloader():
        times.append(time.time()-end)
        end = time.time()

    import numpy as np
    print(np.mean(times))

