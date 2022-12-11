import argparse
import datetime
import os
import random
import time
from re import X

import pytorch_lightning as pl

# Callbacks and loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from Wrappers.SP_CNN_LIN import SP_CNN_LIN_Wrapper
from Wrappers.SP_ETFM_TFM import SP_ETFM_TFM_Wrapper
from Wrappers.SP_GCN import SP_GCN_Wrapper
from Wrappers.SP_GTFM import SP_GTFM_Wrapper
from Wrappers.SP_RTFM_TFM import SP_RTFM_TFM_Wrapper
from Wrappers.SP_TFM import SP_TFM_Wrapper
from Wrappers.SP_TFM_AP import SP_TFM_AP_Wrapper
from Wrappers.SP_TFM_DIL import SP_TFM_DIL_Wrapper
from Wrappers.SP_TFM_NP import SP_TFM_NP_Wrapper
from Wrappers.SP_TFM_PE import SP_TFM_PE_Wrapper
from Wrappers.SP_TFM_TFM import SP_TFM_TFM_Wrapper
from Wrappers.SP_GAT import SP_GAT_Wrapper

# Import dataset modules
from dataset.superpixel import DUTSDataModule,  SPDataModule


# Metric logging

# Deterministic

MODEL_DIRECTORY = {
    # "SPLT": SuperTransformerLightTFM,
    # "SPP": SuperTransformerPos,
    # "SPF": SuperTransformerFCN,
    # "SPSF": SuperTransformerSepFCN,
    # "SPGAT": SuperTransformerGAT,
    # "SPDTNN": SuperTransformerDeepTFMNN,
    # "SPDTBN": SuperTransformerDeepTFMBN,
    # "SPL": SuperLinear,
    # "IT": ImageTransformer,
    # "ITCNN": ImageTransformerCNN,
    # "ITCNNTFM": ImageTransformerCNNTFM,
    # "ITNMP": ImageTransformerNMP,
    # "ITI": ImageLinear,
    # "ITUNET": ImageTransformerUNET,
    # "SC": SuperConvSeg,
    "SP_TFM_DIL": SP_TFM_DIL_Wrapper,
    "SP_RTFM_TFM": SP_RTFM_TFM_Wrapper,
    "SP_CNN_LIN": SP_CNN_LIN_Wrapper,
    "SP_TFM_TFM": SP_TFM_TFM_Wrapper,
    "SP_ETFM_TFM": SP_ETFM_TFM_Wrapper,
    "SP_TFM": SP_TFM_Wrapper,
    "SP_GCN": SP_GCN_Wrapper,
    "SP_TFM_NP": SP_TFM_NP_Wrapper,
    "SP_TFM_AP": SP_TFM_AP_Wrapper,
    "SP_TFM_PE": SP_TFM_PE_Wrapper,
    "SP_GTFM": SP_GTFM_Wrapper,
    "SP_GAT": SP_GAT_Wrapper

}
DATALOADER_DIRECTORY = {
    'SP': SPDataModule,
    'DUTS': DUTSDataModule,
    'SPCNN': SPDataModule,
    'SPE': SPDataModule,
    'SPEmbed': SPDataModule,
    "SPFFT": SPDataModule,
    'SPContour': SPDataModule

} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='Model name to train', required=True, default=None)
    parser.add_argument('--eval', help='Whether to test model on the best iteration after training'
                        , default=False, action="store_true")
    parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)
    parser.add_argument("--load",
                        help="Directory of pre-trained model,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument("--resume_from_checkpoint",
                        help="Directory of pre-trained model,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument('--dataset_tr', help='Directory of your train Dataset', required=True, default=None)
    parser.add_argument('--dataset_val', help='Directory of your validation Dataset', required=True, default=None)
    parser.add_argument('--dataset_test', help='Directory of your test Dataset', default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpus', help="Number of gpus to use for training", default=1, type=int)
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('--num_workers', help="# of dataloader cpu process", default=0, type=int)
    parser.add_argument('--val_freq', help='How often to run validation set within a training epoch, i.e. 0.25 will run 4 validation runs in 1 training epoch', default=0.1, type=float)
    parser.add_argument('--es_patience', help='Max # of consecutive validation runs w/o improvment', default=5, type=int)
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.0001', default=0.0001, type=float)
    parser.add_argument('--num_seg', help='Approximate number of segmentations', default=600, type=int)

    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--size', help='Image size for DUTS', type=int, default=224)


    args = parser.parse_args()
    dict_args = vars(args)
    
    pl.seed_everything(dict_args['seed'])
    # Initialize model to train
    assert dict_args['model'] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[dict_args['model']](**dict_args)
    if dict_args['load']:
        model = model.load_from_checkpoint(dict_args['load'])

    # Initialize logging paths
    random_sec = random.randint(1, 20)
    time.sleep(random_sec)
    now = datetime.datetime.now().strftime('%m%d-%H%M%S')
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now))
    while os.path.exists(weight_save_dir):
        random_sec = random.randint(1, 20)
        time.sleep(random_sec)
        now = datetime.datetime.now().strftime('%m%d%H%M%S')
        weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now))

    os.makedirs(weight_save_dir, exist_ok=True)


    # Callback: early stopping parameters
    early_stopping_callback = EarlyStopping(
        monitor="Validation MAE",
        mode="min",
        verbose=True,
        patience=dict_args["es_patience"],
    )

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_save_dir, save_top_k=5, verbose=True, monitor="Validation MAE", mode="min"
    )

    # Data: load data module
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)

    # Trainer: initialize training behaviour
    profiler = SimpleProfiler()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now, name='lightning_logs', log_graph=True)
    trainer = pl.Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor],
        val_check_interval=dict_args['val_freq'],
        deterministic=False,
        gpus=dict_args['gpus'],
        profiler=profiler,
        logger=logger,
        max_epochs=dict_args["epoch"],
        log_every_n_steps=10,
        gradient_clip_val=dict_args['clip_grad_norm'],
        resume_from_checkpoint=dict_args['resume_from_checkpoint']
    ) 

    # Trainer: train model
    trainer.fit(model, data_module)

    # Evaluate model on best ckpt (defined in 'ModelCheckpoint' callback)
    if dict_args['eval'] and dict_args['dataset_test']:
        trainer.test(model, ckpt_path='best', datamodule=data_module)
    else:
        print("Evaluation skipped")