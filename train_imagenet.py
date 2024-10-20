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
from pytorch_lightning.loggers import TensorBoardLogger

from Wrappers.SP_ImageNet_TFM import SP_ImageNet_TFM_Wrapper
from Wrappers.SP_ImageNet_GAT_PyG import SP_ImageNet_GAT_PyG_Wrapper
from Wrappers.SP_ImageNet_DGAT_PyG import SP_ImageNet_DGAT_PyG_Wrapper
from Wrappers.SP_ImageNet_SWIN import SP_ImageNet_SWIN_Wrapper
from Wrappers.image_CLS_swintransformer import ImageNet_SWIN_Wrapper
from Wrappers.SP_ImageNet_SWIN_RPE import SP_ImageNet_OGSWIN_RPE_Wrapper
from Wrappers.SP_ImageNet_SWIN_APE import SP_ImageNet_OGSWIN_APE_Wrapper
from Wrappers.SP_ImageNet_MBUNET import SP_ImageNet_MBNET_Wrapper
from Wrappers.SP_ImageNet_MBVIT import SP_ImageNet_MBVIT_Wrapper
from Wrappers.image_CLS_mobilevit import ImageNet_MBVIT_Wrapper
from Wrappers.Image_CLS_stopsigns import Image_Stopsigns_Wrapper
from Wrappers.image_CLS_mobilenet import ImageNet_MBNET_Wrapper
from Wrappers.SP_ImageNet_PERF import SP_ImageNet_PERF_Wrapper
from Wrappers.SP_ImageNet_TFM import SP_ImageNet_TFM_Wrapper
from Wrappers.SP_ImageNet_PERFENC import SP_ImageNet_PERFENC_Wrapper
from Wrappers.SP_ImageNet_MAMBA import SP_ImageNet_MAMBA_Wrapper

# Import dataset modules
from dataset.imagenet import SPImageNetDataModule
from dataset.imagenet_pyg import SPGImageNetDataModule
from dataset.imagenet_aug import SPImageNetAugDataModule
from dataset.imagenet_pyg_exp import SPGEImageNetDataModule
from dataset.imagenet_images import ImageNetDataModule
from dataset.imagenet_images_mb import ImageNetMBDataModule
from dataset.imagenet_pyg_swin import SPGSImageNetDataModule
from dataset.stopsigns import SPSpeedLimitsDataModule
from dataset.stopsigns import SpeedLimitsDataModule

import git


# Metric logging

# Deterministic

MODEL_DIRECTORY = {
    'SP_ImageNet': SP_ImageNet_TFM_Wrapper,
    'SP_ImageNet_GAT': SP_ImageNet_GAT_PyG_Wrapper,
    'SP_ImageNet_DGAT': SP_ImageNet_DGAT_PyG_Wrapper,
    'SP_ImageNet_SWIN': SP_ImageNet_SWIN_Wrapper,
    'SP_ImageNet_OGSWIN_APE': SP_ImageNet_OGSWIN_APE_Wrapper,
    'SP_ImageNet_OGSWIN_RPE': SP_ImageNet_OGSWIN_RPE_Wrapper,
    'SWIN': ImageNet_SWIN_Wrapper,
    'SP_ImageNet_MBNET': SP_ImageNet_MBNET_Wrapper,
    'SP_ImageNet_MBVIT': SP_ImageNet_MBVIT_Wrapper,
    'SP_ImageNet_PERF': SP_ImageNet_PERF_Wrapper,
    'SP_ImageNet_PERFENC': SP_ImageNet_PERFENC_Wrapper,
    'SP_ImageNet_MAMBA': SP_ImageNet_MAMBA_Wrapper,
    'MBVIT': ImageNet_MBVIT_Wrapper,
    'Topk': Image_Stopsigns_Wrapper,
    'MBNET': ImageNet_MBNET_Wrapper

}
DATALOADER_DIRECTORY = {
    'ImageNet': SPImageNetDataModule,
    'ImageNet_PyG': SPGImageNetDataModule,
    'ImageNet_Aug': SPImageNetAugDataModule,
    'INPE': SPGEImageNetDataModule,
    'ImageNet_Images': ImageNetDataModule,
    'ImageNet_MB': ImageNetMBDataModule,
    'ImageNet_SWIN': SPGSImageNetDataModule,
    'SPSpeedLimits': SPSpeedLimitsDataModule,
    'SpeedLimits': SpeedLimitsDataModule,
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
    parser.add_argument('--train_dir', help='Directory of your ImageNet Train Dataset', required=True, default=None)
    parser.add_argument('--test_dir', help='Directory of your ImageNet Test Dataset', required=True, default=None)
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
    parser.add_argument('--dropout_edge', help='Dropout edge for Transformers', default=0., type=float)
    parser.add_argument('--drop_path', help='Dropout rate for drop path', default=0., type=float)
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--tag', help='Tag for differentiating runs on CC', default='', type=str)
    parser.add_argument('--tfmhp', default=[8, 4, 6, 128, 32], 
                    nargs=5, metavar=('Heads', 'local heads', 'Number of Layers', 'Embed dim', 'Head dim'),
                    type=int, help='Hyperparameters for Transformer')
    parser.add_argument('--coeff', help='Number of coefficients for fft', type=int, default=10)
    parser.add_argument('--compactness', help='Compactness for SLIC', type=float, default=10)
    parser.add_argument('--dilation', help='Dilation for local transformer', type=int, default=5)
    parser.add_argument('--size', help='Image size for DUTS', type=int, default=224)
    parser.add_argument('--kernels', default=[32, 32, 32, 32], 
                    nargs="*", 
                    type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    parser.add_argument('--window_size', help='Window size for SWIN Transformer', type=int, default=4)
    parser.add_argument('--warmup_epochs', help='Number of epochs for warmup', type=int, default=4)
    parser.add_argument('--debug', help='Whether or not to switch to debug mode, only runs on 100 samples'
                        , default=False, action="store_true")
    parser.add_argument('--heads', default=[3, 6, 12], 
                    nargs="*", 
                    type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    parser.add_argument('--dims', default=[96, 192, 384], 
                    nargs="*", 
                    type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    parser.add_argument('--depths', default=[2, 2, 6], 
                    nargs="*", 
                    type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    parser.add_argument('--mlp_ratio', help='Mlp ratio for FF networks', default=4, type=float)





    args = parser.parse_args()
    dict_args = vars(args)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    dict_args['git'] = sha
    
    pl.seed_everything(dict_args['seed'])
    # Initialize model to train
    assert dict_args['model'] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[dict_args['model']](**dict_args)


    # Initialize logging paths
    random_sec = random.randint(1, 20)
    time.sleep(random_sec)
    now = datetime.datetime.now().strftime('%m%d-%H%M%S')
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now+'_'+dict_args["tag"]))


    # Callback: early stopping parameters
    early_stopping_callback = EarlyStopping(
        monitor="Validation Accuracy",
        mode="max",
        verbose=True,
        patience=dict_args["es_patience"],
    )

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_save_dir, save_top_k=5, verbose=True, monitor="Validation Accuracy", mode="max"
    )

    # Data: load data module
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)

    # Trainer: initialize training behaviour
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now+'_'+dict_args["tag"], name='lightning_logs', log_graph=True)
    # if 'PERF' in dict_args['model']:
    #     precision = 32
    # else:
    #     precision = '16-mixed'
    precision =32
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=dict_args['val_freq'],
        deterministic=False,
        profiler='simple',
        logger=logger,
        max_epochs=dict_args["epoch"],
        log_every_n_steps=10,
        gradient_clip_val=dict_args['clip_grad_norm'],
        devices=-1,
        precision=precision
    ) 

    # Trainer: train model
    if dict_args['resume_from_checkpoint'] is not None:
        trainer.fit(model, data_module, ckpt_path=dict_args['resume_from_checkpoint'])
    else:
        trainer.fit(model, data_module)

    # Evaluate model on best ckpt (defined in 'ModelCheckpoint' callback)
    if dict_args['eval'] and dict_args['test_dir']:
        trainer.test(model, ckpt_path='best', datamodule=data_module) # 
    else:
        print("Evaluation skipped")