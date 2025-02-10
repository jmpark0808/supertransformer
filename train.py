import argparse
import datetime
import os
import random
import time
from re import X
import git

import pytorch_lightning as pl

# Callbacks and loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from Wrappers.SP_CNN import SP_CNN_Wrapper
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
from Wrappers.SP_TFM_FFT import SP_TFM_FFT_Wrapper
from Wrappers.SP_TFM_Contour import SP_TFM_Contour_Wrapper
from Wrappers.SP_GAT import SP_GAT_Wrapper
from Wrappers.SP_Baseline import SP_Baseline_Wrapper
from Wrappers.SP_Baseline_DPE import SP_Baseline_DPE_Wrapper
from Wrappers.SP_Baseline_LAP import SP_Baseline_LAP_Wrapper
from Wrappers.image_transformer import ImageTransformer
from Wrappers.SP_GAT_PyG import SP_GAT_PyG_Wrapper
from Wrappers.SP_TFM_PyG import SP_TFM_PyG_Wrapper
from Wrappers.SP_GUNET_PyG import SP_GUNET_PyG_Wrapper
from Wrappers.SP_GATv2 import SP_GATv2_Wrapper
from Wrappers.SP_GATv3 import SP_GATv3_Wrapper
from Wrappers.SP_GATv4 import SP_GATv4_Wrapper
from Wrappers.SP_CTFM import SP_CTFM_Wrapper
from Wrappers.SP_SWIN import SP_SWIN_Wrapper
from Wrappers.SP_SWINUU import SP_SWINUU_Wrapper
from Wrappers.SP_SWINUM import SP_SWINUM_Wrapper
from Wrappers.SP_SWIN_Kernel import SP_SWIN_Kernel_Wrapper
from Wrappers.SP_SWIN_PyG import SP_SWIN_PyG_Wrapper
from Wrappers.Image_SOD_SWINU import Image_SWINU_Wrapper
from Wrappers.SP_MBUNET import SP_MBUNET_Wrapper
from Wrappers.SP_MBNET import SP_MBNET_Wrapper
from Wrappers.Image_SOD_MBVITU import Image_MBVITU_Wrapper
from Wrappers.Image_SOD_PERFU import Image_PERFUSLIC_Wrapper
from Wrappers.SP_PERFU import SP_PERFU_Wrapper
from Wrappers.SP_MBVITU import SP_MBVITU_Wrapper
from Wrappers.SP_PERF import SP_PERF_Wrapper
from Wrappers.SP_PERFEncDec import SP_PERFEncDec_Wrapper
from Wrappers.SP_SWINUP import SP_SWINUP_Wrapper
from Wrappers.Seg_SOD_PERFU import Seg_PERFUSLIC_Wrapper
from Wrappers.SP_SWINUM_C import SP_SWINUM_C_Wrapper
from Wrappers.SP_SWINUM_C_LPE import SP_SWINUM_C_LPE_Wrapper
from Wrappers.SP_SWINUM_C_ROPE import SP_SWINUM_C_ROPE_Wrapper
from Wrappers.SP_SWINUM_C_CPE import SP_SWINUM_C_CPE_Wrapper
from Wrappers.SP_SWINUM_C_CROPE import SP_SWINUM_C_CROPE_Wrapper
# from Wrappers.SP_MAMBA import SP_MAMBA_Wrapper


# Import dataset modules
from dataset.superpixel import DUTSDataModule,  SPDataModule, SPRSDataModule
from dataset.superpixel_pyg import SPGDataModule
from dataset.superpixel_pyg_image import SPGIDataModule
from dataset.superpixel_fast import SPFDataModule, SPFRSDataModule
from dataset.superpixel_fast_cnn import SPFCDataModule
from dataset.youtube_davis import YDDataModule
from dataset.youtube_davis_swin_pyg import YDGDataModule
from dataset.superpixel_pyg_swin import SPGSWINDataModule




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
    "IT": ImageTransformer,
    # "ITCNN": ImageTransformerCNN,
    # "ITCNNTFM": ImageTransformerCNNTFM,
    # "ITNMP": ImageTransformerNMP,
    # "ITI": ImageLinear,
    # "ITUNET": ImageTransformerUNET,
    # "SC": SuperConvSeg,
    "SP_TFM_DIL": SP_TFM_DIL_Wrapper,
    "SP_RTFM_TFM": SP_RTFM_TFM_Wrapper,
    "SP_CNN": SP_CNN_Wrapper,
    "SP_TFM_TFM": SP_TFM_TFM_Wrapper,
    "SP_ETFM_TFM": SP_ETFM_TFM_Wrapper,
    "SP_TFM": SP_TFM_Wrapper,
    "SP_CTFM": SP_CTFM_Wrapper,
    "SP_GCN": SP_GCN_Wrapper,
    "SP_TFM_NP": SP_TFM_NP_Wrapper,
    "SP_TFM_AP": SP_TFM_AP_Wrapper,
    "SP_TFM_PE": SP_TFM_PE_Wrapper,
    "SP_GTFM": SP_GTFM_Wrapper,
    "SP_GAT": SP_GAT_Wrapper,
    "SP_TFM_FFT": SP_TFM_FFT_Wrapper,
    'SP_TFM_Contour': SP_TFM_Contour_Wrapper,
    'SP_Baseline': SP_Baseline_Wrapper,
    'SP_Baseline_DPE': SP_Baseline_DPE_Wrapper,
    'SP_Baseline_LAP': SP_Baseline_LAP_Wrapper,
    'SP_GAT_PyG': SP_GAT_PyG_Wrapper,
    'SP_TFM_PyG': SP_TFM_PyG_Wrapper,
    'SP_GUNET_PyG': SP_GUNET_PyG_Wrapper,
    'SP_GATv2': SP_GATv2_Wrapper,
    'SP_GATv3': SP_GATv3_Wrapper,
    'SP_GATv4': SP_GATv4_Wrapper,
    'SP_SWIN': SP_SWIN_Wrapper,
    'SP_SWINUU': SP_SWINUU_Wrapper,
    'SP_SWINUM': SP_SWINUM_Wrapper,
    'SP_SWINUP': SP_SWINUP_Wrapper,
    'SP_SWINUM_C': SP_SWINUM_C_Wrapper,
    'SP_SWINUM_C_LPE': SP_SWINUM_C_LPE_Wrapper,
    'SP_SWINUM_C_ROPE': SP_SWINUM_C_ROPE_Wrapper,
    'SP_SWINUM_C_CPE': SP_SWINUM_C_CPE_Wrapper,
    'SP_SWINUM_C_CROPE': SP_SWINUM_C_CROPE_Wrapper,
    # 'SP_MAMBA': SP_MAMBA_Wrapper,
    'SP_SWIN_Kernel': SP_SWIN_Kernel_Wrapper,
    'SP_SWIN_PyG': SP_SWIN_PyG_Wrapper,
    'IM_SWINU': Image_SWINU_Wrapper,
    'IM_PERFUSLIC': Image_PERFUSLIC_Wrapper,
    'IM_MBVIT': Image_MBVITU_Wrapper,
    'SP_MBUNET': SP_MBUNET_Wrapper,
    'SP_MBNET': SP_MBNET_Wrapper,
    'SP_PERFU': SP_PERFU_Wrapper,
    'SP_PERF': SP_PERF_Wrapper,
    'SP_PERFEncDec': SP_PERFEncDec_Wrapper,
    'SP_MBVITU': SP_MBVITU_Wrapper,
    'SEG_PERFUSLIC': Seg_PERFUSLIC_Wrapper
}
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
    'SPGSWIN': SPGSWINDataModule,
    'SPFRS': SPFRSDataModule,
    'SPRS': SPRSDataModule

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
    parser.add_argument("--pretrain",
                        help="Directory of pre-trained model from ImageNet,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
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
    parser.add_argument('--drop_path', help='Dropout rate for drop path', default=0., type=float)
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--compactness', help='Compactness for SLIC', type=float, default=10)
    parser.add_argument('--size', help='Image size for DUTS', type=int, default=224)
    parser.add_argument('--coeff', help='Number of coefficients for fft', type=int, default=7)
    parser.add_argument('--dilation', help='Dilation for local transformer', type=int, default=5)
    parser.add_argument('--downsample', help='Downsample resolution', type=int, default=28)
    parser.add_argument('--tag', help='Tag for differentiating runs on CC', default='', type=str)
    parser.add_argument('--tfmhp', default=[8, 4, 6, 128, 16], 
                    nargs=5, metavar=('Heads', 'local heads', 'Number of Layers', 'Embed dim', 'Head dim'),
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
    parser.add_argument('--heads', default=[3, 6, 12], 
                    nargs="*", 
                    type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    parser.add_argument('--dims', default=[96, 192, 384], 
                    nargs="*", 
                    type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    parser.add_argument('--depths', default=[2, 2, 6], 
                    nargs="*", 
                    type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    parser.add_argument('--window_size', help='Window size for SWIN Transformer', type=int, default=4)
    parser.add_argument('--memory', help='Whether to put the data into memory'
                        , default=False, action="store_true")
    
    parser.add_argument('--warmup_epochs', help='Number of epochs for warmup', type=int, default=4)
    parser.add_argument('--mlp_ratio', help='Mlp ratio for FF networks', default=4, type=float)
    parser.add_argument('--encoder_lr_weight', help='Set LR factor for pre-trained encoder weights', default=1, type=float)
    parser.add_argument('--skip_train', help='Whether to skip training (for evaluation)'
                        , default=False, action="store_true")


    import torch 
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    dict_args = vars(args)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    dict_args['git'] = sha
    
    pl.seed_everything(dict_args['seed'], True)

    # Initialize model to train
    assert dict_args['model'] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[dict_args['model']](**dict_args)
    if dict_args['load']:
        model = model.load_from_checkpoint(dict_args['load'])

    # Data: load data module
    assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)


    

    # Initialize logging paths
    now = datetime.datetime.now().strftime('%m%d-%H%M%S')
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now+'_'+dict_args["tag"]))
 

    # Callback: early stopping parameters
    early_stopping_callback = EarlyStopping(
        monitor="Validation MAE",
        mode="min",
        verbose=True,
        patience=dict_args["es_patience"],
    )

    # Callback: model checkpoint strategy
    checkpoint_callback = ModelCheckpoint(
        dirpath=weight_save_dir, save_top_k=5, verbose=True, save_last=True, monitor="Validation MAE", mode="min"
    )

    

    # Trainer: initialize training behaviour
   
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now+'_'+dict_args["tag"], name='lightning_logs', log_graph=True)
    trainer = pl.Trainer(accelerator="gpu",
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=dict_args['val_freq'],
        deterministic=False,
        profiler='simple',
        logger=logger,
        max_epochs=dict_args["epoch"],
        log_every_n_steps=10,
        gradient_clip_val=dict_args['clip_grad_norm'],
        devices=[dict_args['gpus']],
    ) 

    # Trainer: train model
    if dict_args['resume_from_checkpoint'] is not None:
        trainer.fit(model, data_module, ckpt_path=dict_args['resume_from_checkpoint'])
    else:
        trainer.fit(model, data_module)

    # Evaluate model on best ckpt (defined in 'ModelCheckpoint' callback)
    if dict_args['eval'] and dict_args['dataset_test']:
        trainer.test(model, ckpt_path='best', datamodule=data_module)
    else:
        print("Evaluation skipped")