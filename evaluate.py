from train import MODEL_DIRECTORY, DATALOADER_DIRECTORY

import argparse
import os
import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--checkpoint",
                        help="Directory of pre-trained model,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", required=True)
    parser.add_argument('--dataset', help='Directory of your test Dataset', required=True)
    parser.add_argument('--logdir', help='Where to log the results', required=True)

    # parser.add_argument('--model', help='Model name to train', required=True, default=None)
    # parser.add_argument('--eval', help='Whether to test model on the best iteration after training'
    #                     , default=False, action="store_true")
    # parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)
    # parser.add_argument("--load",
    #                     help="Directory of pre-trained model,  \n"
    #                          "None --> Do not use pre-trained model. Training will start from random initialized model")
    
    # parser.add_argument("--pretrain",
    #                     help="Directory of pre-trained model from ImageNet,  \n"
    #                          "None --> Do not use pre-trained model. Training will start from random initialized model")
    # parser.add_argument('--dataset_tr', help='Directory of your train Dataset', required=True, default=None)
    # parser.add_argument('--dataset_test', help='Directory of your test Dataset', default=None)
    # parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
    #                     default='cuda', choices=['cuda', 'cpu'])
    # parser.add_argument('--gpus', help="Number of gpus to use for training", default=0, type=int)
    # parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    # parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    # parser.add_argument('--num_workers', help="# of dataloader cpu process", default=0, type=int)
    # parser.add_argument('--val_freq', help='How often to run validation set within a training epoch, i.e. 0.25 will run 4 validation runs in 1 training epoch', default=1.0, type=float)
    # parser.add_argument('--es_patience', help='Max # of consecutive validation runs w/o improvment', default=5, type=int)
    # parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    # parser.add_argument('--lr', help='learning_rate for pose. default = 0.0001', default=0.0001, type=float)
    # parser.add_argument('--num_seg', help='Approximate number of segmentations', default=600, type=int)
    # parser.add_argument('--dropout', help='Dropout for Transformers', default=0., type=float)
    # parser.add_argument('--dropout_edge', help='Dropout for dropout_edge', default=0., type=float)
    # parser.add_argument('--drop_path', help='Dropout rate for drop path', default=0., type=float)
    # parser.add_argument('--seed', help='Seed for reproduceability', 
    #                     default=42, type=int)
    # parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    # parser.add_argument('--compactness', help='Compactness for SLIC', type=float, default=10)
    # parser.add_argument('--size', help='Image size for DUTS', type=int, default=224)
    # parser.add_argument('--coeff', help='Number of coefficients for fft', type=int, default=7)
    # parser.add_argument('--dilation', help='Dilation for local transformer', type=int, default=5)
    # parser.add_argument('--downsample', help='Downsample resolution', type=int, default=28)
    # parser.add_argument('--tag', help='Tag for differentiating runs on CC', default='', type=str)
    # parser.add_argument('--tfmhp', default=[8, 4, 6, 128, 16], 
    #                 nargs=5, metavar=('Heads', 'local heads', 'Number of Layers', 'Embed dim', 'Head dim'),
    #                 type=int, help='Hyperparameters for Transformer')
    # parser.add_argument('--ignore_phase', help='Whether or not to use phase of FFT'
    #                     , default=False, action="store_true")
    # parser.add_argument('--force_aug', help='Force data augmentation on select dataloaders'
    #                     , default=False, action="store_true")
    # parser.add_argument('--sigma', help='Sigma for feature augmentation', default=0.04, type=float)
    # parser.add_argument('--debug', help='Whether or not to switch to debug mode, only runs on 100 samples'
    #                     , default=False, action="store_true")
    # parser.add_argument('--dilation_mode', help='Dilation mode, 0 for fully connected, 1 for spotted global', default=0, type=int)
    # parser.add_argument('--gunet_mode', help='GUnet mode, graclus or predefined pooling', default='graclus', type=str)
    # parser.add_argument('--fully_connected', help='Use fully connected neighbourhood'
    #                     , default=False, action="store_true")
    # parser.add_argument('--heads', default=[3, 6, 12], 
    #                 nargs="*", 
    #                 type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    # parser.add_argument('--dims', default=[96, 192, 384], 
    #                 nargs="*", 
    #                 type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    # parser.add_argument('--depths', default=[2, 2, 6], 
    #                 nargs="*", 
    #                 type=int, help='Hyperparameters for kernel sizes of SWIN Transformer')
    # parser.add_argument('--window_size', help='Window size for SWIN Transformer', type=int, default=4)
    # parser.add_argument('--memory', help='Whether to put the data into memory'
    #                     , default=False, action="store_true")
    
    # parser.add_argument('--warmup_epochs', help='Number of epochs for warmup', type=int, default=4)
    # parser.add_argument('--mlp_ratio', help='Mlp ratio for FF networks', default=4, type=float)
    # parser.add_argument('--encoder_lr_weight', help='Set LR factor for pre-trained encoder weights', default=0.1, type=float)


    import torch 
    args = parser.parse_args()
    dict_args = vars(args)
    checkpoint = torch.load(dict_args['checkpoint'])
    checkpoint_args = checkpoint["hyper_parameters"]
    checkpoint_args['pretrain'] = None
    

    # Initialize model to train
    assert checkpoint_args['model'] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[checkpoint_args['model']](**checkpoint_args)
    model.load_state_dict(checkpoint['state_dict'])
    # model = model

    # Data: load data module
    assert checkpoint_args['dataloader'] in DATALOADER_DIRECTORY
    checkpoint_args['dataset_test'] = dict_args['dataset']
    checkpoint_args['skip_train'] = True
    data_module = DATALOADER_DIRECTORY[checkpoint_args['dataloader']](**checkpoint_args)

    now = datetime.datetime.now().strftime('%m%d-%H%M%S')
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now))
    logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now, name='lightning_logs', log_graph=True)
    trainer = pl.Trainer(accelerator="gpu",
        deterministic=False,
        profiler='simple',
        logger=logger,
        devices=-1,
    ) 

    trainer.test(model, datamodule=data_module)



    # args = parser.parse_args()
    # dict_args = vars(args)
    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha
    # dict_args['git'] = sha
    
    # pl.seed_everything(dict_args['seed'], True)

    # # Initialize model to train
    # assert dict_args['model'] in MODEL_DIRECTORY
    # model = MODEL_DIRECTORY[dict_args['model']](**dict_args)
    # if dict_args['load']:
    #     model = model.load_from_checkpoint(dict_args['load'])

    # # Data: load data module
    # assert dict_args['dataloader'] in DATALOADER_DIRECTORY
    # data_module = DATALOADER_DIRECTORY[dict_args['dataloader']](**dict_args)


    

    # # Initialize logging paths
    # now = datetime.datetime.now().strftime('%m%d-%H%M%S')
    # weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now+'_'+dict_args["tag"]))
 

    # # Callback: early stopping parameters
    # early_stopping_callback = EarlyStopping(
    #     monitor="Validation MAE",
    #     mode="min",
    #     verbose=True,
    #     patience=dict_args["es_patience"],
    # )

    # # Callback: model checkpoint strategy
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=weight_save_dir, save_top_k=5, verbose=True, monitor="Validation MAE", mode="min"
    # )

    

    # # Trainer: initialize training behaviour
   
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # logger = TensorBoardLogger(save_dir=dict_args['logdir'], version=now+'_'+dict_args["tag"], name='lightning_logs', log_graph=True)
    # trainer = pl.Trainer(accelerator="gpu",
    #     callbacks=[checkpoint_callback, lr_monitor],
    #     val_check_interval=dict_args['val_freq'],
    #     deterministic=False,
    #     profiler='simple',
    #     logger=logger,
    #     max_epochs=dict_args["epoch"],
    #     log_every_n_steps=10,
    #     gradient_clip_val=dict_args['clip_grad_norm'],
    #     devices=[dict_args['gpus']],
    # ) 

    # # Trainer: train model
    # if dict_args['resume_from_checkpoint'] is not None:
    #     trainer.fit(model, data_module, ckpt_path=dict_args['resume_from_checkpoint'])
    # else:
    #     trainer.fit(model, data_module)

    # # Evaluate model on best ckpt (defined in 'ModelCheckpoint' callback)
    # if dict_args['eval'] and dict_args['dataset_test']:
    #     trainer.test(model, ckpt_path='best', datamodule=data_module)
    # else:
    #     print("Evaluation skipped")