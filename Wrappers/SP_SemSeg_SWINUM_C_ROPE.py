from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
# from Blocks.swintransformer_original_rpe import SwinUTransformer
from Blocks.swinunet_mix_rope_only_semseg import SwinUTransformer
# from Models.SP_SWIN import SP_SWINU
import torch.nn.functional as F
import numpy as np
from dataset.constants import *
from util.util import get_input_dim
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from dataset.mixup import MixupSaliency
from util.util import eval_e, S_object, S_region, TokenDropout
import cv2
import os
import time
class SP_SemSeg_SWINUM_C_ROPE_Wrapper(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.num_seg = kwargs.get('num_seg')
        self.es_patience = kwargs.get('es_patience')
        self.dropout = kwargs.get('dropout')
        self.tfm_hp = kwargs.get('tfmhp')
        self.coeff = kwargs.get('coeff')
        self.dilation = kwargs.get('dilation')
        self.dataloader = kwargs.get('dataloader')
        self.pretrain = kwargs.get('pretrain')
        self.dropout_edge = kwargs.get('dropout_edge')
        self.window_size = kwargs.get('window_size')
        self.image_size = kwargs.get('size')
        self.warmup_epochs = kwargs.get('warmup_epochs')
        self.total_train_epochs = kwargs.get('epoch')
        self.heads = kwargs.get('heads')
        self.dims = kwargs.get('dims')
        self.depths = kwargs.get('depths')
        self.size = kwargs.get('size')
        self.mlp_ratio = kwargs.get('mlp_ratio')
        self.dp = kwargs.get('drop_path')
        self.encoder_lr_weight = kwargs.get('encoder_lr_weight')

        self.aug_strat = kwargs.get('aug_strat')
        

        input_dim = get_input_dim(kwargs)

        res = int(self.num_seg**0.5)
        # Generator that produces the HeatMap
        self.supert = SwinUTransformer(img_size=res, in_chans=input_dim, patch_size=1, window_size=self.window_size,
                                       embed_dim=self.dims, depths=self.depths,
                                         num_heads=self.heads, mlp_ratio=self.mlp_ratio, attn_drop_rate=self.dropout_edge, drop_rate=self.dropout,
                                         qkv_bias=False, drop_path_rate=self.dp)
        # self.supert = SP_SWINU(input_dim, self.tfm_hp[2], self.tfm_hp[0],self.tfm_hp[1], self.dropout, self.dropout_edge, res)

        kwargs['parameters'] = parameter_count(self.supert)['']
        inp = torch.randn([1, input_dim+2, res, res])
        flops = FlopCountAnalysis(self.supert, inp)
        kwargs['flops'] = flops.total()
        self.flops = kwargs['flops']
        self.num_parameters = kwargs['parameters']
        # print(flop_count_table(flops))

        # print(kwargs['parameters'] , kwargs['flops'])
        # assert(0)
        self.mixup = MixupSaliency(
            cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0,  mode='batch',
            )
        self.iteration = 0
        self.test_iteration = 0
        self.num_thresholds = 10
        self.loss_fn = torch.nn.CrossEntropyLoss()
        if self.pretrain:
            checkpoint = torch.load(self.pretrain)
            for key in list(checkpoint['state_dict'].keys()):
                checkpoint['state_dict'][key.replace('supert.', '')] = checkpoint['state_dict'].pop(key)
            
            self.supert.load_state_dict(checkpoint['state_dict'], strict=False)
            for name, param in self.supert.named_parameters():
                if name in checkpoint['state_dict'].keys():
                    param.requires_grad = False
        
        self.save_hyperparameters()
        

    def loss(self, pred, label):
        """
        Defining the loss funcition:
        """
       
        loss = self.loss_fn(pred.permute(0, 2, 1), label)

        return loss

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        
        skip_list = {'absolute_pos_embed'}
        skip_keywords = {'relative_position_bias_table'}
        has_decay_enc = []
        has_decay_dec = []
        no_decay_enc = []
        no_decay_dec = []

        def check_keywords_in_name(name, keywords=()):
            isin = False
            for keyword in keywords:
                if keyword in name:
                    isin = True
            return isin

        for name, param in self.supert.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                    check_keywords_in_name(name, skip_keywords):
                if 'sod_head' in name or 'upsample_layers' in name:
                    no_decay_dec.append(param)
                else:
                    no_decay_enc.append(param)
                # print(f"{name} has no weight decay")
            else:
                if 'sod_head' in name or 'upsample_layers' in name:
                    has_decay_dec.append(param)
                else:
                    has_decay_enc.append(param)
        parameters = [{'params': has_decay_dec},
                      {'params': has_decay_enc, 'lr': self.lr*self.encoder_lr_weight},
                {'params': no_decay_dec, 'weight_decay': 0.},
                {'params': no_decay_enc, 'weight_decay': 0., 'lr': self.lr*self.encoder_lr_weight}]
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=0.05)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.1,
        #     patience=self.es_patience//2,
        #     min_lr=5e-8,
        #     verbose=True)
        
        self.trainer.fit_loop.setup_data()
        dataset= self.trainer.train_dataloader
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(dataset)*(self.total_train_epochs-self.warmup_epochs),
                                                      1, 5e-8)
        return optimizer
      
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        
        dataset= self.trainer.train_dataloader
        # manually warm up lr without a scheduler
        
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / (len(dataset)*self.warmup_epochs))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def forward(self, input):
        """
        Forward pass through model
        :param x: Input features
        :param adj: adjacent matrix 
        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """        
        # first = input[:, :8]
        # second_amp = input[:, 8:8+self.resample_points]
        # second_phase = input[:, 8+self.resample_points:-10]
        # if self.coeff%2!=0: # coeff is odd
        #     second_amp_front = second_amp[:, 1:2+(self.coeff//2)]
        #     second_amp_back = second_amp[:, -(self.coeff//2):]
        #     second_phase_front = second_phase[:,  1:2+(self.coeff//2)]
        #     second_phase_back = second_phase[:,  -(self.coeff//2):]
        # else:
        #     second_amp_front = second_amp[:,  1:1+self.coeff//2]
        #     second_amp_back = second_amp[:,  -self.coeff//2:]
        #     second_phase_front = second_phase[:,  1:1+self.coeff//2]
        #     second_phase_back = second_phase[:,  -self.coeff//2:]
        # third = input[:,  -10:]
        # input = torch.cat((first, second_amp_front, second_amp_back, second_phase_front, second_phase_back, third), dim=1)
        
        pred = self.supert(input)

        return pred

    def on_train_epoch_start(self):
        self.train_acc = 0
        self.num_samples = 0

    def on_train_start(self):
        self.log('Flops', self.flops)
        self.log('Parameters', self.num_parameters)
    
    def on_train_epoch_end(self):
        acc = self.train_acc/self.num_samples
        # thlist = torch.linspace(0, 1 - 1e-10, 256)
        self.log('Train Acc', acc)
        # self.log('Train Max F Threshold', thlist[torch.argmax(fscores)])



    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        """
        features = batch['features']
        seq_mask = batch['seq_mask']
        segments = batch['segments']
        mask = batch['mask']


        res = int(self.num_seg**0.5)
        features = features.reshape(features.size(0), res, res, -1).permute(0, 3, 1, 2)
        if self.aug_strat == 4 and 'RS' not in self.dataloader:
            seq_mask = seq_mask.reshape(seq_mask.size(0), res, res)
            features, seq_mask = self.mixup(features, seq_mask)
            
            seq_mask = seq_mask.reshape(seq_mask.size(0), -1)

        # forward pass
        pred = self.forward(features)

        loss = self.loss(pred, seq_mask)
        
        pred_numpy = pred.argmax(-1).detach().cpu().numpy() # batch, seq_len, 1
       
        
        correct = (pred_numpy == seq_mask)

        acc = correct.sum().float()/correct.numel()


        self.train_acc += acc*features.size(0)
        self.num_samples += features.size(0)
        self.log('loss', loss.item(), prog_bar=True)
        self.iteration += 1
        if self.current_epoch >= self.warmup_epochs:
            self.scheduler.step()
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        features = batch['features']
        seq_mask = batch['seq_mask']
        segments = batch['segments']
        mask = batch['mask']
        
        res = int(self.num_seg**0.5)
        features = features.reshape(features.size(0), res, res, -1).permute(0, 3, 1, 2)
        
        pred = self.forward(features)
        loss = self.loss(pred, seq_mask)
        res = int(self.num_seg**0.5)

        pred_numpy = pred.argmax(-1).detach().cpu() # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = mask.shape[0]
        img_size = self.size
        segments = segments.reshape([batch_size, -1]) # batch, img_size^2

        if torch.sum(segments) != 0 :

            segments = segments.reshape([batch_size, -1]) # batch, img_size^2

            samples = []
            for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
                plt_image = masked[labels-1].reshape([img_size, img_size])
                
                
                samples.append(plt_image)

            samples = torch.stack(samples, dim=0).cuda()
        else:
            samples = torch.sigmoid(pred).reshape(pred.size(0), 1, res, res)
            samples = F.interpolate(samples, (self.image_size, self.image_size), mode='bilinear')
        
        
        
 
        correct = (samples == mask)

        acc = correct.sum().float()/correct.numel()
        # (batch, threshold)
        
        self.val_acc += acc*features.size(0)
        self.mean_num += features.size(0)
        self.validation_step_outputs.append(loss)
        self.test_iteration += 1
        
        return loss


    def on_validation_epoch_end(self):
        acc = self.val_acc/self.mean_num
       
       
        self.log('Validation Acc', acc)
        self.log('Validation MAE', torch.mean(torch.tensor(self.validation_step_outputs)) )

        self.validation_step_outputs.clear()

    def on_validation_start(self):
        self.val_acc = 0
        self.mean_num = 0
        
        self.validation_step_outputs = []

    def on_test_start(self):
        self.test_acc = 0
        self.mean_num = 0
        

    def test_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        features = batch['features']
        seq_mask = batch['seq_mask']
        segments = batch['segments']
        mask = batch['mask']
        names = batch['file_name']


        # forward pass
        res = int(self.num_seg**0.5)
        features = features.reshape(features.size(0), res, res, -1).permute(0, 3, 1, 2)
        start = time.time()
        pred = self.forward(features)
        end = time.time()
        loss = self.loss(pred, seq_mask)
        
        pred_numpy = pred.argmax(-1).detach().cpu() # batch, seq_len, 1

        batch_size = mask.shape[0]
        img_size = self.size
        if torch.sum(segments) != 0 :

            segments = segments.reshape([batch_size, -1]) # batch, img_size^2

            samples = []
            for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
                plt_image = masked[labels-1].reshape([img_size, img_size])
                samples.append(plt_image)

            samples = torch.stack(samples, dim=0).cuda()
        else:
            samples = torch.sigmoid(pred).reshape(pred.size(0), 1, res, res)
            samples = F.interpolate(samples, (self.image_size, self.image_size), mode='bilinear').detach().cpu()
        # tensorboard.add_images('Test Pred', samples, self.test_iteration)

        # tensorboard.add_images('Test GT', samples_mask, self.test_iteration)
        # tensorboard.add_images('Test Image', img, self.test_iteration)

        # for sample, name in zip(samples, names):
        #     name = name.split('/')[-1]
        #     sample = (sample.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)
        #     cv2.imwrite(os.path.join('/home/eddie/Qualitative/SF-S',name), sample)

        correct = (samples == mask)

        acc = correct.sum().float()/correct.numel()
        # (batch, threshold)
        
        self.test_acc += acc*features.size(0)
        self.mean_num += features.size(0)

        self.test_step_outputs.append(loss)
        self.test_iteration += 1
        return loss
    
    def on_test_epoch_end(self):
        acc = self.test_acc/self.mean_num
       
       
        self.log('Test Acc', acc)




if __name__ == "__main__":
    pass