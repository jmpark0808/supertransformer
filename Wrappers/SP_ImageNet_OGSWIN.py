import pytorch_lightning as pl
import torch
from Blocks.swinunet_rpe import SwinTransformer

import torch.nn.functional as F
import numpy as np
from dataset.constants import *
from dataset.constants import NUM_CHUNK
from util.util import get_input_dim
from dataset.mixup import Mixup
from util.optimizers import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count

class SP_ImageNet_OGSWIN_Wrapper(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.num_seg = kwargs.get('num_seg')
        self.es_patience = kwargs.get('es_patience')
        self.dropout = kwargs.get('dropout')
        self.coeff = kwargs.get('coeff')
        self.tfm_hp = kwargs.get('tfmhp')
        self.dataloader = kwargs.get('dataloader')
        self.load = kwargs.get('load', None)
        self.dilation = kwargs.get('dilation')
        self.dropout_edge = kwargs.get('dropout_edge')
        self.kernels = kwargs.get('kernels')
        self.window_size = kwargs.get('window_size')
        self.warmup_epochs = kwargs.get('warmup_epochs')
        self.total_train_epochs = kwargs.get('epoch')
        
        
        input_dim = get_input_dim(kwargs)
        self.res = int(self.num_seg**0.5)
        
        # Generator that produces the HeatMap
        self.supert = SwinTransformer(img_size=self.res, in_chans=16, patch_size=1, window_size=self.window_size,
                                       embed_dim=self.tfm_hp[2], depths=[self.tfm_hp[1], self.tfm_hp[1], self.tfm_hp[1]*3, self.tfm_hp[1]],
                                         num_heads=[self.tfm_hp[0],
                                                    self.tfm_hp[0]*2,
                                                    self.tfm_hp[0]*4,
                                                        self.tfm_hp[0]*8], mlp_ratio=4)
        kwargs['parameters'] = parameter_count(self.supert)['']
        
        inp = torch.randn([1, input_dim+2, self.res, self.res])
        flops = FlopCountAnalysis(self.supert, inp)
        kwargs['flops'] = flops.total()
        # from fvcore.nn import FlopCountAnalysis, flop_count_table
        # inp = torch.randn([1, input_dim+2, 32, 32])
        # flops = FlopCountAnalysis(self.supert, inp)
        # print(flop_count_table(flops))
        # assert(0)
        self.mixup = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=1000)
        if self.load:
            ckpt = torch.load(self.load)
            for key in list(ckpt['state_dict'].keys()):
                ckpt['state_dict'][key.replace('supert.', '')] = ckpt['state_dict'].pop(key)
            self.supert.load_state_dict(ckpt['state_dict'])

        self.validation_step_outputs = []
        self.loss_fn = SoftTargetCrossEntropy()
        self.iteration = 0
        self.test_iteration = 0
        self.save_hyperparameters()
        

    def loss(self, pred, label):
        """
        Defining the loss funcition:
        """
        loss = self.loss_fn(pred, torch.squeeze(label))

        return loss

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        skip_list = {'absolute_pos_embed'}
        skip_keywords = {'relative_position_bias_table'}
        has_decay = []
        no_decay = []

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
                no_decay.append(param)
                # print(f"{name} has no weight decay")
            else:
                has_decay.append(param)
        parameters = [{'params': has_decay},
                {'params': no_decay, 'weight_decay': 0.}]
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=0.05)

        self.trainer.fit_loop.setup_data()
        dataset= self.trainer.train_dataloader
        self.scheduler = CosineAnnealingWarmRestarts(optimizer, len(dataset)*(self.total_train_epochs-self.warmup_epochs),
                                                      1, 5e-6)
        # self.scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr = 5e-6)
        
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

        pred = self.supert(input)

        return pred

    def on_train_epoch_start(self):
        self.train_acc = 0
        self.num_samples = 0
    
    def on_train_epoch_end(self):
        acc = self.train_acc/self.num_samples
        self.log('Train Accuracy', acc, sync_dist=True)


    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        """
        features, target = batch

        features = features.reshape(features.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        features, target = self.mixup(features, target)
        # features = features.permute(0, 2, 3, 1).reshape(features.size(0), 1024, -1)
        # forward pass
        
        pred = self.forward(features)

        loss = self.loss(pred, target)
        
        max_scores, max_idx_class = pred.max(dim=1)
        max_scores, max_idx_label = target.max(dim=1)
        n = pred.size(0)
        acc = (max_idx_class == max_idx_label).sum().item() 

        self.train_acc += acc
        self.num_samples += n

        self.log('loss', loss.item(), sync_dist=True)
        self.iteration += 1
        if self.current_epoch >= self.warmup_epochs:
            self.scheduler.step()
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc/self.val_num_samples
        self.log('Validation Accuracy', acc, sync_dist=True)
        self.validation_step_outputs.clear()

    def on_validation_start(self):
        self.val_acc = 0
        self.val_num_samples = 0

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        features, label = batch


        # forward pass
        features = features.reshape(features.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        
        pred = self.forward(features)

        loss = self.loss(pred, F.one_hot(label, num_classes=1000))
        
        max_scores, max_idx_class = pred.max(dim=1)
        n = pred.size(0)
        acc = (max_idx_class == label).sum().item() 

       
        self.validation_step_outputs.append(loss)
    
        self.val_acc += acc
        self.val_num_samples += n
        return loss


    def on_test_epoch_end(self):
        acc = self.test_acc/self.test_num_samples
        self.log('Final Test Accuracy', acc, sync_dist=True)

    def on_test_start(self):
        self.test_acc = 0
        self.test_num_samples = 0

    def test_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        features, label = batch

        # forward pass
        features = features.reshape(features.size(0), 32, 32, -1).permute(0, 3, 1, 2)
        pred = self.forward(features)

        loss = self.loss(pred, F.one_hot(label, num_classes=1000))
        
        max_scores, max_idx_class = pred.max(dim=1)

        n = pred.size(0)
        acc = (max_idx_class == label).sum().item() 

        self.test_acc += acc
        self.test_num_samples += n
        self.test_iteration += 1
     



if __name__ == "__main__":
    pass