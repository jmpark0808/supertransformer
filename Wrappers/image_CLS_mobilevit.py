import pytorch_lightning as pl
import torch
# import sys
# sys.path.insert(0, '/home/eddie/waterloo/supertransformer')
from Blocks.MobileVitV2 import MobileViTv3_v2
import torch.nn.functional as F
import numpy as np
from dataset.mixup import Mixup
from util.optimizers import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from math import cos, pi


class ImageNet_MBVIT_Wrapper(pl.LightningModule):
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
        
        # Generator that produces the HeatMap

        self.supert = MobileViTv3_v2(image_size=(256, 256), width_multiplier=0.5, num_classes=1000)
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
        # self.loss_fn = SoftTargetCrossEntropy()
        self.loss_fn = torch.nn.CrossEntropyLoss()
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
       
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=4e-5)

        # self.trainer.fit_loop.setup_data()
        # dataset= self.trainer.train_dataloader
        # self.scheduler = CosineAnnealingWarmRestarts(optimizer, len(dataset)*(self.total_train_epochs-self.warmup_epochs),
        #                                               1, 0.0002)
        
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        
        dataset= self.trainer.train_dataloader
        num_iter = len(dataset)
        warmup_epoch = self.warmup_epochs
        warmup_iter = warmup_epoch * num_iter
        current_iter = batch_idx + epoch * num_iter
        

        if epoch < self.warmup_epochs:
            lr = self.lr * current_iter / warmup_iter
        else:
            max_iter = self.total_train_epochs * num_iter
            lr = self.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2


        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
       
        
      

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

        # features, target = self.mixup(features, target)
        
        # forward pass
        
        pred = self.forward(features)

        loss = self.loss(pred, target)
        
        max_scores, max_idx_class = pred.max(dim=1)
        # max_scores, max_idx_label = target.max(dim=1)
        n = pred.size(0)
        acc = (max_idx_class == target).sum().item() 

        self.train_acc += acc
        self.num_samples += n

        self.log('loss', loss.item(), sync_dist=True)
        self.iteration += 1
        # if self.current_epoch >= self.warmup_epochs:
        #     self.scheduler.step()
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc/self.val_num_samples
        self.log('Validation Accuracy', acc, sync_dist=True)

        # self.scheduler.step(torch.mean(torch.stack(self.validation_step_outputs)))
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
        
        pred = self.forward(features)

        loss = self.loss(pred, label)
        
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
        
        pred = self.forward(features)

        loss = self.loss(pred, label)
        
        max_scores, max_idx_class = pred.max(dim=1)

        n = pred.size(0)
        acc = (max_idx_class == label).sum().item() 

        self.test_acc += acc
        self.test_num_samples += n
        self.test_iteration += 1
     



if __name__ == "__main__":


    
    supert = SwinTransformer()
    random_input = torch.ones(32, 3, 224, 224)

    output = supert(random_input)
    print(output.size())