import pytorch_lightning as pl
import torch
from Models.SP_TFM import SP_ImageNet_TFM

import torch.nn.functional as F
import numpy as np
from dataset.constants import *
from dataset.constants import NUM_CHUNK
from util.util import get_input_dim

class SP_ImageNet_TFM_Wrapper(pl.LightningModule):
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
        input_dim = get_input_dim(kwargs)
        # Generator that produces the HeatMap
        self.supert = SP_ImageNet_TFM(input_dim, self.tfm_hp[1], self.tfm_hp[0], self.tfm_hp[2], self.dropout)
        if self.load:
            ckpt = torch.load(self.load)
            for key in list(ckpt['state_dict'].keys()):
                ckpt['state_dict'][key.replace('supert.', '')] = ckpt['state_dict'].pop(key)
            self.supert.load_state_dict(ckpt['state_dict'])

  
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.iteration = 0
        self.test_iteration = 0
        self.save_hyperparameters()
        

    def loss(self, pred, label):
        """
        Defining the loss funcition:
        """
        loss = self.loss_fn(pred, label)

        return loss

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=self.es_patience-3,
            min_lr=1e-8,
            verbose=True)
        return optimizer
      

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
        self.log('Train Accuracy', acc)


    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        """
        features, target = batch
  
        features = features.cuda()
        target = target.cuda()


        # forward pass
        
        pred = self.forward(features)

        loss = self.loss(pred, target)
        
        max_scores, max_idx_class = pred.max(dim=1)
        n = pred.size(0)
        acc = (max_idx_class == target).sum().item() 

        self.train_acc += acc
        self.num_samples += n

        self.log('loss', loss.item())
        self.iteration += 1
        return loss

    def on_validation_epoch_end(self, validation_step_outputs):
        acc = self.val_acc/self.val_num_samples
        self.log('Validation Accuracy', acc)
        self.scheduler.step(torch.mean(torch.stack(validation_step_outputs)))

    def on_validation_start(self):
        self.val_acc = 0
        self.val_num_samples = 0

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        features, label = batch



        features = features.cuda()
        label = label.cuda()

        # forward pass
        
        pred = self.forward(features)

        loss = self.loss(pred, label)
        
        max_scores, max_idx_class = pred.max(dim=1)
        n = pred.size(0)
        acc = (max_idx_class == label).sum().item() 

        self.val_acc += acc
        self.val_num_samples += n
        return loss


    def on_test_epoch_end(self, validation_step_outputs):
        acc = self.test_acc/self.test_num_samples
        self.log('Test Accuracy', acc)

    def on_test_start(self):
        self.test_acc = 0
        self.test_num_samples = 0

    def test_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        features, label = batch



        features = features.cuda()
        label = label.cuda()

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
    pass