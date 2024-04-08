import pytorch_lightning as pl
import torch
from skimage.segmentation import slic
from skimage.measure import regionprops_table
from Blocks.GraphBlocks import GAT
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from Blocks.TransformerBlocks import Transformer

class ImageLinear(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.es_patience = kwargs.get('es_patience')

        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, 3, 28, 28))

        # Generator that produces the HeatMap
        self.linear1 = nn.Linear(28*28*3, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 28*28)

        self.iteration = 0
        self.save_hyperparameters()
        

    def loss(self, pred, label):
        """
        Defining the loss funcition:
        """
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(pred), torch.squeeze(label))

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
      

    def forward(self, x):
        """
        Forward pass through model
        :param x: Input features
        :return: binary pixel-wise predictions
        """        
        x_size = x.size()
        x = x.reshape(x_size[0], -1)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = x.reshape(x_size[0], 1, x_size[2], x_size[3]) # batch, 1, X, Y

        return x

    def training_step(self, batch, batch_idx):
        """
        Compute and return the training loss
        logging resources:
        https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
        """
  
        mask = batch['mask']
        img = batch['image']

        img = img.cuda()
        mask = mask.cuda()
    
        # forward pass
        
        pred = self.forward(img)

        loss = self.loss(pred, mask)

        self.log('loss', loss.item())
        self.iteration += 1
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        tensorboard = self.logger.experiment
        mask = batch['mask']
        img = batch['image']

        img = img.cuda()
        mask = mask.cuda()


        # forward pass
        pred = self.forward(img)

        if batch_idx == 0:
            tensorboard.add_images('Pred', torch.sigmoid(pred), self.iteration)
            tensorboard.add_images('GT', mask, self.iteration)
            tensorboard.add_images('Image', img, self.iteration)

        mae = torch.mean(torch.abs(torch.sigmoid(pred) - mask))
      
        return mae


    def validation_epoch_end(self, validation_step_outputs):
        self.log('Validation MAE', torch.mean(torch.stack(validation_step_outputs)))
        self.scheduler.step(torch.mean(torch.stack(validation_step_outputs)))
  
                    
    def on_test_start(self):
        self.preds = []
        self.masks = []
        self.precs = []
        self.recalls = []

    def test_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        tensorboard = self.logger.experiment
        mask = batch['mask']
        img = batch['image']

        img = img.cuda()
        mask = mask.cuda()


        # forward pass
        pred = self.forward(img)

        mae = torch.mean(torch.abs(pred - mask))
        self.preds.append(pred)
        self.masks.append(mask)
        prec, recall = torch.zeros(mask.shape[0], 256), torch.zeros(mask.shape[0], 256)
        pred = pred.reshape(pred.size(0), -1)
        mask = mask.reshape(mask.size(0), -1)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        for j in range(256):
            y_temp = (pred >= thlist[j]).float()
            tp = (y_temp * mask).sum(dim=-1)
            # avoid prec becomes 0
            prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
        # (batch, threshold)
        self.precs.append(prec)
        self.recalls.append(recall)

        return mae


    def test_epoch_end(self, test_step_outputs):
        prec = torch.cat(self.precs, dim=0).mean(dim=0)
        recall = torch.cat(self.recalls, dim=0).mean(dim=0)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        self.log('Test Max F Score', torch.max(f_score))
        self.log('Test Max F Threshold', thlist[torch.argmax(f_score)])

        pred = torch.cat(self.preds, 0)
        mask = torch.cat(self.masks, 0).round().float()
        self.log('Test MAE', torch.mean(torch.abs(pred-mask)))



if __name__ == "__main__":
    pass