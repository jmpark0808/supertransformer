import pytorch_lightning as pl
import torch
from Models.SP_GAT_PyG import SP_GAT_PyG

import torch.nn.functional as F
import numpy as np
from dataset.constants import *

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from torch_geometric.data import Data
from util.util import get_input_dim

class SP_GAT_PyG_Wrapper(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.num_seg = kwargs.get('num_seg')
        self.es_patience = kwargs.get('es_patience')
        self.dropout = kwargs.get('dropout')
        self.tfm_hp = kwargs.get('tfmhp')
        input_dim = get_input_dim(kwargs)
        # Generator that produces the HeatMap
        self.model = SP_GAT_PyG(input_dim, self.tfm_hp[1], 1, self.dropout, self.tfm_hp[0], self.tfm_hp[2])
        
        # data = Data(x=torch.ones(self.num_seg, input_dim),
        #              edge_index=torch.ones(self.num_seg,self.num_seg),
        #                edge_attr=torch.ones(self.num_seg,self.num_seg, 1))
        # flops = FlopCountAnalysis(self.model, data)
        # print(flop_count_table(flops))
        self.iteration = 0
        self.test_iteration = 0
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
      

    def forward(self, input):
        """
        Forward pass through model
        :param x: Input features
        :param adj: adjacent matrix 
        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """        
        
        pred = self.model(input)
        pred = pred.reshape(-1, self.num_seg)

        return pred

    def on_train_epoch_start(self):
        self.train_fscores = 0
        self.num_samples = 0
    
    def on_train_epoch_end(self):
        fscores = self.train_fscores/self.num_samples
        # thlist = torch.linspace(0, 1 - 1e-10, 256)
        self.log('Train Max F Score', torch.max(fscores))
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
        img = batch['img']


        features = features.cuda()
     
        # forward pass
        
        pred = self.forward(features)

        loss = self.loss(pred, seq_mask)
        
        pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = img.shape[0]
        img_size = img.shape[2]
        segments = segments.reshape([batch_size, -1]) # batch, img_size^2

        samples = []
        for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
            plt_image = masked[labels-1].reshape([img_size, img_size])
            samples.append(plt_image)

        samples = torch.tensor(np.expand_dims(np.array(samples), 1))
        samples_mask = []
        for masked, labels in zip(seq_mask_numpy, segments.cpu().numpy()):
            plt_image = masked[labels-1].reshape([img_size, img_size])
            samples_mask.append(plt_image)

        samples_mask = torch.tensor(np.expand_dims(np.array(samples_mask), 1))

        prec, recall = torch.zeros(samples_mask.shape[0], 1), torch.zeros(samples_mask.shape[0], 1)
        pred = samples.reshape(samples.shape[0], -1)
        mask = samples_mask.reshape(samples_mask.shape[0], -1)
        
        y_temp = (pred >= 0.5).float()
        tp = (y_temp * mask).sum(dim=-1)
        # avoid prec becomes 0
        prec[:, 0], recall[:, 0] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
        # (batch, threshold)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        f_score = f_score.sum(dim=0)
        self.train_fscores += f_score
        self.num_samples += features.size(0)
        self.log('loss', loss.item())
        self.iteration += 1
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
        img = batch['img']


        features = features.cuda()


        # forward pass
        pred = self.forward(features)

        pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = img.shape[0]
        img_size = img.shape[2]
        segments = segments.reshape([batch_size, -1]) # batch, img_size^2

        samples = []
        for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
            plt_image = masked[labels-1].reshape([img_size, img_size])
            samples.append(plt_image)

        samples = torch.tensor(np.expand_dims(np.array(samples), 1))
        # tensorboard.add_images('Test Pred', samples, self.test_iteration)
        samples_mask = []
        for masked, labels in zip(seq_mask_numpy, segments.cpu().numpy()):
            plt_image = masked[labels-1].reshape([img_size, img_size])
            samples_mask.append(plt_image)

        samples_mask = torch.tensor(np.expand_dims(np.array(samples_mask), 1))
        # tensorboard.add_images('Test GT', samples_mask, self.test_iteration)
        # tensorboard.add_images('Test Image', img, self.test_iteration)

        mae = torch.mean(torch.abs(samples - samples_mask))
        self.preds.append(samples)
        self.masks.append(samples_mask)
        prec, recall = torch.zeros(samples_mask.shape[0], 256), torch.zeros(samples_mask.shape[0], 256)
        pred = samples.reshape(samples.shape[0], -1)
        mask = samples_mask.reshape(samples_mask.shape[0], -1)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        for j in range(256):
            y_temp = (pred >= thlist[j]).float()
            tp = (y_temp * mask).sum(dim=-1)
            # avoid prec becomes 0
            prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
        # (batch, threshold)
        self.precs.append(prec)
        self.recalls.append(recall)
        self.test_iteration += 1
        return mae


    def validation_epoch_end(self, validation_step_outputs):
        prec = torch.cat(self.precs, dim=0).mean(dim=0)
        recall = torch.cat(self.recalls, dim=0).mean(dim=0)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        self.log('Validation Max F Score', torch.max(f_score))
        self.log('Validation Max F Threshold', thlist[torch.argmax(f_score)])

        pred = torch.cat(self.preds, 0)
        mask = torch.cat(self.masks, 0).round().float()
        self.log('Validation MAE', torch.mean(torch.abs(pred-mask)))
        self.scheduler.step(torch.mean(torch.stack(validation_step_outputs)))

    def on_validation_start(self):
        self.preds = []
        self.masks = []
        self.precs = []
        self.recalls = []

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
        features = batch['features']
        seq_mask = batch['seq_mask']
        segments = batch['segments']
        mask = batch['mask']
        img = batch['img']


        features = features.cuda()


        # forward pass
        pred = self.forward(features)

        pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = img.shape[0]
        img_size = img.shape[2]
        segments = segments.reshape([batch_size, -1]) # batch, img_size^2

        samples = []
        for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
            plt_image = masked[labels-1].reshape([img_size, img_size])
            samples.append(plt_image)

        samples = torch.tensor(np.expand_dims(np.array(samples), 1))
        # tensorboard.add_images('Test Pred', samples, self.test_iteration)
        samples_mask = []
        for masked, labels in zip(seq_mask_numpy, segments.cpu().numpy()):
            plt_image = masked[labels-1].reshape([img_size, img_size])
            samples_mask.append(plt_image)

        samples_mask = torch.tensor(np.expand_dims(np.array(samples_mask), 1))
        # tensorboard.add_images('Test GT', samples_mask, self.test_iteration)
        # tensorboard.add_images('Test Image', img, self.test_iteration)

        mae = torch.mean(torch.abs(samples - samples_mask))
        self.preds.append(samples)
        self.masks.append(samples_mask)
        prec, recall = torch.zeros(samples_mask.shape[0], 256), torch.zeros(samples_mask.shape[0], 256)
        pred = samples.reshape(samples.shape[0], -1)
        mask = samples_mask.reshape(samples_mask.shape[0], -1)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        for j in range(256):
            y_temp = (pred >= thlist[j]).float()
            tp = (y_temp * mask).sum(dim=-1)
            # avoid prec becomes 0
            prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
        # (batch, threshold)
        self.precs.append(prec)
        self.recalls.append(recall)
        self.test_iteration += 1
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
        self.scheduler.step(torch.mean(torch.stack(test_step_outputs)))
                    
    



if __name__ == "__main__":
    pass