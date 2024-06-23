from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
# from Blocks.swintransformer_original_rpe import SwinUTransformer
# from Blocks.swintransformer_original import SwinUTransformer
from Blocks.swinunet_rpe import SwinUTransformer
# from Blocks.swin_experimental import SwinUTransformer
# from Models.SP_SWIN import SP_SWINU
import torch.nn.functional as F
import numpy as np
from dataset.constants import *
from util.util import get_input_dim
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from dataset.mixup import MixupSaliency

class SP_SWINU_Wrapper(pl.LightningModule):
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
        input_dim = get_input_dim(kwargs)
        res = int(self.num_seg**0.5)
        # Generator that produces the HeatMap
        self.supert = SwinUTransformer(in_chans=16, img_size=res, patch_size=1, window_size=self.window_size,
                                       depths=[self.tfm_hp[1], self.tfm_hp[1], self.tfm_hp[1]*3]
                                       , num_heads=[self.tfm_hp[0],
                                                     self.tfm_hp[0]*2,
                                                     self.tfm_hp[0]*4],
                                       embed_dim=self.tfm_hp[2])
        # self.supert = SwinUTransformer(img_size=res, in_chans=input_dim, patch_size=1, window_size=self.window_size,
        #                                , depths=[self.tfm_hp[1], self.tfm_hp[1], self.tfm_hp[1]*3, self.tfm_hp[1]],
        #                                  num_heads=[self.tfm_hp[0],
        #                                             self.tfm_hp[0]*2,
        #                                             self.tfm_hp[0]*4,
        #                                                 self.tfm_hp[0]*8], mlp_ratio=1)
        # self.supert = SP_SWINU(input_dim, self.tfm_hp[2], self.tfm_hp[0],self.tfm_hp[1], self.dropout, self.dropout_edge, res)

        kwargs['parameters'] = parameter_count(self.supert)['']
        inp = torch.randn([1, input_dim+2, res, res])
        flops = FlopCountAnalysis(self.supert, inp)
        kwargs['flops'] = flops.total()
        self.mixup = MixupSaliency(
            cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0,  mode='batch',
            )
        self.iteration = 0
        self.test_iteration = 0
        self.num_thresholds = 10
        if self.pretrain:
            checkpoint = torch.load(self.pretrain)
            for key in list(checkpoint['state_dict'].keys()):
                checkpoint['state_dict'][key.replace('supert.', '')] = checkpoint['state_dict'].pop(key)
                if 'patch_embed' in key:
                    checkpoint['state_dict'].pop(key.replace('supert.', ''))

            self.supert.load_state_dict(checkpoint['state_dict'], strict=False)
        
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
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.1,
        #     patience=self.es_patience//2,
        #     min_lr=1e-8,
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
        pred = self.supert(input)
        pred = pred.reshape(pred.size(0), -1)
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


        res = int(self.num_seg**0.5)
        features = features.reshape(features.size(0), res, res, -1).permute(0, 3, 1, 2)
        seq_mask = seq_mask.reshape(seq_mask.size(0), res, res)
        features, seq_mask = self.mixup(features, seq_mask)
        
        seq_mask = seq_mask.reshape(seq_mask.size(0), -1)

        # forward pass
        
        pred = self.forward(features)

        loss = self.loss(pred, seq_mask)
        
        pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = mask.shape[0]
        img_size = mask.shape[2]
        if torch.sum(segments) != 0 :

            segments = segments.reshape([batch_size, -1]) # batch, img_size^2

            samples = []
            for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
                plt_image = masked[labels-1].reshape([img_size, img_size])
                samples.append(plt_image)

            samples = torch.tensor(np.expand_dims(np.array(samples), 1)).cuda()
        else:
            samples = torch.sigmoid(pred).reshape(pred.size(0), 1, res, res)
            samples = F.interpolate(samples, (self.image_size, self.image_size), mode='bilinear')
            
        
        prec, recall = torch.zeros(samples.shape[0], 1), torch.zeros(samples.shape[0], 1)
        pred = samples.reshape(samples.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)
        
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
        if self.current_epoch >= self.warmup_epochs:
            self.scheduler.step()
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
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
        res = int(self.num_seg**0.5)
        pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = mask.shape[0]
        img_size = mask.shape[2]
        segments = segments.reshape([batch_size, -1]) # batch, img_size^2

        if torch.sum(segments) != 0 :

            segments = segments.reshape([batch_size, -1]) # batch, img_size^2

            samples = []
            for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
                plt_image = masked[labels-1].reshape([img_size, img_size])
                samples.append(plt_image)

            samples = torch.tensor(np.expand_dims(np.array(samples), 1)).cuda()
        else:
            samples = torch.sigmoid(pred).reshape(pred.size(0), 1, res, res)
            samples = F.interpolate(samples, (self.image_size, self.image_size), mode='bilinear')
        
        mae = torch.mean(torch.abs(samples - mask))
        if dataloader_idx == 0:
            self.preds.append(samples)
            self.masks.append(mask)
        elif dataloader_idx == 1:
            self.preds_test.append(samples)
            self.masks_test.append(mask)

        
        prec, recall = torch.zeros(samples.size(0), self.num_thresholds).cuda(), torch.zeros(samples.size(0), self.num_thresholds).cuda()
        pred = samples.reshape(samples.size(0), -1)
        mask = mask.reshape(mask.size(0), -1)
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds).cuda()
        for j in range(self.num_thresholds):
            y_temp = (pred >= thlist[j]).float()
            tp = (y_temp * mask).sum(dim=-1)
            # avoid prec becomes 0
            prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
        # (batch, threshold)
        if dataloader_idx == 0:
            self.precs.append(prec)
            self.recalls.append(recall)
            self.validation_step_outputs.append(mae)
            self.test_iteration += 1
        elif dataloader_idx == 1:
            self.precs_test.append(prec)
            self.recalls_test.append(recall)
        return mae


    def on_validation_epoch_end(self):
        prec = torch.cat(self.precs, dim=0).cuda().mean(dim=0)
        recall = torch.cat(self.recalls, dim=0).cuda().mean(dim=0)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds).cuda()
        self.log('Validation Max F Score', torch.max(f_score))
        self.log('Validation Max F Threshold', thlist[torch.argmax(f_score)])

        pred = torch.cat(self.preds, 0).cuda()
        mask = torch.cat(self.masks, 0).cuda().round().float()
        self.log('Validation MAE', torch.mean(torch.abs(pred-mask)))

        prec = torch.cat(self.precs_test, dim=0).cuda().mean(dim=0)
        recall = torch.cat(self.recalls_test, dim=0).cuda().mean(dim=0)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds).cuda()
        self.log('Test Max F Score', torch.max(f_score))
        self.log('Test Max F Threshold', thlist[torch.argmax(f_score)])

        pred = torch.cat(self.preds_test, 0)
        mask = torch.cat(self.masks_test, 0).round().float()
        self.log('Test MAE', torch.mean(torch.abs(pred-mask)))
        # if self.current_epoch >= self.warmup_epochs:
        #     self.scheduler.step(torch.mean(torch.stack(self.validation_step_outputs)))
        self.validation_step_outputs.clear()

    def on_validation_start(self):
        self.preds = []
        self.masks = []
        self.precs = []
        self.recalls = []

        self.preds_test = []
        self.masks_test = []
        self.precs_test = []
        self.recalls_test = []

        self.validation_step_outputs = []

    def on_test_start(self):
        self.preds = []
        self.masks = []
        self.precs = []
        self.recalls = []
        self.test_step_outputs = []


    def test_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        features = batch['features']
        seq_mask = batch['seq_mask']
        segments = batch['segments']
        mask = batch['mask'].cpu()


        # forward pass
        res = int(self.num_seg**0.5)
        features = features.reshape(features.size(0), res, res, -1).permute(0, 3, 1, 2)
        pred = self.forward(features)

        pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = mask.shape[0]
        img_size = mask.shape[2]
        if torch.sum(segments) != 0 :

            segments = segments.reshape([batch_size, -1]) # batch, img_size^2

            samples = []
            for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
                plt_image = masked[labels-1].reshape([img_size, img_size])
                samples.append(plt_image)

            samples = torch.tensor(np.expand_dims(np.array(samples), 1))
        else:
            samples = torch.sigmoid(pred).reshape(pred.size(0), 1, res, res)
            samples = F.interpolate(samples, (self.image_size, self.image_size), mode='bilinear').detach().cpu()
        # tensorboard.add_images('Test Pred', samples, self.test_iteration)

        # tensorboard.add_images('Test GT', samples_mask, self.test_iteration)
        # tensorboard.add_images('Test Image', img, self.test_iteration)

        mae = torch.mean(torch.abs(samples - mask))
        self.preds.append(samples)
        self.masks.append(mask)
        prec, recall = torch.zeros(samples.size(0), 256).cuda(), torch.zeros(samples.size(0), 256).cuda()
        pred = samples.reshape(samples.size(0), -1)
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
        self.test_step_outputs.append(mae)
        self.test_iteration += 1
        return mae
    
    def on_test_epoch_end(self):
        prec = torch.cat(self.precs, dim=0).cuda().mean(dim=0)
        recall = torch.cat(self.recalls, dim=0).cuda().mean(dim=0)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, 256).cuda()
        self.log('Final Test Max F Score', torch.max(f_score))
        self.log('Final Test Max F Threshold', thlist[torch.argmax(f_score)])

        pred = torch.cat(self.preds, 0).cuda()
        mask = torch.cat(self.masks, 0).cuda().round().float()
        self.log('Final Test MAE', torch.mean(torch.abs(pred-mask)))
        




if __name__ == "__main__":
    pass