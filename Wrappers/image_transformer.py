import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from Blocks.TransformerBlocks import Transformer

class ImageTransformer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.es_patience = kwargs.get('es_patience')
        self.downsample = kwargs.get('downsample')
        # must be defined for logging computational graph
        self.example_input_array = torch.rand((1, 3, self.downsample, self.downsample))

        # Generator that produces the HeatMap
        self.transformer = Transformer(64, 6, 8, 32, 32)
        self.pos_enc = nn.Parameter(torch.randn(1, self.downsample**2, 64))
        self.linear_proj = nn.Linear(3, 64)
        self.final_linear = nn.Linear(64, 1)
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
        x = x.permute(0, 2, 3, 1) # batch, X, Y, 3
        x_size = x.size()
        x = self.linear_proj(x) # batch, X, Y, channels
        x = x.reshape(x.size(0), -1, x.size(3)) # batch, XY, channels
        x += self.pos_enc
        x = self.transformer(x) # batch, XY, channels
        x = self.final_linear(x) # batch, XY, 1
        x = x.reshape(x_size[0], 1, x_size[1], x_size[2]) # batch, 1, X, Y

        return x

    def on_train_epoch_start(self):
        self.train_fscores = torch.zeros(256)
        self.num_samples = 0
    
    def on_train_epoch_end(self):
        fscores = self.train_fscores/self.num_samples
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        self.log('Train Max F Score', torch.max(fscores))
        self.log('Train Max F Threshold', thlist[torch.argmax(fscores)])



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

        original_size = img.size(-1)
        # forward pass
        img_downsampled = F.interpolate(img, (self.downsample, self.downsample), mode='bilinear')
        mask_downsampled = F.interpolate(mask, (self.downsample, self.downsample), mode='bilinear')
        
        pred = self.forward(img_downsampled)

        loss = self.loss(pred, mask_downsampled)


        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, (original_size, original_size), mode='bilinear')


        prec, recall = torch.zeros(pred.shape[0], 256), torch.zeros(pred.shape[0], 256)
        pred = pred.reshape(pred.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        for j in range(256):
            y_temp = (pred >= thlist[j]).float()
            tp = (y_temp * mask).sum(dim=-1)
            # avoid prec becomes 0
            prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
        # (batch, threshold)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        f_score = f_score.sum(dim=0)
        self.train_fscores += f_score
        self.num_samples += pred.size(0)
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


        original_size = img.size(-1)
        # forward pass
        img_downsampled = F.interpolate(img, (self.downsample, self.downsample), mode='bilinear')
        
        pred = self.forward(img_downsampled)


        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, (original_size, original_size), mode='bilinear')

        mae = torch.sum(torch.mean(torch.abs(pred - mask), dim=(1, 2, 3)))
        self.maes += mae
        prec, recall = torch.zeros(pred.shape[0], 256), torch.zeros(pred.shape[0], 256)
        pred = pred.reshape(pred.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        for j in range(256):
            y_temp = (pred >= thlist[j]).float()
            tp = (y_temp * mask).sum(dim=-1)
            # avoid prec becomes 0
            prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
        # (batch, threshold)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        f_score = f_score.sum(dim=0)
        self.fscores += f_score
        self.num_samples_val += pred.size(0)
      
        return mae


    def validation_epoch_end(self, validation_step_outputs):
        mae = self.maes/self.num_samples_val
        self.log('Validation MAE', mae)

        fscores = self.fscores/self.num_samples_val
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        self.log('Validation Max F Score', torch.max(fscores))
        self.log('Validation Max F Threshold', thlist[torch.argmax(fscores)])

        self.scheduler.step(torch.mean(torch.stack(validation_step_outputs)))

    def on_validation_start(self):
        self.maes = 0
        self.fscores = torch.zeros(256)
        self.num_samples_val = 0      




if __name__ == "__main__":
    pass