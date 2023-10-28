import pytorch_lightning as pl
import torch
from skimage.measure import regionprops_table
from Blocks.GraphBlocks import GAT
import torch.nn.functional as F
import numpy as np
from fast_slic.avx2 import SlicAvx2
from Blocks.TransformerBlocks import SuperTPos

class SuperTransformerPos(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size")
        self.lr = kwargs.get("lr")
        self.num_seg = kwargs.get('num_seg')
        self.es_patience = kwargs.get('es_patience')

        # must be defined for logging computational graph
        def get_seq_len():
            img_np = np.random.rand(300, 300, 3).astype(np.uint8)
            slic = SlicAvx2(num_components=self.num_seg, compactness=10, min_size_factor=0)
            segments = slic.iterate(img_np)+1

            regions = regionprops_table(segments, intensity_image=img_np, properties=('label', 'centroid', 'area', 'intensity_mean', 'extent', 'coords', 'eccentricity'))
            seq_len = len(regions['label'])
            return seq_len
        seq_len = get_seq_len()
        # self.example_input_array = torch.rand((1, seq_len, 8))

        # Generator that produces the HeatMap
        self.supert = SuperTPos(4, seq_len, 64, 6, 4, 64, 300, 300, dim_head=16)

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
        :param adj: adjacent matrix 
        :return: 2D heatmap, 16x3 joint inferences, 2D reconstructed heatmap
        """        

        pred = self.supert(x)

        return pred

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
        adj = batch['neighbor_array']


        features = features.cuda()
        seq_mask = seq_mask.cuda()
        adj = adj.cuda()

        # forward pass
        
        pred = self.forward(features)

        loss = self.loss(pred, seq_mask)

        self.log('loss', loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute the metrics for validation batch
        validation loop: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks
        """
        tensorboard = self.logger.experiment
        features = batch['features']
        seq_mask = batch['seq_mask']
        segments = batch['segments']
        mask = batch['mask']
        img = batch['img']
        adj = batch['neighbor_array']


        features = features.cuda()
        seq_mask = seq_mask.cuda()
        adj = adj.cuda()


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
        
        samples_mask = []
        for masked, labels in zip(seq_mask_numpy, segments.cpu().numpy()):
            plt_image = masked[labels-1].reshape([img_size, img_size])
            samples_mask.append(plt_image)

        samples_mask = torch.tensor(np.expand_dims(np.array(samples_mask), 1))
        if batch_idx == 0:
            tensorboard.add_images('Pred', samples)
            tensorboard.add_images('GT', samples_mask)
            tensorboard.add_images('Image', img)

        mae = torch.mean(torch.abs(samples - samples_mask))
      
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
        features = batch['features']
        seq_mask = batch['seq_mask']
        segments = batch['segments']
        mask = batch['mask']
        img = batch['img']
        adj = batch['neighbor_array']


        features = features.cuda()
        seq_mask = seq_mask.cuda()
        adj = adj.cuda()


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
        tensorboard.add_images('Pred', samples)
        samples_mask = []
        for masked, labels in zip(seq_mask_numpy, segments.cpu().numpy()):
            plt_image = masked[labels-1].reshape([img_size, img_size])
            samples_mask.append(plt_image)

        samples_mask = torch.tensor(np.expand_dims(np.array(samples_mask), 1))
        tensorboard.add_images('GT', samples_mask)
        tensorboard.add_images('Image', img)

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