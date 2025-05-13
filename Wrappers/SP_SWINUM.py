from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
# from Blocks.swintransformer_original_rpe import SwinUTransformer
from Blocks.swinunet_mix_ape import SwinUTransformer
# from Models.SP_SWIN import SP_SWINU
import torch.nn.functional as F
import numpy as np
from dataset.constants import *
from util.util import get_input_dim
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from dataset.mixup import MixupSaliency
import cv2
from util.util import eval_e, S_object, S_region

class SP_SWINUM_Wrapper(pl.LightningModule):
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
        resample_points = int(((self.size**2)//self.num_seg)**0.5)*4
        self.resample_points = resample_points
        
        # if self.coeff == -1: # use all coefficients
        #     kwargs['coeff'] = resample_points-1
        #     self.coeff = resample_points-1
        # else:
        #     assert resample_points-1 >= self.coeff
        input_dim = get_input_dim(kwargs)

        res = int(self.num_seg**0.5)
        # Generator that produces the HeatMap
        self.supert = SwinUTransformer(img_size=res, in_chans=input_dim, patch_size=1, window_size=self.window_size,
                                       embed_dim=self.dims, depths=self.depths,
                                         num_heads=self.heads, mlp_ratio=self.mlp_ratio, attn_drop_rate=self.dropout_edge, drop_rate=self.dropout,
                                         drop_path_rate=self.dp)
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
        if self.pretrain:
            checkpoint = torch.load(self.pretrain)
            for key in list(checkpoint['state_dict'].keys()):
                checkpoint['state_dict'][key.replace('supert.', '')] = checkpoint['state_dict'].pop(key)
            
            self.supert.load_state_dict(checkpoint['state_dict'], strict=False)
            
        
        self.save_hyperparameters()
        

    def loss(self, pred, label, sizes):
        """
        Defining the loss funcition:
        """
    
        # targets = label.float()
        # probs = torch.sigmoid(pred).squeeze()

        # bce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(), targets, reduction='none')

        # p_t = probs * targets + (1 - probs) * (1 - targets)
        # focal_weight = (1 - p_t) ** 2.

        # focal_loss = (focal_weight * bce_loss).mean()


        # pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p if label=1 else 1-p
        # focal_loss = -0.25 * (1 - pt) ** 2.0 * pt.log()
        # focal_loss = focal_loss.mean()

        # intersection = (probs * targets).sum(dim=1)
        # union = probs.sum(dim=1) + targets.sum(dim=1)
        # dice_score = (2 * intersection + 1e-8) / (union + 1e-8)
        # dice_loss = 1 - dice_score
        # dice_loss = dice_loss.mean()
        # loss = focal_loss #+ dice_loss
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(pred), torch.squeeze(label))
        # weights = sizes / (sizes.sum(dim=1, keepdim=True))  # (B, K)
        # weighted_loss = (loss * weights).sum(dim=1).mean()

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
        self.train_fscores = 0
        self.num_samples = 0

    def on_train_start(self):
        self.log('Flops', self.flops)
        self.log('Parameters', self.num_parameters)
    
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

        sizes = features[:, :, -18]
        res = int(self.num_seg**0.5)
        features = features.reshape(features.size(0), res, res, -1).permute(0, 3, 1, 2)
        if self.aug_strat == 4 and 'RS' not in self.dataloader:
            seq_mask = seq_mask.reshape(seq_mask.size(0), res, res)
            features, seq_mask = self.mixup(features, seq_mask)
            
            seq_mask = seq_mask.reshape(seq_mask.size(0), -1)

        # if self.aug_strat == 4 and 'RS' not in self.dataloader:
        #     features, seq_mask = semantic_cutmix(features, seq_mask, self.cutmix_prob)
        #     features = features.reshape(features.size(0), res, res, -1).permute(0, 3, 1, 2)

        
        if torch.sum(sizes[0, :]) != self.size**2:
            assert 'Sizes not aligning'

        # forward pass
        if torch.sum(torch.isnan(features)) > 0:
            assert 0 
        pred = self.forward(features)
        
        loss = self.loss(pred, seq_mask, sizes)
        
        pred_numpy = torch.sigmoid(pred) # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = mask.shape[0]
        img_size = mask.shape[2]
        
        samples = pred_numpy
            
        
        prec, recall = torch.zeros(samples.shape[0], 1), torch.zeros(samples.shape[0], 1)
        pred = samples.reshape(samples.shape[0], -1)
        mask = seq_mask.reshape(mask.shape[0], -1)
        
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
        self.log('loss', loss.item(), prog_bar=True)
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
        pred_numpy = torch.sigmoid(pred).detach().cpu() # batch, seq_len, 1
        seq_mask_numpy = seq_mask.detach().cpu().numpy()
        batch_size = mask.shape[0]
        img_size = self.size
        segments = segments.reshape([batch_size, -1]) # batch, img_size^2

        if torch.sum(segments) != 0 :

            segments = segments.reshape([batch_size, -1]) # batch, img_size^2

            samples = []
            for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
                plt_image = masked[labels-1].reshape([img_size, img_size])
                plt_image = F.interpolate(plt_image.unsqueeze(0).unsqueeze(0), (mask.size(2), mask.size(3)), mode='bilinear')
                
                samples.append(plt_image)

            samples = torch.cat(samples, dim=0).cuda()
        else:
            samples = torch.sigmoid(pred).reshape(pred.size(0), 1, res, res)
            samples = F.interpolate(samples, (self.image_size, self.image_size), mode='bilinear')
        
        mae = torch.sum(torch.mean(torch.abs(samples - mask), dim=tuple(range(1, len(samples.size())))))
        
        if dataloader_idx == 0:
            self.maes += mae
            self.mean_num += features.size(0)
        elif dataloader_idx == 1:
            self.duts_maes_test += mae
            self.duts_mean_num_test += features.size(0)
        elif dataloader_idx == 2:
            self.dutso_maes_test += mae
            self.dutso_mean_num_test += features.size(0)
        elif dataloader_idx == 3:
            self.ecssd_maes_test += mae
            self.ecssd_mean_num_test += features.size(0)
        elif dataloader_idx == 4:
            self.hku_maes_test += mae
            self.hku_mean_num_test += features.size(0)
        elif dataloader_idx == 5:
            self.pascal_maes_test += mae
            self.pascal_mean_num_test += features.size(0)

        
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
            self.precs += prec.sum(0)
            self.recalls += recall.sum(0)
            self.validation_step_outputs.append(mae)
            self.test_iteration += 1
        elif dataloader_idx == 1:
            self.duts_precs_test += prec.sum(0)
            self.duts_recalls_test += recall.sum(0)
        elif dataloader_idx == 2:
            self.dutso_precs_test += prec.sum(0)
            self.dutso_recalls_test += recall.sum(0)
        elif dataloader_idx == 3:
            self.ecssd_precs_test += prec.sum(0)
            self.ecssd_recalls_test += recall.sum(0)
        elif dataloader_idx == 4:
            self.hku_precs_test += prec.sum(0)
            self.hku_recalls_test += recall.sum(0)
        elif dataloader_idx == 5:
            self.pascal_precs_test += prec.sum(0)
            self.pascal_recalls_test += recall.sum(0)
        return mae


    def on_validation_epoch_end(self):
        prec = self.precs/self.mean_num
        recall = self.recalls/self.mean_num
       
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds)
        self.log('Validation Max F Score', torch.max(f_score))
        self.log('Validation Max F Threshold', thlist[torch.argmax(f_score)])

        self.log('Validation MAE', self.maes/self.mean_num)

        prec = self.duts_precs_test/self.duts_mean_num_test
        recall = self.duts_recalls_test/self.duts_mean_num_test
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds)
        self.log('DUTS Max F Score', torch.max(f_score))
        # self.log('Test Max F Threshold', thlist[torch.argmax(f_score)])

        self.log('DUTS MAE', self.duts_maes_test/self.duts_mean_num_test)

        prec = self.dutso_precs_test/self.dutso_mean_num_test
        recall = self.dutso_recalls_test/self.dutso_mean_num_test
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds)
        self.log('DUTSO Max F Score', torch.max(f_score))
        # self.log('Test Max F Threshold', thlist[torch.argmax(f_score)])

        self.log('DUTSO MAE', self.dutso_maes_test/self.dutso_mean_num_test)

        prec = self.ecssd_precs_test/self.ecssd_mean_num_test
        recall = self.ecssd_recalls_test/self.ecssd_mean_num_test
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds)
        self.log('ECSSD Max F Score', torch.max(f_score))
        # self.log('Test Max F Threshold', thlist[torch.argmax(f_score)])

        self.log('ECSSD MAE', self.ecssd_maes_test/self.ecssd_mean_num_test)

        prec = self.hku_precs_test/self.hku_mean_num_test
        recall = self.hku_recalls_test/self.hku_mean_num_test
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds)
        self.log('HKU Max F Score', torch.max(f_score))
        # self.log('Test Max F Threshold', thlist[torch.argmax(f_score)])

        self.log('HKU MAE', self.hku_maes_test/self.hku_mean_num_test)

        prec = self.pascal_precs_test/self.pascal_mean_num_test
        recall = self.pascal_recalls_test/self.pascal_mean_num_test
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, self.num_thresholds)
        self.log('PASCAL Max F Score', torch.max(f_score))
        # self.log('Test Max F Threshold', thlist[torch.argmax(f_score)])

        self.log('PASCAL MAE', self.pascal_maes_test/self.pascal_mean_num_test)
        # if self.current_epoch >= self.warmup_epochs:
        #     self.scheduler.step(torch.mean(torch.stack(self.validation_step_outputs)))
        self.validation_step_outputs.clear()

    def on_validation_start(self):
        self.maes = 0
        self.mean_num = 0
        
        self.precs = torch.zeros(self.num_thresholds).cuda()
        self.recalls = torch.zeros(self.num_thresholds).cuda()

        self.duts_maes_test = 0
        self.duts_mean_num_test = 0

        self.duts_precs_test = torch.zeros(self.num_thresholds).cuda()
        self.duts_recalls_test = torch.zeros(self.num_thresholds).cuda()

        self.dutso_maes_test = 0
        self.dutso_mean_num_test = 0

        self.dutso_precs_test = torch.zeros(self.num_thresholds).cuda()
        self.dutso_recalls_test = torch.zeros(self.num_thresholds).cuda()

        self.ecssd_maes_test = 0
        self.ecssd_mean_num_test = 0

        self.ecssd_precs_test = torch.zeros(self.num_thresholds).cuda()
        self.ecssd_recalls_test = torch.zeros(self.num_thresholds).cuda()

        self.hku_maes_test = 0
        self.hku_mean_num_test = 0

        self.hku_precs_test = torch.zeros(self.num_thresholds).cuda()
        self.hku_recalls_test = torch.zeros(self.num_thresholds).cuda()

        self.pascal_maes_test = 0
        self.pascal_mean_num_test = 0

        self.pascal_precs_test = torch.zeros(self.num_thresholds).cuda()
        self.pascal_recalls_test = torch.zeros(self.num_thresholds).cuda()

        self.validation_step_outputs = []

    def on_test_start(self):
        self.maes = 0
        self.mean_num = 0
        
        self.precs = torch.zeros(256).cuda()
        self.recalls = torch.zeros(256).cuda()
        self.test_step_outputs = []

        self.e_measure_scores = torch.zeros(255).cuda()
        self.s_measure_q = 0.0
        self.times = []



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

        pred = self.forward(features)

 
        pred_numpy = torch.sigmoid(pred).detach().cpu() # batch, seq_len, 1

        batch_size = mask.shape[0]
        img_size = self.size
        if torch.sum(segments) != 0 :

            segments = segments.reshape([batch_size, -1]) # batch, img_size^2

            samples = []
            for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
                plt_image = masked[labels-1].reshape([img_size, img_size])
                plt_image = F.interpolate(plt_image.unsqueeze(0).unsqueeze(0), (mask.size(2), mask.size(3)), mode='bilinear')
                samples.append(plt_image)

            samples = torch.cat(samples, dim=0).cuda()
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

        mae = torch.sum(torch.mean(torch.abs(samples  - mask), dim=tuple(range(1, len(samples.size())))))
        self.maes += mae
        self.mean_num += features.size(0)

        for pred, gt in zip(samples, mask):
            self.e_measure_scores += eval_e(pred, gt, 255)
            y = gt.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 -x
            elif y == 1:
                x = pred.mean()
                Q = x
            else:
                gt[gt>=0.5] = 1
                gt[gt<0.5] = 0
                Q = 0.5 * S_object(pred, gt) + (1-0.5) * S_region(pred, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            self.s_measure_q += Q.item()


        prec, recall = torch.zeros(samples.size(0), 256).cuda(), torch.zeros(samples.size(0), 256).cuda()
        pred = samples.reshape(samples.size(0), -1)
        mask = mask.reshape(mask.size(0), -1)
        thlist = torch.linspace(0, 1 - 1e-10, 256).cuda()
        for j in range(256):
            y_temp = (pred >= thlist[j]).float()
            tp = (y_temp * mask).sum(dim=-1)
            # avoid prec becomes 0
            prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)
        # (batch, threshold)
        self.precs += prec.sum(0)
        self.recalls += recall.sum(0)
        self.test_step_outputs.append(mae)
        self.test_iteration += 1
        return mae
    
    def on_test_epoch_end(self):
        prec = self.precs/self.mean_num
        recall = self.recalls/self.mean_num
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        thlist = torch.linspace(0, 1 - 1e-10, 256)
        self.log('Final Test Max F Score', torch.max(f_score))
        self.log('Final Test Max F Threshold', thlist[torch.argmax(f_score)])

        self.log('Final Test MAE', self.maes/self.mean_num)
        self.log('Final Test E measure', torch.max(self.e_measure_scores)/self.mean_num)
        self.log('Final Test S measure', self.s_measure_q/self.mean_num)

        




if __name__ == "__main__":
    pass