import datetime
import os
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from network import SuperT, MLP
from dataset import DUTSDataset
from GAT import GAT


if __name__ == '__main__':
    torch.cuda.manual_seed_all(1234)
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    print("Default path : " + os.getcwd())
    parser.add_argument("--load",
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/file/d/109a0hLftRZ5at5hwpteRfO1A6xLzf8Na/view?usp=sharing\n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model")
    parser.add_argument('--dataset', help='Directory of your Dataset', required=True, default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 20', default=20, type=int)
    parser.add_argument('-lr', '--learning_rate', help='learning_rate. default = 0.001', default=0.001, type=float)
    parser.add_argument('--lr_decay', help='Learning rate decrease by lr_decay time per decay_step, default = 0.1',
                        default=0.1, type=float)
    parser.add_argument('--decay_step', help='Learning rate decrease by lr_decay time per decay_step,  default = 1E100',
                        default=1E100, type=int)
    parser.add_argument('--display_freq', help='display_freq to display result image on Tensorboard',
                        default=1000, type=int)


    args = parser.parse_args()

    device = torch.device(args.cuda)
    batch_size = args.batch_size
    epoch = args.epoch
    seq_len = 625
    duts_dataset = DUTSDataset(args.dataset, seq_len)
    load = args.load
    start_iter = 0
    # model = SuperT(feature_dim=3, seq_len=seq_len, dim=64, depth=3, heads=1, mlp_dim=128, dropout=0.).cuda()
    model = GAT(8, 8,  0., 0.2, 8).cuda()
    model = model.float() 
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters:', pytorch_total_params)
    now = datetime.datetime.now()
    start_epo = 0 


    if load is not None:
        state_dict = torch.load(load, map_location=args.cuda)

        start_iter = int(load.split('epo_')[1].strip('step.ckpt')) + 1
        start_epo = int(load.split('/')[-1].split('epo')[0])
        now = datetime.datetime.strptime(load.split('/')[-2], '%m%d%H%M')

        print("Loading Model from {}".format(load))
        print("Start_iter : {}".format(start_iter))
        print("now : {}".format(now.strftime('%m%d%H%M')))
        model.load_state_dict(state_dict)

        print('Loading_Complete')

    # Optimizer Setup
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    decay_step = args.decay_step  # from 50000 step
    learning_rate = learning_rate * (lr_decay ** (start_iter // decay_step))
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    # Dataloader Setup
    dataloader = DataLoader(duts_dataset, batch_size, shuffle=True, num_workers=4)
    # Logger Setup
    os.makedirs(os.path.join('log', now.strftime('%m%d%H%M')), exist_ok=True)
    weight_save_dir = os.path.join('models', 'state_dict', now.strftime('%m%d%H%M'))
    os.makedirs(os.path.join(weight_save_dir), exist_ok=True)
    writer = SummaryWriter(os.path.join('log', now.strftime('%m%d%H%M')))
    iterate = start_iter
    display_max = iterate // args.display_freq
    model_save_max = iterate // (args.batch_size * (1000 // args.batch_size))
    decay_max = iterate // (args.batch_size * (decay_step // args.batch_size))
    for epo in range(start_epo, epoch):
        print("\nEpoch : {}".format(epo))
        for i, batch in enumerate(tqdm(dataloader)):
            opt.zero_grad()

            features = batch['features'].to(device)
            seq_mask = batch['seq_mask'].to(device) 
            segments = batch['segments']
            mask = batch['mask']
            img = batch['img']
            adj = batch['neighbor_array'].to(device)
            
      
            pred = model(features, adj)

            loss = F.binary_cross_entropy_with_logits(torch.squeeze(pred), seq_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            opt.step()

            writer.add_scalar('loss', float(loss), global_step=iterate)
            pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1
            seq_mask_numpy = seq_mask.detach().cpu().numpy()
            batch_size = img.shape[0]
            img_size = img.shape[2]
            segments = segments.reshape([batch_size, -1]) # batch, img_size^2

            if iterate // args.display_freq > display_max:
                display_max = iterate // args.display_freq
                samples = []
                for masked, labels in zip(pred_numpy, segments):
                    plt_image = masked[labels-1].reshape([img_size, img_size])
                    samples.append(plt_image)

                samples = np.expand_dims(np.array(samples), 1)
                writer.add_image('Pred', torch.tensor(samples), iterate)
                samples_mask = []
                for masked, labels in zip(seq_mask_numpy, segments):
                    plt_image = masked[labels-1].reshape([img_size, img_size])
                    samples_mask.append(plt_image)

                samples_mask = np.expand_dims(np.array(samples_mask), 1)
                writer.add_image('GT', torch.tensor(samples_mask), iterate)
                writer.add_image('Image', img, iterate)
                

            if iterate // (args.batch_size * (1000 // args.batch_size)) > model_save_max:
                model_save_max = iterate // (args.batch_size * (1000 // args.batch_size))
                if i != 0:
                    torch.save(model.state_dict(),
                               os.path.join(weight_save_dir, '{}epo_{}step.ckpt'.format(epo, iterate)))
                    if len(os.listdir(os.path.join(weight_save_dir))) > 5:
                        model_dict = {}
                        for model_path in os.listdir(os.path.join(weight_save_dir)):
                            iter = model_path.split('epo_')[1].split('step')[0]
                            model_dict[model_path] = int(iter)
                        total_files = len(model_dict)
                        for k, v in sorted(model_dict.items(), key=lambda item: item[1]):
                            os.remove(os.path.join(weight_save_dir, k))
                            total_files -= 1
                            if total_files == 5:
                                break


            if iterate // (args.batch_size * (decay_step // args.batch_size)) > decay_max and i != 0:
                decay_max = (args.batch_size * (decay_step // args.batch_size))
                learning_rate *= lr_decay
                opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            iterate += args.batch_size

