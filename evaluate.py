import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
from Models.EGNet import build_model
from ptflops import get_model_complexity_info
matplotlib.use('Agg')
import time
from train import DATALOADER_DIRECTORY, MODEL_DIRECTORY
from util.util import AverageMeter
import cv2
import os


k = 5
top_k_results = []
top_threshold = 0
worst_threshold = 1
worst_k_results = []

def prepare_input(time_steps):
    x1 = torch.FloatTensor(time_steps[0],time_steps[1] , 11).cuda()
    x2 = torch.FloatTensor(time_steps[0], time_steps[1], time_steps[1], 5).cuda()
    return dict(input = [x1, x2])

def prepare_input_gat(time_steps):
    x1 = torch.FloatTensor(time_steps[0],time_steps[1] , 11).cuda()
    x2 = torch.FloatTensor(time_steps[0], time_steps[1], time_steps[1]).cuda()
    return dict(input = [x1, x2])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='Model name to train', required=True, default=None)
    parser.add_argument('--dataloader', help="Type of dataloader", required=True, default=None)
    parser.add_argument("--model_checkpoint_file",
                        help="Directory of pre-trained model")
    parser.add_argument('--dataset_val', help='Directory of your validation Dataset', required=True, default=None)
    parser.add_argument('--cuda', help="'cuda' for cuda, 'cpu' for cpu, default = cuda",
                        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpus', help="Number of gpus to use for training", default=1, type=int)
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--num_workers', help="# of dataloader cpu process", default=0, type=int)
    parser.add_argument('--num_seg', help='Approximate number of segmentations', default=600, type=int)
    parser.add_argument('--dropout', help='Dropout for Transformers', default=0., type=float)
    parser.add_argument('--seed', help='Seed for reproduceability', 
                        default=42, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--compactness', help='Compactness for SLIC', type=float, default=10)
    parser.add_argument('--size', help='Image size for DUTS', type=int, default=224)
    parser.add_argument('--coeff', help='Number of coefficients for fft', type=int, default=7)
    parser.add_argument('--dilation', help='Dilation for local transformer', type=int, default=5)
    parser.add_argument('--downsample', help='Downsample resolution', type=int, default=28)
    parser.add_argument('--tag', help='Tag for differentiating runs on CC', default='', type=str)
    parser.add_argument('--tfmhp', default=[8, 16, 6], 
                    nargs=3, metavar=('Heads', 'Hidden Dim', 'Number of Layers'),
                    type=int, help='Hyperparameters for Transformer')


    dict_args = vars(parser.parse_args())

    # Data: load validation dataloader
    print("[p] getting val_dataloader")
    assert dict_args["dataloader"] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args["dataloader"]](**dict_args)
    val_dataloader = data_module.val_dataloader()

    # Initialize model to test
    assert dict_args["model"] in MODEL_DIRECTORY
    model = MODEL_DIRECTORY[dict_args["model"]](**dict_args)
    model = model.load_from_checkpoint(
        checkpoint_path=dict_args["model_checkpoint_file"],
        map_location=dict_args["cuda"],
    ).cuda()
    model.eval()
    with torch.no_grad():

        # egnet = build_model('resnet').cuda()
        # egnet.eval()

        # Iterate through each batch to generate visuals
        print("[p] processing batches")
        item_idx = 0
        preds = []
        masks = []
        precs = []
        recalls = []
        for batch in tqdm(val_dataloader):
            features = batch['features']
            seq_mask = batch['seq_mask']
            segments = batch['segments']
            mask = batch['mask']
            img = batch['img']
            file_names = batch['file_name']
            # pos_enc = batch['pos_enc']
            # edge_features = batch['edge_features']


            features = features.cuda()
            seq_mask = seq_mask.cuda()
            # pos_enc = pos_enc.cuda()
            # edge_features = edge_features.cuda()
            
            
            
            

            if dict_args['model'] == 'SP_GAT':
                adj = batch['neighbor_array']
                adj = adj.cuda()

                pred = model([features, adj])
            elif dict_args['model'] == 'SP_CNN_LIN':
                device = 'cuda'
                pred = model(features)
                pred = pred.reshape(pred.size(0), -1)
            elif dict_args['model'] == 'SP_TFM':
                adj = batch['neighbor_array']
                # distances = batch['edge_features']
                adj = adj.cuda()
                pred = model(features, adj, None)
            elif dict_args['model'] == 'SP_Baseline_LAP':
                adj = batch['neighbor_array']
                adj = adj.cuda()

                lap = batch['pos_enc'].cuda()
                device = 'cuda'
                model.cpu()
                a = torch.cuda.memory_allocated(device)
                model.to(device)
                b = torch.cuda.memory_allocated(device)
                model_memory = b - a
                pred = model(features, adj, lap)



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
            preds.append(samples)
            masks.append(samples_mask)
            prec, recall = torch.zeros(samples_mask.shape[0], 256), torch.zeros(samples_mask.shape[0], 256)
            pred = samples.reshape(samples.shape[0], -1)
            mask = samples_mask.reshape(samples_mask.shape[0], -1)
            thlist = torch.linspace(0, 1 - 1e-10, 256)
            for j in range(256):
                y_temp = (pred >= thlist[j]).float()
                tp = (y_temp * mask).sum(dim=-1)
                # avoid prec becomes 0
                prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)

            
            precs.append(prec)
            recalls.append(recall)

            for sample, file_name in zip(samples, file_names):
                output = (sample.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
                if not os.path.exists('./visualization/'+dict_args['model']+'/'+dict_args['dataset_val'].split('/')[-1]):
                    os.makedirs('./visualization/'+dict_args['model']+'/'+dict_args['dataset_val'].split('/')[-1])
                cv2.imwrite('./visualization/'+dict_args['model']+'/'+dict_args['dataset_val'].split('/')[-1]+'/'+file_name, output)


    prec = torch.cat(precs, dim=0).mean(dim=0)
    recall = torch.cat(recalls, dim=0).mean(dim=0)
    beta_square = 0.3
    f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
    print('Max F score:',torch.max(f_score))
    pred = torch.cat(preds, 0)
    mask = torch.cat(masks, 0).round().float()
    print('MAE:',torch.mean(torch.abs(pred-mask)))

        
    # print(flops)
    # (hist, _) = np.histogram(total_score, bins=100, range=(0, 1), density=True)
    # fig = plt.figure(num=1, clear=True)
    # ax1 = fig.add_subplot(111)
    # ax1.bar(list(range(0,100)), hist)
    # ax1.set_xlabel('F1-score bins')
    # ax1.set_title('Histogram of F1-scores on DUTS-TE')
    # ax1.set_ylabel('Normalized frequency')
    # fig.savefig('./results/histogram.png')





        
        

      

if __name__ == "__main__":
    main()
 
