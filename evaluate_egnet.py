import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
from Models.EGNet import build_model
from ptflops import get_model_complexity_info
# matplotlib.use('Agg')
import time
from train import DATALOADER_DIRECTORY, MODEL_DIRECTORY


k = 5
top_k_results = []
top_threshold = 0
worst_threshold = 1
worst_k_results = []

def prepare_input(time_steps):
    x1 = torch.FloatTensor(time_steps[0],time_steps[1] , 11).cuda()
    x2 = torch.FloatTensor(time_steps[0], time_steps[1], time_steps[1], 5).cuda()
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


    parser.add_argument('--size', help='Image size for DUTS', type=int, default=224)


    dict_args = vars(parser.parse_args())

    # Data: load validation dataloader
    print("[p] getting val_dataloader")
    assert dict_args["dataloader"] in DATALOADER_DIRECTORY
    data_module = DATALOADER_DIRECTORY[dict_args["dataloader"]](**dict_args)
    val_dataloader = data_module.val_dataloader()

    # Initialize model to test
    # assert dict_args["model"] in MODEL_DIRECTORY
    # model = MODEL_DIRECTORY[dict_args["model"]](**dict_args)
    # model = model.load_from_checkpoint(
    #     checkpoint_path=dict_args["model_checkpoint_file"],
    #     map_location=dict_args["cuda"],
    # ).cuda()
    # model.eval()

    egnet = build_model('resnet').cuda()
    egnet.load_state_dict(torch.load('/home/abcd/abcde/EGNet/epoch_resnet.pth'))
    egnet.eval()

    # Iterate through each batch to generate visuals
    print("[p] processing batches")
    item_idx = 0
    flops = None
    total_score = []
    # prec, recall = torch.zeros(256).cuda(), torch.zeros(256).cuda()
    for batch in tqdm(val_dataloader):
        features = batch['features']
        seq_mask = batch['seq_mask']
        segments = batch['segments']
        mask = batch['mask'].cuda()
        img = batch['img'].cuda()
        # pos_enc = batch['pos_enc']
        edge_features = batch['edge_features']
        img = img*255
        norm = torch.Tensor([104.00699, 116.66877, 122.67892]).cuda()
        img = img-norm[None, :, None, None]


        features = features.cuda()
        seq_mask = seq_mask.cuda()
        # pos_enc = pos_enc.cuda()
        edge_features = edge_features.cuda()


        # forward pass
        # pred = model([features, edge_features])
        if flops is None:

            flops, params = get_model_complexity_info(egnet, input_res=(3, 512, 512))
            print(flops)
        
        start = time.time()
        _, _, pred_egnet = egnet(img)
        end = time.time()
        print(end-start)
        pred_egnet = torch.sigmoid(pred_egnet[-1])
        # pred_numpy = torch.sigmoid(pred).detach().cpu().numpy() # batch, seq_len, 1
        # seq_mask_numpy = seq_mask.detach().cpu().numpy()
        # batch_size = img.shape[0]
        # img_size = img.shape[2]
        # segments = segments.reshape([batch_size, -1]) # batch, img_size^2

        # samples = []
        # for masked, labels in zip(pred_numpy, segments.cpu().numpy()):
        #     plt_image = masked[labels-1].reshape([img_size, img_size])
        #     samples.append(plt_image)

        # samples = torch.tensor(np.expand_dims(np.array(samples), 1))
        
        # samples_mask = []
        # for masked, labels in zip(seq_mask_numpy, segments.cpu().numpy()):
        #     plt_image = masked[labels-1].reshape([img_size, img_size])
        #     samples_mask.append(plt_image)

        # samples_mask = torch.tensor(np.expand_dims(np.array(samples_mask), 1))

        # prec, recall = torch.zeros(samples_mask.shape[0], 256), torch.zeros(samples_mask.shape[0], 256)
        # pred = samples.reshape(samples.shape[0], -1)
        # mask = samples_mask.reshape(samples_mask.shape[0], -1)
        # thlist = torch.linspace(0, 1 - 1e-10, 256)
        # for j in range(256):
        #     y_temp = (pred >= thlist[j]).float()
        #     tp = (y_temp * mask).sum(dim=-1)
        #     # avoid prec becomes 0
        #     prec[:, j], recall[:, j] = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask.sum(dim=-1) + 1e-10)

        pred_egnet_ = pred_egnet.reshape(pred_egnet.size(0), -1)
        mask_ = mask.reshape(mask.size(0), -1)

        # thlist = torch.linspace(0, 1 - 1e-10, 256)
        # item_idx += pred_egnet.size(0)
        # for j in range(256):
        #     y_temp = (pred_egnet_ >= thlist[j]).float()
        #     tp = (y_temp * mask_).sum(dim=-1)
        #     # avoid prec becomes 0
        #     prec[j] += torch.sum((tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10))
        #     recall[j] += torch.sum((tp + 1e-10) / (mask_.sum(dim=-1) + 1e-10))
            

            

            
  
        
        y_temp = (pred_egnet_ >= 0.4627).float()
        tp = (y_temp * mask_).sum(dim=-1)
        prec, recall = (tp + 1e-10) / (y_temp.sum(dim=-1) + 1e-10), (tp + 1e-10) / (mask_.sum(dim=-1) + 1e-10)

        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        max_f_score = f_score
        


        # for sample, f_score, gt, img in zip(pred_egnet, max_f_score, mask, img):
        #     fig = plt.figure(num=1, clear=True)
        #     ax1 = fig.add_subplot(131)
        #     ax2 = fig.add_subplot(132)
        #     ax3 = fig.add_subplot(133)
    
     
        #     ax1.imshow(np.squeeze(sample.detach().cpu().numpy()), cmap='gray')
        #     ax1.set_title(f"Prediction")
        #     ax1.axis('off')

        #     ax2.imshow(np.squeeze(gt.detach().cpu().numpy()), cmap='gray')
        #     ax2.set_title(f"Ground Truth")
        #     ax2.axis('off')

        #     ax3.imshow(np.transpose((img+norm[:, None, None]).detach().cpu().numpy()/255., axes=(1, 2, 0)))
        #     ax3.set_title(f"Raw Image")
        #     ax3.axis('off')
            

        #     fig.savefig(f'./results/egnet_results/{item_idx}_{round(float(f_score.detach().cpu().numpy().item()), 2)}.png')
        #     total_score.append(float(f_score.detach().cpu().numpy().item()))
        #     item_idx += 1
    print(flops)
    (hist, _) = np.histogram(total_score, bins=100, range=(0, 1), density=False)
    hist = hist/np.sum(hist)
    fig = plt.figure(num=1, clear=True)
    ax1 = fig.add_subplot(111)
    ax1.bar(list(range(0,100)), hist)
    ax1.set_xlabel('F1-score bins')
    ax1.set_title('Histogram of F1-scores on DUTS-TE')
    ax1.set_ylabel('Normalized frequency')
    fig.savefig('./results/egnet_histogram.png')

    # prec /= item_idx
    # recall /= item_idx

    # beta_square = 0.3
    # f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
    # print(torch.max(f_score))
    # thlist = torch.linspace(0, 1 - 1e-10, 256)
    # print(thlist[torch.argmax(f_score)])




        
        

      

if __name__ == "__main__":
    main()
 
