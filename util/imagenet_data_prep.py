from dataset.imagenet import ImageNetDatasetExport
import torch
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--train_dir', help='Directory of your ImageNet Train Dataset', required=True, default=None)
    parser.add_argument('--test_dir', help='Directory of your ImageNet Test Dataset', required=True, default=None)
    parser.add_argument('--train_export_dir', help='Directory of the train saved graphs', required=True, default=None)
    parser.add_argument('--test_export_dir', help='Directory of the test saved graphs', required=True, default=None)
    parser.add_argument('--batch_size', help="batchsize, default = 1", default=1, type=int)
    parser.add_argument('--num_workers', help="# of dataloader cpu process", default=0, type=int)
    parser.add_argument('--debug', help='Whether or not to switch to debug mode, only runs on 100 samples'
                        , default=False, action="store_true")
    parser.add_argument('--num_seg', help="Number of segmentation for SLIC", default=1024, type=int)
    parser.add_argument('--size', help="Resolution for raw image before SLIC", default=320, type=int)
    parser.add_argument('--coeff', help="Number of coefficients used in FFT", default=10, type=int)
    parser.add_argument('--compactness', help="Compactness parameter in SLIC", default=10, type=float)
    parser.add_argument('--ec', help='Whether to enforce connectivity or not'
                        , default=False, action="store_true")
    parser.add_argument('--moments', help='Whether to use moments vs Fourier Descriptors'
                        , default=False, action="store_true")

    

  



    args = parser.parse_args()
    dict_args = vars(args)

    seed = 22
    generator = torch.Generator().manual_seed(seed)
    train_dir = dict_args['train_dir']
    test_dir = dict_args['test_dir']
    num_seg = dict_args['num_seg']
    coeff = dict_args['coeff']
    compactness = dict_args['compactness']
    batch_size = dict_args['batch_size']
    num_workers = dict_args['num_workers']
    train_export_dir = dict_args['train_export_dir']
    test_export_dir = dict_args['test_export_dir']
    size = dict_args['size']
    ec = dict_args['ec']
    moments = dict_args['moments']

    val_test_transform = transforms.Compose(
                            [transforms.Resize([size, size]),
                            transforms.ToTensor()
                            ])

    train_dataset = ImageNetDatasetExport(train_dir, num_seg, coeff, size, compactness,
                                           val_test_transform, train_export_dir, False, ec)

    test_dataset = ImageNetDatasetExport(test_dir, num_seg, coeff, size, compactness,
                                          val_test_transform, test_export_dir, False, ec)

    if dict_args['debug']:
        tr_random_sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=100)
        test_random_sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=100)
        train_source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=tr_random_sampler,
                                                                num_workers =num_workers, drop_last=False)

        test_source_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_random_sampler,
                                                                num_workers=num_workers, drop_last=False)
    else:

        train_source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                                num_workers =num_workers, drop_last=False)

        test_source_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                                num_workers=num_workers, drop_last=False)
        
    for _ in tqdm(train_source_loader):
        pass


    for _ in tqdm(test_source_loader):
        pass




