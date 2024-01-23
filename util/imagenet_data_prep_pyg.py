from dataset.imagenet_pyg_exp import SPGEImageNetDataModule
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
  



    args = parser.parse_args()
    dict_args = vars(args)

    seed = 22
    generator = torch.Generator().manual_seed(seed)
    train_dir = dict_args['train_dir']
    test_dir = dict_args['test_dir']
    num_seg = 400
    coeff = 10
    compactness = 10
    batch_size = dict_args['batch_size']
    num_workers = dict_args['num_workers']
    train_export_dir = dict_args['train_export_dir']
    test_export_dir = dict_args['test_export_dir']

    val_test_transform = transforms.Compose(
                            [transforms.Resize([256, 256]),
                            transforms.ToTensor()
                            ])

    train_dataset = ImageNetDatasetTrainExport(train_dir, num_seg, coeff, compactness, val_test_transform, 'train', train_export_dir)
    class_to_idx = train_dataset.class_to_idx
    train_size = int(0.8*len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)
    val_dataset.dataset.transform = val_test_transform
    val_dataset.mode = 'val'

    test_dataset = ImageNetDatasetTestExport(test_dir, val_test_transform, num_seg, coeff, class_to_idx, compactness, test_export_dir)

    train_source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                            num_workers =num_workers, drop_last=False)

    val_source_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                            num_workers=num_workers, drop_last=False)

    test_source_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                            num_workers=num_workers, drop_last=False)
    
    for _ in tqdm(train_source_loader):
        pass

    for _ in tqdm(val_source_loader):
        pass

    for _ in tqdm(test_source_loader):
        pass




