from torch.utils.data import DataLoader
from dataset.imagenet_aug import ImageNetDatasetTrain
from tqdm import tqdm
from torchvision import transforms

val_test_transform = transforms.Compose(
                        [transforms.Resize([256, 256]),
                         transforms.ToTensor()
                        ])

d = {'train_dir': '/mnt/dragon/Datasets/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
     'test_dir': '/mnt/dragon/Datasets/imagenet-object-localization-challenge/ILSVRC',
     'batch_size': 16,
     'num_seg': 625,
     'coeff': 10,
     'compactness': 10,
     'seed': 22,
     'dilation': 7}
dataset = ImageNetDatasetTrain(d['train_dir'], d['num_seg'], d['coeff'], d['compactness'], val_test_transform, 'train', 7)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=12)

for i in tqdm(loader):
    pass

"""train_dir = kwargs.get('train_dir')
        test_dir = kwargs.get('test_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.num_seg = kwargs.get('num_seg', 600)
        self.coeff = kwargs.get('coeff', 70)
        self.compactness = kwargs.get('compactness', 10)
        self.seed = kwargs.get('seed')
        self.dilation = kwargs.get('dilation')"""