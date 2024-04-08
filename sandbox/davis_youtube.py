import os

path = '/mnt/dragon/Datasets/VideoSaliency/YoutubeVOS/train/JPEGImages'

for root, subdirs, files in os.walk(path):
    for file in files:
        # print(os.path.join(root.split('/')[-1], file))
        tag = os.path.join(root.split('/')[-1], file)
        print(os.path.join(path, tag))
    # print(root, subdirs, files)