
from magic_py import magic_py

from matplotlib import cm
import numpy as np
import os, argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from fast_slic.avx2 import SlicAvx2
from skimage.measure import regionprops_table
from typing import List, Optional, Any
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries
from skimage.segmentation import slic
from skimage import graph


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}



dataset_images = '/mnt/dragon/Datasets/DUTS/DUTS-TE/Image'
masks = '/mnt/dragon/Datasets/DUTS/DUTS-TE/Mask'
all_ious = []
thresholds = [10,  20,  30]
all_nodes = []
for threshold in thresholds:
    IoUs = []
    nodes = []
    for idx, i in tqdm(enumerate(sorted(os.listdir(dataset_images))[:])):
        name = i.split('.jpg')[0]
        image = os.path.join(dataset_images, name+'.jpg')
        mask = os.path.join(masks, name+'.png')

        img = Image.open(image)
        msk = Image.open(mask)
        img = img.convert('RGB').resize((300, 300))
        img = np.uint8(img)

        segments = slic(img, n_segments=625,
                    compactness=10,
                    max_num_iter=10,
                    convert2lab=True,
                    enforce_connectivity=False,
                    slic_zero=False)
                    
        g = graph.rag_mean_color(img, segments)
        segments = graph.merge_hierarchical(segments, g, thresh=threshold, rag_copy=False,
                                    in_place_merge=True,
                                    merge_func=merge_mean_color,
                                    weight_func=_weight_mean_color)
        nodes.append(len(np.unique(segments)))
        msk = msk.convert('L').resize((300, 300))


        msk = np.array(msk)
        msk[msk<=125] = 0
        msk[msk>125] = 1
        

        regions = regionprops_table(segments, img, properties=('label', 'coords'))
        seq_mask = np.zeros([max(regions['label'])])

        for ind, coord in zip(regions['label'], regions['coords']):
            seq_mask[ind-1] = np.sum(msk[coord[:, 0], coord[:, 1]])/len(coord[:, 0])

        plt_image = seq_mask[segments-1].reshape([img.shape[0], img.shape[1]])
        plt_image = np.ravel(plt_image)
            

        msk = np.ravel(msk)
        y_temp = (plt_image >= 0.5).astype(np.float)
        tp = np.sum((y_temp * msk))
        # avoid prec becomes 0
        prec, recall = (tp + 1e-10) / (np.sum(y_temp) + 1e-10), (tp + 1e-10) / (np.sum(msk) + 1e-10)
        beta_square = 0.3
        f_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall)
        IoUs.append(f_score)
    
    all_ious.append(np.mean(IoUs))   
    all_nodes.append(np.mean(nodes))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(thresholds, all_ious, 'g-')
ax2.plot(thresholds, all_nodes, 'r-')

ax1.set_xlabel('RAG Thresholds')
ax1.set_ylabel('F1 score')
ax2.set_ylabel('Average Number of nodes')
plt.show()
   
