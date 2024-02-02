import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import data, io, segmentation, color
from skimage import graph
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.segmentation import mark_boundaries, slic
from skimage.measure import regionprops_table
import os
from fast_slic.avx2 import SlicAvx2
from matplotlib.lines import Line2D

image_path = '/mnt/dragon/Datasets/DUTS/DUTS-TR/Image/'
for i in os.listdir(image_path):
    img = Image.open(os.path.join(image_path, i)).convert('RGB')
    img = img.resize((300, 300))
    img = np.array(img)

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))


    segments = slic(img, n_segments=625,
                compactness=50,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)
    
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



    g = graph.rag_mean_color(img, segments)



    out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    ax[0].imshow(out)
    ax[0].set_title(f'SLIC, # of nodes: {len(np.unique(segments))}')

    segments_10 = graph.merge_hierarchical(segments, g, thresh=10, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    
    out = color.label2rgb(segments_10, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments_10, (0, 0, 0))
    ax[1].imshow(out)
    ax[1].set_title(f'RAG Merging Threshold 10, # of nodes: {len(np.unique(segments_10))}')

    segments_20 = graph.merge_hierarchical(segments, g, thresh=20, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    
    out = color.label2rgb(segments_20, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments_20, (0, 0, 0))
    ax[2].imshow(out)
    ax[2].set_title(f'RAG Merging Threshold 20, # of nodes: {len(np.unique(segments_20))}')
    plt.show()


