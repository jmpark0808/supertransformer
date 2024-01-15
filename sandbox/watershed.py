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

image_path = '/mnt/hdd/Datasets/DUTS/TR/Image/'
for i in os.listdir(image_path):
    img = Image.open(os.path.join(image_path, i)).convert('RGB')
    img = img.resize((300, 300))
    img = np.array(img)
    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # slic = SlicAvx2(num_components=625, compactness=50, min_size_factor=0.)
    # segments = slic.iterate(img, max_iter=0)+1

    segments = slic(img, n_segments=625,
                compactness=50,
                max_num_iter=1,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)

    regions = regionprops_table(segments, img, properties=('label', 'centroid'))
    x = regions['centroid-0']
    y = regions['centroid-1']

    labels = np.array(regions['label'])
    labels_sorted = np.sort(labels)
    if not np.array_equal(labels,labels_sorted):
        assert(0)
    labels = [str(l) for l in labels]

    out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    # ax[0].imshow(out)
    # for x_, y_, l in zip(x, y, labels):
    #     ax[0].text(y_, x_, l)



    segments = slic(img, n_segments=625,
                compactness=50,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)
    regions = regionprops_table(segments, img, properties=('label', 'centroid'))
    x = regions['centroid-0']
    y = regions['centroid-1']
    labels = np.array(regions['label'])
    labels_sorted = np.sort(labels)
    if not np.array_equal(labels,labels_sorted):
        assert(0)
    labels = [str(l) for l in labels]

    out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    # ax[1].imshow(out)
    # for x_, y_, l in zip(x, y, labels):
    #     ax[1].text(y_, x_, l)
    # plt.show()

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



    # g = graph.rag_mean_color(img, segments)



    # out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    # out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    # ax[0].imshow(out)

    # segments = graph.merge_hierarchical(segments, g, thresh=35, rag_copy=False,
    #                                    in_place_merge=True,
    #                                    merge_func=merge_mean_color,
    #                                    weight_func=_weight_mean_color)
    # print(np.max(segments))
    # print(len(np.unique(segments)))
    # out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    # out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    # ax[1].imshow(out)
    # plt.show()


