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
    fig1, ax1 = plt.subplots()

    # slic = SlicAvx2(num_components=625, compactness=50, min_size_factor=0.)
    # segments = slic.iterate(img, max_iter=0)+1

    segments = slic(img, n_segments=625,
                compactness=10,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)
    
    regions = regionprops_table(segments, img, properties=('label', 'centroid', 'intensity_mean'))
  
    x = regions['centroid-0']
    y = regions['centroid-1']
    r = regions['intensity_mean-0']
    g = regions['intensity_mean-1']
    b = regions['intensity_mean-2']

    for x_, y_, r_, g_, b_ in zip(x, y, r, g, b):
        ax1.scatter(y_, x_, c=np.array([[int(r_), int(g_), int(b_)]])/255.0)
    fig1.show()

    labels = np.array(regions['label'])
    labels_sorted = np.sort(labels)
    if not np.array_equal(labels,labels_sorted):
        assert(0)
    labels = [str(l) for l in labels]

    out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    ax[0].imshow(out)
    ax[0].set_title('Compactness 10')
    # for i in range(625):
    #     out_ = segmentation.mark_boundaries(out, segments == (i+1), (255, 0, 0))
    #     ax1.imshow(out_)
    #     ax1.set_title(f'Superpixel {i}')
    #     fig1.savefig(f'/home/eddie/Downloads/gif/{i}')
    # for x_, y_, l in zip(x, y, labels):
    #     ax[0].text(y_, x_, l)



    segments = slic(img, n_segments=625,
                compactness=50,
                max_num_iter=10,
                convert2lab=True,
                enforce_connectivity=False,
                slic_zero=False)
    
    # vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    # vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    # vs_diagonal_r = np.vstack([segments[:-1,:-1].ravel(), segments[1:,1:].ravel()])
    # vs_diagonal_l = np.vstack([segments[1:,:-1].ravel(), segments[:-1,1:].ravel()])
    # bneighbors, counts = np.unique(np.hstack([vs_right, vs_below, vs_diagonal_r, vs_diagonal_l]), axis=1, return_counts=True)
    # segments_ids = np.unique(segments)
    # centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # plt.imshow(mark_boundaries(img, segments))
    # plt.scatter(centers[:,1],centers[:,0], c='y')

    # for i in range(bneighbors.shape[1]):
    #     if bneighbors[0,i] != bneighbors[1,i]:
    #         y0,x0 = centers[bneighbors[0,i]-1]
    #         y1,x1 = centers[bneighbors[1,i]-1]

    #         l = Line2D([x0,x1],[y0,y1], alpha=0.5, linewidth=counts[i]/10.)
    #         ax.add_line(l)

    # plt.show()
    # print(bneighbors, counts)
    # assert(0)



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
    ax[1].imshow(out)
    ax[1].set_title('Compactness 50')




    # for x_, y_, l in zip(x, y, labels):
    #     ax[1].text(y_, x_, l)
    # plt.show()



    segments = slic(img, n_segments=625,
                compactness=100,
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
    ax[2].imshow(out)
    ax[2].set_title('Compactness 100')
    plt.show()

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

    segments = graph.merge_hierarchical(segments, g, thresh=35, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    print(np.max(segments))
    print(len(np.unique(segments)))
    out = color.label2rgb(segments, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, segments, (0, 0, 0))
    ax[1].imshow(out)
    plt.show()


