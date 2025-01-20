
import numpy as np
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt


def _enforce_label_connectivity_cython( segments,
                                        min_size,
                                        max_size,
                                       start_label=1):
    """ Helper function to remove small disconnected regions from the labels

    Parameters
    ----------
    segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    min_size : int
        The minimum size of the segment
    max_size : int
        The maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    start_label : int
        The label indexing start value.

    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    # get image dimensions
    
    depth = segments.shape[0]
    height = segments.shape[1]
    width = segments.shape[2]

    # neighborhood arrays
    ddx = np.array((1, -1, 0, 0, 0, 0), dtype=np.intp)
    ddy = np.array((0, 0, 1, -1, 0, 0), dtype=np.intp)
    ddz = np.array((0, 0, 0, 0, 1, -1), dtype=np.intp)
    
    # new object with connected segments initialized to mask_label
    mask_label = start_label - 1

    connected_segments \
        = np.full_like(segments, mask_label, dtype=np.intp)
    
    connected_segments_nl \
        = np.full_like(segments, mask_label, dtype=np.intp)

    current_new_label = start_label
    

    coord_list = np.empty((max_size, 3), dtype=np.intp)
    coord_list_nl = np.empty((max_size, 3), dtype=np.intp)
    # im = segments == 9
    # plt.imshow(np.squeeze(im))
    # plt.show()
    
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if segments[z, y, x] == mask_label or connected_segments[z, y, x] > mask_label:
                    pass
                else:
                    # find the component size
                    label = segments[z, y, x]
                    adjacent = label
                    connected_segments[z, y, x] = label
                    current_segment_size = 1
                    bfs_visited = 0
                    coord_list[bfs_visited, 0] = z
                    coord_list[bfs_visited, 1] = y
                    coord_list[bfs_visited, 2] = x

                    #perform a breadth first search to find
                    # the size of the connected component
                    while bfs_visited < current_segment_size < max_size:
                        for i in range(6):
                            zz = coord_list[bfs_visited, 0] + ddz[i]
                            yy = coord_list[bfs_visited, 1] + ddy[i]
                            xx = coord_list[bfs_visited, 2] + ddx[i]
                            if (0 <= xx < width and
                                0 <= yy < height and
                                0 <= zz < depth):
                                if (segments[zz, yy, xx] == label and
                                    connected_segments[zz, yy, xx] == mask_label): #if the perturbed location label is the same as current location label
                                    connected_segments[zz, yy, xx] = label
                                    coord_list[current_segment_size, 0] = zz
                                    coord_list[current_segment_size, 1] = yy
                                    coord_list[current_segment_size, 2] = xx
                                    current_segment_size += 1
                                    if current_segment_size >= max_size:
                                        break
                                elif (connected_segments[zz, yy, xx] > mask_label and 
                                        connected_segments[zz, yy, xx] != label):
                                    adjacent = connected_segments[zz, yy, xx]
                                

                                
                        bfs_visited += 1

                    # change to an adjacent one, like in the original paper
                    # print('A', current_segment_size)
                    
                        
                    if current_segment_size < min_size:
                        # if label==adjacent:
                        #     print(label, adjacent)
                        
                        
                        for i in range(current_segment_size):
                            connected_segments[coord_list[i, 0],
                                                coord_list[i, 1],
                                                coord_list[i, 2]] = adjacent
               
                        
                    
                        

                    if segments[z, y, x] == mask_label or connected_segments_nl[z, y, x] > mask_label:
                        pass
                    else:
                        # find the component size
                        adjacent = current_new_label
                        label = segments[z, y, x]
                        connected_segments_nl[z, y, x] = current_new_label
                        current_segment_size = 1
                        bfs_visited = 0
                        coord_list_nl[bfs_visited, 0] = z
                        coord_list_nl[bfs_visited, 1] = y
                        coord_list_nl[bfs_visited, 2] = x

                        #perform a breadth first search to find
                        # the size of the connected component
                        while bfs_visited < current_segment_size < max_size:
                            for i in range(6):
                                zz = coord_list_nl[bfs_visited, 0] + ddz[i]
                                yy = coord_list_nl[bfs_visited, 1] + ddy[i]
                                xx = coord_list_nl[bfs_visited, 2] + ddx[i]
                                if (0 <= xx < width and
                                    0 <= yy < height and
                                    0 <= zz < depth):
                                    if (segments[zz, yy, xx] == label and
                                        connected_segments_nl[zz, yy, xx] == mask_label): #if the perturbed location label is the same as current location label
                                        connected_segments_nl[zz, yy, xx] = current_new_label
                                        coord_list_nl[current_segment_size, 0] = zz
                                        coord_list_nl[current_segment_size, 1] = yy
                                        coord_list_nl[current_segment_size, 2] = xx
                                        current_segment_size += 1
                                        if current_segment_size >= max_size:
                                            break
                                    elif (connected_segments_nl[zz, yy, xx] > mask_label and 
                                          connected_segments_nl[zz, yy, xx] != current_new_label):
                                        adjacent = connected_segments_nl[zz, yy, xx]

                                    
                            bfs_visited += 1

                        # change to an adjacent one, like in the original paper
                        # print('B', current_segment_size)
                        if current_segment_size < min_size:
                            # print('NL triggered merge', adjacent)
                            for i in range(current_segment_size):
                                connected_segments_nl[coord_list_nl[i, 0],
                                                    coord_list_nl[i, 1],
                                                    coord_list_nl[i, 2]] = adjacent
                        else:
                            current_new_label += 1
                        
                    # fig, ax = plt.subplots(1, 2)
                    # valid = np.argwhere(connected_segments != 0)
                    
                    # z,x,y = valid.max(0)
                    # print(connected_segments[0, :x, :y])
                    # print(connected_segments_nl[0, :x, :y])
                    # ax[0].matshow(connected_segments[0, :x, :y])
                    # ax[1].matshow(connected_segments_nl[0, :x, :y])
                    # plt.show()
                    
                    # contours = find_boundaries(connected_segments)
                    # contours_nl = find_boundaries(connected_segments_nl)
                    # contours = np.squeeze(contours.astype(np.float32))
                    # contours_nl = np.squeeze(contours_nl.astype(np.float32))
                    # if np.sum(contours - contours_nl) != 0:
                        # plt.scatter(coord_list[:current_segment_size, 1], coord_list[:current_segment_size, 2], marker="*", alpha=0.5)
                        # plt.scatter(coord_list_nl[:current_segment_size, 1], coord_list_nl[:current_segment_size, 2], alpha=0.5)
                        # plt.show()

                        # valid = np.argwhere(connected_segments == label)
                        # _,x_max,y_max = valid.max(0)
                        # _,x_min,y_min = valid.min(0)
                        # fig, ax = plt.subplots(1, 3)
                        # ax[0].matshow(connected_segments[0, x_min:x_max, y_min:y_max])
                        # ax[1].matshow(connected_segments_nl[0, x_min:x_max, y_min:y_max])
                        # ax[2].matshow(segments[0, x_min:x_max, y_min:y_max])
                        # plt.show()
                        # assert(0)

                        # print(label)
                        # print(connected_segments, connected_segments_nl)
                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].imshow(contours, cmap='gray')
                        # ax[1].imshow(contours_nl, cmap='gray')
                        # plt.show()
                        # assert(0)
    
    return np.asarray(connected_segments)



# segment = np.array([[1, 1, 1, 1, 5, 5, 5, 5],
#                     [1, 2, 2, 1, 1, 5, 5, 5],
#                     [3, 3, 3, 3, 4, 4, 5, 4],
#                     [3, 3, 3, 3, 4, 4, 4, 4]]).reshape(1, 4, 8)
segments = np.load('/home/eddie/waterloo/supertransformer/Analysis/sample_segment.npy').reshape(1, 448, 448)

segments = _enforce_label_connectivity_cython(segments, 76, 256*3, 1)
segments = np.squeeze(segments)

im = np.zeros((448, 448, 3))
for label in range(784):
    if np.sum(segments == label) > 0:

        regions = segments == label
        random_colour = np.random.rand(3)
        
        im[segments==label] = random_colour



plt.imshow(im )
# from skimage.measure import regionprops_table
# regions = regionprops_table(segments, None, properties=('label', 'centroid'))
# centroids_x = regions['centroid-1']
# centroids_y = regions['centroid-0']
# labels = regions['label']



# plt.scatter(centroids_x, centroids_y)

# for x, y, l in zip(centroids_x, centroids_y, labels):
#     plt.text(x+1, y+1, s=l)


plt.show()
