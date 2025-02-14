
import numpy as np
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
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
    

    

    coord_list = np.empty((max_size, 3), dtype=np.intp)
 
    
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
               
                        
                    
                        

    return np.asarray(connected_segments)


im = '/home/eddie/Datasets/DUTS/DUTS-TE/Image/ILSVRC2012_test_00000003.jpg'

img = Image.open(im)
            
img = img.convert('RGB').resize((448, 448))

img = np.array(img)

            

            
           
segments = slic(img, n_segments=784,
compactness=10,
max_num_iter=10,
convert2lab=True,
enforce_connectivity=False,
slic_zero=False)
segments = np.expand_dims(segments, 0)
segments = _enforce_label_connectivity_cython(segments, 76, 256*3, 1)

segments = np.squeeze(segments)-1

windows = np.arange(0, 784).reshape(28, 28)
shifted_windows = np.roll(np.arange(0, 784).reshape(28, 28), (-7, -7), axis=(1, 0))


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
gap = 50
dummy_image1 = np.ones((448+gap, 448+gap, 3))
dummy_image2 = np.ones((448+2*gap, 448+2*gap, 3))
for i in range(2):
    for j in range(2):
       
        start_h = i*14
        start_w = j*14
        indices_window = windows[start_h:start_h+14, start_w:start_w+14].flatten()
        indices_shifted_window = shifted_windows[start_h:start_h+14, start_w:start_w+14].flatten()
        plot_shift_h = gap*i
        plot_shift_w = gap*j

        temp_segments_inner = np.copy(segments)
        temp_segments_outer = np.copy(segments)
        for k in range(784):
            if k in indices_window:
                indices = np.argwhere(temp_segments_inner == k)
                mean_colour = np.mean(img[indices[:, 0], indices[:, 1], :], axis=0)/255.
                
                temp_segments_outer[indices[:, 0], indices[:, 1]] = 1
                temp_segments_inner[indices[:, 0], indices[:, 1]] = k

                indices[:, 0] += plot_shift_h
                indices[:, 1] += plot_shift_w

                dummy_image1[indices[:, 0], indices[:, 1], :] = mean_colour

                
            else:
                indices = np.argwhere(temp_segments_outer == k)
                temp_segments_outer[indices[:, 0], indices[:, 1]] = 0
                temp_segments_inner[indices[:, 0], indices[:, 1]] = 1000

        inner = find_boundaries(temp_segments_inner, mode='inner', background=1000)
        outer = find_boundaries(temp_segments_outer, mode='thick', background=0)
        inner_indices = np.argwhere(inner==1)
        inner_indices[:, 0] += plot_shift_h
        inner_indices[:, 1] += plot_shift_w

        outer_indices = np.argwhere(outer==1)
        outer_indices[:, 0] += plot_shift_h
        outer_indices[:, 1] += plot_shift_w


        dummy_image1[inner_indices[:, 0], inner_indices[:, 1]] = [1, 1, 0]
        if i == 0 and j == 0:
            colour = [1, 0, 0]
        elif i == 0 and j == 1:
            colour = [0, 1, 0]
        elif i == 1 and j == 0:
            colour = [0, 0, 1]
        else:
            colour = [1, 0.5, 0]

        dummy_image1[outer_indices[:, 0], outer_indices[:, 1]] = colour


        
        # SHIFTED        


        temp_segments_inner = np.copy(segments)
        temp_segments_outer = np.copy(segments)
        for k in range(784):
            if i == 0 and j == 0:
                plot_shift_h = gap
                plot_shift_w = gap
            elif i == 0 and j == 1:
                if k%28 < 14:
                    plot_shift_h = gap
                    plot_shift_w = 0 
                else:
                    plot_shift_h = gap
                    plot_shift_w = 2*gap
            elif i == 1 and j == 0:
                if k < 784//2:
                    plot_shift_h = 0
                    plot_shift_w = gap
                else:
                    plot_shift_h = 2*gap
                    plot_shift_w = gap
            else:
                if k%28 < 14:
                    if k < 784//2:
                        plot_shift_h = 0
                        plot_shift_w = 0
                    else:
                        plot_shift_h = 2*gap
                        plot_shift_w = 0 
                else:
                    if k < 784//2:
                        plot_shift_h = 0
                        plot_shift_w = 2*gap
                    else:
                        plot_shift_h = 2*gap
                        plot_shift_w = 2*gap
            if k in indices_shifted_window:
                indices = np.argwhere(temp_segments_inner == k)
                mean_colour = np.mean(img[indices[:, 0], indices[:, 1], :], axis=0)/255.
                temp_segments_outer[indices[:, 0], indices[:, 1]] = 1
                temp_segments_inner[indices[:, 0], indices[:, 1]] = k
                indices[:, 0] += plot_shift_h
                indices[:, 1] += plot_shift_w

                dummy_image2[indices[:, 0], indices[:, 1]] = mean_colour

                
            else:
                indices = np.argwhere(temp_segments_outer == k)
                temp_segments_outer[indices[:, 0], indices[:, 1]] = 0
                temp_segments_inner[indices[:, 0], indices[:, 1]] = 1000

        inner = find_boundaries(temp_segments_inner, mode='inner', background=1000)
        outer = find_boundaries(temp_segments_outer, mode='thick', background=0)
        inner_indices = np.argwhere(inner==1)
        outer_indices = np.argwhere(outer==1)

        if i == 0 and j == 0:
            inner_indices[:, 0] += gap
            inner_indices[:, 1] += gap

            outer_indices[:, 0] += gap
            outer_indices[:, 1] += gap
        elif i == 0 and j == 1:
            for ind, (h, w) in enumerate(inner_indices):
                if w < 224:
                    inner_indices[ind, 0] += gap
                else:
                    inner_indices[ind, 0] += gap
                    inner_indices[ind, 1] += 2*gap


            for ind, (h,w) in enumerate(outer_indices):
                if w < 224:
                    outer_indices[ind, 0] += gap
                else:
                    outer_indices[ind, 0] += gap
                    outer_indices[ind, 1] += 2*gap
            
        elif i == 1 and j == 0:
            for ind, (h,w) in enumerate(inner_indices):
                if h < 224:
                    inner_indices[ind, 1] += gap
                else:
                    inner_indices[ind, 0] += 2*gap
                    inner_indices[ind, 1] += gap
            for ind, (h,w) in enumerate(outer_indices):
                if h < 224:
                    outer_indices[ind, 1] += gap
                else:
                    outer_indices[ind, 0] += 2*gap
                    outer_indices[ind, 1] += gap
            
        else:
            for ind, (h,w) in enumerate(inner_indices):
                if h < 224 and w < 224:
                    pass
                elif h < 224 and w > 224:
                    inner_indices[ind, 1] += 2*gap
                elif h > 224 and w < 224:
                    inner_indices[ind, 0] += 2*gap
                else:
                    inner_indices[ind, 1] += 2*gap
                    inner_indices[ind, 0] += 2*gap
            for ind, (h,w) in enumerate(outer_indices):
                if h < 224 and w < 224:
                    pass
                elif h < 224 and w > 224:
                    outer_indices[ind, 1] += 2*gap
                elif h > 224 and w < 224:
                    outer_indices[ind, 0] += 2*gap
                else:
                    outer_indices[ind, 1] += 2*gap
                    outer_indices[ind, 0] += 2*gap
            

        dummy_image2[inner_indices[:, 0], inner_indices[:, 1]] = [1, 1, 0]
        
        dummy_image2[outer_indices[:, 0], outer_indices[:, 1]] = colour


ax[0].imshow(dummy_image1)
ax[0].set_title('Window Partitioning', fontsize=20)
ax[1].imshow(dummy_image2)
ax[1].set_title('Shifted Window Partitioning', fontsize=20)
ax[0].axis('off')
ax[1].axis('off')
fig.savefig('/home/eddie/Figures/visualize_shifted_windows.pdf', format='pdf')
plt.show()

