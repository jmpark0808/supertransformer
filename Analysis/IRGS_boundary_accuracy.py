
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


class magic_rag:

    """wrapper class which gives a nicer interface to the PyRAG class from the magic_py module"""

    def __init__(self, 
                img: ArrayLike,
                msk: Optional[ArrayLike] = None,
                grd: Optional[ArrayLike] = None,
                wsh: Optional[ArrayLike] = None,
                N_class: Optional[int] = 2,
                sigma: Optional[float] = 1.2,
                verbose: Optional[bool] = False):

        self.verbose = verbose

        if img.ndim == 2:
            self.N_feat = 1
            self.img = np.expand_dims(img, 2)
        elif img.ndim==3:
            self.N_feat = img.shape[2]
            self.img = img
        else:
            raise ValueError("Incorrect image size - expected either 2 or 3 dimensions.")

        self.N_class = N_class

        # if no mask is provided, create a blank one (no masked pixels)
        if msk is None:
            self.msk = np.zeros((img.shape[0], img.shape[1]))
        else:
            self.msk = msk

        # if the image gradient isn't provided, compute it
        if grd is None:
            if self.verbose:
                print("No image gradient provided - computing gradient from source image...")
            self.grd = magic_py.Gradient(self.img)
            if self.verbose:
                print("Done gradient computation.")
        else:
            self.grd = grd
        # grd sometimes has isolated NaNs - zero them out!!!
        self.grd[np.isnan(self.grd)] = 0
        
        # if the watershed map isn't provided, compute it
        if wsh is None:
            if self.verbose:
                print("No watershed provided - computing watershed from source image...")
            
            # if you dont want to filter the gradient before computing the watershed swap the line below with this one
            # self.wsh = magic_py.Watershed(self.grd, self.msk)
            
            # smoothing the gradient makes superpixels larger (tune sigma to your preference)
            self.wsh = magic_py.Watershed(gaussian_filter(self.grd, sigma=sigma, truncate=4), self.msk)
            # slic = SlicAvx2(num_components=625, compactness=10)
            # segments = slic.iterate(np.array(self.img))+1
            # mask = find_boundaries(segments, mode='inner')
            # segments[mask] = -2
            # self.wsh = segments
            if self.verbose:
                print("Done watershed computation.")
        else:
            self.wsh = wsh

    def __getattr__(self, __name: str) -> Any:
        ''' Overload getattr to allow for lazy loading of RAG attributes
        
        Bringing RAG attributes from magic_lib to python has a memory cost,
        so we don't want to instantiate attributes that we're not going to use. 
        We minimize unnecessary instantiations by lazily loading each RAG 
        attribute the first time it is accessed. 
        
        To ensure that mutable attributes are kept up to date, they are 
        automatically deleted by the methods that can modify them. Up-to-date
        versions are then lazily reloaded if they are then accessed at a later time.'''

        self._update_attribute(__name)
        return self.__getattribute__(__name)

    def _update_attribute(self, key: str) -> None:

        # print(f"Updating attribute: {key}")
        
        if key == '_rag':
            self.__dict__['_rag'] = magic_py.PyRAG(self.img, self.grd, self.wsh, self.msk, self.N_class)
            self._delete_attributes(['wsh']) 
        
        elif key == 'edge_idx':
            # edge index: gives the structure of the region adjacency graph 
            edge_idx = self._rag.GetEdges() # (2, N_edges)
            self.__dict__['edge_idx'] = self.consecutive_map[edge_idx.flatten()].reshape(edge_idx.shape)

        elif key == 'N_nodes':
            # number of nodes / regions
            self.__dict__['N_nodes'] = self._rag.GetVertexCount()

        elif key == 'N_edges':
            # edge count (times two since we usually want the number of directed edges)
            self.__dict__['N_edges'] = 2*self._rag.GetEdgeCount()

        elif key in ['wsh', 'consecutive_map']:
            # wsh is the watershed map: defines the regions and region boundaries
            # it's a 2d array of size HxW where each region pixel contains the unique label of its parent region and each boundary pixel has a value of -2

            # consecutive map: remaps region labels to a consecutive sequence (removes any gaps)
            self.__dict__['wsh'], self.__dict__['consecutive_map'] = self._rag.GetWatershed()

        elif key == "bmp":
            # the boundary map (2d array of size HxW. For each pixel on a boundary, the value is the unique label of the boundary. Non-boundary pixels are given a value of -2.)
            self.__dict__['bmp'] = self._rag.GetBoundaryMap()

        elif key == "r_means":
            # the mean vector for each region
            self.__dict__['r_means'] = self._rag.GetRegionMeans()
        
        elif key == "r_comeans":
            # the co-mean of each region, which is 
            # (sum over pixels in the region) yy^T
            self.__dict__['r_comeans'] = self._rag.GetRegionCoMeans()

        elif key == "r_centroids":
            # the centroid of each region
            self.__dict__['r_centroids'] = self._rag.GetRegionCentroids()
        
        elif key == "r_classlabels":
            # the current class label values
            self.r_classlabels = self._rag.GetRegionClassLabels()

        elif key == "r_pixcounts":
            # number of pixels belonging to each region
            self.__dict__['r_pixcounts'] = self._rag.GetRegionPixelCounts()
        
        elif key == "result_image":
            # result image corresponding to the current region labels (boundaries filled in)
            self.__dict__['result_image'] = self._rag.GetConnectedClassMapBoundary()

        elif key == "result_image_with_boundaries":
            # result image corresponding to the current region labels (boundaries not filled in)
            self.__dict__['result_image_with_boundaries'] = self._rag.GetConnectedClassMap()

        elif key == 'merging_lut':
            # merging lookup table: provides a mapping from the original regions to merged regions
            # (a vector which, for each of the original regions in the watershed map, gives the label of the region it is currently a member of.)
            self.__dict__['merging_lut'] = self._rag.GetMergingLUT()
        
        elif key == 'gmm_unary':
            # get the GMM unary potential
            self.__dict__['gmm_unary'] = self._rag.GetGMMUnary()
        
        elif key == 'gmm_params':
            # gets a dictionary which contains the gaussian mixture model parameters
            # entries are 'Mu', 
            self.gmm_params = self._rag.GetGMMParams()._dict()
            self._delete_attributes(['gmm_unary'])
        
        elif key == 'edge_penalty':
            self.__dict__['edge_penalty'] = self._rag.GetEdgePenalty()
        
        elif key == 'beta':
            # beta is multiplier for the discontinuity penalty
            self.beta = self._rag.GetBeta()
        
        elif key == 'boundary_population_ratio':
            # ratio of boundary pixels to non-boundary pixels in the image
            self.__dict__['boundary_population_ratio'] = self._rag.GetBoundaryPopulationRatio()
        
        elif key == 'chain_codes':
            # 4-connected chain code representation of the region boundaries. Is a tuple containing 4 arrays: (start points, end points, chain_indices, chain_values)
            self.__dict__['chain_codes'] = self._rag.GetChainCodes()
        
        else:
            raise AttributeError(f"magic_rag: requested invalid RAG attribute '{key}'.")

    def __setattr__(self, __name: str, __value: Any) -> None:
        
        # Prevents read-only attributes from being set. You can get around this by directly manipulating
        # the entries in self.__dict__, as is done in the _update_attribute() function.
        if __name in [  '_rag', 'r_means', 'r_comeans', 'r_centroids', 'r_pixcounts', 'result_image',
                        'merging_lut', 'gmm_unary', 'edge_penalty', 'boundary_population_ratio']:

            raise AttributeError(f"Cannot set read-only attribute {__name} of magic_rag.")

        if __name == 'r_classlabels':

            self._rag.SetClassLabelsByIndex(__value, np.unique(self.merging_lut))

            # changing the class labels makes the previous result image invalid
            self._delete_attributes(['result_image', 'result_image_with_boundaries'])
        
        super().__setattr__(__name, __value)


    def _delete_attributes(self, keys: List[str]) -> None:
        ''' Deletes object attributes if they exist '''

        for _name in keys:
            # directly check if the key is in self.__dict__ instead of using hasattr.
            # Since __getattr__ is overloaded to lazily load rag attributes, calling
            # hasattr here would create any attributes that didn't already exist, just
            # so that they could be deleted in the next line.  
            if _name in self.__dict__:
                delattr(self, _name)

    def initialize_kmeans(self)-> None:
        '''create an initial labeling over the graph regions using an initialization 
        procedure based on k-means with repeated restarts. This is the initialization
        process described in Yu and Clausi's paper "Multivariate Image Segmentation 
        Using Semantic Region Growing With Adaptive Edge Penalty", IEEE TIP 2010 '''
        # self.r_classlabels = self._rag.KmeansInitialization()
        _ = self._rag.KmeansInitialization()

    def update_gmm(self)->None:
        ''' update the gaussian mixture model parameters with estimates obtained 
        using the current class labels '''
        self._rag.UpdateGMMParam()
        # old gmm unary and parameters become invalid after update
        self._delete_attributes(['gmm_unary', 'gmm_params'])
    
    def update_edge_penalty(self,
                            K:      Optional[float] = 1.1,
                            beta1:  Optional[float] = 2.5,
                            beta2:  Optional[float] = 0.3,
                            itr:    Optional[int]   = 1  ,
                            mode:   Optional[str]   = 'fisher'
                            )->None:

        ''' update the edge penalty '''

        if mode == 'fisher':
            self._rag.UpdateBoundaryParamFisher(K, beta1, beta2, itr)
        # elif mode == 'const':
        else:
            self._rag.UpdateBoundaryParam(K, itr)
        
        # old edge penalties become invalid
        self._delete_attributes(['edge_penalty'])

    def _update_on_merge(self)->List[str]:
        # a list of all the parameters that change after a merging step
        return ['edge_idx', 'N_nodes', 'N_edges', 'wsh', 'consecutive_map', 'bmp', 'r_means',
                'r_comeans', 'r_centroids', 'r_classlabels', 'r_pixcounts', 'result_image','result_image_with_boundaries',
                'merging_lut', 'gmm_unary', 'gmm_params', 'edge_penalty', 'beta',
                'boundary_population_ratio', 'chain_codes']

    def irgs_step(  self,
                    K:      Optional[float] = 1.1,
                    beta1:  Optional[float] = 3.0,
                    beta2:  Optional[float] = 0.4,
                    current_iter: Optional[int] = None)->None:
        
        """ Conduct one iteration of the IRGS algorithm, comprising the following steps:
        
        - Update the gmm parameters based on the current class labels
        - Update the edge penalties using the fisher criterion and the provided parameters
          (K, beta1, beta2 and current_iter)
        - Do a classification step with unary and pairwise costs determined by the new 
          gmm parameters and edge penalties. Note that the simulated annealing algorithm
          is used for classification.
        - Perform region merging by greedily merging all pairs of regions for which the
          merging energy is negative.

        To ensure that all rag attributes remain up to date, all attributes that can change
        during an irgs step are reloaded once the step is complete.

        """
        if current_iter is not None:
            self._rag.SetCurrentIteration(current_iter)

        # advance one step of the IRGS algorithm
        self._rag.IRGSStep(K, beta1, beta2, False)

        # refresh all the attributes that may have changed
        self._delete_attributes(self._update_on_merge())

    def merge_regions( self, current_iter: Optional[int] = None )->None:
        ''' greedy region merging '''
        
        if current_iter is not None:
            self._rag.SetCurrentIteration(current_iter)

        _ = self._rag.MergeRegions()

        # refresh all the attributes that may have changed
        self._delete_attributes(self._update_on_merge())









def Map_labels(labels):
    # map labels to [0, 1, 2,..., n_labels-1]

    id = np.unique(labels)
    aux_id = -1 * np.ones_like(labels)     
    c = 0
    for i in id:                                
        if i != -1:
            aux_id[labels == i] = c
            c += 1
    return aux_id

def IRGS(img, n_classes, n_iter, mask=None):
    # --- RUN IRGS --- #
    rag = None
    if mask is None:
        rag = magic_rag(img, msk=None, N_class=n_classes, verbose=True)
    else:
        rag = magic_rag(img, msk=mask, N_class=n_classes, verbose=True)

    print("Initializing k-means with", n_classes, "classes")
    rag.initialize_kmeans()

    print("Performing", str(n_iter), "IRGS iterations...")
    # for j in tqdm(range(n_iter), ncols=50):
    for j in range(n_iter):
        # rag.irgs_step(beta1=beta1, current_iter=i+1)
        rag.irgs_step(current_iter=j+1)
    
    # final_wsh = rag.wsh

    irgs_output = rag.result_image
    # irgs_output = rag.result_image_with_boundaries
    # boundaries = np.int16(rag.bmp == -2) # Not consistent with rag.result_image_with_boundaries
    boundaries = np.int16(rag.result_image_with_boundaries != -2)

    boundaries[boundaries == 0] = -1
    boundaries[irgs_output < 0] = -1
    irgs_output[irgs_output < 0] = -1           # background and boundaries.
                                                # IRGS returns an aditional class with 
                                                # label -2 for landmask and boundaries\
    irgs_output = Map_labels(irgs_output)

    return irgs_output, boundaries#, final_wsh




if __name__ == '__main__':

    
    dataset_images = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Image'
    masks = '/mnt/hdd/Datasets/DUTS/DUTS-TR/Mask'
    IoUs = []
    for idx, i in enumerate(sorted(os.listdir(dataset_images))[:]):
        print(f'-----------------------------------{idx}------------------------------')
        
                
        name = i.split('.jpg')[0]
        image = os.path.join(dataset_images, name+'.jpg')
        mask = os.path.join(masks, name+'.png')

        img = Image.open(image)
        msk = Image.open(mask)
        img = img.convert('RGB').resize((300, 300))
        img = np.uint8(img)

        segments, boundaries = IRGS(img, 4, 120)


        msk = msk.convert('L').resize((300, 300))


        msk = np.array(msk)
        msk[msk<=125] = 0
        msk[msk>125] = 1
        

        regions = regionprops_table(segments, img, properties=('label', 'centroid', 'area', 'intensity_mean',
                                                                                'coords',))
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
        

    print(np.mean(IoUs))

   
