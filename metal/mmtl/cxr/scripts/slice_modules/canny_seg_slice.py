import cv2
from collections import Counter
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Pool
import dask


# Loading pretrained segnet (from https://github.com/imlab-uiip/lung-segmentation-2d)
from keras.models import load_model
from skimage import exposure, morphology, measure
from skimage.morphology import skeletonize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def canny_edge_detection(x, kwargs):
    x = np.array(x, dtype=np.uint8)
    x = x+np.min(x)
    return cv2.Canny(x, **kwargs)

def get_canny_edges(x, kwargs={'threshold1':100, 'threshold2':150, 'apertureSize':3, 'L2gradient':True}):
    proc_fun = partial(canny_edge_detection, kwargs=kwargs)

    x_out = proc_fun(x*255)
    return x_out

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def get_lung_segmentation(x):
    xx = (cv2.resize(np.array(x), dsize=(256, 256))-mean[0])/std[0]
    im_shape = xx.shape
    xx = xx[None,:,:,None]


    model_name = '../../../../../../lung-segmentation-2d/trained_model.hdf5'
    UNet = load_model(model_name)

    img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
    pred = UNet.predict(xx)[..., 0] #.reshape(inp_shape[:2])

    pr = pred > 0.5

    # Remove regions smaller than 2% of the image
    pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
    return pr, xx

def get_canny_seg_diff_image(x, kwargs):
    edges = get_canny_edges(x, kwargs).squeeze()
    lungs, _ = get_lung_segmentation(x)
    
    # resizing to align with lung segmentation
    edges = cv2.resize(edges.squeeze(), dsize=(256, 256))
    
    # getting difference image
    diff_img = (lungs*255 - edges)*lungs
    
    return diff_img, lungs, edges


###### MAKING MODULE CONTAINING ALL CONSTITUENT FUNCTIONS ######

class CannySegSliceModule(nn.Module):
    def __init__(self, num_procs=1, lng_cutoff=40, slope_cutoff=0.5, canny_kwargs={'apertureSize':3, 'L2gradient':True}):
        super(CannySegSliceModule, self).__init__()
        self.canny_kwargs = canny_kwargs
        self.num_procs = num_procs
        self.lng_cutoff = lng_cutoff
        self.slope_cutoff = slope_cutoff
        
        # Loading unet
        model_name = '../../../../../../lung-segmentation-2d/trained_model.hdf5'
        self.UNet = load_model(model_name)   
        
    def canny_edge_detection(self, x, kwargs):
        x = np.array(x, dtype=np.uint8)
        x = x+np.min(x)
        return cv2.Canny(x, **kwargs)
    
    
    def get_canny_edges(self, x, kwargs):
        v = x.median()*255
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
        kwargs.update({'threshold1':lower_thresh, 'threshold2':upper_thresh})
        proc_fun = partial(self.canny_edge_detection, kwargs=kwargs)
        x_out = proc_fun(x*255)
        return x_out

    
    def remove_small_regions(self, img, size):
        """Morphologically removes small (less than size) connected regions of 0s or 1s."""
        img = morphology.remove_small_objects(img, size)
        img = morphology.remove_small_holes(img, size)
        return img
    
    
    def get_lung_segmentation(self, x):
        xx = (cv2.resize(np.array(x), dsize=(256, 256))-mean[0])/std[0]
        im_shape = xx.shape
        xx = xx[None,:,:,None]

        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = self.UNet.predict(xx)[..., 0]

        pr = pred > 0.5

        # Remove regions smaller than 2% of the image
        pr = self.remove_small_regions(pr, 0.02 * np.prod(im_shape))
        return pr, xx
    
    def get_canny_seg_diff_image(self, x, kwargs):
        edges = self.get_canny_edges(x, kwargs).squeeze()
        lungs, _ = self.get_lung_segmentation(x)

        # resizing to align with lung segmentation
        edges = cv2.resize(edges.squeeze(), dsize=(256, 256))

        # getting difference image
        diff_img = (lungs*255 - edges)*lungs

        return diff_img, lungs, edges
    
    def get_hough_lines_1(self, x):
        diff_img_copy = np.uint8(x.copy()).squeeze()
        line_image = np.zeros(diff_img_copy.shape)
        lines = cv2.HoughLinesP(diff_img_copy,rho = 1,theta = 1*np.pi/180, threshold=4, minLineLength=3, maxLineGap=5)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),1)
            
            return line_image
        else:
            return None
    
    def get_hough_lines_2(self, x):
        lines_2 = cv2.HoughLinesP(x,rho = 1,theta = 1*np.pi/180, threshold=4,minLineLength=10, maxLineGap=4)
        line_image_2 = np.zeros(x.shape)
        if lines_2 is not None:
            for line in lines_2:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image_2,(x1,y1),(x2,y2),(255,255,255),1)

            return line_image_2, lines_2
        else:
            return line_image_2, None
    
    def get_connected_components(self, im,sz=10):
        # 5. perform a connected component analysis on the thresholded image, then store only the "large" blobs into mask
        labels = measure.label(im, neighbors=8, background=0)
        # mask is initially all black
        mask = np.zeros(im.shape, dtype="uint8")
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the mask for this label and count the
            # number of pixels
            labelMask = np.zeros(im.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > sz:
                # add the blob into mask
                mask = cv2.add(mask, labelMask)
        return mask
    
    def get_slope(self,line):
        x1, y1, x2, y2 = line
        slope = (y2-y1)/(x2-x1)
        return slope
    
    def get_dist(self,ln):
        x1, y1, x2, y2 = ln
        dist = np.sqrt((x2-x1)**2 + (y1-y2)**2)
        return dist

    def slope_percentage(self,lns):
        slope_abs = [np.abs(self.get_slope(line[0])) for line in lns]
        vert = [slope_val>1 for slope_val in slope_abs]
        return np.sum(vert)/len(vert)
    
    def heuristic_function(self,x,lng):
        slope_perc = self.slope_percentage(x)
        is_vert = np.abs(slope_perc)>=self.slope_cutoff
        
        # minimum length of connected component -- make an arg?
        lng_cutoff = self.lng_cutoff
        return (is_vert and (lng>lng_cutoff))
    
    def forward_fun(self, args):
        # dimension of x is [1024, 1024]
        x, return_image = args
        
        # Getting canny edges and lung segmentations
        diff_img, lungs, edges = self.get_canny_seg_diff_image(x, kwargs=self.canny_kwargs)
        
        # Computing edges only within the lung segmentation
        edge_reduced_image = lungs.squeeze()*edges.squeeze()
        
        # Getting first hough lines
        hough_line_image_1 = self.get_hough_lines_1(edge_reduced_image)
        
        # If no lines, no drain
        if hough_line_image_1 is None:
            if return_image:
                return (2, None, None)
            else:
                return 2
        
        # Getting morphological closure
        morph_clos_1 = cv2.morphologyEx(hough_line_image_1, cv2.MORPH_CLOSE, kernel=np.array([2,10]),iterations=10)
    
        # Skeletonizing
        skel = skeletonize(morph_clos_1/255)
        
        # Getting labels from skeleton
        labels = measure.label(skel, neighbors=8, background=0)
        
        # Getting largest connected component
        sorted_components = Counter(labels.flatten()).most_common()[1:] # Ignoring background component
        large_comp, length = sorted_components[0]
        conn = self.get_connected_components(skel,length-1)
        
        # Getting hough lines for largest connected component
        hough_line_image_2, hough_lines_2 = self.get_hough_lines_2(conn)
        
        # If no lines, no drain
        if hough_lines_2 is None:
            if return_image:
                return (2, None, None)
            else:
                return 2
        
        # Executing heuristic function
        out = self.heuristic_function(hough_lines_2, length)
       
        # Converting negatives to MeTaL convention
        if out == 0:
            out = 2
        
        if return_image:
            return (out, hough_line_image_2, hough_lines_2)
        else:
            return out
        
    def forward(self, x, return_image=False):
        batch_size = x.shape[0]
        if self.num_procs>1: #DEPRECATED
            pass
            #pool = Pool(processes=self.num_procs)
            #print("Using pool...")
            #args_list = []
            #inds = [a for a in range(x.shape[0])]
            #for ind in inds:
            #    args_list.append([x[ind,0,:,:],return_image])
                
            # pool = mp.Pool(processes=4)
            #results = [pool.apply_async(cube, args=(x,)) for x in range(1,7)]
            #output = [p.get() for p in results]
            #predictions = pool.map(self.forward_fun,args_list)
            #pool.close()
            #pool.join()
            #return torch.Tensor(np.array(predictions).astype(int))
        
           # delayed_results = []
           # for ii in range(batch_size):
           #     x_tmp = x[ii,0,:,:]
           #     pred = dask.delayed(self.forward_fun)([x_tmp,return_image])
           #     delayed_results.append(pred)

           # print('Computing with dask...')
           # predictions = dask.compute(*delayed_results)
           # return torch.Tensor(np.array(predictions).astype(int))
        else:
            predictions = []
            for ii in range(batch_size):
               # print(f"Running sample {ii+1} of {batch_size}...")
                x_tmp = x[ii,0,:,:]
                predictions.append(self.forward_fun([x_tmp,return_image]))
            if not return_image:
                return torch.Tensor(np.array(predictions).astype(int))
            else:
                preds = torch.Tensor(np.array([p[0] for p in predictions]).astype(int))
                ims = [p[1] for p in predictions]
                lines = [p[2] for p in predictions]
                return preds, ims, lines
            
    
    def predict(self, x, return_image=False):
        return self.forward(x, return_image=return_image)
                