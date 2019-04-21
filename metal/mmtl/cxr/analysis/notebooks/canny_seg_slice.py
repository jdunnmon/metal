import cv2
from functools import partial
import numpy as np
import torch


# Loading pretrained segnet (from https://github.com/imlab-uiip/lung-segmentation-2d)
from keras.models import load_model
from skimage import exposure, morphology

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