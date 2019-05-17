import numpy as np

import h5py
from PIL import Image as pil_image

def array_to_img(x, scale=False):
    """Converts a 3D Numpy array to a PIL Image instance.
     Arguments:
         x: Input Numpy array.
         data_format: Image data format.
         scale: Whether to rescale image values
             to be within [0, 255].
     Returns:
         A PIL Image instance.
     Raises:
         ImportError: if PIL is not available.
         ValueError: if invalid `x` or `data_format` is passed.
      """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                             'Got array with shape:', x.shape)

    if scale:
        # ensuring that no values are inf or nan
        x[np.isnan(x)] = 0
        x[~np.isfinite(x)] = 0
        x = x + max(-np.min(x), 0)  # pylint: disable=g-no-augmented-assignment
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
        
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])

#Loading xray images from h5 files
def default_xray_loader_h5(path_h5):
    hf = h5py.File(path_h5,'r')
    xray = np.array(hf.get('data'))
    xray = xray.squeeze()
    xray = np.dstack([xray, xray, xray])
    xray = array_to_img(xray)
    return xray  