
import cv2
import datetime
import math
import numpy as np
import scipy.ndimage
from scipy import signal
import skimage.filters.rank
from skimage.morphology import disk
import warnings

import pdb

def spectralIndices(scene, md = None, improve_mask = True):
    '''
    Calculate a series of spectral indices that describe vegetation state and burned status
    
    Args:
        scene: A sen2mosaic.LoadScene() object
        md: An optional sen2mosaic.Metadata() object to reproject scene to match
        improve_mask: Set False to not apply improvements to the Sentinel-2 L2A mask
    
    #Returns:
        A masked array of spectral indices
    '''
    
    if md is None: md = scene.metadata
    
    mask = scene.getMask(improve = True, md = md)
    
    # Convert mask to a boolean array, allowing only values of 4 (vegetation), 5 (bare sois), and 6 (water)
    mask = np.logical_or(mask < 4, mask > 6)
       
    # Load spectral bands
    
    #B01 = scene.getBand('B01', md = md)[mask == False] / 10000.
    B02 = scene.getBand('B02', md = md)[mask == False] / 10000.
    B04 = scene.getBand('B04', md = md)[mask == False] / 10000.
    B05 = scene.getBand('B05', md = md)[mask == False] / 10000.
    B06 = scene.getBand('B06', md = md)[mask == False] / 10000.
    B11 = scene.getBand('B11', md = md)[mask == False] / 10000.
    B12 = scene.getBand('B12', md = md)[mask == False] / 10000.
    
    if scene.resolution == 10:
        B08 = scene.getBand('B08', md = md)[mask == False] / 10000.
    else:
        B08 = scene.getBand('B8A', md = md)[mask == False] / 10000.
    
    indices = np.zeros((md.nrows, md.ncols, 9), dtype = np.float32)
    
    # Don't report div0 errors
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        
        # Calculate a series of vegetation indices (based loosely on Shultz 2016)
        
        # NDVI (vegetation)
        indices[:,:,0][mask == False] = (B08 - B04) / (B08 + B04)
        
        # EVI (vegetation)
        indices[:,:,1][mask == False] = ((B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1)) * 2.5
        
        # GEMI (vegetation)
        n_val = (2 * (B08 ** 2 - B04 ** 2) + 1.5 * B08 + 0.5 * B04) / (B08 + B04 + 0.5)
        indices[:,:,2][mask == False] = (n_val * (1 - 0.25 * n_val)) - ((B04 - 0.125) / (1 - B04))
        n_val = None
        
        # NDMI (vegetation)
        indices[:,:,3][mask == False] = (B08 - B11) / (B08 + B11)
        
        # SAVI (vegetation)
        indices[:,:,4][mask == False] = (1 + 0.5) * ((B08 - B04) / (B08 + B04 + 0.5))

        # RENDVI (vegetation)
        indices[:,:,5][mask == False] = (B06 - B05) / (B05 + B06) #(2 * B01)
        
        # Moisture Stress Index (MDI) (leaf water content)
        indices[:,:,6][mask == False] = B11 / B08
        
        # SIPI (vegetation)
        #indices[:,:,6][mask == False] = (B08 - B01) / (B08 - B04) 
        
        # NBR (fire)
        indices[:,:,7][mask == False] = (B08 - B12) / (B08 + B12)
        
        # MIRBI (fire)
        indices[:,:,8][mask == False] = (10 * B12) - (9.8 * B11) + 2
        
        # Get rid of inifite values etc.
        indices[np.logical_or(np.isinf(indices), np.isnan(indices))] = 0.
    
    # Turn into a masked array
    indices = np.ma.array(indices, mask = np.repeat(np.expand_dims(mask,2), indices.shape[-1], axis=2))
    
    return indices


def percentileImage(im, md, percentile = 95., radius = 1200):
    """
    Approximates the operation of a percentile filter for a masked numpy array, by downsampling blocks and interpolating across the image. Existing percentile filters operate far too slowly to be practical, and are not designed for masked arays. This function correlates strongly (Pearson's R ~0.85) with the outputs of a percentile filter.
    
    Args:
        im: A masked numpy array
        md: Metadata of im (sen2mosaic.core.Metadata() object().
        percentile: Percentile to extact for each pixel, between 0-100.
        radius: Search radius in m to calculate block size. Larger values are equivalent to a larger filter kernel size.
        
    Returns:
        A local percentile version of im
    """
    
    # Calculate the size of each block
    size = int(round((float(radius) / md.res),0))
    
    # Build output array (stacked samples for each block)
    im_downsampled = np.zeros((int(math.ceil(md.nrows / float(size))), int(math.ceil(md.ncols / float(size))), size * size))
    im_downsampled[:,:,:] = np.nan
    
    # Build blocks
    for row in range(im_downsampled.shape[0]):
        for col in range(im_downsampled.shape[1]):
            this_data = im[row*size:(row*size)+size, col*size:(col*size)+size].flatten()
            
            data_in = np.zeros((size * size), dtype = this_data.dtype)
            data_in[:] = np.nan
            
            data_in[:(this_data.mask == False).sum()] = this_data.data[this_data.mask == False]
            
            im_downsampled[row,col,:] = data_in
    
    # Calculate percentile of blocks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning) # nanpercentile throws warnings where all values are nan. These are safe to ignore
        im_percentile = np.nanpercentile(im_downsampled, percentile, axis = 2)
    
    # Fill in data gaps with nearest valid pixel so as not to upset interpolation
    ind = scipy.ndimage.distance_transform_edt(np.isnan(im_percentile), return_distances = False, return_indices = True)
    im_percentile = im_percentile[tuple(ind)]
    
    # Interpolate
    im_out = scipy.ndimage.interpolation.zoom(im_percentile, (float(md.nrows) / im_percentile.shape[0], float(md.ncols) / im_percentile.shape[1]))
    
    # Output, with a tidy mask
    im_out[im.mask] = 0.
    im_out = np.ma.array(im_out, mask = im.mask)
    
    return im_out


def spatialContext(im, md, percentile = 95., radius = 1200):
    '''
    Calculate pixel value relative to spatial context, following Hamunyela et al. (2017).
    
    Args:
        im: A masked numpy array
        md: Metadata of im (sen2mosaic.core.Metadata() object().
        percentile: Percentile to extact for each pixel, between 0-100.
        radius: Search radius in m to calculate block size. Larger values are equivalent to a larger filter kernel size.
        
    Returns:
        image values divided by local percentile of image values over radius
    '''
    
    return im / percentileImage(im, md, percentile = percentile, radius = radius)

def adaptiveHistogramStretch(im, md, radius = 2400, clip_limit = 0.75):
    '''
    Perform Contrast Limited Adative Histogram Equalisation (CLAHE) on an image. This should accentuate areas of forest/nonforest.
    
    Args:
        im: A numpy array
        md: Metadata of im (sen2mosaic.core.Metadata() object().
        radius: The scale (in m) over which histogram equalisation is performed.
        clip_limit: Maximum histogram intensity. Higher values result in greater contrast in the resulting image.
    
    Returns:
        im, following histogram equalisation
    '''
    
    size = float(radius) / md.res
    kernel_size = (int(round(md.nrows / size, 0)), int(round(md.ncols / size, 0)))
    
    im_min = np.min(im)
    im_max = np.max(im)
    
    data_rescaled = (((im - im_min) / (im_max - im_min)) * 65535).astype(np.uint16)
    
    # Fill in data gaps with nearest valid pixel
    ind = scipy.ndimage.distance_transform_edt(data_rescaled.mask, return_distances = False, return_indices = True)
    data_rescaled = data_rescaled.data[tuple(ind)]
    
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = kernel_size)
    data_rescaled = clahe.apply(data_rescaled) / 65535.
        
    im = np.ma.array((data_rescaled * (im_max - im_min)) + im_min, mask = im.mask)
    
    return im


def meanFilter(im, md, radius = 60):
    '''
    Runs an averaging (mean) filter over an image.
    
    Args:
        im: A numpy array.
        window_size: An integer defining the size of the averaging window. Must be an odd number.
    
    Retuns:
        im, filtered
    '''
    
    # Get nearest odd window_size
    window_size = int(math.floor(float(radius) / md.res))
    if window_size % 2 == 0: window_size += 1
    
    assert window_size >= 3, "Filter radius too small, ensure it's at least 3 times the image resolution."
        
    # Deals with border-effect of filter (floor division to calculate border size)
    border = window_size // 2
    
    im_filt = signal.convolve2d(im, np.ones((window_size, window_size)) / (window_size ** 2), boundary = 'symm')
    
    return im_filt[border:-border, border:-border]
    
    
def standardDeviationFilter(im, md, radius = 60):
    '''
    Run a standard deviation filter over an image.
    Based on https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows and http://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    
    Args:
        im: A numpy array.
        window_size: An integer defining the size of the averaging window. Must be an odd number.
    
    Returns:
        im, filtered
    '''
    
    c1 = meanFilter(im, md, radius = radius)
    c2 = meanFilter(im * im, md, radius = radius)
    
    variance = c2 - c1 * c1
    variance[variance < 0] += 0.0001 # Prevents divide by zero errors.
    
    return np.ma.array(np.sqrt(variance), mask = im.mask)


def coefficientOfVariationFilter(im, md, radius = 60):
    '''
    '''
    
    return standardDeviationFilter(im, md, radius = radius) / meanFilter(im, md, radius = radius)


def timeOfYear(scene_datetime, md, function = 'sin'):
    '''
    Generate a time of year feature, based on trigonometric functions. sin and a cos functions operatate circularly, so may make for better features than a day of year.
    
    Args:
        scene_datetime: a datetime.datetime object
        shape: a tuple for the output shaper
        function: 'sin' or 'cos', defaults to 'sin'
    
    Returns:
        A numpy array
    '''
    
    assert function in ['sin', 'cos'], "Function must be 'sin' or 'cos'."
    
    # Get output shape
    shape = (md.nrows, md.ncols)
    
    # Spring/Autumn
    if function == 'sin':
        return np.zeros(shape, dtype = np.float32) + np.sin(((scene_datetime.date() - datetime.date(scene_datetime.year,1,1)).days / 365.) * 2 * np.pi)
      
    # Winter/Summer
    elif function == 'cos':
        return np.zeros(shape, dtype = np.float32) + np.cos(((scene_datetime.date() - datetime.date(scene_datetime.year,1,1)).days / 365.) * 2 * np.pi)    
