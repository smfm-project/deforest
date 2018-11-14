
import argparse
import csv
import cv2
import datetime
import glob
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from osgeo import gdal
import scipy.ndimage
from scipy import signal
import skimage.filters.rank
from skimage.morphology import disk
import skimage.exposure
import warnings

import sys
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen2mosaic')
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen1mosaic')

import sen2mosaic.utilities
import sen1mosaic.utilities

import pdb


# Declare global variables
global X # Feature array
global clf # Model


def loadScenes(source_files, md = None, sort = True):
    '''
    Load and sort input scenes (Sentinel-2).
    
    Args:
        source_files: List of Sentinel-2 granule directories
        md: sen2mosaic.utilities.Metadata() object. If specified, images from outside the md are not loaded.
        sort: Set True to sort files by date
    
    Returns: 
        A list of scenes of class utilities.LoadScene()
    '''
    
    def getS2Res(resolution):
        '''
        '''
        
        if float(resolution) < 60:
            return 20
        else:
            return 60
    
    # Allow for the processing of one source_file or list of source_files
    if type(source_files) != list: source_files = [source_files]
    
    # Load scenes
    scenes = []
    for source_file in source_files:
        
        try:
            scenes.append(sen2mosaic.utilities.LoadScene(source_file, resolution = getS2Res(md.res)))
        except AssertionError:
            continue
        except IOError:
            try:
                scenes.append(sen1mosaic.utilities.LoadScene(source_file))
            except:
                print 'WARNING: Failed to load scene: %s'%source_file.filename
                continue
    
    if md is not None:
        # Remove scenes that aren't within output extent
        scenes = sen2mosaic.utilities.getSourceFilesInTile(scenes, md)
    
    # Sort by date
    if sort:
        scenes = [scene for _,scene in sorted(zip([s.datetime.date() for s in scenes],scenes))]
    
    return scenes


def _loadS1Single(scene, md = None, polarisation = 'VV'):
    """
    Extract backscatter data given .dim file and polarisation.
    
    Args:
        dim_file: Path to a .dim file from a pre-processed Sentinel-1 image
        polarisation: (i.e. 'VV' or 'VH')
    
    Returns:
        A maked numpy array of backscatter.
    """
        
    mask = scene.getMask(md = md)
    indices = scene.getBand(polarisation, md = md)
    
    # Turn into a masked array
    indices = np.ma.array(indices, mask = mask)
    
    return indices


def _loadS1Dual(scene, md = None):
    """
    Extract backscatter metrics from a dual-polarised Sentinel-1 image.
    
    Args:
        dim_file: Path to a .dim file from a pre-processed Sentinel-1 image
    
    Returns:
        A maked numpy array of VV, VH and VV/VH backscatter.
    """
    
    assert scene.image_type == 'S1dual', "input file %s does not appear to be a dual polarised Sentinel-1 file"%dim_file
    
    VV = _loadS1Single(scene, md = md, polarisation = 'VV')
    VH = _loadS1Single(scene, md = md, polarisation = 'VH')
    
    VV_VH = VH - VV # Proportional difference, logarithmic
    
    indices = np.ma.dstack((VV, VH, VV_VH))
        
    return indices


def _adapthist(im, md, radius, clip_limit = 0.75):
    '''
    Perform Contrast Limited Adative Histogram Equalisation (CLAHE) on an image. This should accentuate areas of forest/nonforest.
    
    Args:
        im: A numpy array
        md: Metadata of im (sen2mosaic.utilities.Metadata() object().
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
    
def _mean_filter(im, window_size = 3):
    '''
    Runs an averaging (mean) filter over an image.
    
    Args:
        im: A numpy array.
        window_size: An integer defining the size of the averaging window. Must be an odd number.
    
    Retuns:
        im, filtered
    '''
    
    assert window_size % 2 == 1 and type(window_size) == int, "Window size must be ean odd number."

    border = window_size / 2
    
    im_filt = signal.convolve2d(im, np.ones((window_size, window_size)) / (window_size ** 2), boundary = 'symm')
    
    return im_filt[border:-border, border:-border]
    
    
def _stdev_filter(im, window_size = 3):
    '''
    Run a standard deviation filter over an image.
    Based on https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows and http://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    
    Args:
        im: A numpy array.
        window_size: An integer defining the size of the averaging window. Must be an odd number.
    
    Returns:
        im, filtered
    '''
    
    c1 = _mean_filter(im, window_size = window_size)
    c2 = _mean_filter(im * im, window_size = window_size)
    
    variance = c2 - c1 * c1
    variance[variance < 0] += 0.0001 # Prevents divide by zero errors.

    return np.ma.array(np.sqrt(variance), mask = im.mask)


def _build_percentile_image(im, md, percentile = 95., radius = 1200):
    """
    Approximates the operation of a percentile filter for a masked numpy array, by downsampling blocks and interpolating across the image. Existing percentile filters operate far too slowly to be practical, and are not designed for masked arays. This function correlates strongly (Pearson's R ~0.85) with the outputs of a percentile filter.
    
    Args:
        im: A masked numpy array
        md: Metadata of im (sen2mosaic.utilities.Metadata() object().
        percentile: Percentile to extact for each pixel, between 0-100.
        radius: Search radius in m to calculate block size. Larger values are equivalent to a larger filter kernel size.
        
    Returns:
        im, filtered
    """
    
    # Calculate the size of each block
    size = int(round((float(radius) / md.res),0))
    
    # Build output array (stacked samples for each block)
    im_downsampled = np.zeros((int(math.ceil(md.nrows / float(size))), int(math.ceil(md.ncols / float(size))), size * size))
    im_downsampled[:,:,:] = np.nan
    
    # build blocks
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


def _time_of_year(scene_datetime, shape, function = 'sin'):
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
    
    # Spring/Autumn
    if function == 'sin':
        return np.zeros(shape, dtype = np.float32) + np.sin(((scene_datetime.date() - datetime.date(scene_datetime.year,1,1)).days / 365.) * 2 * np.pi)
      
    # Winter/Summer
    elif function == 'cos':
        return np.zeros(shape, dtype = np.float32) + np.cos(((scene_datetime.date() - datetime.date(scene_datetime.year,1,1)).days / 365.) * 2 * np.pi)    


def _loadS2(scene, md = None):
    """
    Calculate a range of vegetation features given .SAFE file and resolution.
    
    Args:
        scene: A sen2mosaic.utilities.LoadScene() object
        md: A sen2mosaic.utilities.Metadata() object. Defaults to 'None', which takes the extent of the input file.
    
    Returns:
        A maked numpy array of 20 vegetation features for predicting forest/nonforest probabilities.
    """
       
    mask = scene.getMask(correct = True, md = md)
    
    # Convert mask to a boolean array, allowing only values of 4 (vegetation), 5 (bare sois), and 6 (water)
    mask = np.logical_or(mask < 4, mask > 6)
    
    # To do: getBand down/upscaling
        
    # Load the data (as masked numpy array)
    B02 = scene.getBand('B02', md = md)[mask == False] / 10000.
    B04 = scene.getBand('B04', md = md)[mask == False] / 10000.
    B11 = scene.getBand('B11', md = md)[mask == False] / 10000.
    B12 = scene.getBand('B12', md = md)[mask == False] / 10000.
    
    if scene.resolution == 10:
        B08 = scene.getBand('B08', md = md)[mask == False] / 10000.
    else:
        B08 = scene.getBand('B8A', md = md)[mask == False] / 10000.
    
    #B01 = scene.getBand('B01', md = md)[mask == False] / 10000. # Removed, as necessitates 60 m processing
    B05 = scene.getBand('B05', md = md)[mask == False] / 10000.
    B06 = scene.getBand('B06', md = md)[mask == False] / 10000.
    

    features = np.zeros((mask.shape[0], mask.shape[1], 20), dtype = np.float32)
        
    # Don't report div0 errors
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        
        # Calculate a series of vegetation indices (based loosely on Shultz 2016)
        
        # NDVI (vegetation)
        features[:,:,0][mask == False] = (B08 - B04) / (B08 + B04)
        
        # EVI (vegetation)
        features[:,:,1][mask == False] = ((B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1)) * 2.5
        
        # GEMI (vegetation)
        n = (2 * (B08 ** 2 - B04 ** 2) + 1.5 * B08 + 0.5 * B04) / (B08 + B04 + 0.5)
        features[:,:,2][mask == False] = (n * (1 - 0.25 * n)) - ((B04 - 0.125) / (1 - B04))
        
        # NDMI (vegetation)
        features[:,:,3][mask == False] = (B08 - B11) / (B08 + B11)
        
        # SAVI (vegetation)
        features[:,:,4][mask == False] = (1 + 0.5) * ((B08 - B04) / (B08 + B04 + 0.5))

        # RENDVI (vegetation)
        features[:,:,5][mask == False] = (B06 - B05) / (B05 + B06) #(2 * B01)
        
        # SIPI (vegetation)
        # features[:,:,6][mask == False] = (B08 - B01) / (B08 - B04) 
        
        # NBR (fire)
        features[:,:,7][mask == False] = (B08 - B12) / (B08 + B12)
        
        # MIRBI (fire)
        features[:,:,8][mask == False] = (10 * B12) - (9.8 * B11) + 2
        
        # Get rid of inifite values etc.
        features[np.logical_or(np.isinf(features), np.isnan(features))] = 0.
        
        # Turn into a masked array
        features = np.ma.array(features, mask = np.repeat(np.expand_dims(mask,2), features.shape[-1], axis=2))
    
    # Hamunyela et al. functions (spatial context)
    features[:,:,9]  = features[:,:,0] / _build_percentile_image(features[:,:,0], md, radius = 1200)
    features[:,:,10] = features[:,:,0] / _build_percentile_image(features[:,:,0], md, radius = 2400)
    features[:,:,11] = features[:,:,0] / _build_percentile_image(features[:,:,0], md, radius = 4800)
    features[:,:,12] = features[:,:,0] / _build_percentile_image(features[:,:,0], md, radius = 9600)
    
    # CLAHE (spatial context)
    features[:,:,13] = _adapthist(features[:,:,0], md, 2400, clip_limit = 0.75)
    features[:,:,14] = _adapthist(features[:,:,0], md, 4800, clip_limit = 0.75)
    
    # Texture
    features[:,:,15] = _stdev_filter(features[:,:,0], window_size = 3)
    features[:,:,16] = _stdev_filter(features[:,:,0], window_size = 9)
    
    # Coefficient of variation (seasonality??)
    features[:,:,17] = features[:,:,16] / _mean_filter(features[:,:,16], window_size = 9)
    
    # Time of year
    features[:,:,18] = _time_of_year(scene.datetime, features[:,:,0].shape, function = 'sin')
    features[:,:,19] = _time_of_year(scene.datetime, features[:,:,0].shape, function = 'cos')

    # Tidy up residual nodata values
    features[np.logical_or(np.isinf(features), np.isnan(features))] = 0.
    
    return features


def loadIndices(scene, md = None, force_S1single = False):
    '''
    Load indices from a Sentinel-1 or Sentinel-2 utilities.LoadScene() object.
    
    Args:
        scene: 
        md_dest:
        force_S1single: Force the loading of only a single band from Sentinel-1, even where dual polarised
    Returns:
        A numpy array
    '''
    
    if scene.image_type == 'S2':
        indices = _loadS2(scene, md = md)
    elif scene.image_type == 'S1dual' and force_S1single == False:
        indices = _loadS1Dual(scene, md = md)
    else:
        indices = _loadS1Single(scene, md = md)
    
    return indices


def loadModel(image_type):
    '''
    Loads logistic regression coefficients from config .csv file.
    
    Args:
        image_type: A string wih the image type (i.e. S1single, S1dual, S2)
    Returns:
        An array containing model coefficients, the first element being the intercept, and remaining elements coefficients in layer order.
        Arrays containing the mean and scaling parameters to transform the imput image.
    '''
    
    # Get location of current file
    directory = os.path.dirname(os.path.abspath(__file__))
    
    # Determine name of output file
    filename = '%s/cfg/%s_model.pkl'%('/'.join(directory.split('/')[:-1]),image_type)
    
    from sklearn.externals import joblib
    clf = joblib.load(filename) 
       
    return clf


def classify(data, image_type, nodata = 255):
    """
    Calculate the probability of forest
    
    Args:
        data: A numpy array containing deseasonalised data
        image_type: The source of the image, one of 'S2', 'S1single', or 'S1dual'.
        nodata: Optionally specify a nodata value. Defaults to 255.
        
    Returns:
        A numpy array containing probability of a pixel being forest in that view. Units are percent, with a nodata value set to nodata.
    """
    
    assert image_type in ['S2', 'S1single', 'S1dual'], "image_type must be one of 'S2', 'S1single', or 'S1dual'. The classify() function was given %s."%image_type
        
    #coefs, means, scales = loadCoefficients(image_type)
    
    clf = loadModel(image_type)
        
    if data.ndim == 2:
        data = np.ma.expand_dims(data, 2)
    
    data_shape = data.shape
    mask = data.mask.sum(axis=2) == np.max(data.mask.sum(axis=2))
    
    
    p_forest = (np.zeros((data_shape[0],data_shape[1]), dtype=np.uint8) + 255)
    
    #p_forest = np.ma.array(p_forest, mask = mask, fill_value=255)
    
    X = data.reshape(data_shape[0] * data_shape[1], data_shape[2])
    X = X[X.mask.sum(axis=1) != X.mask.sum(axis=1).max(),:].data

    # Do the classification
    if X.shape[0] == 0:
        y_pred = 0. # In case no data in entire image
    else:
        y_pred = clf.predict_proba(X)[:,1]
    
    p_forest[mask == False] = np.round((y_pred * 100.),0).astype(np.uint8)
    
    p_forest = np.ma.array(p_forest, mask = mask, fill_value = 255)
    
    return p_forest


def getOutputName(scene, output_dir = os.getcwd(), output_name = 'classified'):
    '''
    Build a standardised image output name
    
    Args:
        scene: A sen2mosaic.utilities.LoadScene() object
        output_dir: Output directory, defaults to current working directory
        output_name: A string to prepend to the output images.
    
    Returns:
        Path to the output file
    '''
    
    output_dir = output_dir.rstrip('/')
    
    output_name = '%s/%s_%s_%s_%s_%s.tif'%(output_dir, output_name, scene.image_type, scene.tile, datetime.datetime.strftime(scene.datetime, '%Y%m%d'), datetime.datetime.strftime(scene.datetime, '%H%M%S'))

    return output_name


def _classify_all(input_list):
    '''
    Multiprocessing requires some gymnastics. This is a wrapper function to exexute classify_all() using a single input expressed as a list.
    
    Args:
        input_list: List containing [source_file, target_extent, resolution, EPSG_code, output_dir, output_name]
    '''
    
    source_file = input_list[0]
    target_extent = input_list[1]
    resolution = input_list[2]
    EPSG_code = input_list[3]
    output_dir = input_list[4]
    output_name = input_list[5]
        
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    scene = loadScenes(source_file, md = md_dest, sort = True)
    
    classify_all(scene, md_dest, output_dir = output_dir, output_name = output_name)


def classify_all(scenes, md_dest, output_dir = os.getcwd(), output_name = 'classified'):
    '''
    Classify a list of Sentinel-2 scenes
    
    Args:
        scenes: A list of scenes, of type sen2mosaic.utilites.LoadScene()
        md_dest: Destination image metadata, of type sen2mosaic.utilites.Metadata()
    '''
    
    # Allow for the processing of one scene or list of scenes.
    if type(scenes) != list: scenes = [scenes]
    
    for scene in scenes:
        
        try:
            print 'Doing %s'%scene.filename
            indices = loadIndices(scene, md = md_dest)
        except:
            print 'Error loading %s'%scene.filename
            continue
            
        # Classify the image
        p_forest = classify(indices, scene.image_type)
        
        # Save data to disk
        ds = sen2mosaic.utilities.createGdalDataset(md_dest, data_out = p_forest.filled(255), filename = getOutputName(scene, output_dir = output_dir, output_name = output_name), nodata = 255, driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
        

def main(source_files, target_extent, resolution, EPSG_code, n_processes = 1, output_dir = os.getcwd(), output_name = 'classified'):
    """
    Classify a list of Sentinel-2 input files to forest/nonforest probabilities, reproject and output to GeoTiffs.
    
    Args:
        source_files: A list of directories for Sentinel-2 input tiles. 
        target_extent: Extent of search area, in format [xmin, ymin, xmax, ymax]
        resolution: Resolution to re-sample search area, in meters. Best to be 10 m, 20 m or 60 m to match Sentinel-2 resolution.
        EPSG_code: EPSG code of search area.
        n_processes: Number of processes, defaults to 1.
        output_dir: Directory to output classifier predictors. Defaults to current working directory.
        output_name: Name to precede output file. Defaults to 'processed'. 
        
    """
    
    # Catch case of single source_file input
    if type(source_files) != list: source_files = [source_files]
    
    # Check that files are input
    assert len(source_files) > 0, "No source files found at input location."
    
    # Get absolute path of input files.
    source_files = [os.path.abspath(i) for i in source_files]
    
    # Find files from input directory/granule etc.
    source_files = sen2mosaic.utilities.prepInfiles(source_files, '2A')
    
    # Load scene metadata
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    scenes = loadScenes(source_files, md = md_dest, sort = True)
    
    # Reduce files to those within md_dest
    scenes = sen2mosaic.utilities.getSourceFilesInTile(scenes, md_dest)
    
    # Classify
    if n_processes == 1:
        classify_all(scenes, md_dest, output_dir = output_dir, output_name = output_name)
    
    # Classify in parallel
    elif n_processes > 1:
        instances = multiprocessing.Pool(n_processes)
        instances.map(_classify_all, [[scene.filename, target_extent, resolution, EPSG_code, output_dir, output_name] for scene in scenes])
        instances.close()
    

if __name__ == '__main__':
    '''
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process Sentinel-1 and Sentinel-2 to match a predefined CRS, perform a deseasaonalisation operation to reduce the impact of seasonality on reflectance/backscsatter, and output forest probability images.")
    
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    # Required arguments
    required.add_argument('-te', '--target_extent', nargs = 4, metavar = ('XMIN', 'YMIN', 'XMAX', 'YMAX'), type = float, help = "Extent of output image tile, in format <xmin, ymin, xmax, ymax>.")
    required.add_argument('-e', '--epsg', type=int, help="EPSG code for output image tile CRS. This must be UTM. Find the EPSG code of your output CRS as https://www.epsg-registry.org/.")
    optional.add_argument('-res', '--resolution', metavar = 'N', type=int, help="Specify a resolution to output.")
    
    # Optional arguments
    optional.add_argument('infiles', metavar = 'FILES', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files (level 2A) in .SAFE format, Sentinel-1 input files in .dim format, or a mixture. Specify one or more valid Sentinel-2 .SAFE, a directory containing .SAFE files, or multiple granules through wildcards (e.g. *.SAFE/GRANULE/*). Defaults to processing all granules in current working directory.')
    optional.add_argument('-p', '--n_processes', type = int, metavar = 'N', default = 1, help = "Maximum number of tiles to process in paralell. Bear in mind that more processes will require more memory. Defaults to 1.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'CLASSIFIED', help="Optionally specify a string to precede output filename.")
    
    # Get arguments
    args = parser.parse_args()
        
    # Execute script
    main(args.infiles, args.target_extent, args.resolution, args.epsg,n_processes = args.n_processes, output_dir = args.output_dir, output_name = args.output_name)
    
    # Examples
    #~/anaconda2/bin/python ~/DATA/deforest/deforest/classify.py ../chimanimani/L2_files/S1 -r 60 -e 32736 -te 499980 7790200 609780 7900000 -n S1_test
    #~/anaconda2/bin/python ~/DATA/deforest/deforest/classify.py ../chimanimani/L2_files/S2/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -n S2_test
