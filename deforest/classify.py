
import argparse
import csv
import datetime
import glob
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from osgeo import gdal
import scipy.ndimage
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


def loadLandcover(landcover_map, md_dest):
    '''
    Load a landcover map, and reproject it to the CRS defined in md.
    For test purposes only.
    '''
    
    from osgeo import osr
    
    # Load landcover map
    ds_source = gdal.Open(landcover_map,0)
    geo_t = ds_source.GetGeoTransform()
    proj = ds_source.GetProjection()

    # Get extent and resolution    
    nrows = ds_source.RasterXSize
    ncols = ds_source.RasterYSize
    ulx = float(geo_t[4])
    uly = float(geo_t[5])
    xres = float(geo_t[0])
    yres = float(geo_t[3])
    lrx = ulx + (xres * ncols)
    lry = uly + (yres * nrows)
    extent = [ulx, lry, lrx, uly]
    
    # Get EPSG
    srs = osr.SpatialReference(wkt = proj)
    srs.AutoIdentifyEPSG()
    EPSG = int(srs.GetAttrValue("AUTHORITY", 1))
    
    # Add source metadata to a dictionary
    md_source =sen2mosaic.utilities.Metadata(extent, xres, EPSG)
    
    # Build an empty destination dataset
    ds_dest = sen2mosaic.utilities.createGdalDataset(md_dest,dtype = 1)
    
    # And reproject landcover dataset to match input image
    landcover = sen2mosaic.utilities.reprojectImage(ds_source, ds_dest, md_source, md_dest)
    
    return np.squeeze(landcover)
    
    

def loadS1Single(scene, md = None, polarisation = 'VV', normalisation_type = 'NONE', reference_scene = None):
    """
    Extract backscatter data given .dim file and polarisation
    
    Args:
        dim_file:
        polarisation: (i.e. 'VV' or 'VH')
    
    Returns:
        A maked numpy array of backscatter.
    """
        
    mask = scene.getMask(md = md)
    indices = scene.getBand(polarisation, md = md)
    
    # Turn into a masked array
    indices = np.ma.array(indices, mask = mask)
    
    # Normalise data
    if normalisation_type != 'NONE':
        indices = normalise(indices, md = md, normalisation_type = normalisation_type, reference_scene = reference_scene)
    
    return indices


def loadS1Dual(scene, md = None, normalisation_type = 'NONE', reference_scene = None):
    """
    Extract backscatter metrics from a dual-polarised Sentinel-1 image.
    
    Args:
        dim_file: 
    
    Returns:
        A maked numpy array of VV, VH and VV/VH backscatter.
    """
    
    assert scene.image_type == 'S1dual', "input file %s does not appear to be a dual polarised Sentinel-1 file"%dim_file
    
    VV = loadS1Single(scene, md = md, polarisation = 'VV')
    
    VH = loadS1Single(scene, md = md, polarisation = 'VH')
    
    VV_VH = VH - VV # Proportional difference, logarithmic
    
    indices = np.ma.dstack((VV, VH, VV_VH))
    
    # Normalise data
    if normalisation_type != 'NONE':
        indices = normalise(indices, md = md, normalisation_type = normalisation_type, reference_scene = reference_scene)
        
    return indices



def adapthist(im, md, radius = 4800, clip_limit = 0.75):
    '''
    '''
    
    size = float(radius) / md.res
    kernel_size = (int(round(md.nrows / size, 0)), int(round(md.ncols / size, 0)))
    
    im_min = np.min(im)
    im_max = np.max(im)
    
    data_rescaled = (((im - im_min) / (im_max - im_min)) * 65535).astype(np.uint16)
    
    # Fill in data gaps with nearest valid pixel
    ind = scipy.ndimage.distance_transform_edt(data_rescaled.mask, return_distances = False, return_indices = True)
    data_rescaled = data_rescaled.data[tuple(ind)]
    
    import cv2
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = kernel_size)
    data_rescaled = clahe.apply(data_rescaled) / 65535.
        
    im = np.ma.array((data_rescaled * (im_max - im_min)) + im_min, mask = im.mask)
    
    return im
    
    
    
    
def stdev_filter(im, window_size = 3):
    '''
    Based on https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
    and http://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    '''

    from scipy import signal

    c1 = signal.convolve2d(im, np.ones((window_size, window_size)) / (window_size ** 2), boundary = 'symm')
    c2 = signal.convolve2d(im*im, np.ones((window_size, window_size)) / (window_size ** 2), boundary = 'symm')

    border = window_size / 2

    variance = c2 - c1 * c1
    variance[variance < 0] += 0.0001 # Prevents divide by zero errors.

    return np.ma.array(np.sqrt(variance)[border:-border, border:-border], mask = im.mask)


def build_percentile_image(im, md, percentile = 95., radius = 1200):
    """
    This approximates the operation of a percentile filter, by downsampling blocks and interpolating across the image. A percetnile file operates far too slowly to be practical.
    
    Args:
        im:
        md:
        percentile:
        radius:
        
    Returns:
        
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
        warnings.simplefilter("ignore", RuntimeWarning)
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
    
    

def loadS2(scene, md = None, normalisation_type = 'none', reference_scene = None):
    """
    Calculate a range of vegetation indices given .SAFE file and resolution.
    
    Args:
        scene: 
    
    Returns:
        A maked numpy array of vegetation indices.
    """
       
    mask = scene.getMask(correct = True, md = md)
    
    # Convert mask to a boolean array, allowing only values of 4 (vegetation), 5 (bare sois), and 6 (water)
    mask = np.logical_or(mask < 4, mask > 6)
    
    # To do: getBand down/upscaling
        
    # Load the data (as masked numpy array)
    B02 = scene.getBand('B02', md = md)[mask == False] / 10000.
    #B03 = scene.getBand('B03', md = md)[mask == False] / 10000.
    B04 = scene.getBand('B04', md = md)[mask == False] / 10000.
    B11 = scene.getBand('B11', md = md)[mask == False] / 10000.
    B12 = scene.getBand('B12', md = md)[mask == False] / 10000.
    
    if scene.resolution == 10:
        B08 = scene.getBand('B08', md = md)[mask == False] / 10000.
    else:
        B08 = scene.getBand('B8A', md = md)[mask == False] / 10000.
    
    B01 = scene.getBand('B01', md = md)[mask == False] / 10000.
    B05 = scene.getBand('B05', md = md)[mask == False] / 10000.
    B06 = scene.getBand('B06', md = md)[mask == False] / 10000.
    
    # Calculate vegetation indices from Shultz 2016
    indices = np.zeros((mask.shape[0], mask.shape[1], 16), dtype = np.float32)
    
    # Don't report div0 errors
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        
        # NDVI
        indices[:,:,0][mask == False] = (B08 - B04) / (B08 + B04)
        
        # EVI
        indices[:,:,1][mask == False] = ((B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 1)) * 2.5
        
        # GEMI
        n = (2 * (B08 ** 2 - B04 ** 2) + 1.5 * B08 + 0.5 * B04) / (B08 + B04 + 0.5)
        indices[:,:,2][mask == False] = (n * (1 - 0.25 * n)) - ((B04 - 0.125) / (1 - B04))
        
        # NDMI
        indices[:,:,3][mask == False] = (B08 - B11) / (B08 + B11)
        
        # SAVI
        indices[:,:,4][mask == False] = (1 + 0.5) * ((B08 - B04) / (B08 + B04 + 0.5))

        # RENDVI
        indices[:,:,5][mask == False] = (B06 - B05) / (B05 + B06) #(2 * B01)
        
        # SIPI
        indices[:,:,6][mask == False] = (B08 - B01) / (B08 - B04) 
        
        # NBR
        indices[:,:,7][mask == False] = (B08 - B12) / (B08 + B12)
        
        # MIRBI
        indices[:,:,8][mask == False] = (10 * B12) - (9.8 * B11) + 2
        
        # Get rid of inifite values etc.
        indices[np.logical_or(np.isinf(indices), np.isnan(indices))] = 0.
        
        # Turn into a masked array
        indices = np.ma.array(indices, mask = np.repeat(np.expand_dims(mask,2), indices.shape[-1], axis=2))
    
    # Hamunyela et al. functions
    indices[:,:,9] = indices[:,:,0] / build_percentile_image(indices[:,:,0], md, radius = 1200)

    indices[:,:,10] = indices[:,:,0] / build_percentile_image(indices[:,:,6], md, radius = 2400)

    indices[:,:,11] = indices[:,:,0] / build_percentile_image(indices[:,:,0], md, radius = 4800)
    
    # CLAHE
    indices[:,:,12] = adapthist(indices[:,:,0], md, radius = 2400, clip_limit = 0.75)
    indices[:,:,13] = adapthist(indices[:,:,0], md, radius = 4800, clip_limit = 0.75)

    # Texture
    indices[:,:,14] = stdev_filter(indices[:,:,0], window_size = 3)
    indices[:,:,15] = stdev_filter(indices[:,:,0], window_size = 9)
        
    # Tidy up residual nodata values
    indices[np.logical_or(np.isinf(indices), np.isnan(indices))] = 0.
    
    return indices


def loadIndices(scene, md = None, normalisation_type = 'none', reference_scene = None, force_S1single = False):
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
        indices = loadS2(scene, md = md, normalisation_type = normalisation_type, reference_scene = reference_scene)
    elif scene.image_type == 'S1dual' and force_S1single == False:
        indices = loadS1Dual(scene, md = md, normalisation_type = normalisation_type, reference_scene = reference_scene)
    else:
        indices = loadS1Single(scene, md = md, normalisation_type = normalisation_type, reference_scene = reference_scene)
    
    return indices


def _createGdalDataset(md, data_out = None, filename = '', driver = 'MEM', dtype = 3, RasterCount = 1, nodata = None, options = []):
    '''
    Function to create an empty gdal dataset with georefence info from metadata dictionary.

    Args:
        md: Object from Metadata() class.
        data_out: Optionally specify an array of data to include in the gdal dataset.
        filename: Optionally specify an output filename, if image will be written to disk.
        driver: GDAL driver type (e.g. 'MEM', 'GTiff'). By default this function creates an array in memory, but set driver = 'GTiff' to make a GeoTiff. If writing a file to disk, the argument filename must be specified.
        dtype: Output data type. Default data type is a 16-bit unsigned integer (gdal.GDT_Int16, 3), but this can be specified using GDAL standards.
        options: A list containing other GDAL options (e.g. for compression, use [compress='LZW'].

    Returns:
        A GDAL dataset.
    '''
    from osgeo import gdal, osr
        
    gdal_driver = gdal.GetDriverByName(driver)
    ds = gdal_driver.Create(filename, md.ncols, md.nrows, RasterCount, dtype, options = options)
    
    ds.SetGeoTransform(md.geo_t)
    
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(md.EPSG_code)
    ds.SetProjection(proj.ExportToWkt())
    
    # If a data array specified, add data to the gdal dataset
    if type(data_out).__module__ == np.__name__:
        
        if len(data_out.shape) == 2:
            data_out = np.ma.expand_dims(data_out,2)
        
        for feature in range(RasterCount):
            ds.GetRasterBand(feature + 1).WriteArray(data_out[:,:,feature])
            
            if nodata != None:
                ds.GetRasterBand(feature + 1).SetNoDataValue(nodata)
    
    # If a filename is specified, write the array to disk.
    if filename != '':
        ds = None
    
    return ds

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


def _classifyChunk(input_list):
    '''
    '''
    
    chunks = input_list[0]
    i = input_list[1]
    print str(i)
    try:
        out = clf.predict_proba(X[i::chunks,:])[:,1]
    except:
        out = np.zeros_like(X[i::chunks,0]) + 0.5 # In case nodata
    return out


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
    
    #chunks = 100
    #instances = multiprocessing.Pool(25)
    #y_pred = instances.map(_classifyChunk, [[chunks, i] for i in range(chunks)])
    #instances.close()
    
    # Extract pixel predictions, and place on the map
    #for i,y in enumerate(y_pred):
    #    p_forest[i::chunks] = np.round((y * 100.),0).astype(np.uint8)
    
    #p_forest_out = np.zeros((data_shape[0], data_shape[1]), dtype = np.uint8)
    #p_forest_out[mask == False] = p_forest
    
    #p_forest_out = np.ma.array(p_forest_out,mask = mask, fill_value = 255)
    
    return p_forest
    
         

def getOutputName(scene, output_dir = os.getcwd(), output_name = 'classified'):
    '''
    '''
    
    output_dir = output_dir.rstrip('/')
    
    output_name = '%s/%s_%s_%s_%s_%s.tif'%(output_dir, output_name, scene.image_type, scene.tile, datetime.datetime.strftime(scene.datetime, '%Y%m%d'), datetime.datetime.strftime(scene.datetime, '%H%M%S'))

    return output_name



def loadScenes(source_files, md = None, sort = True):
    '''
    Load and sort input scenes (Sentinel-1 or Sentinel-2).
    
    Args:
        source_files: 
        md: 
        sort: Set True to sort files by date
    
    Returns: 
        A list of scenes of class utilitiesLoadScene()
    '''
    
    def getS2Res(resolution):
        '''
        '''
        
        if float(resolution) < 60:
            return 20
        else:
            return 60
    
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


def _classify_all(input_list):
    '''
    '''
    
    source_file = input_list[0]
    target_extent = input_list[1]
    resolution = input_list[2]
    EPSG_code = input_list[3]
    output_dir = input_list[4]
    output_name = input_list[5]
    
    print 'Res: %s, %s'%(str(resolution), source_file)
    
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    scene = loadScenes([source_file], md = md_dest, sort = True)[0]
    
    print 'Doing %s'%scene.filename
    
    indices = loadIndices(scene, md = md_dest, normalisation_type = 'local')
        
    p_forest = classify(indices, scene.image_type)
    
    # Save data to disk
    ds = sen2mosaic.utilities.createGdalDataset(md_dest, data_out = p_forest.filled(255), filename = getOutputName(scene, output_dir = output_dir, output_name = output_name), nodata = 255, driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
    
    return p_forest
    

def main(source_files, target_extent, resolution, EPSG_code, output_dir = os.getcwd(), output_name = 'classified'):
    """
    """
       
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    scenes = loadScenes(source_files, md = md_dest, sort = True)
       
    _classify_all([[scene.filename, target_extent, resolution, EPSG_code, output_dir, output_name] for scene in scenes][0])
    
    instances = multiprocessing.Pool(25)
    p_forest = instances.map(_classify_all, [[scene.filename, target_extent, resolution, EPSG_code, output_dir, output_name] for scene in scenes])
    instances.close()
    
    """
    # Determine output extent and projection
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    
    # Load and sort input scenes
    scenes = loadScenes(source_files, md = md_dest, sort = True)
    
    for n, scene in enumerate(scenes):
        print 'Doing %s'%scene.filename.split('/')[-1]
        
        # Load data
        indices = loadIndices(scene, md = md_dest, normalisation_type = 'local')#, reference_scene = reference_scenes[inds[n]])
        
        # Classify to probability of forest
        p_forest = classify(indices, scene.image_type)
        
         # Save data to disk
        ds = sen2mosaic.utilities.createGdalDataset(md_dest, data_out = p_forest.data, filename = getOutputName(scene, output_dir = output_dir, output_name = output_name), driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
    """


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
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'CLASSIFIED', help="Optionally specify a string to precede output filename.")
    
    # Get arguments
    args = parser.parse_args()
    
    # Get absolute path of input files.
    infiles = [os.path.abspath(i) for i in args.infiles]
    
    # Find files from input directory/granule etc.
    infiles_S2 = sen2mosaic.utilities.prepInfiles(infiles, '2A')
    infiles_S1 = sen1mosaic.utilities.prepInfiles(infiles)
    
    # Execute script
    main(infiles_S2 + infiles_S1, args.target_extent, args.resolution, args.epsg, output_dir = args.output_dir, output_name = args.output_name)
    
    #~/anaconda2/bin/python ~/DATA/deforest/deforest/classify.py ../chimanimani/L2_files/S1 -r 60 -e 32736 -te 499980 7790200 609780 7900000 -n S1_test
    #~/anaconda2/bin/python ~/DATA/deforest/deforest/classify.py ../chimanimani/L2_files/S2/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -n S2_test
