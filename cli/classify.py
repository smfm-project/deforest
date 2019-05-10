
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

import deforest.classify
import sen2mosaic
import sen2mosaic.IO

#import sen1mosaic.utilities

import pdb


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
        scene: A sen2mosaic.core.LoadScene() object
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
        
    md_dest = sen2mosaic.core.Metadata(target_extent, resolution, EPSG_code)
    
    res_S2 = 20 if resolution < 60 else 60
    
    #scene = sen2mosaic.IO.loadSceneList([source_file], resolution = res_S2, md_dest = md_dest, sort_by = 'date')
    
    scene = sen2mosaic.LoadScene(source_file, resolution = res_S2)
    
    
    #scene = loadScenes(source_file, md = md_dest, sort = True)
    
    classify_all([scene], md_dest, output_dir = output_dir, output_name = output_name)


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
                
        print('Doing %s'%scene.granule)
        indices = deforest.classify.loadS2(scene, md = md_dest)
                    
        # Classify the image
        p_forest = classify(indices, scene.image_type)
        
        # Save data to disk
        ds = sen2mosaic.IO.createGdalDataset(md_dest, data_out = p_forest.filled(255), filename = getOutputName(scene, output_dir = output_dir, output_name = output_name), nodata = 255, driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
        

def main(source_files, target_extent, resolution, EPSG_code, n_processes = 1, output_dir = os.getcwd(), output_name = 'classified', level = '2A'):
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
    source_files = sen2mosaic.IO.prepInfiles(source_files, level = level)
    
    # Load output image metadata
    md_dest = sen2mosaic.core.Metadata(target_extent, resolution, EPSG_code)
    
    # Load a list of scenes
    scenes = sen2mosaic.IO.loadSceneList(source_files, md_dest = md_dest, level = '2A', sort_by = 'date')
    
    # Classify
    if n_processes == 1:
        classify_all(scenes, md_dest, output_dir = output_dir, output_name = output_name)
    
    # Classify in parallel
    elif n_processes > 1:
        instances = multiprocessing.Pool(n_processes)
        instances.map(_classify_all, [[scene.granule, target_extent, resolution, EPSG_code, output_dir, output_name] for scene in scenes])
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
    optional.add_argument('-l', '--level', type=str, metavar = '1C/2A', default = '2A', help="Processing level to use, either '1C' or '2A'. Defaults to level 2A.")
    optional.add_argument('-p', '--n_processes', type = int, metavar = 'N', default = 1, help = "Maximum number of tiles to process in paralell. Bear in mind that more processes will require more memory. Defaults to 1.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'CLASSIFIED', help="Optionally specify a string to precede output filename.")
    
    # Get arguments
    args = parser.parse_args()
        
    # Execute script
    main(args.infiles, args.target_extent, args.resolution, args.epsg,n_processes = args.n_processes, output_dir = args.output_dir, output_name = args.output_name, level = args.level)
    
    # Examples
    #~/anaconda2/bin/python ~/DATA/deforest/deforest/classify.py ../chimanimani/L2_files/S1 -r 60 -e 32736 -te 499980 7790200 609780 7900000 -n S1_test
    #~/anaconda2/bin/python ~/DATA/deforest/deforest/classify.py ../chimanimani/L2_files/S2/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -n S2_test
