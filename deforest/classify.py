
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
import skimage.exposure

import deforest.feature

import sen2mosaic
import sen2mosaic.IO

#import sen1mosaic.utilities

import pdb


# Declare global variables
global X # Feature array
global clf # Model



def loadS2(scene, md = None):
    """
    Calculate a range of vegetation features given .SAFE file and resolution.
    
    Args:
        scene: A sen2mosaic.core.LoadScene() object
        md: A sen2mosaic.core.Metadata() object. Defaults to 'None', which takes the extent of the input file.
    
    Returns:
        A maked numpy array of 20 vegetation features for predicting forest/nonforest probabilities.
    """
    
    if md is None: md = scene.metadata
    
    # Set up blank array to contain features
    features = np.ma.array(np.zeros((md.nrows, md.ncols, 20), dtype = np.float32), mask= np.ones((md.nrows, md.ncols, 20), dtype = np.bool))
    
    # Calculate vegetation and burn scar indices
    features[:,:,:9] = deforest.feature.spectralIndices(scene, md = md)
        
    # Calculates features representing Hamunyela et al. (2017) approach (spatial context)
    features[:,:,9]   = deforest.feature.percentileImage(features[:,:,0], md, percentile = 95, radius = 600)
    features[:,:,10]  = deforest.feature.percentileImage(features[:,:,0], md, percentile = 95, radius = 2400)
    features[:,:,11]  = deforest.feature.percentileImage(features[:,:,0], md, percentile = 95, radius = 9600)
    features[:,:,12]  = deforest.feature.percentileImage(features[:,:,0], md, percentile = 95, radius = 38400)
    
    # CLAHE (spatial context)
    features[:,:,13] = deforest.feature.adaptiveHistogramStretch(features[:,:,0], md, radius = 2400, clip_limit = 0.75)
    features[:,:,14] = deforest.feature.adaptiveHistogramStretch(features[:,:,0], md, radius = 4800, clip_limit = 0.75)  
    
    # Texture
    features[:,:,15] = deforest.feature.standardDeviationFilter(features[:,:,0], md, radius = md.res * 3)
    features[:,:,16] = deforest.feature.standardDeviationFilter(features[:,:,0], md, radius = md.res * 9)

    # Coefficient of variation (seasonality?)    
    features[:,:,17] = deforest.feature.coefficientOfVariationFilter(features[:,:,0], md, radius = md.res * 3)
        
    features[:,:,18] = deforest.feature.timeOfYear(scene.datetime, md, function = 'sin')
    features[:,:,19] = deforest.feature.timeOfYear(scene.datetime, md, function = 'cos')
    
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
        indices = loadS2(scene, md = md)
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
    
    scene = sen2mosaic.LoadScene(source_file, resolution = res_s2)
    
    
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
        
        try:
            print('Doing %s'%scene.granule)
            indices = loadIndices(scene, md = md_dest)
        except:
            print('Error loading %s'%scene.granule)
            continue
            
        # Classify the image
        p_forest = classify(indices, scene.image_type)
        
        # Save data to disk
        ds = sen2mosaic.IO.createGdalDataset(md_dest, data_out = p_forest.filled(255), filename = getOutputName(scene, output_dir = output_dir, output_name = output_name), nodata = 255, driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
        