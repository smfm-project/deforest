import argparse
import csv
import multiprocessing
import numpy as np
import os
import random

import sen2mosaic
import sen2mosaic.IO

import deforest.classify

import pdb
import traceback


def _outputData(forest_px, nonforest_px, output_name = 'S2', output_dir = None, verbose = False):
    """
    Save data to a .npz file for analysis in model fitting script.
    
    Args:
        forest_px: List of pixel values representing forest.
        nonforest_px: List of pixel values representing nonforest
        image_name: String to represent image type (e.g. S1single, S1dual, S2). Defaults to 'S2'.
        output_dir: Directory to output .npz file. Defaults to current working directory.
    """
    
    # Ensure that px are formatted as arrays
    forest_px = np.array(forest_px)
    nonforest_px = np.array(nonforest_px)
    
    # Find output directory
    if output_dir == None:
        output_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Test that it's possible to output
    assert os.path.exists(output_dir), "output_dir does not exist. output_dir is set to: %s"%output_dir
    assert os.access(output_dir, os.W_OK | os.X_OK), "output_dir does not have write access. output_dir is set to: %s."%output_dur
    
    # Save
    np.savez('%s/%s_training_data.npz'%(output_dir, output_name), forest_px = forest_px, nonforest_px = nonforest_px) 
    
    if verbose: print('Done!')
    
    return '%s/%s_training_data.npz'%(output_dir, output_name)


def _extractData(input_list):
    '''
    Multiprocessing requires some gymnastics. This is a wrapper function to initiate extractData() for multiprocessing.
    
    Args:
        input_list: A list of inputs for a single source_file, in the format: [source_file. trainging_data, target_extent, resolution, EPSG_code, subset].
    
    Returns:
        A tuple with (a list of forest pixel values, a list of nonforest pixel values)
    '''
    
    source_file = input_list[0]
    s2_res = input_list[1]
    training_data = input_list[2]
    target_extent = input_list[3]
    resolution = input_list[4]
    EPSG_code = input_list[5]
    forest_values = input_list[6]
    nonforest_values = input_list[7]
    field_name = input_list[8]
    subset = input_list[9]
    verbose = input_list[10]
    
    # Load input scene
    md_dest = sen2mosaic.Metadata(target_extent, resolution, EPSG_code)
        
    # Load scene with sen2mosaic
    scene = sen2mosaic.core.LoadScene(source_file, resolution = s2_res)
           
    return extractData([scene], training_data, md_dest, forest_values, nonforest_values, field = field_name, subset = subset)


def _unpackOutputs(outputs):
    '''
    Multiprocessing requires some gymnastics. This is a wrapper function to unpack data from multiprocessing outputs.
    
    Args:
        outputs: outputs from multiprocessing.Pool.map
    
    Returns:
        A tuple with (a list of forest pixel values, a list of nonforest pixel values)
    '''
    
    forest_px, nonforest_px = [], []
    
    for f, nf in outputs:
        forest_px.extend(f)
        nonforest_px.extend(nf)
    
    return forest_px, nonforest_px


def _getPixels(indices, mask, subset = 5000):
    '''
    Extract a sample of pixels from a 2/3d array with a 2d mask.
    
    Args:
        indices: Numpy array of forest/nonforest predictors from classify.loadIndices()
        mask: A boolean array of the same 2d extent as indices
        subset: Maximum number of pixels to extract for each class from each image
        
    Returns:
        A list containing a sample of pixel values for use as training data.
        
    '''
        
    if indices.ndim == 2:
        indices = np.ma.expand_dims(indices, 2)
        
    sub = np.logical_and(mask, np.sum(indices.mask,axis=2)==0)
    indices_subset = indices[sub].data
    sub = np.zeros(indices_subset.shape[0], dtype = np.bool)
    sub[:subset] = True
    np.random.shuffle(sub)
    
    data_out = np.squeeze(indices_subset[sub,:]).tolist()
    
    if np.unique(np.array([len(i) for i in data_out])).shape[0] > 1:
        print('    Unexpected number of dimensions, skipping')
        data_out = []
    
    return data_out


def extractData(scenes, training_data, md_dest, forest_values, nonforest_values, field = '', subset = 5000, n_processes = 1, verbose = False, output = False, output_dir = os.getcwd(), output_name = 'S2'):
    '''
    Extract pixel values from a list of scenes.
    
    Args:
        scenes: A list of scenes of type deforest.classify.loadScenes()
        training_data: A GeoTiff, .vrt of .shp file containing training pixels/polygons.
        md_dest: A metadata file from sen2mosaic.core.Metadata().
        forest_values: A list of raster classes (integers) or shapefile attribute values (str) indicating forest in training_data.
        nonforest_values: A list of raster classes (integers) or shapefile attribute values (str) indicating nonforest in training_data.
        subset: Maximum number of pixels to extract for each class from each image
    
    Returns:
        A tuple with (a list of forest pixel values, a list of nonforest pixel values)
    '''
    
    # If more than one process, re-initiate via _extractData
    if n_processes > 1:
        
        # Initiate extractData with multiprocessing
        instances = multiprocessing.Pool(n_processes)
        results = instances.map(_extractData, [[scene.granule, scene.resolution, training_data, md_dest.extent, md_dest.res, md_dest.EPSG_code, forest_values, nonforest_values, field, subset, verbose] for scene in scenes])
        instances.close()
        
        # Unpack outputs to usable form
        forest, nonforest = _unpackOutputs(results)
    
    # Or if a single process (or single instance)
    else:
        
        # Establish list for outputs
        forest, nonforest = [], []
        
        for scene in scenes:
            
            print('Doing %s'%scene.granule)
                            
            # Get indices
            try:
                indices = deforest.classify.loadFeatures(scene, md = md_dest)
            except:
                traceback.print_exc()
                print('Missing data, continuing')
                continue
            
            if training_data.split('.')[-1] == 'shp':
                forest_mask = sen2mosaic.IO.loadShapefile(training_data, md_dest, field = field, field_values = forest_values)
                nonforest_mask = sen2mosaic.IO.loadShapefile(training_data, md_dest, field = field, field_values = nonforest_values)
                
            elif training_data.split('.')[-1] in ['tif', 'tiff', 'vrt']:
                
                # Load and reproject land cover map
                landcover = sen2mosaic.IO.loadRaster(training_data, md_dest)
                
                # Select matching values, and reshape to mask
                forest_mask = np.in1d(landcover, np.array(forest_values)).reshape(landcover.shape)
                nonforest_mask = np.in1d(landcover, np.array(nonforest_values)).reshape(landcover.shape)
                    
            # Get random subset of pixels
            forest.extend(_getPixels(indices, forest_mask, subset = subset))
            nonforest.extend(_getPixels(indices, nonforest_mask, subset = subset))

    # Output data    
    if output:
        _outputData(forest, nonforest, output_name = output_name, output_dir = output_dir)
    
    return forest, nonforest


