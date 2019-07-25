
import argparse
import datetime
import multiprocessing
import numpy as np
import os
from osgeo import gdal

import sen2mosaic.core
import sen2mosaic.IO

import deforest.classify


import pdb

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
    model = input_list[4]
    output_dir = input_list[5]
    output_name = input_list[6]
        
    md_dest = sen2mosaic.core.Metadata(target_extent, resolution, EPSG_code)
    
    res_S2 = 20 if resolution < 60 else 60
        
    scene = sen2mosaic.core.LoadScene(source_file, resolution = res_S2)
        
    classify_all([scene], md_dest, model = model, output_dir = output_dir, output_name = output_name)


def classify_all(scenes, md_dest, model = deforest.classify.getCfgDir()+'/S2_model.pkl', output_dir = os.getcwd(), output_name = 'classified'):
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
        features = deforest.classify.loadFeatures(scene, md = md_dest)
                    
        # Classify the image
        p_forest = deforest.classify.classify(features, model)
        
        # Save data to disk
        ds = sen2mosaic.IO.createGdalDataset(md_dest, data_out = p_forest.filled(255), filename = getOutputName(scene, output_dir = output_dir, output_name = output_name), nodata = 255, driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
        

def main(source_files, target_extent, resolution, EPSG_code, n_processes = 1, model = deforest.classify.getCfgDir()+'/S2_model.pkl', output_dir = os.getcwd(), output_name = 'classified', level = '2A'):
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
        classify_all(scenes, md_dest, model = model, output_dir = output_dir, output_name = output_name)
    
    # Classify in parallel
    elif n_processes > 1:
        instances = multiprocessing.Pool(n_processes)
        instances.map(_classify_all, [[scene.granule, target_extent, resolution, EPSG_code, model,  output_dir, output_name] for scene in scenes])
        instances.close()
    

if __name__ == '__main__':
    '''
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process Sentinel-2 to match a predefined CRS and classify each to show a probability of forest (0-100%) in each pixel.")
    
    parser._action_groups.pop()
    
    positional = parser.add_argument_group('positional arguments')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    # Positional arguments
    optional.add_argument('infiles', metavar = 'FILES', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files in .SAFE format. Specify one or more valid Sentinel-2 .SAFE files, a directory containing .SAFE files, or multiple granules through wildcards (e.g. *.SAFE/GRANULE/*). Defaults to processing all granules in current working directory.')
    
    # Required arguments
    required.add_argument('-te', '--target_extent', nargs = 4, metavar = ('XMIN', 'YMIN', 'XMAX', 'YMAX'), type = float, help = "Extent of output image tile, in format <xmin, ymin, xmax, ymax>.")
    required.add_argument('-e', '--epsg', type=int, help="EPSG code for output image tile CRS. This must be UTM. Find the EPSG code of your output CRS as https://www.epsg-registry.org/.")
    required.add_argument('-r', '--resolution', metavar = 'N', type=int, help="Specify a resolution to output.")
    
    # Optional arguments
    optional.add_argument('-m', '--model', type=str, metavar = 'PKL', default = deforest.classify.getCfgDir()+'/S2_model.pkl', help="Path to .pkl model, produced with train.py. Defaults to a test model, trained on data from Chimanimani in Mozambique.")
    optional.add_argument('-l', '--level', type=str, metavar = '1C/2A', default = '2A', help="Processing level to use, either '1C' or '2A'. Defaults to level 2A.")
    optional.add_argument('-p', '--n_processes', type = int, metavar = 'N', default = 1, help = "Maximum number of tiles to process in paralell. Bear in mind that more processes will require more memory. Defaults to 1.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'S2', help="Specify a string to precede output filename. Defaults to 'S2'.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory")
    
    
    # Get arguments
    args = parser.parse_args()
        
    # Execute script
    main(args.infiles, args.target_extent, args.resolution, args.epsg,n_processes = args.n_processes, model = args.model, output_dir = args.output_dir, output_name = args.output_name, level = args.level)
    
    # Examples
    
    #deforest classify ~/SMFM/chimanimani/L2_files/S2 -r 20 -e 32736 -te 399980 7790200 609780 7900000 -n S2_test -p 1