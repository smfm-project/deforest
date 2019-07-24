
import argparse
import multiprocessing
import os
import random

import deforest.extraction
import sen2mosaic

import pdb

def main(source_files, target_extent, resolution, EPSG_code, training_data, forest_values, nonforest_values, level = '2A', field_name = '', n_processes = 1, max_images = 0, max_pixels = 5000, output_dir = os.getcwd(), output_name = 'S2', verbose = True):
    '''main(source_files, training_data, target_extent, resolution, EPSG_code, n_processes = 1, max_pixels = 5000, output_dir = os.getcwd(), output_name = 'S2')
    
    Extract pixel values from source_files and output as a np.savez() file. This is the function that is initiated from the command line.
    
    Args:
        source_files: A list of directories for Sentinel-2 input tiles. 
        target_extent: Extent of search area, in format [xmin, ymin, xmax, ymax]
        resolution: Resolution to re-sample search area, in meters. Best to be 10 m, 20 m or 60 m to match Sentinel-2 resolution.
        EPSG_code: EPSG code of search area.
        training_data: A GeoTiff, .vrt of .shp file containing training pixels/polygons.
        forest_values: A list of raster classes (integers) or shapefile attribute values (str) indicating forest in training_data.
        nonforest_values: A list of raster classes (integers) or shapefile attribute values (str) indicating nonforest in training_data.
        field_name: Shapefile attribute under which forest_values and nonforest_values can be found
        n_processes: Number of processes, defaults to 1.
        max_images: Maximum number of input tiles to extract data from. Defaults to 0, meaning all valid tiles.
        max_pixels: Maximum number of pixels to extract for each class from each image. Defaults to 5000.
        output_dir: Directory to output classifier predictors. Defaults to current working directory.
        output_name: Name to precede output file. Defaults to 'S2'.
        
    '''
    
    assert type(n_processes) == int and n_processes > 0, "n_processes must be an integer > 0."
    
    # Get output metadata
    md_dest = sen2mosaic.Metadata(target_extent, resolution, EPSG_code)
    
    # Load and sort input scenes    
    scenes = sen2mosaic.IO.loadSceneList(source_files, md_dest = md_dest, level = level, sort_by = 'date')#, verbose = verbose)
    
    # Reduce number of inputs to max_images
    if max_images > 0 and len(scenes) > max_images:
        scenes =  [scenes[i] for i in sorted(random.sample(range(len(scenes)), max_images))]
    
    assert len(scenes) > 0, "No valid input files found at specified location."
    
    # Extract pixel values
    forest_px, nonforest_px = deforest.extraction.extractData(scenes, training_data, md_dest, forest_values, nonforest_values, field = field_name, subset = max_pixels, n_processes = n_processes, output = True, output_dir = output_dir, output_name = output_name, verbose = verbose)
    

if __name__ == '__main__':
    '''
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Extract indices from Sentinel-2 data to train a classifier of forest cover. Returns a numpy .npz file containing pixel values for forest/nonforest.")

    parser._action_groups.pop()
    positional = parser.add_argument_group('positional arguments')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Positional arguments
    positional.add_argument('infiles', metavar = 'FILES', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files (level 2A) in .SAFE format. Specify one or more valid Sentinel-2 .SAFE, a directory containing .SAFE files, multiple tiles through wildcards (e.g. *.SAFE/GRANULE/*), or a text file listing files. Defaults to processing all tiles in current working directory.')
    
    # Required arguments
    required.add_argument('-te', '--target_extent', nargs = 4, metavar = ('XMIN', 'YMIN', 'XMAX', 'YMAX'), type = float, help = "Extent of output image tile, in format <xmin, ymin, xmax, ymax>.")
    required.add_argument('-e', '--epsg', type = int, help="EPSG code for output image tile CRS. This must be UTM. Find the EPSG code of your output CRS as https://www.epsg-registry.org/.")
    required.add_argument('-res', '--resolution', metavar = 'N', type=int, help = "Specify a resolution to output.")
    required.add_argument('-t', '--training_data', metavar = 'SHP/TIF', type = str, help = 'Path to training data geotiff/shapefile.')
    required.add_argument('-f', '--forest_values', metavar = 'VALS', type = str, nargs = '*', help = 'Values indicating forest in the training GeoTiff or shapefile')
    required.add_argument('-nf', '--nonforest_values', metavar = 'VALS', type = str, nargs = '*', help = 'Values indicating nonforest in the training GeoTiff or shapefile')
    
    # Optional arguments
    optional.add_argument('-fn', '--field_name', metavar = 'NAME', type = str, default = '', help = 'Shapefile attribute name to search for training data polygons. Defaults to all polygons. Required where inputting a shapefile as training_data.')
    optional.add_argument('-l', '--level', type=str, metavar='1C/2A', default = '2A', help = "Input image processing level, '1C' or '2A'. Defaults to '2A'.")
    optional.add_argument('-mi', '--max_images', type = int, metavar = 'N', default = 0, help = "Maximum number of input tiles to extract data from. Defaults to all valid tiles.")
    optional.add_argument('-mp', '--max_pixels', type = int, metavar = 'N', default = 5000, help = "Maximum number of pixels to extract from each image per class. Defaults to 5000.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Output directory. Defaults to current working directory.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'S2', help="Specify a string to precede output filename. Defaults to 'S2'.")
    optional.add_argument('-p', '--n_processes', type = int, metavar = 'N', default = 1, help = "Maximum number of tiles to process in paralell. Bear in mind that more processes will require more memory. Defaults to 1.")
    optional.add_argument('-v', '--verbose', action = 'store_true', help="Make script verbose.")
    
    # Get arguments
    args = parser.parse_args()
    
    # Format training data values
    if args.training_data.split('.')[-1] in ['tiff', 'tif', 'vrt']:
        args.forest_values = [int(v) for v in args.forest_values]
        args.nonforest_values = [int(v) for v in args.nonforest_values]
    
    # Execute script
    main(args.infiles, args.target_extent, args.resolution, args.epsg, args.training_data, args.forest_values, args.nonforest_values, level = args.level, field_name = args.field_name, n_processes = args.n_processes, max_pixels = args.max_pixels, max_images = args.max_images, output_dir = args.output_dir, output_name = args.output_name, verbose = args.verbose)
    
    # Example:
    # ~/anaconda2/bin/python ~/DATA/deforest/deforest/extract_training_data.py ../chimanimani/L2_files/S2/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t ~/SMFM/landcover/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif -o ./ --max_images 100 -p 20 -f 1 -nf 2 3 4 5 6 7 8 10
    
    # Example 2:
    # deforest extract ~/SMFM/chimanimani/L2_files/S2 -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t /home/sbowers3/SMFM/landcover/mozambique/moz_lulc2016_28032019_wgs84.tif -o ./ --max_images 150 -p 10 -f 21 23 24 25 26 70 71 72 73 74 75 76 77 78 79 -nf 11 12 31 33 41 42 44 51 61 62 -n S2_reorganisation
    
