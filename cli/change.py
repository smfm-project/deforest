import argparse
import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal, osr
import scipy.stats

import pdb

import deforest.change

    
    
def main(source_files, deforestation_threshold = 0.99, block_weight = 0.1, output_dir = os.getcwd(), output_name = 'OUTPUT'):
    '''
    Process a list of probability maps to generate a map of deforestation year and warning estimates of upcoming events. Outputs a GeoTiff of confirmed deforestation years and warnings of deforestation.
    
    Args:
        source_files: A list of GDAL-compatible input files from deforest.classify()
        deforestation_threshold: Threshold probability for flagging deforestation. Defaults to 0.99. Must be between > 0.5 an < 1.0.
        block_weight: Limits the probability range of input files. For example, default block_weight 0.1 allows of a probability range of 0.1 to 0.9 in input files. This reduces the probability of false positives.
        output_dir = Directory to output GeoTiffs
        output_name = String to prepend to output images
        '''
    
    # Get an output_name from source_files
    if output_name is None: output_name = '_'.join(source_files[0].split('/')[-1].split('_')[:-4])
        
    # Get absolute path of input files
    source_files = [os.path.abspath(i) for i in source_files]
    
    # Run through images to identify changes
    deforestation_confirmed, deforestation_warning = deforest.change.calculateDeforestation(source_files, deforestation_threshold = deforestation_threshold, block_weight = block_weight)
    
    # Output images
    deforest.change.outputImage(deforestation_confirmed, source_files[0], '%s/%s_%s.tif'%(output_dir, output_name, 'confirmed'))
    deforest.change.outputImage(deforestation_warning, source_files[0], '%s/%s_%s.tif'%(output_dir, output_name, 'warning'))


if __name__ == '__main__':
    '''
    '''

    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process probability maps to generate a map of deforestation year and warning estimates of upcoming events.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('infiles', metavar = 'FILES', type = str, nargs = '+', help = 'A list of files output by classify.py, specifying multiple files using wildcards.')
    
    # Optional arguments
    optional.add_argument('-t', '--threshold', metavar = 'N', type = float, default = 0.99, help = 'Set a threshold probability to identify deforestation (between 0 and 1). High thresholds are more strict in the identification of deforestation. Defaults to 0.99.')
    optional.add_argument('-b', '--block_weight', metavar = 'N', type = float, default = 0.1, help = 'Set a block weighting threshold to limit the range of forest/nonforest probabilities. Set to 0 for no block-weighting. Parameter cannot be set higher than 0.5.')
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory. If nothing specified, downloads will output to the present working directory, given a standard filename.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = None, help="Optionally specify a string to precede output filename. Defaults to the same as input files.")
    
    # Get arguments
    args = parser.parse_args()
        
    main(args.infiles, deforestation_threshold = args.threshold, block_weight = args.block_weight, output_dir = args.output_dir, output_name = args.output_name)
    
    # Example:
    #deforest change S2_test*.tif
