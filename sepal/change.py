import argparse
import csv
import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal, osr
import scipy.stats

import pdb

import deforest.change


def _bayesUpdate(prior, likelihood, scale_factor = 10000):
    '''
    Update Baysian prior based on new observation.

    Args:
        prior: Prior probability of non-forest
        likelihood: Probability of forest from new observation
        scale_factor: Where probability not represented on 0 - 1 scale                                               
    Returns:
        posterior probability
    '''

    posterior = (prior * likelihood) / ((prior * likelihood) + ((scale_factor - prior) * (scale_factor - likelihood)))

    return posterior * scale_factor



def calculateDeforestation(stack_file, dates_file, deforestation_threshold = 0.99, block_weight = 0.1, scale_factor = 10000):
    '''
    '''
    
    # Read stack of forest probabilities
    ds = gdal.Open(stack_file)
    
    # Initialise output arrays
    deforestation = np.zeros((ds.RasterYSize, ds.RasterXSize), dtype = bool)
    warning = np.zeros_like(deforestation, dtype = bool)
    deforestation_date = np.zeros_like(deforestation, dtype='datetime64[D]')
    pchange = np.zeros_like(deforestation, dtype=np.int16)

    # Read dates list
    with open(dates_file) as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        dates = [dt.datetime.strptime(d[0], '%Y-%m-%d') for d in reader]
    

    for n, date in enumerate(dates):
       
        PF = ds.GetRasterBand(n+1).ReadAsArray().astype(np.int16)
        PNF = scale_factor - PF

        mask = PF == ds.GetRasterBand(n+1).GetNoDataValue()
        
        # Apply block weighting
        PNF[PNF < block_weight * scale_factor] = np.round(block_weight * scale_factor, 0).astype(np.int16)
        PNF[PNF > (1 - block_weight) * scale_factor] = np.round((1 - block_weight) * scale_factor, 0).astype(np.int16)

        # Step 1: Flag potential changes
        flag = PNF > 0.5 * scale_factor

        # Step 2: Update pchange for current time step
        
        # Case A: A new flag appears. Set pchange to PNF
        sel = (warning == False) & flag & (deforestation == False) & (mask == False)
        pchange[sel] = PNF[sel]
        deforestation_date[sel] = date
        warning[sel] = True

        # Case B: There is a warning in place, but no confirmation. Update pchange using PNF
        sel = warning & (deforestation == False) & (mask == False)
        pchange[sel] = _bayesUpdate(pchange[sel], PNF[sel], scale_factor = scale_factor)

        # Step 3: Reject or accept warnings

        # Case A: Reject warning where pchange falls below 50%
        sel = warning & (pchange < 0.5 * scale_factor) & (deforestation == False) & (mask == False)
        warning[sel] = False

        # Tidy up
        deforestation_date[sel] = dt.date(1970,1,1)
        pchange[sel] = 0

        # Case B: Confirm warning where pchange > deforestation_threshold
        sel = warning & (deforestation == False) & (pchange >= deforestation_threshold * scale_factor) & (mask == False)
        deforestation[sel] = True

    # Get day of year of change
    change_day = (deforestation_date == deforestation_date.astype('datetime64[Y]')).astype(np.float32)

    # Outut year of deforestation for confirmed events (units = year)
    confirmed_deforestation = deforestation_date.astype('datetime64[Y]').astype(np.float32) + 1970. + (change_day / 365.)
    confirmed_deforestation[deforestation == False] = 0.
    confirmed_deforestation[confirmed_deforestation == 1970.] - 0.

    # Return final probability of early warning (units = %)
    warning_deforestation = pchange 
    warning_deforestation[deforestation] = 0
    warning_deforestation[pchange < (0.5 * scale_factor)] = 0

    return confirmed_deforestation, warning_deforestation

    
    
def main(source_dir, deforestation_threshold = 0.99, block_weight = 0.1, output_dir = os.getcwd(), output_name = 'SMFMDeforest'):
    '''
    Process a list of probability maps to generate a map of deforestation year and warning estimates of upcoming events. Outputs a GeoTiff of confirmed deforestation years and warnings of deforestation.
    
    Args:
        source_dir: A time series directory from SEPAL
        deforestation_threshold: Threshold probability for flagging deforestation. Defaults to 0.99. Must be between > 0.5 an < 1.0.
        block_weight: Limits the probability range of input files. For example, default block_weight 0.1 allows of a probability range of 0.1 to 0.9 in input files. This reduces the probability of false positives.
        output_dir = Directory to output GeoTiffs
        output_name = String to prepend to output images
        '''
     
    # Get absolute path of input files
    stack_file = os.path.abspath(source_dir) + '/stack.vrt'
    dates_file = os.path.abspath(source_dir) + '/dates.csv'
    
    # Run through images to identify changes
    deforestation_confirmed, deforestation_warning = calculateDeforestation(stack_file, dates_file, deforestation_threshold = deforestation_threshold, block_weight = block_weight)
 
    # Output images
    deforest.change.outputImage(deforestation_confirmed, stack_file, '%s/%s_%s.tif'%(output_dir, output_name, 'confirmed'), dtype = gdal.GDT_Int16)
    deforest.change.outputImage(deforestation_warning,   stack_file, '%s/%s_%s.tif'%(output_dir, output_name, 'warning'),   dtype = gdal.GDT_Int16)


if __name__ == '__main__':
    '''
    '''

    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process probability maps to generate a map of deforestation year and warning estimates of upcoming events.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('source_dir', metavar = 'DIR', type = str, help = 'Input directory from sepal.')
    
    # Optional arguments
    optional.add_argument('-t', '--threshold', metavar = 'N', type = float, default = 0.99, help = 'Set a threshold probability to identify deforestation (between 0 and 1). High thresholds are more strict in the identification of deforestation. Defaults to 0.99.')
    optional.add_argument('-b', '--block_weight', metavar = 'N', type = float, default = 0.1, help = 'Set a block weighting threshold to limit the range of forest/nonforest probabilities. Set to 0 for no block-weighting. Parameter cannot be set higher than 0.5. Defaults to 0.1.')
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory. If nothing specified, downloads will output to the present working directory, given a standard filename.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'SMFMDeforest', help="Optionally specify a string to precede output filename. Defaults to the same as input files.")
    
    # Get arguments
    args = parser.parse_args()

    main(args.source_dir, deforestation_threshold = args.threshold, block_weight = args.block_weight, output_dir = args.output_dir, output_name = args.output_name)
    
