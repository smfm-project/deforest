import argparse
import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal, osr
import scipy.stats

import pdb



def getImageDate(infiles):
    '''
    '''
    
    datestrings = ['_'.join(x.split('/')[-1].split('.')[0].split('_')[3:5]) for x in infiles]
    datetimes = np.array([dt.datetime.strptime(date,'%Y%m%d_%H%M%S') for date in datestrings],dtype='datetime64[s]')
    dates = datetimes.astype('datetime64[D]')
    
    return dates


def getImageType(infiles):
    '''
    '''
    
    image_type = np.array(['_'.join(x.split('/')[-1].split('.')[0].split('_')[1:3]) for x in infiles])
    
    return image_type
    


def combineObservations(probability, mask):
    '''
    For cases where there are more than one observations for a given day, here their forest probabilities are combined.
    
    '''
    
    # Set masked elements to 1, so they have no impact on the multiplication
    probability[mask] = 1
    probability_inv = 1 - probability
    probability_inv[mask] = 1
     
    # Calculate probability and inverse probability
    prod = np.prod(probability, axis = 2)
    prod_inv = np.prod(probability_inv, axis = 2)
    
    # Combine into a single probability
    probability = prod / (prod + prod_inv)
    
    # Identify pixels without a single measurement
    mask = np.sum(mask == False, axis = 2) == 0
    
    # Keep things tidy
    probability[mask] = 0.
    
    return probability, mask

    

def bayesUpdate(prior, likelihood):
    '''
    '''
    
    posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood))) 
    
    return posterior




def calculateDeforestation(infiles, deforestation_threshold = 0.995):
    '''
    '''
    
    # Get datetimes for each image
    dates = getImageDate(infiles)
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1
    dates_include = dates[months<10]
    
    # Get the image_type for each image
    image_type = getImageType(infiles)
        
    # Initialise arrays
    ds = gdal.Open(infiles[0])

    YSize = ds.RasterYSize
    XSize = ds.RasterXSize
        
    deforestation = np.zeros((YSize, XSize), dtype=np.bool)
    warning = np.zeros_like(deforestation, dtype=np.bool)
    #PNF_last = np.zeros_like(deforestation, dtype = np.float) + 0.5 # Initialise to 0.5 probability of no forest
    deforestation_date = np.zeros_like(deforestation, dtype='datetime64[D]')
    pchange = np.zeros_like(deforestation, dtype=np.float)


    # Run for each unique date
    for date in sorted(np.unique(dates)):
        
        unique_datetimes = np.unique(dates[dates == date])
        unique_images = np.unique(image_type[dates == date])
        
        #TODO find repeat data in Sentinel-2 imagery. This is tricky, as datetimes can be different yet imagery the same in tile overlap.
        
        # One layer per unique image type. This should allow only one overpass per satellite per granule/track. Needs work to accomodate Sentinel-2 tiles
                
        p_forest = np.zeros((YSize, XSize, unique_images.shape[0]), dtype = np.uint8) + 255
            
        for n, this_image_type in enumerate(unique_images):
            
            # Only select one data file per image type
            for infile in infiles[np.logical_and(dates == date, image_type == this_image_type)]:
                    
                print 'Loading %s'%infile
                data = gdal.Open(infile,0).ReadAsArray()#[3000:5000,1000:3000]
                
                # Select areas that still have the nodata value
                s = p_forest[:,:,n] == 255
                
                # Paste the new data into these locations
                p_forest[:,:,n][s] = data[s]
        #pdb.set_trace()
        # Change percent probability of forest to probability of non forest
        mask = p_forest == 255
        
        PNF = (100 - p_forest) / 100.
        PNF[mask] = 0.
        
        # Remove length-1 axes
        PNF = np.squeeze(PNF)
        mask = np.squeeze(mask)
            
        ## Apply block weighting
        PNF[PNF < 0.1] = 0.1
        PNF[PNF > 0.9] = 0.9
        
        # If multiple observations from the same date exist, combine them (not performed where only one observation)
        if unique_images.shape[0] > 1:
            
            PNF, mask = combineObservations(PNF, mask)
        
        # Step 1: Flag potential changes
        flag = PNF > 0.5
            
        # Step 2: Update pchange for current time step
        
        # Case A: A new flag appears
        s = np.logical_and(np.logical_and(np.logical_and(warning == False, flag == True), deforestation == False), mask == False)
        pchange[s] = PNF[s]
        deforestation_date[s] = date
        warning[s] = True
        
        # Case B: There is a warning in place, but no confirmation
        s = np.logical_and(np.logical_and(warning == True, deforestation == False),  mask == False)
        pchange[s] = bayesUpdate(pchange[s], PNF[s])
        
        
        # Step 3: Reject or accept warnings
        
        # Case A: Reject warning where pchange falls below 50 %
        s = np.logical_and(np.logical_and(np.logical_and(warning == True, pchange <= 0.5), deforestation == False), mask == False)
        warning[s] = False
        
        # Tidy up
        deforestation_date[s] = dt.date(1970,1,1)
        pchange[s] = 0.
        
        # Case B: Confirm warning where pchange > chi (hardwired to 99 %)
        s = np.logical_and(np.logical_and(np.logical_and(warning == True, pchange > deforestation_threshold), deforestation == False), mask == False)
        deforestation[s] = True
                
            
    change_day = (deforestation_date - deforestation_date.astype('datetime64[Y]')).astype(np.float32)
    
    confirmed_deforestation = deforestation_date.astype('datetime64[Y]').astype(np.float32) + 1970. + (change_day/365.)
    confirmed_deforestation[deforestation == False] = 0.
    confirmed_deforestation[confirmed_deforestation == 1970] = 0.

    warning_deforestation = pchange
    warning_deforestation[deforestation == True] = 0
    warning_deforestation[pchange<0.5] = 0
    warning_deforestation[warning_deforestation == 1970] = 0

    return confirmed_deforestation, warning_deforestation


def outputImage(array, image_like, filename):
    '''
    '''
    
    data_ds = gdal.Open(image_like)
    gdal_driver = gdal.GetDriverByName('GTiff')
    ds = gdal_driver.Create(filename, data_ds.RasterXSize, data_ds.RasterYSize, 1, 6, options = ['COMPRESS=LZW'])
    ds.SetGeoTransform(data_ds.GetGeoTransform())
    ds.SetProjection(data_ds.GetProjection())
    ds.GetRasterBand(1).WriteArray(array)
    ds = None
        


def main(infiles, output_dir = os.getcwd(), output_name = 'OUTPUT'):
    '''
    '''
    
    infiles = np.array(infiles)
    
    # Run through images
    deforestation_confirmed, deforestation_warning = calculateDeforestation(infiles)
    
    # Output images
    outputImage(deforestation_confirmed, infiles[0], '%s/%s_%s.tif'%(output_dir, output_name, 'confirmed'))
    
    outputImage(deforestation_warning, infiles[0], '%s/%s_%s.tif'%(output_dir, output_name, 'warning'))


if __name__ == '__main__':
    '''
    '''

    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process probability maps to generate a map of deforestation year and warning estimates of upcoming events.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('infiles', metavar = 'FILES', type = str, nargs = '+', help = 'Files output by classify.py, including wildcards where desired.')
    
    # Optional arguments
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory. If nothing specified, downloads will output to the present working directory, given a standard filename.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'deforestation', help="Optionally specify a string to precede output filename.")
        
    # Get arguments
    args = parser.parse_args()

    # Get absolute path of input .safe files.
    infiles = [os.path.abspath(i) for i in args.infiles]
    
    main(infiles, output_dir = args.output_dir, output_name = args.output_name)


