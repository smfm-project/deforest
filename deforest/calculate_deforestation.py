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
    
    datestrings = ['_'.join(x.split('/')[-1].split('_')[1:3]) for x in infiles]
    datetimes = np.array([dt.datetime.strptime(date,'%Y%m%d_%H%M%S') for date in datestrings],dtype='datetime64[s]')
    dates = datetimes.astype('datetime64[D]')
    
    return dates


def getImageType(infiles):
    '''
    '''
    image_type = np.array(['_'.join(x.split('/')[-1].split('.')[-2].split('_')[3:5]) for x in infiles])
    
    return image_type
    


def combineObservations(probability, mask):
    '''
    For cases where there are more than one observations for a given day, here their forest probabilities are combined.
    '''
    
    probability = np.ma.array(probability, mask = mask)
    
    probability_combined = np.prod(probability, axis = 2) / (np.prod(probability, axis = 2) + np.prod(1 - probability, axis = 2))
       
    return probability_combined.data, probability_combined.mask


def bayesUpdate(prior, likelihood):
    '''
    '''
    
    posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood))) 
    
    return posterior


def calculateDeforestation(infiles):
    '''
    '''
    
    # Get datetimes for each image
    dates = getImageDate(infiles)
    
    # Get the image_type for each image
    image_type = getImageType(infiles)
        
    # Initialise arrays
    ds = gdal.Open(infiles[0])

    YSize = ds.RasterYSize
    XSize = ds.RasterXSize

    deforestation = np.zeros((YSize, XSize), dtype=np.bool)
    previous_flag = np.zeros_like(deforestation, dtype=np.bool)
    false_alarms = np.zeros_like(deforestation, dtype=np.bool)
    PNF_last = np.zeros_like(deforestation, dtype = np.float) + 0.5 # Initialise to 0.5 probability of no forest
    deforestation_date = np.zeros_like(deforestation, dtype='datetime64[D]')
    pchange = np.zeros_like(deforestation, dtype=np.float)


    # Run for each unique date
    for date in sorted(np.unique(dates)):
        
        unique_datetimes = np.unique(dates[dates == date])
        unique_images = np.unique(image_type[dates == date])
        
        #TODO find repeat data in Sentinel-2 imagery. This is tricky, as datetimes can be different yet imagery the same in tile overlap.
        
        # One layer per unique image type. This should allow only one overpass per satellite per granule/track. Needs work to accomodate Sentinel-2 tiles
        
        #NB: Sentinel-2 currently contributing too much deforestation. Re-calibrate?
        
        p_forest = np.zeros((YSize, XSize, unique_images.shape[0]), dtype = np.uint8) + 255
            
        for n, this_image_type in enumerate(unique_images):
            
            # Only select one data file per image type
            for infile in infiles[np.logical_and(dates == date, image_type == this_image_type)]:
                    
                print 'Loading %s'%infile
                data = gdal.Open(infile,0).ReadAsArray()
                
                # Select areas that still have the nodata value
                s = p_forest[:,:,n] == 255
                
                # Paste the new data into these locations
                p_forest[:,:,n][s] = data[s]
                
            
        # Change percent probability of forest to probability of non forest
        mask = p_forest == 255
        PNF = (100 - p_forest) / 100.
        PNF[mask] = 0.
        
        # Remove length-1 axes
        PNF = np.squeeze(PNF)
        mask = np.squeeze(mask)
            
        ## Apply block weighting/
        PNF[PNF < 0.1] = 0.1
        PNF[PNF > 0.9] = 0.9
        
        # If multiple observations from the same date exist, combine them (not performed where only one observation)
        if unique_images.shape[0] > 1:
            
            PNF, mask = combineObservations(PNF, mask)
        
        
        # Flag potential changes
        flag = PNF > 0.5
            
        # Step 2.1: Update flag and pchange for current time step
        # Case A: A new change appears which is flagged. but not confirmed
        s = np.logical_and(np.logical_and(np.logical_and(flag == True, previous_flag == False), mask == False), deforestation == False)
        pchange[s] = bayesUpdate(PNF_last[s], PNF[s])
        deforestation_date[s] = date
            
        # Case B: There is a previously flagged change
        s = np.logical_and(previous_flag == True, mask == False)
        pchange[s] = bayesUpdate(pchange[s], PNF[s])
        
        
        # Step 2.2: Reject or accept previously flagged cases
        # Case A: Reject, where pchange falls below 50 %
        s = np.logical_and(np.logical_and(np.logical_and(pchange < 0.5, mask == False), previous_flag == True), deforestation == False)
        deforestation_date[s] = dt.date(1970,1,1)
        pchange[s] = 0.1
        previous_flag[s] = False
        
        # Case B: Confirm, where pchange > chi (hardwired to 99 %)
        s = np.logical_and(np.logical_and(pchange > 0.99, deforestation == False), mask == False)
        deforestation[s] = True
        
        # Update arrays for next round
        previous_flag = flag.copy()
        PNF_last = PNF.copy()    

    confirmed_deforestation = deforestation_date.astype('datetime64[Y]').astype(int) + 1970
    confirmed_deforestation[deforestation == False] = 0
    confirmed_deforestation[confirmed_deforestation == 1970] = 0

    warning_deforestation = deforestation_date.astype('datetime64[Y]').astype(int) + 1970
    warning_deforestation[deforestation == True] = 0
    warning_deforestation[pchange<0.5] = 0
    warning_deforestation[warning_deforestation == 1970] = 0

    return confirmed_deforestation, warning_deforestation


def outputImage(array, image_like, filename):
    '''
    '''
    
    data_ds = gdal.Open(image_like)
    gdal_driver = gdal.GetDriverByName('GTiff')
    ds = gdal_driver.Create(filename, data_ds.RasterXSize, data_ds.RasterYSize, 1, 3, options = ['COMPRESS=LZW'])
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




##############
## The code ##
##############

"""

# Load each image in turn, and calculate the probability of forest (from a start point of everything being forest)


data_files = glob.glob('/exports/csce/datastore/geos/groups/SMFM/chimanimani/L3_data/chimanimaniGlobal_*S1*.tif')
data_files.sort()
data_files = np.array(data_files)

datestrings = ['_'.join(x.split('/')[-1].split('_')[1:3]) for x in data_files]
datetimes = np.array([dt.datetime.strptime(date,'%Y%m%d_%H%M%S') for date in datestrings],dtype='datetime64[s]')
dates = datetimes.astype('datetime64[D]')

image_type = np.array(['_'.join(x.split('/')[-1].split('.')[-2].split('_')[3:5]) for x in data_files])


# Initialise arrays
ds = gdal.Open(data_files[0])

YSize = ds.RasterYSize
XSize = ds.RasterXSize

deforestation = np.zeros((YSize, XSize), dtype=np.bool)
previous_flag = np.zeros_like(deforestation, dtype=np.bool)
false_alarms = np.zeros_like(deforestation, dtype=np.bool)
PNF_last = np.zeros_like(deforestation, dtype = np.float) + 0.5 # Initialise to 0.5 probability of no forest
deforestation_date = np.zeros_like(deforestation, dtype='datetime64[D]')
pchange = np.zeros_like(deforestation, dtype=np.float)


# Run for each unique date
for date in sorted(np.unique(dates)):
       
    unique_datetimes = np.unique(datetimes[dates == date])
    unique_images = np.unique(image_type[dates == date])
    
    #TODO find repeat data in Sentinel-2 imagery. This is tricky, as datetimes can be different yet imagery the same in tile overlap.
    
    # One layer per unique image type. This should allow only one overpass per satellite per granule/track. Needs work to accomodate Sentinel-2 tiles
    
    #NB: Sentinel-2 currently contributing too much deforestation. Re-calibrate?
    
    p_forest = np.zeros((YSize, XSize, unique_images.shape[0]), dtype = np.uint8) + 255
        
    for n, this_image_type in enumerate(unique_images):
        
        # Only select one data file per image type
        for data_file in data_files[np.logical_and(dates == date, image_type == this_image_type)]:
                   
            print 'Loading %s'%data_file
            data = gdal.Open(data_file,0).ReadAsArray()
            
            # Select areas that still have the nodata value
            s = p_forest[:,:,n] == 255
            
            # Paste the new data into these locations
            p_forest[:,:,n][s] = data[s]
            
        
    # Change percent probability of forest to probability of non forest
    mask = p_forest == 255
    PNF = (100 - p_forest) / 100.
    PNF[mask] = 0.
    
    # Remove length-1 axes
    PNF = np.squeeze(PNF)
    mask = np.squeeze(mask)
        
    ## Apply block weighting/
    PNF[PNF < 0.1] = 0.1
    PNF[PNF > 0.9] = 0.9
    
    # If multiple observations from the same date exist, combine them (not performed where only one observation)
    if unique_images.shape[0] > 1:
        
        PNF, mask = combineObservations(PNF, mask)
    
    
    # Flag potential changes
    flag = PNF > 0.5
        
    # Step 2.1: Update flag and pchange for current time step
    # Case A: A new change appears which is flagged. but not confirmed
    s = np.logical_and(np.logical_and(np.logical_and(flag == True, previous_flag == False), mask == False), deforestation == False)
    pchange[s] = bayesUpdate(PNF_last[s], PNF[s])
    deforestation_date[s] = date
        
    # Case B: There is a previously flagged change
    s = np.logical_and(previous_flag == True, mask == False)
    pchange[s] = bayesUpdate(pchange[s], PNF[s])
    
    
    # Step 2.2: Reject or accept previously flagged cases
    # Case A: Reject, where pchange falls below 50 %
    s = np.logical_and(np.logical_and(np.logical_and(pchange < 0.5, mask == False), previous_flag == True), deforestation == False)
    deforestation_date[s] = dt.date(1970,1,1)
    pchange[s] = 0.1
    previous_flag[s] = False
    
    # Case B: Confirm, where pchange > chi (hardwired to 99 %)
    s = np.logical_and(np.logical_and(pchange > 0.99, deforestation == False), mask == False)
    deforestation[s] = True
    
    # Update arrays for next round
    previous_flag = flag.copy()
    PNF_last = PNF.copy()    


confirmed_deforestation = deforestation_date.astype('datetime64[Y]').astype(int) + 1970
confirmed_deforestation[deforestation == False] = 0
confirmed_deforestation[confirmed_deforestation == 1970] = 0

warning_deforestation = deforestation_date.astype('datetime64[Y]').astype(int) + 1970
warning_deforestation[deforestation == True] = 0
warning_deforestation[pchange<0.5] = 0
warning_deforestation[warning_deforestation == 1970] = 0


data_ds = gdal.Open(data_files[0])
gdal_driver = gdal.GetDriverByName('GTiff')
ds = gdal_driver.Create('deforestation_confirmed.tif', data_ds.RasterXSize, data_ds.RasterYSize, 1, 3, options = ['COMPRESS=LZW'])
proj = osr.SpatialReference()
proj.ImportFromEPSG(32736)
ds.SetGeoTransform(data_ds.GetGeoTransform())
ds.SetProjection(proj.ExportToWkt())
ds.GetRasterBand(1).WriteArray(confirmed_deforestation)
ds = None
    
data_ds = gdal.Open(data_files[0])
gdal_driver = gdal.GetDriverByName('GTiff')
ds = gdal_driver.Create('deforestation_warning.tif', data_ds.RasterXSize, data_ds.RasterYSize, 1, 3, options = ['COMPRESS=LZW'])
proj = osr.SpatialReference()
proj.ImportFromEPSG(32736)
ds.SetGeoTransform(data_ds.GetGeoTransform())
ds.SetProjection(proj.ExportToWkt())
ds.GetRasterBand(1).WriteArray(warning_deforestation)
ds = None
    
"""