import argparse
import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal, osr
import scipy.stats

import pdb



def _testSourceFiles(source_files):
    '''
    Test that a list of raster images have the same CRS and are preprended by the same name.
    
    Args:
        source_files: A list or array of paths to raster files from deforest.classify.py()
    
    Returns:
        A boolean: True if all of source_files match, False where one or more of source_files don't match
    '''
    
    # Get metadata from the first file
    name = source_files[0].split('_')[0]
    ds = gdal.Open(source_files[0],0)
    geo_t = ds.GetGeoTransform()
    proj = ds.GetProjection()
    xSize, ySize = ds.RasterXSize, ds.RasterYSize
    
    # Test whether all subsequent files match metadata from the first file
    for source_file in source_files[1:]:
        if source_file.split('_')[0] != name:
            return False
        ds = gdal.Open(source_file, 0)
        if ds.GetGeoTransform() != geo_t:
            return False
        if ds.GetProjection() != proj:
            return False
        if ds.RasterXSize != xSize or ds.RasterYSize != ySize:
            return False
    
    # All tests passed, return True
    return True


def _getImageDate(infiles):
    '''
    Extract image date from outputs of deforest.classify()
    
    Args:
        infiles: List of files from deforest.classify()
    
    Returns:
        A list of image dates
    '''
    
    datestrings = ['_'.join(i.split('/')[-1].split('.')[0].split('_')[-2:]) for i in infiles]
    datetimes = np.array([dt.datetime.strptime(date,'%Y%m%d_%H%M%S') for date in datestrings],dtype='datetime64[s]')
    dates = datetimes.astype('datetime64[D]')
    
    return dates


def _getImageType(infiles):
    '''
    Extract image type from outputs of deforest.classify()
    
    Args:
        infiles: List of files from deforest.classify()
    
    Returns:
        A list of image types
    '''
    
    image_type = np.array(['_'.join(x.split('/')[-1].split('.')[0].split('_')[1:3]) for x in infiles])
    
    return image_type
    


def _combineObservations(PNF, mask):
    '''
    For cases where there are more than one observations for a given day, their forest probabilities should be combined.
    
    Args:
        PNF: proability of non forest, 3-dimensional for multiple images
        mask: The mask for PNF
    
    Returns:
        Combined PNF and mask
    '''
        
    # Set masked elements to 1, so they have no impact on the multiplication
    PNF[mask] = 1
    PNF_inv = 1 - PNF
    PNF_inv[mask] = 1
     
    # Calculate probability and inverse probability
    prod = np.prod(PNF, axis = 2)
    prod_inv = np.prod(PNF_inv, axis = 2)
    
    # Combine into a single probability
    PNF = prod / (prod + prod_inv)
    
    # Identify pixels without a single measurement
    mask = np.sum(mask == False, axis = 2) == 0
    
    # Keep things tidy
    PNF[mask] = 0.
    
    return PNF, mask

    

def _bayesUpdate(prior, likelihood):
    '''
    Update Baysian prior based on new observation.
    
    Args:
        prior: Prior probability of non-forest
        likelihood: Probability of forest from new observation
    
    Returns:
        posterior probability
    '''
    
    posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood))) 
    
    return posterior


def calculateDeforestation(infiles, deforestation_threshold = 0.99, block_weight = 0.1):
    '''
    Calculate deforestation using the 'bayts' method of Reiche et al. (2018), re-written into Pyton by Samuel Bowers.
    
    Function takes a list of paths to input rasters in the same CRS/resolution, where 0-100 = probability of a pixel being forested and 255 = nodata. Deforestation is identified based on a Bayesian method, where deforestation_threshold is the treshold probability for identifiying deforestation and block_weight limits the range of probabilities in input files.
    
    Credit for method: Reiche et al. (2018)
    
    Args:
        infiles: A list of GDAL-compatible input files from deforest.classify()
        deforestation_threshold: Threshold probability for flagging deforestation. Defaults to 0.99. Must be between > 0.5 an < 1.0.
        block_weight: Limits the probability range of input files. For example, default block_weight 0.1 allows of a probability range of 0.1 to 0.9 in input files. This reduces the probability of false positives.
    
    Returns
        confirmed deforestation (years), warning deforestation (%)
    
    '''
    
    assert deforestation_threshold < 1. and deforestation_threshold > 0.5, "Deforestation theshold must be greater than 0.5 and less than 1."
    assert block_weight < 0.5, "Block weight must be less than 0.5."
       
    # Get absolute path of input .safe files, an update to an array
    infiles = [os.path.abspath(i) for i in infiles]
    infiles = np.array(infiles)
    
    assert _testSourceFiles(infiles), "Source files must all have the same CRS, extent and resolution, and the same name."
    
    # Get datetime and image type for each image
    dates = _getImageDate(infiles)
    image_type = _getImageType(infiles)
        
    # Initialise output arrays
    ds = gdal.Open(infiles[0])
    deforestation = np.zeros((ds.RasterYSize, ds.RasterXSize), dtype=np.bool)
    warning = np.zeros_like(deforestation, dtype=np.bool)
    deforestation_date = np.zeros_like(deforestation, dtype='datetime64[D]')
    pchange = np.zeros_like(deforestation, dtype=np.float)

    # For each date in turn
    for date in sorted(np.unique(dates)):
        
        # Possible future functionality to not consider some months with poor data quality.
        #if date.astype('datetime64[M]').astype(int) % 12 + 1 in [10,11]: continue
        
        # Get unique image types (max one measurement per pixel per sensor per day permitted)
        image_types = np.unique(image_type[dates == date])
        
        # Build blank probability of forest image
        PF = np.zeros((ds.RasterYSize, ds.RasterXSize, image_types.shape[0]), dtype = np.uint8) + 255
        
        for n, this_image_type in enumerate(image_types):
            
            # Build a single composite p_forest image for each image type per date
            for infile in infiles[np.logical_and(dates == date, image_type == this_image_type)]:
                    
                print('Loading %s'%infile)
                data = gdal.Open(infile,0).ReadAsArray()
                
                # Select areas that still have the nodata value
                s = PF[:,:,n] == 255
                
                # Paste the new data into these locations
                PF[:,:,n][s] = data[s]
        
        # Change percent probability of forest (PF) to probability of non forest (PNF)
        PNF = (100 - PF) / 100.
        mask = PF == 255
        
        PNF = np.squeeze(PNF)
        mask = np.squeeze(mask)
        
        # Apply block weighting
        PNF[PNF < block_weight] = block_weight
        PNF[PNF > 1 - block_weight] = 1 - block_weight
        
        # If multiple observations from the same date exist, combine them (not performed where only one observation)
        if image_types.shape[0] > 1:
            PNF, mask = _combineObservations(PNF, mask)
        
        ##################################
        # Step 1: Flag potential changes #
        ##################################
        
        flag = PNF > 0.5
        
        ################################################
        # Step 2: Update pchange for current time step #
        ################################################
        
        # Case A: A new flag appears. Set pchange to PNF.
        sel = (warning == False) & flag & (deforestation == False) & (mask == False)
        pchange[sel] = PNF[sel]
        deforestation_date[sel] = date
        warning[sel] = True
        
        # Case B: There is a warning in place, but no confirmation. Update pchange using PNF.
        sel = warning & (deforestation == False) & (mask == False)
        pchange[sel] = _bayesUpdate(pchange[sel], PNF[sel])
        
        #####################################
        # Step 3: Reject or accept warnings #
        #####################################
        
        # Case A: Reject warning where pchange falls below 50%
        sel = warning & (pchange < 0.5) & (deforestation == False) & (mask == False)
        warning[sel] = False
        
        # Tidy up
        deforestation_date[sel] = dt.date(1970,1,1)
        pchange[sel] = 0.
        
        # Case B: Confirm warning where pchange > deforestation_threshold
        sel = warning & (deforestation == False) & (pchange >= deforestation_threshold) & (mask == False)
        deforestation[sel] = True
    
    # Get day of year of change
    change_day = (deforestation_date - deforestation_date.astype('datetime64[Y]')).astype(np.float32)
    
    # Output year of deforestation for confirmed events (units = year)
    confirmed_deforestation = deforestation_date.astype('datetime64[Y]').astype(np.float32) + 1970. + (change_day/365.)
    confirmed_deforestation[deforestation == False] = 0.
    confirmed_deforestation[confirmed_deforestation == 1970] = 0.
    
    # Return the final probability of early warning pixels (units = %)
    warning_deforestation = pchange * 100.
    warning_deforestation[deforestation == True] = 0.
    warning_deforestation[pchange < 0.5] = 0.
    
    return confirmed_deforestation, warning_deforestation


def outputImage(array, image_like, filename):
    '''
    Output a numpy array to a GeoTiff using an example projected image.
    
    Args:
        array: A numpy array
        image_like: Path to a GDAL-compatible raste file of the same CRS as array
        filename: Output filename
    '''
    
    data_ds = gdal.Open(image_like)
    gdal_driver = gdal.GetDriverByName('GTiff')
    ds = gdal_driver.Create(filename, data_ds.RasterXSize, data_ds.RasterYSize, 1, 6, options = ['COMPRESS=LZW'])
    ds.SetGeoTransform(data_ds.GetGeoTransform())
    ds.SetProjection(data_ds.GetProjection())
    ds.GetRasterBand(1).WriteArray(array)
    ds = None
