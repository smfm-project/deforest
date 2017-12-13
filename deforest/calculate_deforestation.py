from osgeo import gdal
import numpy as np
import glob
import datetime as dt
import matplotlib.pyplot as plt
from osgeo import osr
import scipy.stats
import csv

import pdb

def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)

def getPixel(data_files, mask_files, E, N):
    '''
    '''
    
    pixel, line = world2Pixel(gdal.Open(data_files[0],0).GetGeoTransform(),E,N)
        
    data_out = []
    date = []
    
    for data_file, mask_file in zip(data_files,mask_files):
        
        print 'Doing %s'%data_file.split('/')[-1]
        mask = gdal.Open(mask_file,0).ReadAsArray()[line,pixel]
        
        if mask != 1:
            print 'Got one!'
            data_out.append(gdal.Open(data_file,0).ReadAsArray()[line,pixel])
            datestring = data_file.split('/')[-1].split('.tif')[0].split('_')[3]
            date.append(dt.date(int(datestring[:4]),int(datestring[4:6]),int(datestring[6:])))
    
    return data_out, date


def bayesUpdate(prior, likelihood):
    '''
    '''
    
    posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood))) 
    
    return posterior


def combineObservations(probability, mask):
    '''
    For cases where there are more than one observations for a given day, here their forest probabilities can be combined.
    '''
        
    # Calculate number of observations
    nobs = np.sum(mask, axis=2)
    
    probability_out = np.zeros_like(nobs, dtype = np.float)
    
    probability_out[nobs > 1] = np.array([np.prod(probability[nobs > 1, :], axis = 1) / (np.prod(probability[nobs > 1, :], axis = 1) + np.prod(1 - probability[nobs > 1, :], axis = 1))])[0,:]
    
    for observation in range(probability.shape[2]):
        
        probability_out[np.logical_and(mask[:,:,observation], nobs == 1)] =  probability[:,:,observation][np.logical_and(mask[:,:,observation], nobs == 1)]
    
        
    # Update mask where there's at least one valid measurement
    mask_out = nobs >= 1
    
    return probability_out, mask_out



##############
## The code ##
##############

# Load each image in turn, and calculate the probability of forest (from a start point of everything being forest)


data_files = glob.glob('/exports/eddie/scratch/sbowers3/chimanimani/L3_files/chimanimaniGlobal*_data.tif')
data_files.sort(key = lambda x: x.split('_')[4])
data_files = np.array(data_files)

mask_files = np.array([i[:-8] + 'mask.tif' for i in data_files])

datestrings = [x.split('/')[-1].split('_')[3] for x in data_files]
dates = np.array([dt.date(int(x[:4]), int(x[4:6]), int(x[6:])) for x in datestrings])

sensors = np.array([x.split('/')[-1].split('_')[1] for x in data_files])
pols = np.array([x.split('/')[-1].split('_')[2] for x in data_files])

# Get selection that removes VV S1 files where a VH file exists
sel = np.logical_and(np.in1d(dates,np.intersect1d(dates[pols == 'VH'],dates[pols == 'VV'])), pols == 'VV') == False

data_files = data_files[sel]
mask_files = mask_files[sel]
dates = dates[sel]
sensors = sensors[sel]
pols = pols[sel]

# Initialise arrays
deforestation = np.zeros_like(gdal.Open(data_files[0]).ReadAsArray(), dtype=np.bool)
previous_flag = np.zeros_like(deforestation, dtype=np.bool)
false_alarms = np.zeros_like(deforestation, dtype=np.bool)
PNF_last = np.zeros_like(deforestation, dtype = np.float) + 0.5 # Initialise to 0.5 probability of no forest
deforestation_date = np.zeros_like(deforestation, dtype='datetime64[D]')
pchange = np.zeros_like(deforestation, dtype=np.float)

    
# Get unique dates
for date in sorted(np.unique(dates)):
    
    sensor = sensors[dates == date]
    pol = pols[dates == date]    

    F_mean = np.zeros_like(sensors[dates == date], dtype = np.float64)
    F_sd = np.zeros_like(F_mean)
    NF_mean = np.zeros_like(F_mean)
    NF_sd = np.zeros_like(F_mean)
    
    # Sentinel-1 (VV)
    F_mean[np.logical_and(sensor=='S1',pol=='VV')] = -0.91 #-1.19#-0.81#-7.
    F_sd[np.logical_and(sensor=='S1',pol=='VV')] = 1.48 #1.8#0.94#0.75
    NF_mean[np.logical_and(sensor=='S1',pol=='VV')] = -2.38 #-4.7#-4.38#-11.5
    NF_sd[np.logical_and(sensor=='S1',pol=='VV')] = 1.84 #1.6#1.17#1

    # Sentinel-1 (VH)
    F_mean[np.logical_and(sensor=='S1',pol=='VH')] = -0.66
    F_sd[np.logical_and(sensor=='S1',pol=='VH')] = 1.38
    NF_mean[np.logical_and(sensor=='S1',pol=='VH')] = -2.51
    NF_sd[np.logical_and(sensor=='S1',pol=='VH')] = 2.09
    
    # Sentinel-2
    F_mean[sensor=='S2'] = -0.097#-0.11#-0.1#0.85
    F_sd[sensor=='S2'] = 0.17#0.1337#0.08#0.075
    NF_mean[sensor=='S2'] = -0.3#-0.28#-0.34 #0.4
    NF_sd[sensor=='S2'] = 0.17#0.168#0.14#0.125      
    
    # Load files (axis 2 when > 1 observation at a given date)
    ds = gdal.Open(data_files[dates == date][0])
    data = np.zeros((ds.RasterYSize, ds.RasterXSize, (dates == date).sum()))
    mask = np.zeros_like(data, dtype = np.bool)
    
    for n, (data_file, mask_file) in enumerate(zip(data_files[dates == date], mask_files[dates == date])):
        print 'Loading %s'%data_file
        data[:,:,n] = gdal.Open(data_file,0).ReadAsArray()
        mask[:,:,n] = gdal.Open(mask_file,0).ReadAsArray()
    
    # Remove length-1 axes
    data = np.squeeze(data)
    mask = np.squeeze(mask)
    
    # Get probability of an observation being forest/nonforest
    PF = scipy.stats.norm.pdf(data, F_mean, F_sd)
    PNF = scipy.stats.norm.pdf(data, NF_mean, NF_sd)
    
    # Determine conditinal probability of an observation being from NF (From Reiche et al. 2018)
    PNF[PNF < 1E-10000] = 0
    PNF[PNF > 0] = (PNF[PNF > 0] / (PF[PNF > 0] + PNF[PNF > 0]))
    
    ## Apply block weighting function fudge.
    PNF[PNF < 0.1] = 0.1
    PNF[PNF > 0.9] = 0.9
    
    # If multiple observations from the same date exist, combine them (not performed where only one observation)
    if (dates == date).sum() > 1:
        
        PNF, mask = combineObservations(PNF, mask)
        
 
    # Flag potential changes
    flag = PNF > 0.5
        
    # Step 2.1: Update flag and pchange for current time step
    # Case A: A change appears which is flagged. but not confirmed
    s = np.logical_and(np.logical_and(flag == True, previous_flag == False), mask == False)
    pchange[s] = bayesUpdate(PNF_last[s], PNF[s])
    deforestation_date[np.logical_and(s, deforestation == False)] = date
        
    # Case B: There is a previously flagged change
    s = np.logical_and(previous_flag == True, mask == False)
    pchange[s] = bayesUpdate(pchange[s], PNF[s])
          
    # Step 2.2: Reject or accept previously flagged cases    
    s = np.logical_and(pchange < 0.5, mask == False)
    deforestation_date[s] = dt.date(1970,1,1)
    #pchange[s] = 0.1
    
    # Confirm change where pchange > chi (hardwired to 0.99)
    s = np.logical_and(np.logical_and(pchange > 0.99, deforestation == False), mask == False)
    deforestation[s] = True
    
    # Update arrays for next round
    previous_flag = flag.copy()
    PNF_last = PNF.copy()
    
    # Where > 1 observation, may need to apply block weighting function again
    #PNF_last[PNF_last > 0.9] = 0.9
    #PNF_last[PNF_last < 0.1] = 0.1



confirmed_deforestation = deforestation_date.astype('datetime64[Y]').astype(int) + 1970
confirmed_deforestation[deforestation == False] = 0
confirmed_deforestation[confirmed_deforestation == 1970] = 0

warning_deforestation = deforestation_date.astype('datetime64[Y]').astype(int) + 1970
warning_deforestation[deforestation == True] = 1970
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
    
