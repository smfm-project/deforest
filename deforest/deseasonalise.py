
import glob
import glymur
import numpy as np
import scipy.ndimage.filters



def loadS2Mask(L2A_file, res):
    """
    Load classification mask given .SAFE file granule and resolution.
    
    Args:
        L2A_file: A granule from a level 2A Sentinel-2 .SAFE file, processed with sen2cor.
        res: Integer of resolution to be processed (i.e. 20 m or 60 m).
    
    Returns:
        A numpy array of the classified mask image.
        The directory location of the .jp2 mask file.
    """
        
    # Remove trailing slash from input filename, if it exists
    L2A_file = L2A_file.rstrip('/')
    
    # Identify the cloud mask following the standardised file pattern
    image_path = glob.glob('%s/IMG_DATA/R%sm/*_SCL_*m.jp2'%(L2A_file,str(res)))
    
    # In case of old file format structure, the SCL file is stored elsewhere
    if len(image_path) == 0:
        image_path = glob.glob('%s/IMG_DATA/*_SCL_*%sm.jp2'%(L2A_file,str(res)))
    
    # Load the cloud mask (.jp2 format)
    jp2 = glymur.Jp2k(image_path[0])
    
    mask = jp2[:]
    
    mask = np.logical_or(mask < 4, mask > 6)
    
    return mask



def calcS2NDVI(L2A_file, res):
    """
    Calculate NDVI given .SAFE file and resolution.
    
    Args:
        L2A_file: A granule from a level 2A Sentinel-2 .SAFE file, processed with sen2cor.
        res: Integer of resolution to be processed (i.e. 10 m, 20 m, or 60 m). 
    
    Returns:
        A numpy array of the classified mask image.
        The directory location of the .jp2 mask file.
    """
        
    # Remove trailing slash from input filename, if it exists
    L2A_file = L2A_file.rstrip('/')
    
    # Identify the red/NIR bands following the standardised file pattern
    red_path = glob.glob('%s/IMG_DATA/R%sm/*_B04_*m.jp2'%(L2A_file,str(res)))
    nir_path = glob.glob('%s/IMG_DATA/R%sm/*_B8A_*m.jp2'%(L2A_file,str(res)))
        
    # Load the data (as numpy array)
    red = glymur.Jp2k(red_path[0])[:] / 10000.
    nir = glymur.Jp2k(nir_path[0])[:] / 10000.
    
    ndvi = (nir - red) / (nir + red)
    
    return ndvi



def deseasonalise(data, mask, res, area = 200000.):
    '''
    Deseasonalises an array by dividing pixels by the 95th percentile value in the vicinity of each pixel
    
    Args:
        data: A numpy array containing the data
        mask: A boolean mask
        res: Integer of resolution to be processed (i.e. 10 m, 20 m, or 60 m).         
        area: Area in m^2 to determine the kernel size. This should be greater than the size of a deforestation event. Defaults to 200,000 m^2 (20 ha).
    
    Returns:
        The deseasonalised numpy array
    
    '''
    
    filter_size = int(round((float(area) / (res ** 2)) ** 0.5,0))
        
    data_95pc = scipy.ndimage.filters.percentile_filter(np.ma.array(data, mask = mask), 95, size=(filter_size,filter_size))
    
    data_deseasonalised = data / data_95pc   
    
    return data_deseasonalised
    


if __name__ == '__main__':
     
     infile = '/home/sbowers3/DATA/testing/S2B_MSIL2A_20170630T072949_N0205_R049_T36KXE_20170630T074249.SAFE/GRANULE/L2A_T36KXE_A001648_20170630T074249/'
     tile = '36KXE'
     
     mask = loadS2Mask(infile, 20)
     
     ndvi = calcS2NDVI(infile, 20)
     
     ndvi_des = deseasonalise(ndvi, mask, 20)
     
     plt.imshow(np.ma.array(ndvi_des, mask=mask),interpolation='nearest',vmin=0,vmax=1)
     plt.colorbar()
     plt.show()