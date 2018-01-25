
import argparse
import csv
import datetime as dt
import glob
import glymur
import lxml.etree as ET
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from osgeo import gdal
import scipy.ndimage

import pdb


def loadS2(L2A_file, res):
    """
    Calculate a range of vegetation indices given .SAFE file and resolution.
    
    Args:
        L2A_file: A granule from a level 2A Sentinel-2 .SAFE file, processed with sen2cor.
        res: Integer of resolution to be processed (i.e. 20 or 60 m). 
    
    Returns:
        A maked numpy array of NDVI.
    """
    
    # Remove trailing slash from input filename, if it exists
    L2A_file = L2A_file.rstrip('/')
    
    # Identify the cloud mask following the standardised file pattern
    mask_path = glob.glob('%s/IMG_DATA/R%sm/*_SCL_*m.jp2'%(L2A_file,str(res)))
    
    # In case of old file format structure, the SCL file is stored elsewhere
    if len(mask_path) == 0:
        mask_path = glob.glob('%s/IMG_DATA/*_SCL_*%sm.jp2'%(L2A_file,str(res)))
            
    # Load the cloud mask as a numpy array
    jp2 = glymur.Jp2k(mask_path[0])
    mask = jp2[:]
    
    # Convert mask to a boolean array, allowing only values of 4 (vegetation), 5 (bare sois), and 6 (water)
    mask = np.logical_or(mask < 4, mask > 6)  
       
    
    # Identify the red/NIR bands following the standardised file pattern
    blue_path = glob.glob('%s/IMG_DATA/R%sm/*_B02_*m.jp2'%(L2A_file,str(res)))
    green_path = glob.glob('%s/IMG_DATA/R%sm/*_B03_*m.jp2'%(L2A_file,str(res)))
    red_path = glob.glob('%s/IMG_DATA/R%sm/*_B04_*m.jp2'%(L2A_file,str(res)))
    nir_path = glob.glob('%s/IMG_DATA/R%sm/*_B8A_*m.jp2'%(L2A_file,str(res)))
    swir1_path = glob.glob('%s/IMG_DATA/R%sm/*_B11_*m.jp2'%(L2A_file,str(res)))
    swir2_path = glob.glob('%s/IMG_DATA/R%sm/*_B12_*m.jp2'%(L2A_file,str(res)))

    # Load the data (as numpy array)
    blue = np.ma.array(glymur.Jp2k(blue_path[0])[:] / 10000., mask = mask)
    green = np.ma.array(glymur.Jp2k(green_path[0])[:] / 10000., mask = mask)
    red = np.ma.array(glymur.Jp2k(red_path[0])[:] / 10000., mask = mask)
    nir = np.ma.array(glymur.Jp2k(nir_path[0])[:] / 10000., mask = mask)
    swir1 = np.ma.array(glymur.Jp2k(swir1_path[0])[:] / 10000., mask = mask)
    swir2 = np.ma.array(glymur.Jp2k(swir2_path[0])[:] / 10000., mask = mask)
   
    # Calculate vegetation indices from Shultz 2016
    indices = np.ma.zeros((mask.shape[0], mask.shape[1], 9), dtype = np.float32)
    
    # NDVI
    indices[:,:,0] = (nir - red) / (nir + red)
    
    # EVI
    indices[:,:,1] = ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)) * 2.5
    
    # GEMI
    n = (2 * (nir ** 2 - red ** 2) + 1.5 * nir + 0.5 * red) / nir + red + 0.5
    indices[:,:,2] = (n * (1 - 0.25 * n)) - ((red - 0.125) / (1 - red))
    
    # NDMI
    indices[:,:,3] = (nir - swir1) / (nir + swir1)
    
    # SAVI
    indices[:,:,4] = (1 + 0.5) * ((nir - red) / (nir + red + 0.5))
    
    # TC wetness
    indices[:,:,5] = 0.0315 * blue + 0.2021 * green + 0.3102 * red + 0.1594 * nir - 0.6806 * swir1 - 0.6109 * swir2

    # TC greenness
    indices[:,:,6] = -0.1603 * blue - 0.2819 * green - 0.4934 * red + 0.7940 * nir - 0.0002 * swir1 - 0.1446 * swir2
    
    # Load date layers
    extent, EPSG, datetime, tile = getS2Metadata(L2A_file, resolution = res)
    
    # Calculate day of year 
    doy = (datetime.date() - dt.date(datetime.year,1,1)).days
    
    # Two seasonal predictors (trigonometric)
    indices[:,:,7] = np.sin(2 * np.pi * (doy / 365.))
    indices[:,:,8] = np.cos(2 * np.pi * (doy / 365.))
    
    # Set masked data to 0
    indices.data[indices.mask] = 0.
    
    return indices



def loadS1Single(dim_file, polarisation = 'VV', return_date = True):
    """
    Extract backscatter data given .dim file and polarisation
    
    Args:
        dim_file:
        polarisation: (i.e. 'VV' or 'VH')
    
    Returns:
        A maked numpy array of backscatter.
    """
    
    # Remove trailing slash from input filename, if it exists, and determine path of data file
    dim_file = dim_file.rstrip('/')
    data_file = dim_file[:-4] + '.data'
    
    # Load data
    data = gdal.Open('%s/Gamma0_%s.img'%(data_file,polarisation)).ReadAsArray()
        
    # Get mask (where backscatter is 0)
    mask = data == 0
    
    # Convert from natural units to dB
    data[mask] = 1E-10 # -100 dB, prevents error messages in np.log10
    data = 10 * np.log10(data)
    
    # Add day of year
    if return_date == True:
        data = np.dstack((data, np.zeros_like(data), np.zeros_like(data)))
    
        extent, EPSG, res, datetime, overpass = getS1Metadata(dim_file)
    
        # Calculate day of year 
        doy = (datetime.date() - dt.date(datetime.year,1,1)).days
        
        data[:,:,1] = np.sin(2 * np.pi * (doy / 365.))
        data[:,:,2] = np.cos(2 * np.pi * (doy / 365.))
        
        mask = np.dstack((mask, mask, mask))
                     
    data[mask] = 0
    
    return np.ma.array(data, mask = mask)


def loadS1Dual(dim_file):
    """
    Extract backscatter metrics from a dual-polarised Sentinel-1 image.
    
    Args:
        dim_file: 
    
    Returns:
        A maked numpy array of VV, VH and VV/VH backscatter.
    """
    
    assert getImageType(dim_file) == 'S1dual', "input file %s does not appear to be a dual polarised Sentinel-1 file"%dim_file
        
    VV = loadS1Single(dim_file, polarisation = 'VV', return_date = False)
    
    VH = loadS1Single(dim_file, polarisation = 'VH', return_date = False)
    
    VV_VH = VV / VH
    
    extent, EPSG, res, datetime, overpass = getS1Metadata(dim_file)
    
    # Calculate day of year 
    doy = (datetime.date() - dt.date(datetime.year,1,1)).days
    
    doy_X = np.zeros_like(VV) + np.sin(2 * np.pi * (doy / 365.))
    doy_Y = np.zeros_like(VV) + np.cos(2 * np.pi * (doy / 365.))
    
    return np.ma.dstack((VV, VH, VV_VH, doy_X, doy_Y))

    

def getS1Metadata(dim_file):
    '''
    Function to extract georefence info from level 2A Sentinel 2 data in .SAFE format.
    
    Args:
        dim_file: 
        polarisation: Defaults to 'VV'.

    Returns:
        A list describing the extent of the .dim file, in the format [xmin, ymin, xmax, ymax].
        EPSG code of the coordinate reference system of the image
        The image resolution
    '''
    
    from osgeo import gdal, osr
    
    assert os.path.exists(dim_file), "The location %s does not contain a Sentinel-1 .dim file."%dim_file
    
    tree = ET.ElementTree(file = dim_file)
    root = tree.getroot()
    
    # Get array size
    size = root.find("Raster_Dimensions")  
    nrows = int(size.find('NROWS').text)
    ncols = int(size.find('NCOLS').text)
    
    geopos = root.find("Geoposition/IMAGE_TO_MODEL_TRANSFORM").text.split(',')
    ulx = float(geopos[4])
    uly = float(geopos[5])
    xres = float(geopos[0])
    yres = float(geopos[3])
    lrx = ulx + (xres * ncols)
    lry = uly + (yres * nrows)
    extent = [ulx, lry, lrx, uly]
    
    res = abs(xres)
        
    wkt = root.find("Coordinate_Reference_System/WKT").text
    
    srs = osr.SpatialReference(wkt = wkt)
    srs.AutoIdentifyEPSG()
    EPSG = int(srs.GetAttrValue("AUTHORITY", 1))
    
    # Extract date string from filename
    datestring = root.find("Production/PRODUCT_SCENE_RASTER_START_TIME").text.split('.')[0]
    datetime = dt.datetime.strptime(datestring, '%d-%b-%Y %H:%M:%S')
    
    # Get ascending/descending overpass
    overpass = root.find("Dataset_Sources/MDElem/MDElem/MDATTR[@name='PASS']").text
    
    return extent, EPSG, res, datetime, overpass



def getS2Metadata(granule_file, resolution = 20):
    '''
    Function to extract georefence info from level 2A Sentinel 2 data in .SAFE format.
    
    Args:
        granule_file: String with /path/to/the/granule folder bundled in a .SAFE file.
        resolution: Integer describing pixel size in m (10, 20, or 60). Defaults to 20 m.

    Returns:
        A list describing the extent of the .SAFE file granule, in the format [xmin, ymin, xmax, ymax].
        EPSG code of the coordinate reference system of the granule
    '''

    import lxml.etree as ET
    
    # Remove trailing / from granule directory if present 
    granule_file = granule_file.rstrip('/')
    
    assert len(glob.glob((granule_file + '/*MTD*.xml'))) > 0, "The location %s does not contain a metadata (*MTD*.xml) file."%granule_file
    
    # Find the xml file that contains file metadata
    xml_file = glob.glob(granule_file + '/*MTD*.xml')[0]
    
    # Parse xml file
    tree = ET.ElementTree(file = xml_file)
    root = tree.getroot()
            
    # Define xml namespace (specific to level 2A Sentinel 2 .SAFE files)
    ns = {'n1':root.tag[1:].split('}')[0]}
    
    # Get array size
    size = root.find("n1:Geometric_Info/Tile_Geocoding[@metadataLevel='Brief']/Size[@resolution='%s']"%str(resolution),ns)
    nrows = int(size.find('NROWS').text)
    ncols = int(size.find('NCOLS').text)
    
    # Get extent data
    geopos = root.find("n1:Geometric_Info/Tile_Geocoding[@metadataLevel='Brief']/Geoposition[@resolution='%s']"%str(resolution),ns)
    ulx = float(geopos.find('ULX').text)
    uly = float(geopos.find('ULY').text)
    xres = float(geopos.find('XDIM').text)
    yres = float(geopos.find('YDIM').text)
    lrx = ulx + (xres * ncols)
    lry = uly + (yres * nrows)
    
    extent = [ulx, lry, lrx, uly]
    
    # Find EPSG code to define projection
    EPSG = root.find("n1:Geometric_Info/Tile_Geocoding[@metadataLevel='Brief']/HORIZONTAL_CS_CODE",ns).text
    EPSG = int(EPSG.split(':')[1])
    
    # Get datetime
    datestring = root.find("n1:General_Info/SENSING_TIME[@metadataLevel='Standard']",ns).text.split('.')[0]
    datetime = dt.datetime.strptime(datestring,'%Y-%m-%dT%H:%M:%S')
    
    # Get tile from granule filename
    if granule_file.split('/')[-1].split('_')[1] == 'USER':
        
        # If old file format
        tile = granule_file.split('/')[-1].split('_')[-2]
        
    else:
        
        # If new file format
        tile = granule_file.split('/')[-1].split('_')[1]
    
    return extent, EPSG, datetime, tile



def buildMetadataDictionary(extent_dest, res, EPSG):
    '''
    Build a metadata dictionary to describe the destination georeference info
    
    Args:
        extent_dest: List desciribing corner coordinate points in destination CRS [xmin, ymin, xmax, ymax]
        res: Integer describing pixel size in m
        EPSG: EPSG code of destination coordinate reference system. Must be a UTM projection. See: https://www.epsg-registry.org/ for codes.
    
    Returns:
        A dictionary containg projection info.
    '''
    
    from osgeo import osr
    
    # Set up an empty dictionary
    md = {}
    
    # Define projection from EPSG code
    md['EPSG_code'] = EPSG

    # Get GDAL projection string
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(EPSG)
    md['proj'] = proj
    
    # Get image extent data
    md['ulx'] = float(extent_dest[0])
    md['lry'] = float(extent_dest[1])
    md['lrx'] = float(extent_dest[2])
    md['uly'] = float(extent_dest[3])
    md['xres'] = float(res)
    md['yres'] = float(-res)

    # Save current resolution for future reference
    md['res'] = res
    
    # Calculate array size
    md['nrows'] = int(round((md['lry'] - md['uly']) / md['yres']))
    md['ncols'] = int(round((md['lrx'] - md['ulx']) / md['xres']))
    
    # Define gdal geotransform (Affine)
    md['geo_t'] = (md['ulx'], md['xres'], 0, md['uly'], 0, md['yres'])
    
    return md



def _testOutsideTile(md_source, md_dest):
    '''
    Function that uses metadata dictionaries from buildMetadatadisctionary() metadata to test whether any part of a source data falls inside destination tile.
    
    Args:
        md_source: A metadata dictionary created by buildMetadataDictionary() representing the source image.
        md_dest: A metadata dictionary created by buildMetadataDictionary() representing the destination image.
        
    Returns:
        A boolean (True/False) value.
    '''
    
    from osgeo import osr
            
    # Set up function to translate coordinates from source to destination
    tx = osr.CoordinateTransformation(md_source['proj'], md_dest['proj'])
         
    # And translate the source coordinates
    md_source['ulx'], md_source['uly'], z = tx.TransformPoint(md_source['ulx'], md_source['uly'])
    md_source['lrx'], md_source['lry'], z = tx.TransformPoint(md_source['lrx'], md_source['lry'])   
    
    out_of_tile =  md_source['ulx'] >= md_dest['lrx'] or \
                   md_source['lrx'] <= md_dest['ulx'] or \
                   md_source['uly'] <= md_dest['lry'] or \
                   md_source['lry'] >= md_dest['uly']
    
    return out_of_tile



def getImageType(infile):
    '''
    Determines the type of image from a filepath.
    
    Args:
        infile: '/path/to/input_file'
    
    Returns
        A string indicating the file type. This can be 'S2' (Sentinel-2), 'S1single' (Sentinel-1, VV polarised), or 'S1dual' (Sentinel-1, VV/VH polarised).
    '''
    
    if infile.split('/')[-3].split('.')[-1] == 'SAFE':
        
        image_type = 'S2'
    
    elif infile.split('/')[-1].split('.')[-1] == 'dim':
        
        # Use metadata to determine number of bands
        tree = ET.ElementTree(file = infile)
        root = tree.getroot()
    
        # Get array size
        bands = int(root.find("Raster_Dimensions/NBANDS").text)
        
        if bands > 1:
            image_type = 'S1dual'
        else:
            image_type = 'S1single'
     
    else: 
        print 'WARNING: File %s does not match any expected file pattern'%infile
        image_type = None
    
    return image_type
    
    


def getFilesInTile(source_files, md_dest):
    '''
    Takes a list of source files as input, and determines where each falls within extent of output tile.
    
    Args:
        source_files: A list of S1/S2 input files.
        md_dest: Dictionary from buildMetaDataDictionary() containing output projection details.

    Returns:
        A reduced list of source_files containing only files that will contribute to each tile.
    '''
      
    do_file = []
     
    for infile in source_files:
        
        if getImageType(infile) == 'S2':
            
            # Extract this image's resolution from md_dest
            if md_dest['res'] < 20:
                res_source = 10
            elif md_dest['res'] >= 20 and md_dest['res'] < 60:
                res_source = 20
            elif md_dest['res'] >= 60:
                res_source = 60
            
            # Get source file metadata if from Sentinel-2
            extent_source, EPSG_source, datetime, tile = getS2Metadata(infile, resolution = res_source)
            
        
        elif getImageType(infile) == 'S1single' or getImageType(infile) == 'S1dual':
                        
            # Get source file metadata if from Sentinel-1
            extent_source, EPSG_source, res_source, datetime, overpass = getS1Metadata(infile)
             
        # Define source file metadata dictionary
        md_source = buildMetadataDictionary(extent_source, res_source, EPSG_source)

        # Skip processing the file if image falls outside of tile area
        if _testOutsideTile(md_source, md_dest):
            do_file.append(False)
            continue
        
        #print '    Found one: %s'%input_file
        do_file.append(True)
    
    # Get subset of source_files in specified limits
    source_files_tile = list(np.array(source_files)[np.array(do_file)])
        
    return source_files_tile





def _createGdalDataset(md, data_out = None, filename = '', driver = 'MEM', dtype = 3, RasterCount = 1, nodata = None, options = []):
    '''
    Function to create an empty gdal dataset with georefence info from metadata dictionary.

    Args:
        md: A metadata dictionary created by buildMetadataDictionary().
        data_out: Optionally specify an array of data to include in the gdal dataset.
        filename: Optionally specify an output filename, if image will be written to disk.
        driver: GDAL driver type (e.g. 'MEM', 'GTiff'). By default this function creates an array in memory, but set driver = 'GTiff' to make a GeoTiff. If writing a file to disk, the argument filename must be specified.
        dtype: Output data type. Default data type is a 16-bit unsigned integer (gdal.GDT_Int16, 3), but this can be specified using GDAL standards.
        options: A list containing other GDAL options (e.g. for compression, use [compress='LZW'].

    Returns:
        A GDAL dataset.
    '''
    from osgeo import gdal, osr
       
    gdal_driver = gdal.GetDriverByName(driver)
    ds = gdal_driver.Create(filename, md['ncols'], md['nrows'], RasterCount, dtype, options = options)
    
    ds.SetGeoTransform(md['geo_t'])
    
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(md['EPSG_code'])
    ds.SetProjection(proj.ExportToWkt())
        
    # If a data array specified, add data to the gdal dataset
    if type(data_out).__module__ == np.__name__:
        
        if len(data_out.shape) == 2:
            data_out = np.ma.expand_dims(data_out,2)
        
        for feature in range(RasterCount):
            ds.GetRasterBand(feature + 1).WriteArray(data_out[:,:,feature])
            
            if nodata != None:
                ds.GetRasterBand(feature + 1).SetNoDataValue(nodata)
    
    # If a filename is specified, write the array to disk.
    if filename != '':
        ds = None
    
    return ds


def _reprojectImage(ds_source, ds_dest, md_source, md_dest, nodatavalue = 0):
    '''
    Reprojects a source image to match the coordinates of a destination GDAL dataset.
    
    Args:
        ds_source: A gdal dataset from _createGdalDataset() containing data to be repojected.
        ds_dest: A gdal dataset from _createGdalDataset(), with destination coordinate reference system and extent.
        md_source: A metadata dictionary created by buildMetadataDictionary() representing the source image.
        md_dest: A metadata dictionary created by buildMetadataDictionary() representing the destination image.
        nodatavalue: New array is initalised to nodatavalue (default = 0). Optionally change this to another value. For example, set nodatavalue = 1 for a mask.
        
    Returns:
        A GDAL array with resampled data
    '''
    
    from osgeo import gdal
    
    proj_source = md_source['proj'].ExportToWkt()
    proj_dest = md_dest['proj'].ExportToWkt()
    
    # Set nodata value
    for feature in range(ds_dest.RasterCount):
        ds_dest.GetRasterBand(feature + 1).Fill(nodatavalue)
        ds_dest.GetRasterBand(feature + 1).SetNoDataValue(nodatavalue)
    
    # Reproject source into dest project coordinates
    gdal.ReprojectImage(ds_source, ds_dest, proj_source, proj_dest, gdal.GRA_NearestNeighbour)
    
    data_resampled = np.zeros((ds_dest.RasterYSize,ds_dest.RasterXSize, ds_dest.RasterCount))
    
    for feature in range(ds_dest.RasterCount):
        data_resampled[:,:,feature] = ds_dest.GetRasterBand(feature + 1).ReadAsArray()
    
    return data_resampled


def subset(data, md_source, md_dest, dtype = 3):
    '''
    '''
       
    if len(data.shape) == 2:
        RasterCount = 1
    else:
        RasterCount = data.shape[2]
        
    # Write array to a gdal dataset
    ds_source = _createGdalDataset(md_source, data_out = data.data, dtype = dtype, RasterCount = RasterCount)
    ds_dest = _createGdalDataset(md_dest, dtype = dtype, RasterCount = RasterCount)
    data_resampled = _reprojectImage(ds_source, ds_dest, md_source, md_dest)
    
    ds_source = _createGdalDataset(md_source, data_out = data.mask, dtype = 1, RasterCount = RasterCount)
    ds_dest = _createGdalDataset(md_dest, dtype = 1, RasterCount = RasterCount)
    mask_resampled = _reprojectImage(ds_source, ds_dest, md_source, md_dest, nodatavalue = 1) # No data values should be added to the mask
    
    return np.ma.array(data_resampled, mask = mask_resampled)

                

def deseasonalise(data, md, normalisation_type = 'none', normalisation_percentile = 95, area = 200000.):
    '''
    Deseasonalises an array by dividing pixels by the 95th percentile value in the vicinity of each pixel
    
    Args:
        data: A masked numpy array containing the data
        md: metadata dictionary
        normalisation_type: Select one of 'none' (no normalisation), 'global' (subtract percentile of entire scene), or 'local' (subtract percentile from the area surrounding each pixel).
        percentile: Data percentile to subtract, if normalisation_type == 'local' or 'global'. Defaults to 95%.
        area: Area in m^2 to determine the kernel size if normalisation_type == 'local'. This should be greater than the size of expected deforestation events. Defaults to 200,000 m^2 (20 ha).
    
    Returns:
        The deseasonalised numpy array
    
    '''
    
    assert normalisation_type in ['none','local','global'], "normalisation_type must be one of 'none', 'local' or 'global'. It was set to %s."%str(normalisation_type)  
    
        
    # Takes care of case where only a 2d array is input, allowing us to loop through the third axis
    if data.ndim == 2:
        data = np.ma.expand_dims(data,2)
        
    # No normalisation
    if normalisation_type == 'none':
        data_percentile = np.zeros_like(data)
    
    # Following Hamunyela et al. 2016
    if normalisation_type == 'local':
                
        data_percentile = np.zeros_like(data)
        
        for feature in range(data.shape[2]):
        
            # Fill in data gaps with nearest valid pixel (percentile_filter doesn't understand masked arrays)
            ind = scipy.ndimage.distance_transform_edt(data.mask[:,:,feature], return_distances = False, return_indices = True)
            data_percentile[:,:,feature] = data.data[:,:,feature][tuple(ind)]
        
            # Calculate filter size
            res = md['res']
            filter_size = int(round((float(area) / (res ** 2)) ** 0.5,0))
            
            # Filter by percentile
            data_percentile[:,:,feature] = scipy.ndimage.filters.percentile_filter(data_percentile[:,:,feature], normalisation_percentile, size = (filter_size, filter_size))
            
            # Replace the mask
            data_percentile[:,:,feature] = np.ma.array(data_percentile[:,:,feature], mask = data.mask[:,:,feature])
        
    # Following Reiche et al. 2018
    if normalisation_type == 'global':
        
        data_percentile = np.zeros_like(data)
        
        for feature in range(data.shape[2]):
            
            if (data.mask==False).sum() != 0:         
                # np.percentile doesn't understand masked arrays, so calculate percentile one feature at a time
                data_percentile[:,:,feature] = np.percentile(data.data[:,:,feature][data.mask[:,:,feature]==False], normalisation_percentile)
            else:
                # This catches the case where there's no usable data in an image
                data_percentile[:,:,feature] = 0
    
    # Get rid of residual dimensions where 2d array was input
    data = np.squeeze(data)
    data_percentile = np.squeeze(data_percentile)
        
    # And subtract the seasonal effect from the array
    data_deseasonalised = data - data_percentile 
    
    return data_deseasonalised


def loadCoefficients(image_type):
    '''
    Loads logistic regression coefficients from config .csv file.
    
    Args:
        image_type: A string wih the image type (i.e. S1single, S1dual, S2)
    Returns:
        An array containing model coefficients, the first element being the intercept, and remaining elements coefficients in layer order.
        Arrays containing the mean and scaling parameters to transform the imput image.
    '''
    
    # Get location of current file
    directory = os.path.dirname(os.path.abspath(__file__))
    
    # Determine name of output file
    filename = '%s/cfg/%s_coef.csv'%('/'.join(directory.split('/')[:-1]),image_type)
    
    # Read csv file
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        header = reader.next()
        
        coef = []
        mean = []
        scale = []

        for row in reader:
            coef.append(float(row[1]))
            mean.append(float(row[2]))
            scale.append(float(row[3]))
       
    return np.array(coef), np.array(mean), np.array(scale)


def rescaleData(data, means, scales):
    '''
    Rescale data to match the scaling of training data
    
    Args:
        data: A numpy array containing deseasonalised data
        means:  A numpy array containing the mean parameters to transform the imput image.
        scales:  A numpy array containing the scaling parameters to transform the imput image.
    Returns:
        The rescaled data array
    '''
    
    return (data - means[1:]) / scales[1:]


def classify(data, image_type, nodata = 255):
    """
    Calculate the probability of forest
    
    Args:
        data: A numpy array containing deseasonalised data
        image_type: The source of the image, one of 'S2', 'S1single', or 'S1dual'.
        nodata: Optionally specify a nodata value. Defaults to 255.
        
    Returns:
        A numpy array containing probability of a pixel being forest in that view. Units are percent, with a nodata value set to nodata.
    """
    
    assert image_type in ['S2', 'S1single', 'S1dual'], "image_type must be one of 'S2', 'S1single', or 'S1dual'. The classify() function was given %s."%image_type
    
    coefs, means, scales = loadCoefficients(getImageType(infile))
    
    if getImageType(infile) == 'S1single':
        # TODO: Update me
        p_forest = (0.644 * data) + 1.182
        
    elif getImageType(infile) == 'S1dual':
        # TODO: Update me
        p_forest = (0.316 * data[:,:,0]) + (0.462 * data[:,:,1]) + 1.205
    
    elif getImageType(infile) == 'S2':
        
        p_forest = np.sum(coefs[1:] * rescaleData(data, means, scales), axis = 2) + coefs[0]
    
    # Convert from odds to probability
    p_forest = np.exp(p_forest) / (1 + np.exp(p_forest))
    
    # Reduce the file size by converting data to integer
    p_forest_out = np.round(p_forest.data * 100 , 0).astype(np.uint8)
    p_forest_out[p_forest.mask] = nodata
    
    return p_forest_out
    
    
def loadData(infile, S2_res = 20, output_dir = os.getcwd(), output_name = 'CLASSIFIED'):
    '''
    Loads data and metadata from a Sentinel-1 or Sentinel-2 file
    
    Args:
        infile: Path to a Sentinel-1 .dim file or a Sentinel-2 GRANULE directory.
        output_res: Output resolution, important for selecting correct Sentinel-2 file.
    Returns:
        A numpy array containing data, a dictionary containing file metadata, and a proposed output filename.
    '''
    
    # Get source metadata
    if getImageType(infile) == 'S1single' or getImageType(infile) == 'S1dual':
        extent_source, EPSG_source, res_source, datetime, overpass = getS1Metadata(infile)
    
    elif getImageType(infile) == 'S2':
    
        # Select appropriate Sentinel-2 resolution. TODO: Deal with loading most appropriate scale
        res_source = S2_res
        extent_source, EPSG_source, datetime, tile = getS2Metadata(infile, resolution = S2_res)
    
    else:
        print 'WARNING: infile %s does not match any known ImageTypes'%infile

    # Build metadata dictionary    
    md_source = buildMetadataDictionary(extent_source, res_source, EPSG_source)
    
    # Load data using appropriate function
    if getImageType(infile) == 'S1single':    
        data = loadS1Single(infile)
        output_filename = '%s/%s_%s_%s_%s.tif'%(output_dir, output_name, getImageType(infile), overpass, datetime.strftime("%Y%m%d_%H%M%S"))
    
    elif getImageType(infile) == 'S1dual':
        data = loadS1Dual(infile)
        output_filename = '%s/%s_%s_%s_%s.tif'%(output_dir, output_name, getImageType(infile), overpass, datetime.strftime("%Y%m%d_%H%M%S"))
    
    elif getImageType(infile) == 'S2':
        data = loadS2(infile, res_source)
        output_filename = '%s/%s_%s_%s_%s.tif'%(output_dir, output_name, getImageType(infile), tile, datetime.strftime("%Y%m%d_%H%M%S"))

    else:
        print 'WARNING: infile %s does not match any known ImageTypes'%infile

    return data, md_source, output_filename


    
        
    

def main(infile, extent_dest, EPSG_dest, output_res, output_dir = os.getcwd(), output_name = 'CLASSIFIED', normalisation_type = 'none', normalisation_percentile = 95):
    """
    """
    
    from osgeo import gdal
        
    assert '_' not in output_name, "Sorry, output_name may not include the character '_'."
    
    # Load data, source metadata, and generate an output filename. S2_res currently hardwired. TODO: Deal with loading most appropriate data
    data, md_source, output_filename = loadData(infile, S2_res = 20, output_dir = output_dir, output_name = output_name)
    
    # Generate output metadata
    md_dest = buildMetadataDictionary(extent_dest, output_res, EPSG_dest)
    
    # Deseasonalise data
    data_deseasonalised = deseasonalise(data, md_source, normalisation_type = normalisation_type, normalisation_percentile = normalisation_percentile)

    # Resample data to output CRS
    data_resampled = subset(data_deseasonalised, md_source, md_dest, dtype = 7)
    
    # Classify the data into probability of a pixel being forest
    p_forest = classify(data_resampled, getImageType(infile))
        
    # Save data to disk
    ds = _createGdalDataset(md_dest, data_out = p_forest, filename = output_filename, nodata = 255, driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
    
    

if __name__ == '__main__':
    '''
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process Sentinel-1 and Sentinel-2 to match a predefined CRS, and perform a deseasaonalisation operation to reduce the impact of seasonality on relfectance/backscsatter.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('infiles', metavar = 'FILES', type = str, nargs = '+', help = 'Sentinel-1 processed input files in .dim format or a Sentinel-2 granule. Specify a valid S1/S2 input file or multiple files through wildcards (e.g. PATH/TO/*.dim, PATH/TO.SAFE/GRANULE/*/).')
    required.add_argument('-te', '--target_extent', nargs = 4, metavar = ('XMIN', 'YMIN', 'XMAX', 'YMAX'), type = float, help = "Extent of output image tile, in format <xmin, ymin, xmax, ymax>.")
    required.add_argument('-e', '--epsg', type=int, metavar = 'EPSG', help="EPSG code for output image tile CRS. This must be UTM. Find the EPSG code of your output CRS as https://www.epsg-registry.org/.")
    required.add_argument('-r', '--resolution', type=float, metavar = 'RES', help="Output resolution in m.")
    
    # Optional arguments
    optional.add_argument('-nt', '--normalisation_type', type=str, metavar = 'STR', default = 'none', help="Normalisation type. Set to one of 'none', 'local' or 'global'. Defaults to 'none'.")
    optional.add_argument('-np', '--normalisation_percentile', type=int, metavar = 'N', default = 95, help="Normalisation percentile, in case where normalisation type set to  'local' or 'global'. Defaults to 95 percent.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory. If nothing specified, downloads will output to the present working directory, given a standard filename.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'CLASSIFIED', help="Optionally specify a string to precede output filename.")
        
    # Get arguments
    args = parser.parse_args()

    # Get absolute path of input .safe files.
    infiles = [os.path.abspath(i) for i in args.infiles]
    
    # Slim down input files to only those that fall within tile
    md_dest = buildMetadataDictionary(args.target_extent, args.resolution, args.epsg)
    
    infiles = getFilesInTile(args.infiles, md_dest)
    
    for infile in infiles:
        
        # Execute script
        main(infile, args.target_extent, args.epsg, args.resolution, output_dir = args.output_dir, output_name = args.output_name, normalisation_type = args.normalisation_type, normalisation_percentile = args.normalisation_percentile)
