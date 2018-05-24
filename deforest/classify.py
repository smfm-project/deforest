
import argparse
import csv
import datetime
import glob
import glymur
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from osgeo import gdal
import scipy.ndimage

import sys
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen2mosaic')

import sen2mosaic.utilities

import pdb



def loadS1Single(scene, polarisation = 'VV', normalise = True):
    """
    Extract backscatter data given .dim file and polarisation
    
    Args:
        dim_file:
        polarisation: (i.e. 'VV' or 'VH')
    
    Returns:
        A maked numpy array of backscatter.
    """
    
    
    # Remove trailing slash from input filename, if it exists, and determine path of data file
    data_file = scene.filename[:-4] + '.data'
     
    # Load data
    data = gdal.Open('%s/Gamma0_%s.img'%(data_file,polarisation)).ReadAsArray()
        
    # Get mask (where backscatter is 0)
    data = np.ma.array(data, mask = data == 0)
    
    # Normalise
    if normalise:
        data = normalise(data, scene.metadata, normalisation_type = scene.normalisation_type, normalisation_percentile = scene.normalisation_percentile, normalisation_area = scene.normalisation_area)
    
    return data


def loadS1Dual(scene):
    """
    Extract backscatter metrics from a dual-polarised Sentinel-1 image.
    
    Args:
        dim_file: 
    
    Returns:
        A maked numpy array of VV, VH and VV/VH backscatter.
    """
    
    assert scene.image_type == 'S1dual', "input file %s does not appear to be a dual polarised Sentinel-1 file"%dim_file
        
    VV = loadS1Single(scene.filename, polarisation = 'VV', normalise = False)
    
    VH = loadS1Single(scene.filename, polarisation = 'VH', normalise = False)
    
    VV_VH = VH - VV # Proportional difference, logarithmic
    
    data = np.ma.vstack((VV, VH, VV_VH))
    
    data = normalise(data, scene.metadata, normalisation_type = scene.normalisation_type, normalisation_percentile = scene.normalisation_percentile, normalisation_area = scene.normalisation_area)
    
    return data





def loadS2(scene, normalisation_type = 'none', normalisation_percentile = 95, normalisation_area = 200000, reference_scene = None):
    """
    Calculate a range of vegetation indices given .SAFE file and resolution.
    
    Args:
        scene: 
    
    Returns:
        A maked numpy array of vegetation indices.
    """
       
    mask = scene.getMask(correct = True)
    
    # Convert mask to a boolean array, allowing only values of 4 (vegetation), 5 (bare sois), and 6 (water)
    mask = np.logical_or(mask < 4, mask > 6)
    
    # Load the data (as masked numpy array)
    blue = scene.getBand('B02')[mask == False] / 10000.
    green = scene.getBand('B03')[mask == False] / 10000.
    red = scene.getBand('B04')[mask == False] / 10000.
    swir1 = scene.getBand('B11')[mask == False] / 10000.
    swir2 = scene.getBand('B12')[mask == False] / 10000.
    
    if scene.resolution == 10:
        nir = scene.getBand('B08')[mask == False] / 10000.
    else:
        nir = scene.getBand('B8A')[mask == False] / 10000.
       
    # Calculate vegetation indices from Shultz 2016
    indices = np.zeros((mask.shape[0], mask.shape[1], 5), dtype = np.float32)
    
    # Don't report div0 errors
    with np.errstate(divide='ignore'):
        
        # NDVI
        indices[:,:,0][mask == False] = (nir - red) / (nir + red)
        
        # EVI
        indices[:,:,1][mask == False] = ((nir - red) / (nir + 6 * red - 7.5 * blue + 1)) * 2.5
        
        # GEMI
        n = (2 * (nir ** 2 - red ** 2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5)
        indices[:,:,2][mask == False] = (n * (1 - 0.25 * n)) - ((red - 0.125) / (1 - red))
        
        # NDMI
        indices[:,:,3][mask == False] = (nir - swir1) / (nir + swir1)
        
        # SAVI
        indices[:,:,4][mask == False] = (1 + 0.5) * ((nir - red) / (nir + red + 0.5))
        
        # TC wetness
        #indices[:,:,5][mask == False] = 0.0315 * blue + 0.2021 * green + 0.3102 * red + 0.1594 * nir - 0.6806 * swir1 - 0.6109 * swir2
        
        # TC greenness
        #indices[:,:,6][mask == False] = -0.1603 * blue - 0.2819 * green - 0.4934 * red + 0.7940 * nir - 0.0002 * swir1 - 0.1446 * swir2
        
        # Turn into a masked array
        indices = np.ma.array(indices, mask = np.repeat(np.expand_dims(mask,2), indices.shape[-1], axis=2))
        
    # Normalise data
    indices = normalise(indices, scene.metadata, normalisation_type = normalisation_type, normalisation_percentile = normalisation_percentile, normalisation_area = normalisation_area, reference_scene = reference_scene)
    
    return indices



def _createGdalDataset(md, data_out = None, filename = '', driver = 'MEM', dtype = 3, RasterCount = 1, nodata = None, options = []):
    '''
    Function to create an empty gdal dataset with georefence info from metadata dictionary.

    Args:
        md: Object from Metadata() class.
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
    ds = gdal_driver.Create(filename, md.ncols, md.nrows, RasterCount, dtype, options = options)
    
    ds.SetGeoTransform(md.geo_t)
    
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(md.EPSG_code)
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
                

def normalise(data, md, normalisation_type = 'none', normalisation_percentile = 95, normalisation_area = 200000., reference_scene = None):
    '''
    Normalises an array by dividing pixels by the 95th percentile value in the vicinity of each pixel
    
    Args:
        data: A masked numpy array containing the data
        md: metadata dictionary
        normalisation_type: Select one of 'none' (no normalisation), 'global' (subtract percentile of entire scene), or 'local' (subtract percentile from the area surrounding each pixel).
        percentile: Data percentile to subtract, if normalisation_type == 'local' or 'global'. Defaults to 95%.
        area: Area in m^2 to determine the kernel size if normalisation_type == 'local'. This should be greater than the size of expected deforestation events. Defaults to 200,000 m^2 (20 ha).
    
    Returns:
        The deseasonalised numpy array
    
    '''
    
    assert normalisation_type in ['none','local','global','match'], "normalisation_type must be one of 'none', 'local', 'global' or 'match' (test only). It was set to %s."%str(normalisation_type)  
    
    if normalisation_type == 'match':
        assert reference_scene is not None, "A matching scene must be specified if using match for image normalisation."
        
    # Takes care of case where only a 2d array is input, allowing us to loop through the third axis
    if data.ndim == 2:
        data = np.ma.expand_dims(data,2)
        
    # No normalisation        
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
    
    # Normalise to reference_scene
    if normalisation_type == 'match':
        
        # Histogram matching with overlap
        #data = sen2mosaic.utilities.histogram_match(data, reference_scene)
        
        # Gain compensation at overlap
        """
        for i in range(data.shape[-1]):
            pdb.set_trace()
            overlap = np.logical_and(data.mask[:,:,i] == False, reference_scene.mask[:,:,i] == False)
                        
            # Gain compensation (simple inter-scene correction)                    
            this_intensity = np.mean(data.data[:,:,i][overlap])
            ref_intensity = np.mean(reference_scene.data[:,:,i][overlap])
            
            sel = data.mask[:,:,i]==False
            data.data[sel] = data[sel] * (ref_intensity/this_intensity)
        """
        
        for i in range(data.shape[-1]):
            
            overlap = np.logical_and(data.mask[:,:,i] == False, reference_scene.mask[:,:,i] == False)

            # Standardise to match mean/stdev of overlap
            ref_stdev = np.ma.std(reference_scene[:,:,i][overlap])
            ref_mean = np.ma.mean(reference_scene[:,:,i][overlap])
            this_stdev = np.ma.std(data.data[:,:,i][overlap])
            this_mean = np.ma.mean(data.data[:,:,i][overlap])
            
            sel = data.mask[:,:,i]==False
            data[sel, i] = ref_mean + (data[sel,i] - this_mean) * (ref_stdev / this_stdev)
        
    # Get rid of residual dimensions where 2d array was input
    data = np.squeeze(data)
    data_percentile = np.squeeze(data_percentile)
        
    # And subtract the seasonal effect from the array
    data_normalised = data - data_percentile
    
    return data_normalised


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


def _rescaleData(data, means, scales):
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
    
    coefs, means, scales = loadCoefficients(image_type)
    
    p_forest = np.sum(coefs[1:] * _rescaleData(data, means, scales), axis = 2) + coefs[0]
    
    # Convert from odds to probability
    p_forest = np.exp(p_forest) / (1 + np.exp(p_forest))
    
    # Reduce the file size by converting data to integer
    p_forest_out = np.round(p_forest.data * 100 , 0).astype(np.uint8)
    p_forest_out[p_forest.mask] = nodata
    
    return p_forest_out
    
    

def findMatch(source_files, ref_month):
    '''
    Scan through a set of Sentinel-2 GRANULE files, and locate the most appropriate scene to match each to for each one.
    '''
    
    assert ref_month in range(1,13), "Normalisation month must be between 1 and 12."
    
    nodata, datetimes = [], []
    
    source_files = np.array(source_files)
    
    for scene in source_files:
        nodata.append(scene.nodata_percent)
        datetimes.append(scene.datetime)
    
    nodata = np.array(nodata)
    datetimes = np.array(datetimes)
    
    years = datetimes.astype('datetime64[Y]').astype(int) + 1970
    months = datetimes.astype('datetime64[M]').astype(int) % 12 + 1
    
    masked = 1
    this_month = 1
    
    out = []
    im_date = []
    
    for year in np.unique(years):
        
        for n, scene in enumerate(source_files[years == year]):
                   
            this_month = months[years == year][n]
            
            if np.logical_or(this_month < ref_month, this_month > ref_month + 1):
                continue
            
            print scene.filename
            
            indices = loadS2(scene)
            
            if 'composite' not in locals():
                composite = indices
                im_date.append(scene.datetime)
                
            else:
                
                sel = np.logical_and(composite.mask, indices.mask==False)
                
                composite[sel] = indices[sel]
            
            # Re-calculate maked pixels
            masked = float(composite.mask.sum()) / ((composite.mask==False).sum() + composite.mask.sum())
            
            # After 1 month data, quit if enough data present, or continue for another month until enough data
            # Break if two months passed
            if this_month > ref_month + 1:
                break
            if this_month == ref_month + 1 and masked > 0.75:
                break
                
        # If year had any data
        if 'composite' in locals():
            out.append(composite)
        
            # Reset composite image
            del composite
      
    # Determine which composite image should be used by each infile
    inds = np.zeros_like(source_files, dtype=np.int)
    
    # Use most recently measured composite (or upcoming if month not yet passed)
    for n, date in enumerate(im_date):
        inds[datetimes >= date] = n
    
    return out, inds

    
def findMatch2(source_files, ref_month):
    '''
    Scan through a set of Sentinel-2 GRANULE files, and locate the most appropriate scene to match each to for each one.
    '''
    
    assert ref_month in range(1,13), "Normalisation month must be between 1 and 12."
    
    nodata, datetimes = [], []
    
    source_files = np.array(source_files)
    
    for scene in source_files:
        nodata.append(scene.nodata_percent)
        datetimes.append(scene.datetime)
    
    nodata = np.array(nodata)
    datetimes = np.array(datetimes)
    
    years = datetimes.astype('datetime64[Y]').astype(int) + 1970
    months = datetimes.astype('datetime64[M]').astype(int) % 12 + 1
    
    masked = 1
    this_month = 1
    
    out = []
    im_date = []
    
    for year in np.unique(years):
        
        for n, scene in enumerate(source_files[years == year]):
                   
            this_month = months[years == year][n]
                        
            print scene.filename
            
            indices = loadS2(scene)
            
            if 'composite' not in locals():
                composite = np.ma.array(np.zeros_like(indices).data)
                composite[indices.mask == False] = indices[indices.mask == False]
                im_date.append(scene.datetime)
                count = np.zeros_like(composite[:,:,0].data)
                
            else:
                                
                composite[indices.mask == False] += indices[indices.mask == False]
            
            count[(indices.mask == False)[:,:,0]] += 1
        
        composite = composite / count.astype(np.float)[:,:,np.newaxis]       
        
        # If year had any data
        if 'composite' in locals():
            out.append(composite)
        
            # Reset composite image
            del composite
        
    # Determine which composite image should be used by each infile
    inds = np.zeros_like(source_files, dtype=np.int)
    
    # Use most recently measured composite (or upcoming if month not yet passed)
    for n, date in enumerate(im_date):
        inds[datetimes >= date] = n
    
    return out, inds

          

def getOutputName(scene, output_dir = os.getcwd(), output_name = 'classified'):
    '''
    '''
    
    output_dir = output_dir.rstrip('/')
    
    output_name = '%s/%s_%s_%s_%s_%s.tif'%(output_dir, output_name, scene.image_type, scene.tile, datetime.datetime.strftime(scene.datetime, '%Y%m%d'), datetime.datetime.strftime(scene.datetime, '%H%M%S'))

    return output_name


"""
def getTileMetadata(tile, resolution):
    '''
    '''
    
    from osgeo import ogr, osr
    import lxml.etree as ET
    
    assert sen2mosaic.utilities.validateTile(tile), "Invalid tile format. Tile must take the format '##XXX'."
    assert resolution in [10, 20, 60], "Resolution must be 10, 20 or 60 m."
    
    # Get location of current file
    directory = os.path.dirname(os.path.abspath(__file__))
    #directory = os.getcwd()
    kml_file =  glob.glob('%s/cfg/S2A_OPER_GIP_*_B00.kml'%'/'.join(directory.split('/')[:-1]))[0]
    
    tree = ET.parse(kml_file)
    root = tree.getroot()
    
    # Define xml namespace
    xmlns = {'xmlns':root.tag[1:].split('}')[0]}
    
    placemarks = root.findall('.//xmlns:Placemark', xmlns)
    tiles = np.array([placemark.find('xmlns:name',xmlns).text for placemark in placemarks])
    
    # For this tile
    placemark = placemarks[np.where(tiles == tile)[0][0]]
    
    polygon = placemark.find('.//xmlns:Polygon', xmlns)
    
    coordinates = polygon.find('.//xmlns:coordinates', xmlns).text
    coordinates = coordinates.replace('\n','')
    coordinates = coordinates.replace('\t','')
    
    minlon = min(float(coordinates.split(' ')[0].split(',')[0]), float(coordinates.split(' ')[3].split(',')[0]))
    maxlon = max(float(coordinates.split(' ')[1].split(',')[0]), float(coordinates.split(' ')[2].split(',')[0]))
    minlat = min(float(coordinates.split(' ')[3].split(',')[1]), float(coordinates.split(' ')[4].split(',')[1]))
    maxlat = max(float(coordinates.split(' ')[0].split(',')[1]), float(coordinates.split(' ')[1].split(',')[1]))
    
    # N or S hemisphere:
    point = placemark.find('.//xmlns:Point', xmlns)
    coordinates = point.find('.//xmlns:coordinates', xmlns).text

    utm_zone = int(tile[:2])
    
    if float(coordinates.split(',')[1]) < 0:
        EPSG_code = 32700 + utm_zone
    else:
        EPSG_code = 32600 + utm_zone
            
    # create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(4326)

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(EPSG_code)
    
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    
    xmin, ymin, z = coordTransform.TransformPoint(minlon, minlat)
    xmax, ymax, z = coordTransform.TransformPoint(maxlon, maxlat)
    
    return sen2mosaic.utilities.Metadata([xmin, ymin, xmax, ymax], resolution, EPSG_code)

"""

def getTileMetadata(tile, resolution):
    '''
    '''
    
    import pandas as pd
    
    assert sen2mosaic.utilities.validateTile(tile), "Invalid tile format. Tile must take the format '##XXX'."
    assert resolution in [10, 20, 60], "Resolution must be 10, 20 or 60 m."
    
    # Get location of metadata csv file
    directory = os.path.dirname(os.path.abspath(__file__))
    csv_file =  glob.glob('%s/cfg/S2_tile_metadata.csv'%'/'.join(directory.split('/')[:-1]))[0]
    
    # Load S2 tile metadata as pandas dataframe
    df = pd.read_csv(csv_file, sep = ',')
    
    assert sum(df['Tile'] == tile) > 0, "Sentinel-2 tile %s not recognised (see S2_tile_metadata.csv for Sentinel-2 tile metadata)."%tile
    
    # Get extent
    xmin = int(df.loc[df['Tile'] == tile, 'Xmin'])
    ymin = int(df.loc[df['Tile'] == tile, 'Ymin'])
    xmax = int(df.loc[df['Tile'] == tile, 'Xmax'])
    ymax = int(df.loc[df['Tile'] == tile, 'Ymax'])
    
    # Get EPSG code
    EPSG_code = int(df.loc[df['Tile'] == tile, 'EPSG_Code'])

    return sen2mosaic.utilities.Metadata([xmin, ymin, xmax, ymax], resolution, EPSG_code)


def main(tile, source_files, resolution = 20, output_dir = os.getcwd(), output_name = 'classified', normalisation_type = 'none', normalisation_percentile = 95, normalisation_area = 200000., normalisation_month = 7):
    """
    """
    
    from osgeo import gdal
    
    # Determine output extent and projection
    #md_dest = getTileMetadata(tile, resolution)
    
    # Load scenes
    scenes = [sen2mosaic.utilities.LoadScene(source_file, resolution = resolution) for source_file in source_files]
    
    # Remove scenes that aren't within output extent
    #scenes = sen2mosaic.utilities.getSourceFilesInTile(scenes, md_dest)
    
    # Sort by date
    scenes = sen2mosaic.utilities.sortScenes(scenes, by = 'date')
    
    # To be formalised: remove any scene which isn't from this S2 tile
    scenes_out = []
    for scene in scenes:
        if scene.tile == 'T%s'%tile:
            scenes_out.append(scene)
    scenes = scenes_out
    
    if normalisation_type == 'match' or normalisation_type == 'stratify':
        reference_scene, inds = findMatch(scenes, normalisation_month)
        if normalisation_type == 'stratify':
            
            from sklearn import cluster
            k_means = cluster.KMeans(n_clusters=6)
            X = reference_scene[1].reshape((1830*1830,5))
            mask = np.sum(X.mask,axis=1) > 0
            k_means.fit(X.data[mask==False])
            
            normalisation_class = np.zeros_like(reference_scene[1][:,:,0].data)
            normalisation_class[mask.reshape((1830,1830))==False] = k_means.labels_ + 1
            
            #normalisation_class = np.array(np.round(reference_scene[0][:,:,0] * 10,0),dtype=np.int)
            #normalisation_class[normalisation_class < 1] = 0
            
            ds = _createGdalDataset(scene.metadata, data_out = normalisation_class, filename = output_dir + 'STRATIFY_CLASS.tif', driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
        
    for n, scene in enumerate(scenes):
        print 'Doing %s'%scene.filename.split('/')[-1]
                
        # Load indices
        if normalisation_type == 'match':
            indices = loadS2(scene, normalisation_type = normalisation_type, reference_scene = reference_scene[inds[n]])
        
        elif normalisation_type == 'stratify':
            indices = loadS2(scene, normalisation_type = 'none')
        
        else:
            indices = loadS2(scene, normalisation_type = normalisation_type, normalisation_percentile = normalisation_percentile, normalisation_area = normalisation_area)
        
        if normalisation_type != 'stratify':
            print np.ma.mean(indices[:,:,0])
            p_forest = classify(indices, scene.image_type)
            p_forest = np.ma.array(p_forest,mask = p_forest == 255)
        
        else:
            import scipy.stats
            p_forest = np.zeros_like(reference_scene[0].data[:,:,0])
            for i in range(1,7):
                from statsmodels.robust.scale import huber
                try:
                    mean_robust, std_robust = huber(indices[:,:,0][normalisation_class==i])
                    z_score = (indices[:,:,0][normalisation_class==i] - mean_robust) / std_robust
                    
                except:
                    z_score = scipy.stats.mstats.zscore(indices[:,:,0][normalisation_class==i])
                
                p_forest[normalisation_class==i] = z_score#scipy.special.ndtr()
                #pdb.set_trace()
                
                #from sklearn.covariance import MinCovDet
                #X = indices[normalisation_class==i]
                #robust_cov = MinCovDet().fit(X)
                #robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
                
                #from sklearn.neighbors import LocalOutlierFactor
                #clf = LocalOutlierFactor(n_neighbors=20)
                #y_pred = clf.fit_predict(X)
            
            p_forest = np.ma.array(p_forest,mask = np.logical_or(normalisation_class==0, indices[:,:,0].mask))
                    
        
        this_output_name = getOutputName(scene, output_dir = output_dir, output_name = output_name)
        
        # Save data to disk
        #ds = _createGdalDataset(scene.metadata, data_out = p_forest, filename = this_output_name, nodata = 255, driver='GTiff', dtype = gdal.GDT_Byte, options=['COMPRESS=LZW'])
        ds = _createGdalDataset(scene.metadata, data_out = p_forest.data, filename = this_output_name, driver='GTiff', dtype = gdal.GDT_Float32, options=['COMPRESS=LZW'])

if __name__ == '__main__':
    '''
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process Sentinel-1 and Sentinel-2 to match a predefined CRS, and perform a deseasaonalisation operation to reduce the impact of seasonality on relfectance/backscsatter.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('-t', '--tile', metavar = '##XXX', type = str, help = "Deforest uses the Sentinel-2 tiling grid to produce outputs. Specify a valid Sentinel-2 tile in the format ##XXX.")
    
    # Optional arguments
    optional.add_argument('infiles', metavar = 'L2A_FILES', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files (level 2A) in .SAFE format. Specify one or more valid Sentinel-2 .SAFE, a directory containing .SAFE files, or multiple granules through wildcards (e.g. *.SAFE/GRANULE/*). Defaults to processing all granules in current working directory.')
    optional.add_argument('-r', '--resolution', metavar = 'N', type = int, default = 20, help = 'Resolution to process. Must be 10, 20 or 60 m.')
    optional.add_argument('-nt', '--normalisation_type', type=str, metavar = 'STR', default = 'none', help="Normalisation type. Set to one of 'none', 'local' or 'global'. Defaults to 'none'.")
    optional.add_argument('-np', '--normalisation_percentile', type=int, metavar = 'N', default = 95, help="Normalisation percentile, in case where normalisation type set to  'local' or 'global'. Defaults to 95 percent.")
    optional.add_argument('-na', '--normalisation_area', type=float, metavar = 'N', default = 200000., help="Normalisation area. Defaults to 200000 m^2.")
    optional.add_argument('-nm', '--normalisation_month', type=int, metavar = 'N', default = 7, help="If using 'match' image normalisation, each image will be corrected using the histogram of a single month. Specify months as 1-12 (January - December), defaults to 7 (July).")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'CLASSIFIED', help="Optionally specify a string to precede output filename.")
        
    # Get arguments
    args = parser.parse_args()

    # Get absolute path of input .safe files.
    infiles = [os.path.abspath(i) for i in args.infiles]
    
    # Find files from input directory/granule etc.
    infiles = sen2mosaic.utilities.prepInfiles(infiles, '2A')
    
    # Execute script
    main(args.tile, infiles, resolution = args.resolution, output_dir = args.output_dir, output_name = args.output_name, normalisation_type = args.normalisation_type, normalisation_percentile = args.normalisation_percentile, normalisation_area = args.normalisation_area, normalisation_month = args.normalisation_month)
