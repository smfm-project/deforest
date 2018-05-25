
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
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen1mosaic')

import sen2mosaic.utilities
import sen1mosaic.utilities

import pdb



def loadS1Single(scene, md_dest, polarisation = 'VV'):
    """
    Extract backscatter data given .dim file and polarisation
    
    Args:
        dim_file:
        polarisation: (i.e. 'VV' or 'VH')
    
    Returns:
        A maked numpy array of backscatter.
    """
    
    mask = scene.getMask(md = md_dest)
    data = scene.getBand(polarisation, md = md_dest)
    
    # Normalise
    #if normalise:
    #    wdata = normalise(data, scene.metadata, normalisation_type = scene.normalisation_type, normalisation_percentile = scene.normalisation_percentile, normalisation_area = scene.normalisation_area)
    
    return data


def loadS1Dual(scene, md_dest):
    """
    Extract backscatter metrics from a dual-polarised Sentinel-1 image.
    
    Args:
        dim_file: 
    
    Returns:
        A maked numpy array of VV, VH and VV/VH backscatter.
    """
    
    assert scene.image_type == 'S1dual', "input file %s does not appear to be a dual polarised Sentinel-1 file"%dim_file
        
    VV = loadS1Single(scene.filename, md_dest, polarisation = 'VV')
    
    VH = loadS1Single(scene.filename, md_dest, polarisation = 'VH')
    
    VV_VH = VH - VV # Proportional difference, logarithmic
    
    data = np.ma.vstack((VV, VH, VV_VH))
    
    #data = normalise(data, scene.metadata, normalisation_type = scene.normalisation_type, normalisation_percentile = scene.normalisation_percentile, normalisation_area = scene.normalisation_area)
    
    return data





def loadS2(scene, md_dest):
    """
    Calculate a range of vegetation indices given .SAFE file and resolution.
    
    Args:
        scene: 
    
    Returns:
        A maked numpy array of vegetation indices.
    """
       
    mask = scene.getMask(correct = True, md = md_dest)
    
    # Convert mask to a boolean array, allowing only values of 4 (vegetation), 5 (bare sois), and 6 (water)
    mask = np.logical_or(mask < 4, mask > 6)
    
    # Load the data (as masked numpy array)
    blue = scene.getBand('B02', md = md_dest)[mask == False] / 10000.
    green = scene.getBand('B03', md = md_dest)[mask == False] / 10000.
    red = scene.getBand('B04', md = md_dest)[mask == False] / 10000.
    swir1 = scene.getBand('B11', md = md_dest)[mask == False] / 10000.
    swir2 = scene.getBand('B12', md = md_dest)[mask == False] / 10000.
    
    if scene.resolution == 10:
        nir = scene.getBand('B08', md = md_dest)[mask == False] / 10000.
    else:
        nir = scene.getBand('B8A', md = md_dest)[mask == False] / 10000.
       
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
    #indices = normalise(indices, scene.metadata, normalisation_type = normalisation_type, normalisation_percentile = normalisation_percentile, normalisation_area = normalisation_area, reference_scene = reference_scene)
    
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
    
    

def buildReferenceScenes(source_files, ref_month, md_dest):
    '''
    Scan through a set of Sentinel-2 GRANULE files, and locate the most appropriate scene to match each to for each one.
    '''
    
    assert ref_month in range(1,13), "Normalisation month must be between 1 and 12."
    
    datetimes = []
    
    source_files = np.array(source_files)
    
    for scene in source_files:
        datetimes.append(scene.datetime)
    
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
            
            if scene.image_type == 'S1single':
                indices = loadS1Single(scene, md_dest)
            elif scene.image_type == 'S1dual':
                indices = loadS1Dual(scene, md_dest)
            elif scene.image_type == 'S2':
                indices = loadS2(scene, md_dest)
            else:
                raise IOError
                        
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


def getS2Res(resolution):
    '''
    '''
    
    if resolution < 60:
        return 20
    else:
        return 60
    

def main(source_files, target_extent, resolution, EPSG_code, output_dir = os.getcwd(), output_name = 'classified'):
    """
    """
    
    from osgeo import gdal
       
    # Determine output extent and projection
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    
    # Load scenes
    scenes = []
    for source_file in source_files:
        try:
            scenes.append(sen2mosaic.utilities.LoadScene(source_file, resolution = getS2Res(resolution)))
        except AssertionError:
            continue
        except IOError:
            try:
                scenes.append(sen1mosaic.utilities.LoadScene(source_file))
            except:
                continue
    
    # Remove scenes that aren't within output extent
    scenes = sen2mosaic.utilities.getSourceFilesInTile(scenes, md_dest)
    
    # Sort by date
    scenes = [scene for _,scene in sorted(zip([s.datetime.date() for s in scenes],scenes))]
    
    # Build reference scenes
    reference_scene, inds = buildReferenceScenes(scenes, 7, md_dest)
    
    
    for n, scene in enumerate(scenes):
        print 'Doing %s'%scene.filename.split('/')[-1]
        
        indices = loadS2(scene, md_dest)
        
        indices = normalise(indices, md_dest, normalisation_type = 'match', reference_scene = reference_scene[inds[n]])
        
        p_forest = classify(indices, scene.image_type)
        p_forest = np.ma.array(p_forest,mask = p_forest == 255)                   
        
        this_output_name = getOutputName(scene, output_dir = output_dir, output_name = output_name)
        
        # Save data to disk
        ds = _createGdalDataset(md_dest, data_out = p_forest.data, filename = this_output_name, driver='GTiff', dtype = gdal.GDT_Float32, options=['COMPRESS=LZW'])



if __name__ == '__main__':
    '''
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process Sentinel-1 and Sentinel-2 to match a predefined CRS, and perform a deseasaonalisation operation to reduce the impact of seasonality on relfectance/backscsatter.")
    
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    # Required arguments
    required.add_argument('-te', '--target_extent', nargs = 4, metavar = ('XMIN', 'YMIN', 'XMAX', 'YMAX'), type = float, help = "Extent of output image tile, in format <xmin, ymin, xmax, ymax>.")
    required.add_argument('-e', '--epsg', type=int, help="EPSG code for output image tile CRS. This must be UTM. Find the EPSG code of your output CRS as https://www.epsg-registry.org/.")
    optional.add_argument('-res', '--resolution', metavar = 'N', type=int, help="Specify a resolution to output.")
    
    # Optional arguments
    optional.add_argument('infiles', metavar = 'FILES', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files (level 2A) in .SAFE format, Sentinel-1 input files in .dim format, or a mixture. Specify one or more valid Sentinel-2 .SAFE, a directory containing .SAFE files, or multiple granules through wildcards (e.g. *.SAFE/GRANULE/*). Defaults to processing all granules in current working directory.')
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'CLASSIFIED', help="Optionally specify a string to precede output filename.")
    
    # Get arguments
    args = parser.parse_args()
    
    # Get absolute path of input .safe files.
    infiles = [os.path.abspath(i) for i in args.infiles]
    
    # Find files from input directory/granule etc.
    infiles_S2 = sen2mosaic.utilities.prepInfiles(infiles, '2A')
    infiles_S1 = sen1mosaic.utilities.prepInfiles(infiles)
    
    # Execute script
    main(infiles_S2 + infiles_S1, args.target_extent, args.resolution, args.epsg, output_dir = args.output_dir, output_name = args.output_name)
