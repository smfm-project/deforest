
import argparse
import glob
import glymur
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from osgeo import gdal
import scipy.ndimage.filters
import xml.etree.ElementTree as ET

import pdb



def loadS2NDVI(L2A_file, res):
    """
    Calculate NDVI given .SAFE file and resolution.
    
    Args:
        L2A_file: A granule from a level 2A Sentinel-2 .SAFE file, processed with sen2cor.
        res: Integer of resolution to be processed (i.e. 10 m, 20 m, or 60 m). 
    
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
    red_path = glob.glob('%s/IMG_DATA/R%sm/*_B04_*m.jp2'%(L2A_file,str(res)))
    nir_path = glob.glob('%s/IMG_DATA/R%sm/*_B8A_*m.jp2'%(L2A_file,str(res)))
    
    # Load the data (as numpy array)
    red = np.ma.array(glymur.Jp2k(red_path[0])[:] / 10000., mask = mask)
    nir = np.ma.array(glymur.Jp2k(nir_path[0])[:] / 10000., mask = mask)
    
    # Calculate NDVI
    ndvi = (nir - red) / (nir + red)
    
    # Set masked data to 0
    ndvi.data[ndvi.mask] = 0.
    
    return ndvi


def loadS1Gamma0(dim_file, polarisation = 'VV'):
    """
    Extract Gamma0 data given .dim file and polarisation
    
    Args:
        dim_file: 
        polarisation: (i.e. 'VV' or 'VH')
    
    Returns:
        A maked numpy array of Gamma0 backscatter.
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
    data[mask] = 0
    
    return np.ma.array(data, mask = mask)


def getS1Metadata(dim_file, polarisation = 'VV'):
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
    
    # Remove trailing / from granule directory if present 
    dim_file = dim_file.rstrip('/')
    data_file = dim_file[:-4] + '.data'
    
    assert polarisation == 'VV' or polarisation == 'VH', "Polarisation must be VV or VH. getS1Metadata() was given %s"%polarisation    
    assert os.path.exists(dim_file), "The location %s does not contain a Sentinel-1 .dim file."%dim_file
    assert os.path.exists(data_file), "The location %s does not contain a Sentinel-1 .data file."%data_file

    ds = gdal.Open('%s/Gamma0_%s.img'%(data_file,polarisation))

    geo_t = ds.GetGeoTransform()
    
    # Get array size
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    
    # Get extent data
    ulx = geo_t[0]
    uly = geo_t[3]
    xres = geo_t[1]
    yres = geo_t[5]
    lrx = ulx + (xres * ncols)
    lry = uly + (yres * nrows)
    extent = [ulx, lry, lrx, uly]
    
    res = abs(xres)
    
    # Find EPSG code to define projection
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    srs.AutoIdentifyEPSG()
    EPSG = int(srs.GetAttrValue("AUTHORITY", 1))
    
    # Extract date string from filename
    date = dim_file.split('/')[-1].split('_')[-5]
    
    return extent, EPSG, res, date



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
        
    # Remove trailing / from granule directory if present 
    granule_file = granule_file.rstrip('/')
    
    assert len(glob.glob((granule_file + '/*MTD*.xml'))) > 0, "The location %s does not contain a metadata (*MTD*.xml) file."%granule_file
    
    # Find the xml file that contains file metadata
    xml_file = glob.glob(granule_file + '/*MTD*.xml')[0]
        
    # Define xml namespace (specific to level 2A Sentinel 2 .SAFE files)
    ns = {'n1':'https://psd-12.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd'}
    
    # Parse xml file
    tree = ET.ElementTree(file = xml_file)
    root = tree.getroot()
    
    # Get array size
    size = root.find("n1:Geometric_Info/Tile_Geocoding/Size[@resolution='%s']"%str(resolution),ns)
    nrows = int(size.find('NROWS').text)
    ncols = int(size.find('NCOLS').text)
    
    # Get extent data
    geopos = root.find("n1:Geometric_Info/Tile_Geocoding/Geoposition[@resolution='%s']"%str(resolution),ns)
    ulx = float(geopos.find('ULX').text)
    uly = float(geopos.find('ULY').text)
    xres = float(geopos.find('XDIM').text)
    yres = float(geopos.find('YDIM').text)
    lrx = ulx + (xres * ncols)
    lry = uly + (yres * nrows)
    
    extent = [ulx, lry, lrx, uly]
    
    # Find EPSG code to define projection
    EPSG = root.find('n1:Geometric_Info/Tile_Geocoding/HORIZONTAL_CS_CODE',ns).text
    EPSG = int(EPSG.split(':')[1])
    
    # Extract date string from filename
    
    if granule_file.split('/')[-1].split('_')[1] == 'USER':
        # If old file format
        date = granule_file.split('/')[-1].split('__')[1].split('_')[0].split('T')[0]
    else:
        # If new file format
        date = granule_file.split('/')[-1].split('_')[-1].split('T')[0]
    
    return extent, EPSG, date



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




def _createGdalDataset(md, data_out = None, filename = '', driver = 'MEM', dtype = 3, options = []):
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
    from osgeo import gdal
       
    gdal_driver = gdal.GetDriverByName(driver)
    ds = gdal_driver.Create(filename, md['ncols'], md['nrows'], 1, dtype, options = options)
    ds.SetGeoTransform(md['geo_t'])
    ds.SetProjection(md['proj'].ExportToWkt())
       
    # If a data array specified, add it to the gdal dataset
    if type(data_out).__module__ == np.__name__:
        ds.GetRasterBand(1).WriteArray(data_out)
    
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
    ds_dest.GetRasterBand(1).Fill(nodatavalue)
    ds_dest.GetRasterBand(1).SetNoDataValue(nodatavalue)
    
    # Reproject source into dest project coordinates
    gdal.ReprojectImage(ds_source, ds_dest, proj_source, proj_dest, gdal.GRA_NearestNeighbour)
            
    ds_resampled = ds_dest.GetRasterBand(1).ReadAsArray()
    
    return ds_resampled


def subset(data, md_source, md_dest, dtype = 3):
    '''
    '''
    
    # Write array to a gdal dataset
    ds_source = _createGdalDataset(md_source, data_out = data.data, dtype = dtype)
    ds_dest = _createGdalDataset(md_dest, dtype = dtype)
    data_resampled = _reprojectImage(ds_source, ds_dest, md_source, md_dest)
    
    ds_source = _createGdalDataset(md_source, data_out = data.mask, dtype = 1)
    ds_dest = _createGdalDataset(md_dest, dtype = 1)
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
        area: Area in m^2 to determine the kernel size if normalisation_type == 'local'. This should be greater than the size of a deforestation event. Defaults to 200,000 m^2 (20 ha).
    
    Returns:
        The deseasonalised numpy array
    
    '''
    
    assert normalisation_type in ['none','local','global'], "normalisation_type must be one of 'none', 'local' or 'global'. It was set to %s."%str(normalisation_type)  
    
    # No normalisation
    if normalisation_type == 'none':
        data_percentile = 0.
    
    # Following Hamunyela et al. 2016
    if normalisation_type == 'local':  
        res = md['res']
        filter_size = int(round((float(area) / (res ** 2)) ** 0.5,0))
        data_percentile = scipy.ndimage.filters.percentile_filter(data, normalisation_percentile, size = (filter_size, filter_size))
    
    # Following Reiche et al. 2017
    if normalisation_type == 'global':
        data_percentile = np.percentile(data, normalisation_percentile)
    
    data_deseasonalised = data - data_data_percentile 
    
    return data_deseasonalised



def main(infile, sensor, extent_dest, EPSG_dest, output_res, output_dir = os.getcwd(), output_name = 'DESEASONALISED', S1_pol = 'VV', S2_res = 20, normalisation_type = 'none', normalisation_percentile = 95):
    """
    """
    
    assert sensor == 'S1' or sensor == 'S2', "Specified sensor %s is invalid."%str(sensor)
    assert '_' not in output_name, "Sorry, output_name may not include the character '_'."
        
    md_dest = buildMetadataDictionary(extent_dest, output_res, EPSG_dest)
              
    if sensor == 'S1':
        
        # Get source metadata
        extent_source, EPSG_source, res_source, date = getS1Metadata(infile, polarisation = S1_pol)
        
        # Build source metadata dictionary
        md_source = buildMetadataDictionary(extent_source, res_source, EPSG_source)
        
        # Load data
        data = loadS1Gamma0(infile, polarisation = S1_pol)

        # Resample data
        data_resampled = subset(data, md_source, md_dest, dtype = 7)

        # Deseasonalise data
        data_deseasonalised = deseasonalise(data_resampled, md_dest, normalisation_type = normalisation_type, normalisation_percentile = normalisation_percentile)        
               
        # Output data
        output_uid = '_'.join(infile.split('/')[-1][:-4].split('_')[-4:])
        output_filename = '%s/%s_%s_%s_%s_%s_%s.tif'%(output_dir, output_name, sensor, S1_pol, date, output_uid, 'data')       
        ds = _createGdalDataset(md_dest, data_out = data_deseasonalised.data, filename = output_filename, driver='GTiff', dtype=7, options=['COMPRESS=LZW'])

        output_filename = '%s/%s_%s_%s_%s_%s_%s.tif'%(output_dir, output_name, sensor, S1_pol, date, output_uid, 'mask')
        ds = _createGdalDataset(md_dest, data_out = data_deseasonalised.mask, filename = output_filename, driver='GTiff', dtype=1, options=['COMPRESS=LZW'])
        
    if sensor == 'S2':
        
        # Get source metadata
        extent_source, EPSG_source, date = getS2Metadata(infile, resolution = S2_res)
        
        # Build source metadata dictionary
        md_source = buildMetadataDictionary(extent_source, S2_res, EPSG_source)
        
        # Load data
        data = loadS2NDVI(infile, S2_res)

        # Resample data
        data_resampled = subset(data, md_source, md_dest, dtype = 7)
        
        # Deseasonalise data
        data_deseasonalised = deseasonalise(data_resampled, md_dest, normalisation_type = normalisation_type, normalisation_percentile = normalisation_percentile)
        
        # Output data
        output_uid = ''.join((infile.split('/')[-1]).split('.'))
        output_filename = '%s/%s_%s_%s_%s_%s_%s.tif'%(output_dir, output_name, sensor, str(S2_res), date, output_uid, 'data')
        ds = _createGdalDataset(md_dest, data_out = data_deseasonalised.data, filename = output_filename, driver='GTiff', dtype=7, options=['COMPRESS=LZW'])

        output_filename = '%s/%s_%s_%s_%s_%s.tif'%(output_dir, output_name, sensor, str(S2_res), date, output_uid, 'mask')
        ds = _createGdalDataset(md_dest, data_out = data_deseasonalised.mask, filename = output_filename, driver='GTiff', dtype=1, options=['COMPRESS=LZW'])



if __name__ == '__main__':
    '''
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Process Sentinel-1 and Sentinel-2 to match a predefined CRS, and perform a deseasaonalisation operation to reduce the impact of seasonality on relfectance/backscsatter.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('infiles', metavar = 'FILES', type = str, nargs = '+', help = 'Sentinel-1 processed input files in .dim format or a Sentinel-2 granule. Specify a valid S1/S2 input file or multiple files through wildcards (e.g. PATH/TO/*.dim, PATH/TO/GRANULE/*/).')
    required.add_argument('-s', '--sensor', metavar = 'SENSOR', type = str, help = 'The name of the sensor to be processed (S1 or S2)')
    required.add_argument('-te', '--target_extent', nargs = 4, metavar = ('XMIN', 'YMIN', 'XMAX', 'YMAX'), type = float, help = "Extent of output image tile, in format <xmin, ymin, xmax, ymax>.")
    required.add_argument('-e', '--epsg', type=int, metavar = 'EPSG', help="EPSG code for output image tile CRS. This must be UTM. Find the EPSG code of your output CRS as https://www.epsg-registry.org/.")
    required.add_argument('-r', '--resolution', type=float, metavar = 'RES', help="Output resolution in m.")
    
    # Optional arguments
    optional.add_argument('-nt', '--normalisation_type', type=str, metavar = 'STR', default = 'none', help="Normalisation type. Set to one of 'none', 'local' or 'global'. Defaults to 'none'.")
    optional.add_argument('-np', '--normalisation_percentile', type=float, metavar = 'N', default = 95, help="Normalisation percentile, in case where normalisation type set to  'local' or 'global'. Defaults to 95%.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Optionally specify an output directory. If nothing specified, downloads will output to the present working directory, given a standard filename.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'S1_output', help="Optionally specify a string to precede output filename.")
    optional.add_argument('-i', '--S2resolution', type=int, metavar = 'RES', default = 20, help="Optionally specify an input resolution for Sentinel-2 data (10, 20, or 60). Defaults to 20 m.")
    optional.add_argument('-p', '--polarisation', type=str, metavar = 'POL', default = 'VV', help="Optionally specify a polarisation for Sentinel-2 (VV or VH). Defaults to VV.")
    
    # Get arguments
    args = parser.parse_args()

    # Get absolute path of input .safe files.
    args.infiles = [os.path.abspath(i) for i in args.infiles]
    
    for infile in args.infiles:
        
        # Execute script
        main(infile, args.sensor, args.target_extent, args.epsg, args.resolution, output_dir = args.output_dir, output_name = args.output_name, S1_pol = args.polarisation, S2_res = args.S2resolution, )