# This is a second attempt at producing the training script.
# It works by importing functions from classify.py, and outputting a config file to inform classify.py

import argparse
import csv
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal
from PIL import Image, ImageDraw
import shapefile
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen2mosaic')
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen1mosaic')
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/deforest/deforest')

import sen1mosaic.utilities
import sen2mosaic.utilities
import classify

import pdb


def getFilesOfType(infiles, image_type):
    '''
    Reduce an input file list to just those files with type image_type.
    
    Args:
        infiles: A list of input files of types .dim (Sentinel-1) or granules (Sentinel-2)
        image_type: A string describing image type. This can be 'S1single', 'S1dual', or 'S2'.
    
    Returns:
        A reduced list of input files
    '''
    
    s = np.array([classify.getImageType(i) == image_type for i in infiles])
        
    return np.array(infiles)[s].tolist()




def _coordinateTransformer(shp, EPSG_out):
    """
    Generates function to transform coordinates from a source shapefile CRS to EPSG.
    
    Args:
        shp: Path to a shapefile.
    
    Returns:
        A function that transforms shapefile points to EPSG.
    """
    
    from osgeo import ogr, osr
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shp)
    layer = ds.GetLayer()
    spatialRef = layer.GetSpatialRef()
    
    # Create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromWkt(spatialRef.ExportToWkt())

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(EPSG_out)

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    
    return coordTransform


def _world2Pixel(geo_t, x, y):
    """
    Uses a gdal geomatrix (ds.GetGeoTransform()) to calculate the pixel location of a geospatial coordinate.
    Modified from: http://geospatialpython.com/2011/02/clip-raster-using-shapefile.html.
    
    Args:
        geo_t: A gdal geoMatrix (ds.GetGeoTransform().
        x: x coordinate in map units.
        y: y coordinate in map units.
        buffer_size: Optionally specify a buffer size. This is used when a buffer has been applied to extend all edges of an image, as in rasterizeShapfile().
    
    Returns:
        A tuple with pixel/line locations for each input coordinate.
    """
    ulX = geo_t[0]
    ulY = geo_t[3]
    xDist = geo_t[1]
    yDist = geo_t[5]
    
    pixel = int((x - ulX) / xDist)
    line = int((y - ulY) / yDist)
    
    return (pixel, line)




def rasterizeShapefile(shp, landcover, md):
    """
    Rasterize polygons from a shapefile to match a gdal raster.
        
    Args:
        shp: Path to a shapefile consisting of points, lines and/or polygons. This does not have to be in the same projection as ds
        md: A metadata file

    Returns:
        A numpy array with a boolean mask delineating locations inside (True) and outside (False) the shapefile [and optional buffer].
    """
    
    from osgeo import gdalnumeric
       
    # Create output image
    rasterPoly = Image.new("I", (md.ncols , md.nrows), 0)
    rasterize = ImageDraw.Draw(rasterPoly)
    
    # The shapefile may not have the same CRS as the data, so this will generate a function to reproject points.
    coordTransform = _coordinateTransformer(shp, md.EPSG_code)
    
    # Read shapefile
    sf = shapefile.Reader(shp) 
    
    # Get names of fields
    fields = sf.fields[1:] 
    field_names = [field[0] for field in fields] 
    
    # For each shape in shapefile...
    for r in sf.shapeRecords():
        
        atr = dict(zip(field_names, r.record))
        
        if atr['landcover'] != landcover:
            continue
        
        shape = r.shape
                
        # Get shape bounding box
        sxmin, symin, sxmax, symax = shape.bbox
        
        # Transform points
        sxmin, symin, z = coordTransform.TransformPoint(sxmin, symin)
        sxmax, symax, z = coordTransform.TransformPoint(sxmax, symax)
                
        # Go to the next record if out of bounds
        geo_t = md.geo_t
        if sxmax < geo_t[0]: continue
        if sxmin > geo_t[0] + (geo_t[1] * md.ncols): continue
        if symax < geo_t[3] + (geo_t[5] * md.nrows): continue
        if symin > geo_t[3]: continue
        
        #Separate polygons with list indices
        n_parts = len(shape.parts) #Number of parts
        indices = shape.parts #Get indices of shapefile part starts
        indices.append(len(shape.points)) #Add index of final vertex
        
        for part in range(n_parts):
            
            start_index = shape.parts[part]
            end_index = shape.parts[part+1]
            
            points = shape.points[start_index:end_index] #Map coordinates
            pixels = [] #Pixel coordinantes
            
            # Transform coordinates to pixel values
            for p in points:
                
                # First update points from shapefile projection to the new projection
                E, N, z = coordTransform.TransformPoint(p[0], p[1])

                # Then convert map to pixel coordinates using geo transform
                pixels.append(_world2Pixel(geo_t, E, N))

            # Draw the mask for this shape...
            # if a point...
            if shape.shapeType == 0:
                rasterize.point(pixels, 1)

            # a line...
            elif shape.shapeType == 3:
                rasterize.line(pixels, 1)
  
            # or a polygon.
            elif shape.shapeType == 5:  
                rasterize.polygon(pixels, 1)
    
    #Converts a Python Imaging Library array to a gdalnumeric image.
    mask = gdalnumeric.fromstring(rasterPoly.tobytes(),dtype=np.uint32)
    mask.shape = rasterPoly.im.size[1], rasterPoly.im.size[0]
        
    return mask


def loadLandcover(landcover_map, md_dest):
    '''
    Load a landcover map, and reproject it to the CRS defined in md.
    For test purposes only.
    '''
    
    from osgeo import osr
    
    # Load landcover map
    ds_source = gdal.Open(landcover_map,0)
    geo_t = ds_source.GetGeoTransform()
    proj = ds_source.GetProjection()

    # Get extent and resolution    
    nrows = ds_source.RasterXSize
    ncols = ds_source.RasterYSize
    ulx = float(geo_t[4])
    uly = float(geo_t[5])
    xres = float(geo_t[0])
    yres = float(geo_t[3])
    lrx = ulx + (xres * ncols)
    lry = uly + (yres * nrows)
    extent = [ulx, lry, lrx, uly]
    
    # Get EPSG
    srs = osr.SpatialReference(wkt = proj)
    srs.AutoIdentifyEPSG()
    EPSG = int(srs.GetAttrValue("AUTHORITY", 1))
    
    # Add source metadata to a dictionary
    md_source =sen2mosaic.utilities.Metadata(extent, xres, EPSG)
    
    # Build an empty destination dataset
    ds_dest = sen2mosaic.utilities.createGdalDataset(md_dest,dtype = 1)
    
    # And reproject landcover dataset to match input image
    landcover = sen2mosaic.utilities.reprojectImage(ds_source, ds_dest, md_source, md_dest)
    
    return np.squeeze(landcover)
    

def getPixels(indices, mask, subset = 5000):
    '''
    '''
        
    if indices.ndim == 2:
        indices = np.ma.expand_dims(indices, 2)
        
    sub = np.logical_and(mask, np.sum(indices.mask,axis=2)==0)
    indices_subset = indices[sub].data
    sub = np.zeros(indices_subset.shape[0], dtype = np.bool)
    sub[:subset] = True
    np.random.shuffle(sub)
    
    data_out = np.squeeze(indices_subset[sub,:]).tolist()
    
    if np.unique(np.array([len(i) for i in data_out])).shape[0] > 1:
        print '    Unexpected number of dimensions, skipping'
        data_out = []
    
    return data_out


def outputData(forest_px, nonforest_px, image_type, output_dir = os.getcwd()):
    """
    Save data to a .npz file for analysis in model fitting script.
    
    Args:
        forest_px: List of pixel values representing forest.
        nonforest_px: List of pixel values representing nonforest
        image_type: String to represent image type (e.g. S1single, S1dual, S2)
        output_dir: Directory to output .npz file. Defaults to current working directory.
    """
    
    forest_px = np.array(forest_px)
    nonforest_px = np.array(nonforest_px)
    
    np.savez('%s/%s_training_data.npz'%(output_dir, image_type), forest_px = forest_px, nonforest_px = nonforest_px) 



def _extractData(input_list):
    '''
    '''
    
    source_file = input_list[0]
    training_data = input_list[1]
    target_extent = input_list[2]
    resolution = input_list[3]
    EPSG_code = input_list[4]
    
    forest_key = [1]
    nonforest_key = [2, 3, 4, 7]
    
    print 'Doing %s'%source_file

    # Determine output extent and projection
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
        
    # Load and sort input scenes
    scene = classify.loadScenes([source_file], md = md_dest)[0]
    indices = classify.loadIndices(scene, md = md_dest)
    
    if training_data.split('.')[-1] == 'shp':
        forest_mask = rasterizeShapefile(training_data, 'forest', md_dest)
        nonforest_mask = rasterizeShapefile(training_data, 'nonforest', md_dest)
        
    elif training_data.split('.')[-1] == 'tif' or training_data.split('.')[-1] == 'tiff':
        landcover = loadLandcover(training_data, md_dest)
            
        forest_mask = np.in1d(landcover, np.array(forest_key)).reshape(landcover.shape)
        nonforest_mask = np.in1d(landcover, np.array(nonforest_key)).reshape(landcover.shape)

    
    forest = getPixels(indices, forest_mask)
    nonforest = getPixels(indices, nonforest_mask)
    
    return [forest, nonforest]

   

def main(source_files, training_data, target_extent, resolution, EPSG_code, output_dir = os.getcwd()):
    """
    """
        
    # Determine output extent and projection
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    
    # Load and sort input scenes
    scenes = classify.loadScenes(source_files[::2], md = md_dest, sort = True)
    
    import multiprocessing
        
    # Get pixel values
    instances = multiprocessing.Pool(20)
    pixel_values = instances.map(_extractData, [[scene.filename, training_data, target_extent, resolution, EPSG_code] for scene in scenes])
    instances.close()
    
    #forest_px, forest_image_type, nonforest_px, nonforest_image_type = [], [], [], []
    px = {}
    for x in ['forest','nonforest']:
        px[x] = {}
        for y in ['S1single','S1dual','S2']:
            px[x][y] = []
    
    for scene, pixels in zip(scenes,pixel_values):
        px['forest'][scene.image_type] += pixels[0]
        px['nonforest'][scene.image_type] += pixels[1]
        
    # This currently only outputs S2
    outputData(px['forest'][scene.image_type], px['nonforest']['S2'], 'S2', output_dir = output_dir)
    

if __name__ == '__main__':
    
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Extract data from Sentinel-1 and Sentinel-2 data to train logistic regression functions.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('-t', '--training_data', metavar = 'SHP/TIF', type = str, help = 'Path to training data geotiff/shapefile.')
    #required.add_argument('-t', '--image_type', metavar = 'TYPE', type = str, help = 'Image type to train (S1single, S1dual, or S2')
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
    main(infiles_S2 + infiles_S1, args.training_data, args.target_extent, args.resolution, args.epsg, output_dir = args.output_dir)
    
    # /anaconda2/bin/python ~/DATA/deforest/deforest/extract_training_data.py ../chimanimani/L2_files/S2/ -r 60 -e 32736 -te 450000 7790200 550000 7900000 -s /home/sbowers3/SMFM/chimanimani/training_areas/training_areas.shp
    # ~/anaconda2/bin/python ~/DATA/deforest/deforest/extract_training_data.py ../chimanimani/L2_files/S2/ -r 60 -e 32736 -te 499980 7790200 609780 7900000 -s /home/sbowers3/SMFM/chimanimani/training_areas/training_areas.shp
    # ~/anaconda2/bin/python ~/DATA/deforest/deforest/extract_training_data.py ../chimanimani/L2_files/S2/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t ~/SMFM/landcover/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif
    