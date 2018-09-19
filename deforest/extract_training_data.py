import argparse
import csv
import datetime as dt
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from osgeo import gdal
import random
import shapefile

import sys
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen2mosaic')
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen1mosaic')
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/deforest/deforest')

import sen1mosaic.utilities
import sen2mosaic.utilities
import classify

import pdb


def loadShapefile(shp, md_dest, attribute = '', attribute_value = ''):
    """
    Rasterize polygons from a shapefile to match a specified CRS.
        
    Args:
        shp: Path to a shapefile consisting of points, lines and/or polygons. This does not have to be in the same projection as ds.
        md_dest: A metadata file from sen2mosaic.utilities.Metadata().
        attribute: Attribute name to include as part of mask. Defaults to all. If specifying an attribute, you must also specify an attribute_value.
        attribute_value: Attribute value (from attribute 'attribute_name') to include in the mask. Defaults to all. If specifying an attribute_value, you must also specify an attribute.
        
    Returns:
        A numpy array with a boolean mask delineating locations inside (True) and outside (False) the shapefile given attribute and attribute_value.
    """
    
    if attribute != '' or attribute_value != '':
        assert attribute != '' and attribute_value != '', "Both  `attribute` and `attribute_value` must be specified."
        
    from PIL import Image, ImageDraw
    from osgeo import gdalnumeric
    
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
    
    # Create output image
    rasterPoly = Image.new("I", (md_dest.ncols , md_dest.nrows), 0)
    rasterize = ImageDraw.Draw(rasterPoly)
    
    # The shapefile may not have the same CRS as the data, so this will generate a function to reproject points.
    coordTransform = _coordinateTransformer(shp, md_dest.EPSG_code)
    
    # Read shapefile
    sf = shapefile.Reader(shp) 
    
    # Get names of fields
    fields = sf.fields[1:] 
    field_names = [field[0] for field in fields] 
    
    # For each shape in shapefile...
    for r in sf.shapeRecords():
        
        atr = dict(zip(field_names, r.record))
        
        if attribute_value != '' and atr[attribute] != attribute_value:
            continue
        
        shape = r.shape
                
        # Get shape bounding box
        sxmin, symin, sxmax, symax = shape.bbox
        
        # Transform points
        sxmin, symin, z = coordTransform.TransformPoint(sxmin, symin)
        sxmax, symax, z = coordTransform.TransformPoint(sxmax, symax)
                
        # Go to the next record if out of bounds
        geo_t = md_dest.geo_t
        if sxmax < geo_t[0]: continue
        if sxmin > geo_t[0] + (geo_t[1] * md_dest.ncols): continue
        if symax < geo_t[3] + (geo_t[5] * md_dest.nrows): continue
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


def loadRaster(raster, md_dest, classes):
    '''
    Build a mask from a landcover map indicating specified land cover classes.
    
    Args:
        raster: Path to a GeoTiff or .vrt file.
        md_dest: A metadata file from sen2mosaic.utilities.Metadata().
        classes: Pixel values to use as part of the mask
    
    Returns:
        A numpy array with a boolean mask delineating locations inside (True) and outside (False) the chosen classes.
    '''
    
    from osgeo import osr
    
    # Load landcover map
    ds_source = gdal.Open(raster,0)
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
    md_source = sen2mosaic.utilities.Metadata(extent, xres, EPSG)
    
    # Build an empty destination dataset
    ds_dest = sen2mosaic.utilities.createGdalDataset(md_dest,dtype = 1)
    
    # And reproject landcover dataset to match input image
    landcover = np.squeeze(sen2mosaic.utilities.reprojectImage(ds_source, ds_dest, md_source, md_dest))
    
    # Select matching values, and reshape to mask
    mask = np.in1d(landcover, np.array(classes)).reshape(landcover.shape)
    
    return mask
    

def _getPixels(indices, mask, subset = 5000):
    '''
    Extract a sample of pixels from a 2/3d array with a 2d mask.
    
    Args:
        indices: Numpy array of forest/nonforest predictors from classify.loadIndices()
        mask: A boolean array of the same 2d extent as indices
        subset: Maximum number of pixels to extract for each class from each image
        
    Returns:
        A list containing a sample of pixel values for use as training data.
        
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


def outputData(forest_px, nonforest_px, output_name = 'S2', output_dir = os.getcwd()):
    """
    Save data to a .npz file for analysis in model fitting script.
    
    Args:
        forest_px: List of pixel values representing forest.
        nonforest_px: List of pixel values representing nonforest
        image_name: String to represent image type (e.g. S1single, S1dual, S2). Defaults to 'S2'.
        output_dir: Directory to output .npz file. Defaults to current working directory.
    """
    
    forest_px = np.array(forest_px)
    nonforest_px = np.array(nonforest_px)
    
    np.savez('%s/%s_training_data.npz'%(output_dir, output_name), forest_px = forest_px, nonforest_px = nonforest_px) 
    
    print 'Done!'



def _extractData(input_list):
    '''
    Multiprocessing requires some gymnastics. This is a wrapper function to initiate extractData() for multiprocessing.
    
    Args:
        input_list: A list of inputs for a single source_file, in the format: [source_file. trainging_data, target_extent, resolution, EPSG_code, subset].
    
    Returns:
        A tuple with (a list of forest pixel values, a list of nonforest pixel values)
    '''
    
    source_file = input_list[0]
    training_data = input_list[1]
    target_extent = input_list[2]
    resolution = input_list[3]
    EPSG_code = input_list[4]
    subset = input_list[5]
    
    # Load input scene
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    scene = classify.loadScenes([source_file], md = md_dest)
       
    return extractData(scene, training_data, md_dest, subset = subset)


def _unpackOutputs(outputs):
    '''
    Multiprocessing requires some gymnastics. This is a wrapper function to unpack data from multiprocessing outputs.
    
    Args:
        outputs: outputs from multiprocessing.Pool.map
    
    Returns:
        A tuple with (a list of forest pixel values, a list of nonforest pixel values)
    '''
    
    forest_px, nonforest_px = [], []
    
    for f, nf in outputs:
        forest_px.append(f)
        nonforest_px.append(nf)
    
    return forest_px, nonforest_px
    

def extractData(scenes, training_data, md_dest, subset = 5000):
    '''
    Extract pixel values from a list of scenes.
    
    Args:
        scenes: A list of scenes of type classify.loadScenes()
        training_data: A GeoTiff, .vrt of .shp file containing training pixels/polygons.
        md_dest: A metadata file from sen2mosaic.utilities.Metadata().
        subset: Maximum number of pixels to extract for each class from each image
    
    Returns:
        A tuple with (a list of forest pixel values, a list of nonforest pixel values)
    '''
       
    forest_key = [1]
    nonforest_key = [2, 3, 4, 7]
    
    forest, nonforest = [], []
    
    for scene in scenes:
        
        print 'Doing %s'%scene.filename
                        
        # Get indices
        try:
            indices = classify.loadIndices(scene, md = md_dest)
        except:
            print 'Missing data, continuing'
        
        if training_data.split('.')[-1] == 'shp':
            forest_mask = loadShapefile(training_data, md_dest, attribute = 'landcover', attribute_value = 'forest')
            nonforest_mask = loadShapefile(training_data, md_dest, attribute = 'landcover', attribute_value = 'nonforest')
            
        elif training_data.split('.')[-1] in ['tif', 'tiff', 'vrt']:
            forest_mask = loadRaster(training_data, md_dest, forest_key)
            nonforest_mask = loadRaster(training_data, md_dest, nonforest_key)
        
        # Get random subset of pixels
        forest.append(_getPixels(indices, forest_mask, subset = subset))
        nonforest.append(_getPixels(indices, nonforest_mask, subset = subset))
    
    return forest, nonforest


def main(source_files, training_data, target_extent, resolution, EPSG_code, n_processes = 1, max_pixels = 5000, output_dir = os.getcwd(), output_name = 'S2'):
    '''main(source_files, training_data, target_extent, resolution, EPSG_code, n_processes = 1, max_pixels = 5000, output_dir = os.getcwd(), output_name = 'S2')
    
    Extract pixel values from source_files and output as a np.savez() file. This is the function that is initiated from the command line.
    
    Args:
        source_files: A list of directories for Sentinel-2 input tiles. 
        training_data: A GeoTiff, .vrt of .shp file containing training pixels/polygons.
        target_extent: Extent of search area, in format [xmin, ymin, xmax, ymax]
        resolution: Resolution to re-sample search area, in meters. Best to be 10 m, 20 m or 60 m to match Sentinel-2 resolution.
        EPSG_code: EPSG code of search area.
        n_processes: Number of processes, defaults to 1.
        max_pixels: Maximum number of pixels to extract for each class from each image. Defaults to 5000.
        output_dir: Directory to output classifier predictors. Defaults to current working directory.
        output_name: Name to precede output file. Defaults to 'S2'.
        
    '''
    
    assert type(n_processes) == int and n_processes > 0, "n_processes must be an integer > 0."
        
    # Load and sort input scenes
    md_dest = sen2mosaic.utilities.Metadata(target_extent, resolution, EPSG_code)
    scenes = classify.loadScenes(source_files, md = md_dest, sort = True)
    
    # Extract pixel values
    if n_processes == 1:
        forest_px, nonforest_px = extractData(scenes, training_data, md_dest, subset = max_pixels)
    
    # Extract pixels by multi-processing
    else:
        instances = multiprocessing.Pool(n_processes)
        forest_px, nonforest_px = _unpackOutputs(instances.map(_extractData, [[scene.filename, training_data, target_extent, resolution, EPSG_code, max_pixels] for scene in scenes]))
        instances.close()
        
    # Output data (currently only outputs S2)
    outputData(forest_px, nonforest_px, output_name = output_name, output_dir = output_dir)
    

if __name__ == '__main__':
    '''
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Extract indices from Sentinel-2 data to train a classifier of forest cover. Returns a numpy .npz file containing pixel values for forest/nonforest.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('-te', '--target_extent', nargs = 4, metavar = ('XMIN', 'YMIN', 'XMAX', 'YMAX'), type = float, help = "Extent of output image tile, in format <xmin, ymin, xmax, ymax>.")
    required.add_argument('-e', '--epsg', type=int, help="EPSG code for output image tile CRS. This must be UTM. Find the EPSG code of your output CRS as https://www.epsg-registry.org/.")
    required.add_argument('-res', '--resolution', metavar = 'N', type=int, help = "Specify a resolution to output.")
    required.add_argument('-t', '--training_data', metavar = 'SHP/TIF', type = str, help = 'Path to training data geotiff/shapefile.')
        
    # Optional arguments
    optional.add_argument('infiles', metavar = 'FILES', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files (level 2A) in .SAFE format. Specify one or more valid Sentinel-2 .SAFE, a directory containing .SAFE files, multiple tiles through wildcards (e.g. *.SAFE/GRANULE/*), or a text file listing files. Defaults to processing all tiles in current working directory.')
    optional.add_argument('-p', '--n_processes', type = int, metavar = 'N', default = 1, help = "Specify a maximum number of tiles to process in paralell. Bear in mind that more processes will require more memory. Defaults to 1.")
    optional.add_argument('-mi', '--max_images', type = int, metavar = 'N', default = 0, help = "Specify a maximum number of input tiles to extract data from. Defaults to all valid tiles.")
    optional.add_argument('-mp', '--max_pixels', type = int, metavar = 'N', default = 5000, help = "Specify a maximum number of pixels to extract from each image per class. Defaults to 5000.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Specify an output directory. Defaults to current working directory.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'S2', help="Specify a string to precede output filename. Defaults to 'S2'.")
    
    # Get arguments
    args = parser.parse_args()
    
    # Get absolute path of input .safe files.
    infiles = [os.path.abspath(i) for i in args.infiles]
    
    # Find files from input directory/granule etc.
    infiles = sen2mosaic.utilities.prepInfiles(infiles, '2A')
    #infiles_S1 = sen1mosaic.utilities.prepInfiles(infiles)
    
    # Reduce number of inputs to max_images
    if args.max_images > 0 and len(infiles) > args.max_images:
        infiles =  [infiles[i] for i in sorted(random.sample(range(len(infiles)), args.max_images))]
    
    # Execute script
    main(infiles, args.training_data, args.target_extent, args.resolution, args.epsg, n_processes = args.n_processes, max_pixels = args.max_pixels, output_dir = args.output_dir, output_name = args.output_name)
    
    # Example:
    # ~/anaconda2/bin/python ~/DATA/deforest/deforest/extract_training_data.py ../chimanimani/L2_files/S2/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t ~/SMFM/landcover/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif -o ./ -mi 100 -p 20
    