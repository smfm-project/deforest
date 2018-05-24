# This is a second attempt at producing the training script.
# It works by importing functions from classify.py, and outputting a config file to inform classify.py

import argparse
import classify
import csv
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw
import shapefile
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, '/exports/csce/datastore/geos/users/sbowers3/sen2mosaic')

import sen2mosaic.utilities

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


def main(source_files, shp, normalisation_type = 'none', normalisation_percentile = 95, normalisation_area = 200000, output_dir = os.getcwd()):
    """
    """
    
    from osgeo import gdal
    
    # Slim down input files to only those that fall within shapefile #TODO
    # md_dest = buildMetadataDictionary(args.target_extent, args.resolution, args.epsg)
    # infiles = getFilesInTile(args.infiles, md_dest)
    
    # Slim down input files to only those of given image type
    #infiles = getFilesOfType(infiles, image_type)
    
    scenes = [sen2mosaic.utilities.LoadScene(source_file, resolution=60) for source_file in source_files]
    scenes = sen2mosaic.utilities.sortScenes(scenes, by = 'date')
    
    forest_px = []
    nonforest_px = []
    
    if normalisation_type == 'match':
        reference_scene, inds = classify.findMatch(scenes, 7)
    
    for n, scene in enumerate(scenes):
        
        print 'Reading file %s'%scene.filename.split('/')[-1]
                
        # Load scene
        if normalisation_type == 'match':
            indices = classify.loadS2(scene, normalisation_type = normalisation_type, reference_scene = reference_scene[inds[n]])
        
        else:
            indices = classify.loadS2(scene, normalisation_type = normalisation_type, normalisation_percentile = normalisation_percentile, normalisation_area = normalisation_area)
                
        # Where only one predictor
        if indices.ndim == 2:
            indices = indices[:,:,np.newaxis]
        
        # Get areas within shapefile
        forest_mask = rasterizeShapefile(shp, 'forest', scene.metadata)
        nonforest_mask = rasterizeShapefile(shp, 'nonforest', scene.metadata)
        
        ### Landcover mask test
        # Get forest and nonforest pixels from landcover map
        #landcover = classify.loadLandcover('/home/sbowers3/SMFM/DATA/landcover/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif', md_source)
        #pdb.set_trace()
        # These are the values that represent forest. In the test map, it's only 1.
        #forest_key = np.array([1])
        #forest_mask = np.in1d(landcover, forest_key).reshape(landcover.shape)
        #nonforest_key = np.array([2, 3, 4, 7]) # Those landcovers to which forest could feasible change into
        #nonforest_mask = np.in1d(landcover, nonforest_key).reshape(landcover.shape)
        
        # Get nonforest pixels within 5 pixels of a forest. TODO: specify minimum extent of forest
        #from scipy import ndimage
        #nonforest_mask = np.logical_and(nonforest_mask,ndimage.morphology.binary_dilation(forest_mask.astype(np.int), iterations = 5))
        ####
        #pdb.set_trace() 
        # Hack date out of output_filename. TODO: Return date from loadData?
        #date = output_filename.split('/')[-1].split('.')[0].split('_')[-2]
        #date = dt.datetime.strptime(date, '%Y%m%d').date()
        
        # Extract data for forest training pixels
        sub = np.logical_and(forest_mask==1, np.sum(indices.mask,axis=2)==0)
        data_subset = indices[sub].data
        sub = np.zeros(data_subset.shape[0], dtype = np.bool)
        sub[:5000] = True
        np.random.shuffle(sub)
        
        forest_px.extend(data_subset[sub,:].tolist())
                
        # Extract data for nonforest training pixels
        sub = np.logical_and(nonforest_mask==1, np.sum(indices.mask,axis=2)==0)
        data_subset = indices[sub].data
        sub = np.zeros(data_subset.shape[0], dtype = np.bool)
        sub[:5000] = True
        np.random.shuffle(sub)
        
        nonforest_px.extend(data_subset[sub,:].tolist())
        
        # Output data (as we go)
        outputData(forest_px, nonforest_px, 'S2', output_dir = output_dir)


if __name__ == '__main__':
    
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Extract data from Sentinel-1 and Sentinel-2 data to train logistic regression functions.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('-s', '--shapefile', metavar = 'SHP', type = str, help = 'Path to training data shapefile.')
    #required.add_argument('-t', '--image_type', metavar = 'TYPE', type = str, help = 'Image type to train (S1single, S1dual, or S2')
    
    # Optional arguments
    optional.add_argument('infiles', metavar = 'L2A_FILES', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files (level 2A) in .SAFE format. Specify one or more valid Sentinel-2 .SAFE, a directory containing .SAFE files, or multiple granules through wildcards (e.g. *.SAFE/GRANULE/*). Defaults to processing all granules in current working directory.')
    optional.add_argument('-nt', '--normalisation_type', type=str, metavar = 'STR', default = 'none', help="Normalisation type. Set to one of 'none', 'local', 'global', or 'match'. Defaults to 'none'.")
    optional.add_argument('-np', '--normalisation_percentile', type=int, metavar = 'N', default = 95, help="Normalisation percentile, in case where normalisation type set to  'local' or 'global'. Defaults to 95 percent.")
    optional.add_argument('-na', '--normalisation_area', type=float, metavar = 'N', default = 200000., help="Normalisation area. Defaults to 200000 m^2.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'PATH', default = os.getcwd(), help="Directory to output training data.")

    # Get arguments
    args = parser.parse_args()

    # Get absolute path of input .safe files.
    infiles = [os.path.abspath(i) for i in args.infiles]
    
    # Find files from input directory/granule etc.
    infiles = sen2mosaic.utilities.prepInfiles(infiles, '2A')
    
    # Execute script
    main(infiles, args.shapefile, normalisation_type = args.normalisation_type, normalisation_percentile = args.normalisation_percentile, output_dir = args.output_dir)
