# Script to extract training pixels using a shapefule to parameterise forest and nonforest PDFs

import numpy as np
from osgeo import gdal, ogr, osr, gdalnumeric
import glob
import shapefile
import datetime as dt
from PIL import Image, ImageDraw
import pdb
import matplotlib.pyplot as plt
import scipy.stats

def _coordinateTransformer(shp, EPSG_out):
    """
    Generates function to transform coordinates from a source shapefile CRS to EPSG.
    
    Args:
        shp: Path to a shapefile.
    
    Returns:
        A function that transforms shapefile points to EPSG.
    """
    
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




def rasterizeShapefile(ds, shp, landcover):
    """
    Rasterize polygons from a shapefile to match a gal raster
        
    Args:
        ds: A gdal file (gdal.Open())
        shp: Path to a shapefile consisting of points, lines and/or polygons. This does not have to be in the same projection as ds

    Returns:
        A numpy array with a boolean mask delineating locations inside (True) and outside (False) the shapefile [and optional buffer].
    """
    
    
    # Create output image
    rasterPoly = Image.new("I", (ds.RasterYSize , ds.RasterXSize), 0)
    rasterize = ImageDraw.Draw(rasterPoly)
    
    # The shapefile may not have the same CRS as the data, so this will generate a function to reproject points.
    coordTransform = _coordinateTransformer(shp, 32736)
    
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
        geo_t = ds.GetGeoTransform()
        if sxmax < geo_t[0]: continue
        if sxmin > geo_t[0] + (geo_t[1] * ds.RasterYSize): continue
        if symax < geo_t[3] + (geo_t[5] * ds.RasterXSize): continue
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
                
                # First update points from shapefile projection to ALOS mosaic projection
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




##############
## The code ##
##############


data_dir = '/home/sbowers3/DATA/chimanimani/test_data/'
shp = '/home/sbowers3/Documents/SMFM/chimanimani/training_areas.shp'

# Load each image in turn, and calculate the probability of forest (from a start point of everything being forest)

data_files = glob.glob(data_dir + 'chimanimaniGlobal*S2*_data.tif')
data_files.sort(key = lambda x: x.split('_')[3])
data_files = np.array(data_files)

mask_files = glob.glob(data_dir + 'chimanimaniGlobal*S2*_mask.tif')
mask_files.sort(key = lambda x: x.split('_')[3])
mask_files = np.array(mask_files)

datestrings = [x.split('/')[-1].split('_')[3] for x in data_files]
dates = np.array([dt.date(int(x[:4]), int(x[4:6]), int(x[6:])) for x in datestrings])

sensors = np.array([x.split('/')[-1].split('_')[1] for x in data_files])


forest = []
nonforest = []


# Get unique dates
for data_file, mask_file, date, sensor in zip(data_files, mask_files, dates, sensors):
    
    # Load file    
    print 'Loading %s'%data_file
    data = gdal.Open(data_file,0).ReadAsArray()
    mask = gdal.Open(mask_file,0).ReadAsArray()
    
    
    # Get pixels of land cover type
    pixels = rasterizeShapefile(gdal.Open(data_files[0]), shp, 'forest')
    forest += list(data[np.logical_and(pixels, mask == False)])
    
    pixels = rasterizeShapefile(gdal.Open(data_files[0]), shp, 'woodland')
    forest += list(data[np.logical_and(pixels, mask == False)])

    pixels = rasterizeShapefile(gdal.Open(data_files[0]), shp, 'agriculture')
    nonforest += list(data[np.logical_and(pixels, mask == False)])
    
    pixels = rasterizeShapefile(gdal.Open(data_files[0]), shp, 'scrubland')
    nonforest += list(data[np.logical_and(pixels, mask == False)])
    
    pixels = rasterizeShapefile(gdal.Open(data_files[0]), shp, 'smallholders')
    nonforest += list(data[np.logical_and(pixels, mask == False)])   
    

x = np.linspace(-10, 4.5, 1000)
#x = np.linspace(-1, 1, 1000)

plt.hist(forest,normed=True, bins = 25, alpha = 0.5, label = 'Forest', color = 'green')
forest_mu, forest_std = scipy.stats.norm.fit(forest)
p = scipy.stats.norm.pdf(x, forest_mu, forest_std)
plt.plot(x, p, 'green', linewidth=2)

plt.hist(nonforest,normed=True, bins =25, alpha = 0.5, label = 'Nonforest', color = 'orange')
nonforest_mu, nonforest_std = scipy.stats.norm.fit(nonforest)
p = scipy.stats.norm.pdf(x, nonforest_mu, nonforest_std)
plt.plot(x, p, 'orange', linewidth=2)

plt.legend()
plt.show()

