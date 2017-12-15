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


data_dir = '/home/sbowers3/scratch/chimanimani/L3_files/'
shp = '/home/sbowers3/Documents/chimanimani/training_areas/training_areas.shp'

# Load each image in turn, and calculate the probability of forest (from a start point of everything being forest)

data_files = glob.glob(data_dir + 'chimanimaniGlobal*_data.tif')
data_files.sort(key = lambda x: x.split('_')[4])
data_files = np.array(data_files)

mask_files = np.array([i[:-8] + 'mask.tif' for i in data_files])

datestrings = [x.split('/')[-1].split('_')[3] for x in data_files]
dates = np.array([dt.date(int(x[:4]), int(x[4:6]), int(x[6:])) for x in datestrings])

sensors = np.array([x.split('/')[-1].split('_')[1] for x in data_files])

pols = np.array([x.split('/')[-1].split('_')[2] for x in data_files])


forest_S2 = []
nonforest_S2 = []
forest_S1_dual_VV, forest_S1_dual_VH, nonforest_S1_dual_VV, nonforest_S1_dual_VH = [], [], [], []
forest_S1_VV, nonforest_S1_VV = [], []


forest_landcovers = ['forest','woodland']
nonforest_landcovers = ['agriculture','scrub','smallholders']

# Get unique dates
for date in np.unique(dates):
     
    if date >= dt.date(2017,1,1):
        continue
    
    if 'S2' in sensors[dates == date]:
        
        s = np.logical_and(dates == date, sensors == 'S2')

        for data_file, mask_file in zip(data_files[s], mask_files[s]):        
             
            # Load file    
            print 'Loading %s'%data_file
            data = gdal.Open(data_file,0).ReadAsArray()
            mask = gdal.Open(mask_file,0).ReadAsArray()
        
            for landcover in forest_landcovers:

                # Get pixels of land cover type
                pixels = rasterizeShapefile(gdal.Open(data_file), shp, landcover)
                forest_S2 += list(data[np.logical_and(pixels, mask == False)])

            for landcover in nonforest_landcovers:
            
                pixels = rasterizeShapefile(gdal.Open(data_file), shp, landcover)
                nonforest_S2 += list(data[np.logical_and(pixels, mask == False)])

    if 'S1' in sensors[dates == date]:
        
        if ('VV' in pols[dates == date]) and ('VH' in pols[dates == date]):
            
            s = np.logical_and(dates == date, sensors == 'S1')
        
            for data_file, mask_file, pol in zip(data_files[s], mask_files[s], pols[s]):
                
                # Load file    
                print 'Loading %s'%data_file
                data = gdal.Open(data_file,0).ReadAsArray()
                mask = gdal.Open(mask_file,0).ReadAsArray()
                
                for landcover in forest_landcovers:

                    # Get pixels of land cover type
                    if pol == 'VH':
                       pixels = rasterizeShapefile(gdal.Open(data_file), shp, landcover)
                       forest_S1_dual_VH += list(data[np.logical_and(pixels, mask == False)])
                    else:
                       pixels = rasterizeShapefile(gdal.Open(data_file), shp, landcover)
                       forest_S1_dual_VV += list(data[np.logical_and(pixels, mask == False)])

                for landcover in nonforest_landcovers:

                    # Get pixels of land cover type
                    if pol == 'VH':
                       pixels = rasterizeShapefile(gdal.Open(data_files[0]), shp, landcover)
                       nonforest_S1_dual_VH += list(data[np.logical_and(pixels, mask == False)])
                    else:
                       pixels = rasterizeShapefile(gdal.Open(data_files[0]), shp, landcover)
                       nonforest_S1_dual_VV += list(data[np.logical_and(pixels, mask == False)])

        else:
            
            s = np.logical_and(dates == date, sensors == 'S1')
            
            for data_file, mask_file, pol in zip(data_files[s], mask_files[s], pols[s]):
                
                # Load file    
                print 'Loading %s'%data_file
                data = gdal.Open(data_file,0).ReadAsArray()
                mask = gdal.Open(mask_file,0).ReadAsArray()
        
                for landcover in forest_landcovers:        
                    # Get pixels of land cover type
                    pixels = rasterizeShapefile(gdal.Open(data_file), shp, landcover)
                    forest_S1_VV += list(data[np.logical_and(pixels, mask == False)])

                for landcover in nonforest_landcovers:

                    pixels = rasterizeShapefile(gdal.Open(data_files[0]), shp, landcover)
                    nonforest_S1_VV += list(data[np.logical_and(pixels, mask == False)])



            
###########
# Plot S2 #
###########

x = np.linspace(-1, 1, 1000)

plt.hist(forest_S2,normed=True, bins = 25, alpha = 0.5, label = 'Forest', color = 'green')
forest_mu, forest_std = scipy.stats.norm.fit(forest_S2)
p = scipy.stats.norm.pdf(x, forest_mu, forest_std)
plt.plot(x, p, 'green', linewidth=2)

plt.hist(nonforest_S2,normed=True, bins = 25, alpha = 0.5, label = 'Nonforest', color = 'orange')
nonforest_mu, nonforest_std = scipy.stats.norm.fit(nonforest_S2)
p = scipy.stats.norm.pdf(x, nonforest_mu, nonforest_std)
plt.plot(x, p, 'orange', linewidth=2)

plt.legend()
#plt.show()


# Alternatively, we might be able to use logistic regression. This means multiple features per image is possible, which might well improve things

# balanced classes
logistic = LogisticRegression(class_weight='balanced')
y = np.array(([1] * len(forest_S2)) + ([0] * len(nonforest_S2)))
X = np.array(forest_S2 + nonforest_S2)[:, np.newaxis]
logistic.fit(X,y)
plt.plot(x, logistic.predict_proba(x[:, np.newaxis])[:,0], 'red', linewidth=2)


plt.show()


###########
# Plot S1 #
###########

x = np.linspace(-10, 10, 1000)

fig, ax1 = plt.subplots()

ax1.hist(forest_S1_VV,normed=True, bins = 25, alpha = 0.5, label = 'Forest', color = 'green')
forest_mu, forest_std = scipy.stats.norm.fit(forest_S1_VV)
p = scipy.stats.norm.pdf(x, forest_mu, forest_std)
ax1.plot(x, p, 'green', linewidth=2)

ax1.hist(nonforest_S1_VV,normed=True, bins = 25, alpha = 0.5, label = 'Nonforest', color = 'orange')
nonforest_mu, nonforest_std = scipy.stats.norm.fit(nonforest_S1_VV)
p = scipy.stats.norm.pdf(x, nonforest_mu, nonforest_std)
ax1.plot(x, p, 'orange', linewidth=2)

plt.legend()
#plt.show()

ax2 = ax1.twinx()

# Alternatively, we might be able to use logistic regression. This means multiple features per image is possible, which might well improve things

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(class_weight='balanced')

y = np.array(([1] * len(forest_S1_VV)) + ([0] * len(nonforest_S1_VV)))
X = np.array(forest_S1_VV + nonforest_S1_VV)[:, np.newaxis]

logistic.fit(X,y)

ax2.plot(x, logistic.predict_proba(x[:, np.newaxis])[:,0], 'blue', linewidth=2)

import scipy.stats
# Compare this to the result from probability:
PF = scipy.stats.norm.pdf(x, np.mean(forest_S1_VV), np.std(forest_S1_VV))
PNF = scipy.stats.norm.pdf(x, np.mean(nonforest_S1_VV), np.std(nonforest_S1_VV))
PNF[PNF < 1E-10000] = 0
PNF[PNF > 0] = (PNF[PNF > 0] / (PF[PNF > 0] + PNF[PNF > 0]))

ax2.plot(x, PNF, 'red', linewidth = 2)

plt.show()





###################
# Plot S1 (HH/HV) #
###################

x = np.linspace(-10, 10, 1000)

fig, ax1 = plt.subplots()

ax1.hist(forest_S1_dual_VV,normed=True, bins = 25, alpha = 0.5, label = 'Forest', color = 'darkgreen')
forest_mu, forest_std = scipy.stats.norm.fit(forest_S1_VV)
p = scipy.stats.norm.pdf(x, forest_mu, forest_std)
ax1.plot(x, p, 'green', linewidth=2)

ax1.hist(nonforest_S1_dual_VV,normed=True, bins = 25, alpha = 0.5, label = 'Nonforest', color = 'darkorange')
nonforest_mu, nonforest_std = scipy.stats.norm.fit(nonforest_S1_VV)
p = scipy.stats.norm.pdf(x, nonforest_mu, nonforest_std)
ax1.plot(x, p, 'orange', linewidth=2)


ax1.hist(forest_S1_dual_VH,normed=True, bins = 25, alpha = 0.5, label = 'Forest', color = 'blue')
forest_mu, forest_std = scipy.stats.norm.fit(forest_S1_dual_VV)
p = scipy.stats.norm.pdf(x, forest_mu, forest_std)
ax1.plot(x, p, 'blue', linewidth=2)

ax1.hist(nonforest_S1_dual_VH,normed=True, bins = 25, alpha = 0.5, label = 'Nonforest', color = 'red')
nonforest_mu, nonforest_std = scipy.stats.norm.fit(nonforest_S1_dual_VH)
p = scipy.stats.norm.pdf(x, nonforest_mu, nonforest_std)
ax1.plot(x, p, 'red', linewidth=2)


plt.legend()
#plt.show() 

ax2 = ax1.twinx()

# Alternatively, we might be able to use logistic regression. This means multiple features per image is possible, which might well improve things

logistic = LogisticRegression(class_weight='balanced')

# Use every 17th forest measurement to equalise sample sizes
y = np.array(([1] * len(forest_S1_dual_VV)) + ([0] * len(nonforest_S1_dual_VV)))
X = np.transpose(np.array([forest_S1_dual_VV + nonforest_S1_dual_VV, forest_S1_dual_VH + nonforest_S1_dual_VH]))
#X = np.hstack((X,(X[:,0]/X[:,1])[:,None]))

logistic.fit(X,y)

ax2.plot(x, logistic.predict_proba(np.transpose(np.vstack((x,np.zeros_like(x)-2))))[:,0], 'blue', linewidth=2)

import scipy.stats
# Compare this to the result from probability:
PF = scipy.stats.norm.pdf(x, np.mean(forest_S1_VV), np.std(forest_S1_VV))
PNF = scipy.stats.norm.pdf(x, np.mean(nonforest_S1_VV), np.std(nonforest_S1_VV))
PNF[PNF < 1E-10000] = 0
PNF[PNF > 0] = (PNF[PNF > 0] / (PF[PNF > 0] + PNF[PNF > 0]))

ax2.plot(x, PNF, 'red', linewidth = 2)

plt.show()

for i in range(-10,10,1):
   plt.plot(x, logistic.predict_proba(np.transpose(np.vstack((x,np.zeros_like(x)+i))))[:,0], 'blue', linewidth=2)

# VV only
logistic = LogisticRegression(class_weight='balanced')
y = np.array(([1] * len(forest_S1_VV)) + ([0] * len(nonforest_S1_VV)))
X = np.array(forest_S1_VV + nonforest_S1_VV)[:, np.newaxis]

logistic.fit(X,y)

plt.plot(x, logistic.predict_proba(x[:,np.newaxis])[:,0], 'red', linewidth=2)

logodds = (logistic.coef_[0][0]*(x[:,np.newaxis]))+logistic.intercept_[0]

plt.plot(x, 1-np.exp(logodds)/(1+np.exp(logodds)), 'darkred', linewidth=2)

plt.show()


"""
# Plot variation
def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

mu_list = []
std_list = []

cmap = plt.cm.get_cmap('Spectral')
for m in range(12):
    mu, std = scipy.stats.norm.fit(forest[m])
    mu_list.append(mu)
    std_list.append(std)
    p = scipy.stats.norm.pdf(x, mu, std)
    RGB = cmap(float(m)/11.)
    plt.plot(x, p, rgb2hex(RGB[0],RGB[1],RGB[2]), linewidth=2)

plt.show()
"""
