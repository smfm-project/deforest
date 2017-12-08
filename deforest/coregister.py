import numpy as np
import cv2 #pip install opencv-python
from matplotlib import pyplot as plt
from osgeo import gdal


from skimage import data
from skimage import transform as tf
from skimage.feature import CENSURE

reference_image = '/home/sbowers3/DATA/gorongosa/gorongosaGlobal_S2_20_20170916_L2A_T36KXE_A011672_20170916T075748_data.tif'
reference_mask = '/home/sbowers3/DATA/gorongosa/gorongosaGlobal_S2_20_20170916_L2A_T36KXE_A011672_20170916T075748_mask.tif'

target_image = '/home/sbowers3/DATA/gorongosa/gorongosaGlobal_S1_VV_20141005_161532_161622_002696_003033_data.tif'
target_mask = '/home/sbowers3/DATA/gorongosa/gorongosaGlobal_S1_VV_20141005_161532_161622_002696_003033_mask.tif'


ref_data = gdal.Open(reference_image).ReadAsArray()
ref_mask = gdal.Open(reference_mask).ReadAsArray()

tar_data = gdal.Open(target_image).ReadAsArray()
tar_mask = gdal.Open(target_mask).ReadAsArray()

# Feature detection with skimage
ref_points = CENSURE()
ref_points.detect(ref_data)

tar_points = CENSURE()
tar_points.detect(tar_data)











"""
ref_data = cv2.imread(reference_image, -1)
ref_mask = cv2.imread(reference_mask, -1)

tar_data = cv2.imread(target_image, -1)
tar_mask = cv2.imread(target_mask, -1)

# Initiate ORB feature detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(tar_data.astype(np.float32),None)
kp2, des2 = orb.detectAndCompute(ref_data.astype(np.float32),None)

"""