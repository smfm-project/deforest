.. _worked_example_commandline:

Worked example on the command line
==================================

Here we'll show you by example how the deforest processing chain works in practice. We will focus on an example from Zambezia province of Mozambique, with the aim of producing a remote sensing product for historic deforestation and near real-time warnings of deforestation.

We'll run this example in the Chimanimani region, covering Zimbabwe and Mozambique. We'll use a dense time series of two Sentinel-2 tiles, **36KWC** and **36KWD**. This location has the CRS **UTM 36S** (EPSG: 32736), and an extent of **400,000 - 600,000** m Eastings, and **7,800,000 - 7,900,000** m Northings. We'll use all data from the start of the Sentinel-2 era to mid-July 2019, the time of writing.

Preparation
-----------

First ensure that you've followed :ref:`setup` successfully.

Open a terminal, and use ``cd`` to navigate to the location you'd like to store data.

.. code-block:: console
    
    cd /home/user/DATA
    mkdir worked_example
    cd worked_example

Use mkdir to make a separate directory to contain the data you wish to download.

.. code-block:: console
    
    mkdir DATA
    
To begin, navigate to the DATA folder.

.. code-block:: console
    
    cd DATA

Data preparation
----------------

Downloading data
~~~~~~~~~~~~~~~~

The first step is to download Sentinel-2 level 1C data from the `Copernicus Open Access Data Hub <https://scihub.copernicus.eu/>`_.

For this we use the ``sen2mosaic`` ``download.py`` tool. See [sen2mosaic instructions] for more details.

Here, we'll download all L2A data for the period 1st July 2018 to 2019 for the tile ``36KWD`` specifying a maximum cloud cover percetage of 30%:

.. code-block:: console
    
    s2m download -u user.name -p supersecret -t 36KWD -c 30 -s 20180701 -l L2A

.. note::  Not all Sentinel-2 data are available in L2A format, meaning that you can either use L1C data, or preprocess data yourself with sen2cor. See `sen2mosaic <https://www.bitbucket.org/sambowers/sen2mosaic>`_ for more details.

.. note:: Data from more than 1 year in the past may have been moved to the `Long Term Archive <https://earth.esa.int/web/sentinel/news/-/article/activation-of-long-term-archive-lta-access>`_. To access this data it will be necessary to order it from Copernicus. For more practical purposes, to use dense time series you should consider using these scripts on an online platform (e.g. F-TEP, DIAS, AWS) to access Sentinel-2 data.
    
Ensure that you have at least 2 years of data before proceeding to the next step. Ensuere that you have a directory (i.e. ``DATA``) containing a series of Sentinel-2 .SAFE files:

.. code-block:: console

    [user@linuxpc DATA]$ ls
    S2B_MSIL2A_20170630T072949_N0205_R049_T36KWC_20170630T075509.SAFE
    S2B_MSIL2A_20170710T072619_N0205_R049_T36KWC_20170710T074330.SAFE
    S2B_MSIL2A_20170713T075209_N0205_R092_T36KVD_20170713T075751.SAFE
    S2B_MSIL2A_20170713T075209_N0205_R092_T36KWC_20170713T075751.SAFE
    S2B_MSIL2A_20170713T075209_N0205_R092_T36KWC_20170713T080544.SAFE
    S2B_MSIL2A_20170713T075209_N0205_R092_T36KWD_20170713T075751.SAFE
    S2B_MSIL2A_20170720T074239_N0205_R049_T36KWC_20170720T074942.SAFE
    S2B_MSIL2A_20170723T073609_N0205_R092_T36KVD_20170723T075425.SAFE
    ...
    S2B_MSIL2A_20190703T073619_N0212_R092_T36KWD_20190703T122423.SAFE
    S2B_MSIL2A_20190713T073619_N0213_R092_T36KWD_20190713T111309.SAFE
    S2B_MSIL2A_20190723T073619_N0213_R092_T36KWD_20190723T115930.SAFE

Training the classifier
-----------------------

Training of the classifier is performed in two steps. 1) Extracting data from a series of training pixels of stable forest and nonforest, 2) Calibrating a classifier to separate the spectral characteristics of forest from those of nonforest.

Extracting training data
~~~~~~~~~~~~~~~~~~~~~~~~

The first step to using the ``deforest`` algorithm is to extract training data. We perform this task with the ``deforest extract.py`` tool.

There are two options for specification of locations to extract training data, either using a shapefile or a raster image. In each case we need to specify the attributes of a 'forest' and a 'nonforest' pixel, and these should be associated with locations of stable forest/nonforest.

For ease, here we'll use a pre-existing land cover map to train our classifier (download on registration `here <http://2016africalandcover20m.esrin.esa.int/>`_). This map covers Africa at 20 m resolution, with numbered land cover classes with meaning:

+-----------------------------------------+-------+
| Land cover                              | Value |
+-----------------------------------------+-------+
| No data                                 | 0     |
+-----------------------------------------+-------+
| Tree cover areas                        | 1     |
+-----------------------------------------+-------+
| Shrubs cover areas                      | 2     |
+-----------------------------------------+-------+
| Grassland                               | 3     |
+-----------------------------------------+-------+
| Cropland                                | 4     |
+-----------------------------------------+-------+
| Vegetation aquatic or regularly flooded | 5     |
+-----------------------------------------+-------+
| Lichens Mosses / Sparse vegetation      | 6     |
+-----------------------------------------+-------+
| Bare areas                              | 7     |
+-----------------------------------------+-------+
| Built up areas                          | 8     |
+-----------------------------------------+-------+
| Snow and/or ice                         | 9     |
+-----------------------------------------+-------+
| Open water                              | 10    |
+-----------------------------------------+-------+

To use this with our existing directory containing Sentinel-2 data, we can use the following command:

.. code-block:: console
    
    deforest extract path/to/DATA/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t path/to/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif -o ./ -f 1 -nf 2 3 4 5 6 7 8 10 -v

If resources are limited, input training data can be limited to fewer images:

.. code-block:: console
    
    deforest extract path/to/DATA/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t path/to/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif -o ./ --max_images 100 -f 1 -nf 2 3 4 5 6 7 8 10 -v
    
If resources are available, this process can be sped up by increasing the number of processes to, for instance, to run 8 similtaneous processes:

.. code-block:: console
    
    deforest extract path/to/DATA/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t path/to/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif -o ./ -f 1 -nf 2 3 4 5 6 7 8 10 -v -p 8

Be aware, the more processes used the more computational resources will be required.



The output of this command will be a ``.npz`` file, which contains the pixel values for each classification feature.

.. code-block:: console
    
    [user@linuxpc directory]$ ls
    S2_training_data.npz

Calibrating the classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step it to use this training data to calibrate the classifier of forest cover. This is performed with the ``deforest train.py`` tool.

To train the classifier, run:

.. code-block:: console
    
    deforest train S2_training_data.npz

Once complete there will be two new files

.. code-block:: console

    [user@linuxpc directory]$ ls
    S2_model.pkl
    S2_quality_assessment.png

``S2_model.pkl`` is an object that defines the classifier, ``S2_quality_assessment.png`` can be used to assess the quality of the model. See **MODEL QUALITY ASSESSMENT** (to follow).

Classifing the data
-------------------

First, we'll make a new directory to store classified images:

.. code-block:: console
    
    [user@linuxpc directory]$ mkdir classified_images

We can then run the classification algorithm we just calibrated to produce probability of forest for each image. This operates very similarly to ``training.py``, here we'll use the same output extents:
    
.. code-block:: console
    
    deforest classify path/to/DATA/ -m S2_model.pkl -r 20 -e 32736 -te 399980 7790200 609780 7900000 -o classified_images

If resources are available, classification can can be sped up by allocating additional processes:
    
.. code-block:: console
    
    deforest classify path/to/DATA/ -m S2_model.pkl -r 20 -e 32736 -te 399980 7790200 609780 7900000 -o classified_images -p 8

Once complete, images will be output to the ``classified_images`` directory.

.. code-block:: console

    [user@linuxpc classified_images]$ ls
    S2_S2_T36KVD_20151126_075714.tif  S2_S2_T36KWC_20171001_075742.tif
    S2_S2_T36KVD_20151206_075547.tif  S2_S2_T36KWC_20171006_075832.tif
    S2_S2_T36KVD_20151226_080933.tif  S2_S2_T36KWC_20171008_075024.tif
    S2_S2_T36KVD_20151229_082023.tif  S2_S2_T36KWC_20171016_075320.tif
    S2_S2_T36KVD_20160105_080719.tif  S2_S2_T36KWC_20171023_074855.tif
    S2_S2_T36KVD_20160108_082023.tif  S2_S2_T36KWC_20171026_080348.tif
    S2_S2_T36KVD_20160125_080606.tif  S2_S2_T36KWC_20171031_075502.tif
    S2_S2_T36KVD_20160204_080212.tif  S2_S2_T36KWC_20171107_075205.tif
    S2_S2_T36KVD_20160207_080537.tif  S2_S2_T36KWC_20171120_075322.tif
    ...                               ...
    S2_S2_T36KWC_20170926_075507.tif  S2_S2_T36KWD_20180906_075434.tif
    S2_S2_T36KWC_20170928_074401.tif


IMAGE

Change detection
----------------

The final step is to combine these classified images into an estimate of forest cover and forest cover change. For this we use the ``change.py`` command line tool:

.. code-block:: console
    
    deforest change classified_images/*.tif

This process will output two images:

.. code-block:: console
    
    [user@linuxpc directory]$ ls
    ...
    S2_confirmed.tif
    S2_warning.tif

IMAGE