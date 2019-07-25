.. _worked_example_commandline:

Worked example on the command line
==================================

Here we'll show you by example how the deforest processing chain works in practice. We will focus on an example from Zambezia province of Mozambique, with the aim of producing a remote sensing product for historic deforestation and near real-time warnings of deforestation.

We'll run this example in Gile, Mozambique, an area covered by Sentinel-2 tile 37LDC. This location has the CRS **UTM 37S** (EPSG: 32737), and an extent of **XXX,XXX - XXX,XXX** m Eastings, and **X,XXX,XXX - X,XXX,XXX** m Eastings. We'll use all data from the start of the Sentinel-2 era to May 2019, the time of writing.

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

Here, we'll download all L1C data for the tile ``37LDC`` specifying a maximum cloud cover percetage of 30%:

.. code-block:: console
    
    s2m download -u user.name -p supersecret -t 37LDC -c 30

Wait for all files to finish downloading before proceeding to the next step. By the time the processing is complete, your ``DATA`` directory should contain a series of Sentinel-2 .SAFE files:

.. code-block:: console
    
    ...
    ...

.. note:: If available, you can also download L2A from the Copernicus Open Access Data Hub (using option ``-l 2A``). At the time of writing, this data weren't widely available.    
    
Atmopsheric correction and cloud masking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code-block:: console
    
    ...
    ...


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
    
    deforest extract path/to/DATA/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t path/to/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif -o ./ --max_images 100 -f 1 -nf 2 3 4 5 6 7 8 10 -v

If resources are available, this process can be sped up by increasing the number of processes to, for instance, to run 8 similtaneous processes:

.. code-block:: console
    
    deforest extract path/to/DATA/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t path/to/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif -o ./ --max_images 100 -f 1 -nf 2 3 4 5 6 7 8 10 -v -p 8

Be aware, the more processes used the more computational resources will be required.

The output of this command will be a ``.npz`` file, which contains the pixel values for each classification feature.

.. code-block:: console
    
    [user@linuxpc directory] ls
    S2_training_data.npz

Calibrating the classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step it to use this training data to calibrate the classifier of forest cover. This is performed with the ``deforest train.py`` tool.

To train the classifier, run:

.. code-block:: console
    
    deforest train S2_training_data.npz

Once complete there will be two new files

.. code-block:: console

    [user@linuxpc directory] ls
    S2_model.pkl
    S2_quality_assessment.png

``S2_model.pkl`` is an object that defines the classifier, ``S2_quality_assessment.png`` can be used to assess the quality of the model. See **MODEL QUALITY ASSESSMENT** (to follow).

Classifing the data
-------------------

First, we'll make a new directory to store classified images:

.. code-block:: console
    
    [user@linuxpc directory] mkdir classified_images

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
    
    [user@linuxpc directory] ls
    ...
    S2_confirmed.tif
    S2_warning.tif

IMAGE