Using deforest on the command line
==================================

Downloading and Preprocessing Data
----------------------------------

Instructions for downloaded and pre-processing data from Sentinel-2 can be found at `sen2mosaic <https://bitbucket.org/sambowers/sen2mosaic/>`_.

.. NOTE::
    SMFM deforest is designed for analysis of dense time series of Sentinel-2 data, which will require access to subsantial file storage and processing power. We recommend use of a cloud platform, where data do not have to downloaded of pre-processed on a local machine. See, for example, the 'Data and Information Access Services' (`DIAS <https://www.copernicus.eu/en/access-data/dias>`_) platforms that provide centralised access to Copernicus data.

Calibrating SMFM deforest
-------------------------

SMFM deforest will work best where local data are used for calibration. This is performed in two steps: (i) Extracting training data, and (ii) Training the classifier.

Extracting training data
~~~~~~~~~~~~~~~~~~~~~~~~

Training pixels are defined by either a geotiff image or shapefile defining locations of stable forest and stable non-forest. Classification features are derived for a series of Sentinel-2 images, and a random selection of pixel values extracted for forest and non-forest classes. The output is an array of feature values used to train a classifier to predict probabilities of forest and non-forest.

.. code-block:: console
    
    usage: extract.py [-h] [-te XMIN YMIN XMAX YMAX] [-e EPSG] [-res N]
                      [-t SHP/TIF] [-f [VALS [VALS ...]]] [-nf [VALS [VALS ...]]]
                      [-fn NAME] [-l 1C/2A] [-mi N] [-mp N] [-o DIR] [-n NAME]
                      [-p N] [-v]
                      [FILES [FILES ...]]
    
    Extract indices from Sentinel-2 data to train a classifier of forest cover.
    Returns a numpy .npz file containing pixel values for forest/nonforest.
    
    positional arguments:
      FILES                 Sentinel 2 input files (level 2A) in .SAFE format.
                            Specify one or more valid Sentinel-2 .SAFE, a
                            directory containing .SAFE files, multiple tiles
                            through wildcards (e.g. *.SAFE/GRANULE/*), or a text
                            file listing files. Defaults to processing all tiles
                            in current working directory.
    
    required arguments:
      -te XMIN YMIN XMAX YMAX, --target_extent XMIN YMIN XMAX YMAX
                            Extent of output image tile, in format <xmin, ymin,
                            xmax, ymax>.
      -e EPSG, --epsg EPSG  EPSG code for output image tile CRS. This must be UTM.
                            Find the EPSG code of your output CRS as
                            https://www.epsg-registry.org/.
      -res N, --resolution N
                            Specify a resolution to output.
      -t SHP/TIF, --training_data SHP/TIF
                            Path to training data geotiff/shapefile.
      -f [VALS [VALS ...]], --forest_values [VALS [VALS ...]]
                            Values indicating forest in the training GeoTiff or
                            shapefile
      -nf [VALS [VALS ...]], --nonforest_values [VALS [VALS ...]]
                            Values indicating nonforest in the training GeoTiff or
                            shapefile

    optional arguments:
      -fn NAME, --field_name NAME
                              Shapefile attribute name to search for training data
                              polygons. Defaults to all polygons. Required where
                              inputting a shapefile as training_data.
      -l 1C/2A, --level 1C/2A
                              Input image processing level, '1C' or '2A'. Defaults
                              to '2A'.
      -mi N, --max_images N
                              Maximum number of input tiles to extract data from.
                              Defaults to all valid tiles.
      -mp N, --max_pixels N
                              Maximum number of pixels to extract from each image
                              per class. Defaults to 5000.
      -o DIR, --output_dir DIR
                              Output directory. Defaults to current working
                              directory.
      -n NAME, --output_name NAME
                              Specify a string to precede output filename. Defaults
                              to 'S2'.
      -p N, --n_processes N
                              Maximum number of tiles to process in paralell. Bear
                              in mind that more processes will require more memory.
                              Defaults to 1.
      -v, --verbose         Make script verbose.

    
For example, for a directory containing Sentinel-2 data from tile ``36KWD`` (``~/S2_data/``), specifying an appropriate bounding box and resolution (``-r``, ``-e``, ``-te``), training data contained in a geotiff (``~/training_data.tif``) coded as stable forest (``-f 1``) and stable non-forest (``-nf 2``), using 10 processes (``-p 10``):

.. code-block:: console
    
    deforest extract ~/S2_data/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t ~/training_data.tif --max_images 100 -f 1 -nf 2 -v -p 10

Training the model
~~~~~~~~~~~~~~~~~~

SMFM deforest uses a Random Forest model to predict the probability of forest in each input Sentinel-2 image. This model can be calibrated using training data from the region of interest.

The training function takes a series of labelled forest and non-forest pixels (see 'Extracting training data') as input and returns a calibrated model (a ``.pkl`` file). The process also returns a series of plots that can b eused to assess model performance.

.. code-block:: console
    
    usage: train.py [-h] [-m N] [-n NAME] [-o PATH] DATA
    
    Ingest Sentinel-2 data to train a random forest model to predict the
    probability of a pixel being forested. Returns a calibrated model and QA
    graphics.
    
    positional arguments:
      DATA                  Path to .npz file containing training data, generated
                            by extract.py
    
    optional arguments:
      -m N, --max_samples N
                            Maximum number of samples to train the classifier
                            with. Smaller sample sizes will run faster and produce
                            a simpler model, possibly at the cost of predictive
                            power. Defaults to 100,000 points.
      -n NAME, --output_name NAME
                            Specify a string to precede output filename. Defaults
                            to name of input training data.
      -o PATH, --output_dir PATH
                            Directory to save the classifier. Defaults to the
                            current working directory.

For example, using the output of ``deforest extract``:

.. code-block:: console
    
    deforest train S2_training_data.npz

Classification and change detection
-----------------------------------

SMFM deforest uses a two-step process to produce change maps: (i) classification of individual Sentinel-2 images, and (ii) change detection.

Image classification
~~~~~~~~~~~~~~~~~~~~

Sentinel-2 images are classified into a continuous probability of forest in each non-masked pixel. Inputs can be either Sentinel-1 L1C data or L2A data (preferable). The output is a set of geotiffs numbered 0 - 100%, with a set extent, resolution and coordinate reference system (UTM). 

.. code-block:: console
    
    usage: classify.py [-h] [-te XMIN YMIN XMAX YMAX] [-e EPSG] [-r N] [-m PKL]
                       [-l 1C/2A] [-p N] [-n NAME] [-o DIR]
                       [FILES [FILES ...]]
    
    Process Sentinel-2 to match a predefined CRS and classify each to show a
    probability of forest (0-100%) in each pixel.
    
    required arguments:
      -te XMIN YMIN XMAX YMAX, --target_extent XMIN YMIN XMAX YMAX
                            Extent of output image tile, in format <xmin, ymin,
                            xmax, ymax>.
      -e EPSG, --epsg EPSG  EPSG code for output image tile CRS. This must be UTM.
                            Find the EPSG code of your output CRS as
                            https://www.epsg-registry.org/.
      -r N, --resolution N  Specify a resolution to output.
    
    optional arguments:
      FILES                 Sentinel 2 input files in .SAFE format. Specify one or
                            more valid Sentinel-2 .SAFE files, a directory
                            containing .SAFE files, or multiple granules through
                            wildcards (e.g. *.SAFE/GRANULE/*). Defaults to
                            processing all granules in current working directory.
      -m PKL, --model PKL   Path to .pkl model, produced with train.py. Defaults
                            to a test model, trained on data from Chimanimani in
                            Mozambique.
      -l 1C/2A, --level 1C/2A
                            Processing level to use, either '1C' or '2A'. Defaults
                            to level 2A.
      -p N, --n_processes N
                            Maximum number of tiles to process in paralell. Bear
                            in mind that more processes will require more memory.
                            Defaults to 1.
      -n NAME, --output_name NAME
                            Specify a string to precede output filename. Defaults
                            to 'S2'.
      -o DIR, --output_dir DIR
                            Optionally specify an output directory

For example, to classify probability of forest in all images in a directory containing Sentinel-2 data from tile ``36KWD`` (``~/S2_data/``), specifying an appropriate bounding box and resolution (``-r``, ``-e``, ``-te``),  and a calibrated model named ``S2_model.pkl``:

.. code-block:: console
    
    deforest classify ~/S2_data/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -m S2_model.pkl
    
Change detection
~~~~~~~~~~~~~~~~

The final step is to combine the time series of forest probability images under a Bayesian framework to detect changes in forest cover. The output is two geotiffs, one providing the year of change, the other an early warning of pixels flagged as possible changes at the final time step.

.. code-block:: console
    
    usage: change.py [-h] [-t N] [-b N] [-o DIR] [-n NAME] FILES [FILES ...]
    
    Process probability maps to generate a map of deforestation year and warning
    estimates of upcoming events.
    
    required arguments:
      FILES                 A list of files output by classify.py, specifying
                            multiple files using wildcards.
    
    optional arguments:
      -t N, --threshold N   Set a threshold probability to identify deforestation
                            (between 0 and 1). High thresholds are more strict in
                            the identification of deforestation. Defaults to 0.99.
      -b N, --block_weight N
                            Set a block weighting threshold to limit the range of
                            forest/nonforest probabilities. Set to 0 for no block-
                            weighting. Parameter cannot be set higher than 0.5.
      -o DIR, --output_dir DIR
                            Optionally specify an output directory. If nothing
                            specified, downloads will output to the present
                            working directory, given a standard filename.
      -n NAME, --output_name NAME
                            Optionally specify a string to precede output
                            filename. Defaults to the same as input files.

For example, using default change detection parameters and a set of classified images from ``classify.py``:

.. code-block:: console
    
    deforest change ./*.tif 