Using deforest on the command line
==================================

Downloading and Preprocessing Data
----------------------------------

See sen2mosaic.

Extracting training data
------------------------


.. code-block:: console
    
    usage: extract.py [-h] [-te XMIN YMIN XMAX YMAX] [-e EPSG] [-res N]
                    [-t SHP/TIF] [-f [VALS [VALS ...]]] [-nf [VALS [VALS ...]]]
                    [-fn NAME] [-p N] [-mi N] [-mp N] [-o DIR] [-n NAME] [-v]
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
    -p N, --n_processes N
                            Maximum number of tiles to process in paralell. Bear
                            in mind that more processes will require more memory.
                            Defaults to 1.
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
    -v, --verbose         Make script verbose.
    
For example:

.. code-block:: console
    
    deforest extract ~/SMFM/chimanimani/L2_files/S2/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -t ~/SMFM/landcover/ESACCI-LC-L4-LC10-Map-20m-P1Y-2016-v1.0.tif -o ./ --max_images 100 -p 10 -f 1 -nf 2 3 4 5 6 7 8 10 -v

Training the model
------------------

.. code-block:: console
    
    usage: train.py [-h] [-m N] [-o PATH] DATA

    Ingest Sentinel-2 data to train a random forest model to predict the
    probability of a pixel being forested. Returns a calibrated model and QA
    graphics.

    positional arguments:
    DATA                  Path to .npz file containing training data, generated
                            by extract_training_data.py

    optional arguments:
    -m N, --max_samples N
                            Maximum number of samples to train the classifier
                            with. Smaller sample sizes will run faster and produce
                            a simpler model, possibly at the cost of predictive
                            power.
    -o PATH, --output_dir PATH
                            Directory to save the classifier. Defaults to the
                            deforest/cfg directory.

For example:

.. code-block:: console
    
    deforest train S2_test_training_data.npz

Classifying the images
----------------------

.. code-block:: console
    
    usage: classify.py [-h] [-te XMIN YMIN XMAX YMAX] [-e EPSG] [-res N]
                    [-l 1C/2A] [-p N] [-o DIR] [-n NAME]
                    [FILES [FILES ...]]

    Process Sentinel-1 and Sentinel-2 to match a predefined CRS, perform a
    deseasaonalisation operation to reduce the impact of seasonality on
    reflectance/backscsatter, and output forest probability images.

    required arguments:
    -te XMIN YMIN XMAX YMAX, --target_extent XMIN YMIN XMAX YMAX
                            Extent of output image tile, in format <xmin, ymin,
                            xmax, ymax>.
    -e EPSG, --epsg EPSG  EPSG code for output image tile CRS. This must be UTM.
                            Find the EPSG code of your output CRS as
                            https://www.epsg-registry.org/.

    optional arguments:
    -res N, --resolution N
                            Specify a resolution to output.
    FILES                 Sentinel 2 input files (level 2A) in .SAFE format,
                            Sentinel-1 input files in .dim format, or a mixture.
                            Specify one or more valid Sentinel-2 .SAFE, a
                            directory containing .SAFE files, or multiple granules
                            through wildcards (e.g. *.SAFE/GRANULE/*). Defaults to
                            processing all granules in current working directory.
    -l 1C/2A, --level 1C/2A
                            Processing level to use, either '1C' or '2A'. Defaults
                            to level 2A.
    -p N, --n_processes N
                            Maximum number of tiles to process in paralell. Bear
                            in mind that more processes will require more memory.
                            Defaults to 1.
    -o DIR, --output_dir DIR
                            Optionally specify an output directory
    -n NAME, --output_name NAME
                            Optionally specify a string to precede output
                            filename.

For example:

.. code-block:: console
    
    deforest classify ~/SMFM/chimanimani/L2_files/S2/ -r 20 -e 32736 -te 399980 7790200 609780 7900000 -n S2_test

    
    
