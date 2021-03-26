.. _worked_example_sepal:

Worked example on SEPAL
=======================

Preparation
-----------

Ensure that you are signed up to sepal.

Image classification
--------------------

Creating a composite image
~~~~~~~~~~~~~~~~~~~~~~~~~~

Training a classifier
~~~~~~~~~~~~~~~~~~~~~


Dowloading the time series
---------------------------


Change detection
----------------

The final step is to combine these classified images into an estimate of forest cover and forest cover change. For use the ``change.py`` command line tool:

.. code-block:: console
    
    [user@linuxpc worked_example]$ deforest change classified_images/*.tif

This process will combine all the probability images in ``classified_images``, and identify changes in the time series.

Options are available to change the parameters of change detection. For example, to apply a stricter probability threshold detection of confirmed changes:

.. code-block:: console
    
    [user@linuxpc worked_example]$ deforest change -t 0.9995 classified_images/*.tif

Or to alter block-weighting, which reduces the impact of very high or low probability outliers:

.. code-block:: console
    
    [user@linuxpc worked_example]$ deforest change -b 0.2 classified_images/*.tif

The ``change.py`` script will output two images:

.. code-block:: console
    
    [user@linuxpc worked_example]$ ls
    ...
    S2_confirmed.tif
    S2_warning.tif

The image ``S2_confirmed.tif`` shows the year of changes that have been detected in the time series (in this case 2017-2019.5). Locations of remaining forest are numbered 0. Note that the changes in the first year or two will be mostly spurious, as the landscape is initially considered entirely forested. It is recommended that the user either discards the first 2-3 years of changes, or uses a high-quality forest baseline map to mask out locations that weren't forest at the start of the time series.

Shown below is a section of ``S2_confirmed.tif``. Forest is shown in green, and non-forest in yellow (change date < 2018.5). Confirmed changes (2018.5 - 2019.5) are indicated in blue, with lighter blues changing earlier in the time series.

.. image:: _static/S2_confirmed.png

The image ``S2_warning.tif`` shows the combined probability of non-forest existing at the end of the time series in locations that have not yet been flagged as deforested. This can be used to provide information on locations that have not yet reached the threshold for confirmed changes, but are looking likely to possible. A simple probability threshold can be applied to supply early warnings.

Shown below is a section of ``S2_warning.tif``, with warning locations where probability of non-forest is greater than 85% shown in red. Early warnings show pixels that have yet to be confirmed as change, at the cost of an increased false positive rate relative to confirmed changes.

.. image:: _static/S2_warning.png
