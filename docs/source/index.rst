.. deforest documentation master file, created by
   sphinx-quickstart on Tue Jan 16 18:16:52 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to deforest's documentation!
====================================

The DEnse FOREst Time Series (DEFOREST) tool is a method for detecting changes in forest cover in a time series of Sentinel-2 data.

The processing chain:

* Takes Sentinel-2 data (L1C/L2A) as input.
* Trains a model to robustly identify forest/nonforest across any phenological state.
* Classifies each image to a probability of forest cover.
* Combines these images under a Bayesian framework to identify forest cover change.
* Outputs maps of forest cover change, and 'early warnings' of change in recent imagery.

Contents:
---------

.. toctree::
   :maxdepth: 2
   setup.rst
   command_line.rst
   worked_example.rst