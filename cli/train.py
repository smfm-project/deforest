import argparse
import os

import deforest.training

import pdb


def main(data, max_samples = 100000, output_name = 'S2', output_dir = deforest.training.getCfgDir()):
    '''
    Train a random forest model to predict the probability of forest/nonforest given data fextracted from imagery by extract_training_data.py.
    
    Args:
        data: A .npz file from extract.py
        max_samples: Maximum number of pixels to use in training the classifier
        output_dir: Directory to save the calibrated model. Defaults to deforest/cfg/.
    '''
        
    # Get data
    forest_px, nonforest_px = deforest.training.loadData(data)
    
    # Fit an RF model
    clf = deforest.training.fitModel(forest_px, nonforest_px, output_name, max_pixels = max_samples, output_QA = True, output_dir = output_dir)
    
    # Save the classifier
    deforest.training.saveModel(clf, output_name, output_dir = output_dir)
    

if __name__ == '__main__':
    '''
    Script to train a Random Forest model to classify S2 images into forest/nonforest probabilities.
    Returns a calibrated model and QA graphics.
    
    Requires a .npz file from extract_training_data.py.
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Ingest Sentinel-2 data to train a random forest model to predict the probability of a pixel being forested. Returns a calibrated model and QA graphics.")

    parser._action_groups.pop()
    positional = parser.add_argument_group('positional arguments')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Positional arguments
    positional.add_argument('data', metavar = 'DATA', type = str, help = 'Path to .npz file containing training data, generated by extract_training_data.py')
    
    # Required arguments
    
    # Optional arguments
    optional.add_argument('-m', '--max_samples', type = int, metavar = 'N', default = 100000, help = "Maximum number of samples to train the classifier with. Smaller sample sizes will run faster and produce a simpler model, possibly at the cost of predictive power. Defaults to 100,000 points.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = None, help="Specify a string to precede output filename. Defaults to name of input training data.")
    optional.add_argument('-o', '--output_dir', type = str, metavar = 'PATH', default = os.getcwd(), help = "Directory to save the classifier. Defaults to the current working directory.")
        
    # Get arguments
    args = parser.parse_args()
    
    if args.output_name is None:
        output_name = '_'.join(args.data.split('_')[:-2])
    else:
        output_name = args.output_name
    
    # Execute script
    main(args.data, max_samples = args.max_samples, output_name = output_name, output_dir = args.output_dir)
    
    # Example:
    # 
    # deforest train S2_training_data.npz
