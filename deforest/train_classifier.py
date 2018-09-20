import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import pdb


def _getCfgDir():
    '''
    Returns the directory of the cfg directory to output model coefficients.
    '''
    
    return '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]) + '/cfg/'


def _undersampleArray(array1, array2):
    '''
    Function to randomly undersample an array to match the size of the larger array.
    
    Args:
        array1: A numpy array.
        array2: Another numpy array.
        
    Returns:
        array1 and array2, with the larger input array resampled to the size of the smaller array.
    '''

    assert array1.ndim == 1 or array1.ndim == 2, "Input arrays must have either one or two dimensions. array1 has %s dimensions"%str(array1.ndim)
    assert array2.ndim == 1 or array2.ndim == 2, "Input arrays must have either one or two dimensions. array2 has %s dimensions"%str(array2.ndim)
    assert array1.ndim == 1 or array1.ndim == 2, "Input arrays must have the same number of dimensions. array1 has %s dimensions and array2 has %s dimensions."%(str(array1.ndim), str(array2.ndim))

    # Determine which array is larger
    if array1.shape[0] > array2.shape[0]:
        larger_array = array1
        smaller_array = array2
    elif array1.shape[0] <= array2.shape[0]:
        larger_array = array2
        smaller_array = array2

    # Randomly resample it
    s = np.arange(larger_array.shape[0])
    np.random.shuffle(s)

    if larger_array.ndim == 2:
        larger_array_subsample = larger_array[s < smaller_array.shape[0], :]
    else:
        larger_array_subsample = larger_array[s < smaller_array.shape[0]]

    # Rename the subsampled array to the input names
    if array1.shape[0] > array2.shape[0]:
        array1 = larger_array_subsample
    elif array2.shape[0] <= array2.shape[0]:
        array2 = larger_array_subsample

    return array1, array2



def loadData(data):
    '''
    Load data from the .npz file output by extract_training_data.py
    
    Args:
       data: Path to the .npz file
       
    Returns:
        a list of forest pixel values, a list of nonforest pixel values
    '''
    
    data = np.load(data)
    forest_px = data['forest_px']
    nonforest_px = data['nonforest_px']
    
    return forest_px, nonforest_px


def fitModel(forest_px, nonforest_px, output_name, max_pixels = 100000, output_QA = True, output_dir = _getCfgDir()):
    '''
    '''
        
    # Balance data by undersampling the larger class
    #forest_px, nonforest_px = _undersampleArray(forest_px, nonforest_px)
    
    # Randomise sample
    np.random.shuffle(forest_px); np.random.shuffle(nonforest_px)
    
    # Limit sample size to max_pixels
    forest_px = forest_px[:max_pixels,:]
    nonforest_px = nonforest_px[:max_pixels, :]
    
    # Prepare data for sklearn
    y = np.array(([1] * forest_px.shape[0]) + ([0] * nonforest_px.shape[0]))
    X = np.vstack((forest_px,nonforest_px))
    
    X[np.logical_or(np.isinf(X), np.isnan(X))] = 0.
    
    # Split into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)
    #X_test1, X_test2, y_test1, y_test2 = train_test_split(X, y, test_size = 0.5, random_state = 42)
    
    # Fir the random forest model.
    clf = RandomForestClassifier(random_state = 42, n_estimators = 100, class_weight = 'balanced')
    clf.fit(X_train, y_train)

    #from sklearn.ensemble import GradientBoostingClassifier
    #clf = GradientBoostingClassifier(random_state = 42, n_estimators = 100)
    #clf.fit(X_train, y_train)
        
    # Post-hoc probability calibration. This performs poorly with seasonality, so we leave it out.
    # from sklearn.calibration import CalibratedClassifierCV
    # clf_cal = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    # #X_test1_bal, y_test1_bal = undersampleArray(X_test1, y_test1)
    #' clf_cal.fit(X_test1, y_test1)
    
    # Quality assessment:
    # We strongly recommend you carefully consider the quality of your models. These are the metrics that worked for us, but it's perfeclty possible they'll work poorly in your case. Proceed with caution! 

    if output_QA: buildQAPlot(clf, X_test, y_test, output_name, output_dir = output_dir)
        
    return clf


def _plotConfusionMatrix(cm, classes, normalize=False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.i
    Normalization can be applied by setting `normalize=True`.
    # From: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def buildQAPlot(clf, X_test, y_test, output_name, output_dir = _getCfgDir()):
    '''
    Function to construct a plot with quality assessment metrics for the random forest model.
    
    Args:
        clf: The classifier object
        X_test: Test predictors (separate from training data)
        y_tesy: Test labels (separate from training data)
        output_name: String to prepend to ourput file
        output_dir: Location to output QA plot. Defaults to deforest/cfg/
    '''
    
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn import metrics
    
    
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    ax3 = plt.subplot2grid((3, 2), (2, 1))
    
    ax1.plot([0, 1], [0, 1], "b:", label="Perfectly calibrated")
    
    prob_pos = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
    
    ax1.plot(mean_predicted_value, fraction_of_positives, "bs-",
                 label="%s" % 'Logistic regression')
    ax1.plot(fpr, tpr, "r-", label='ROC curve')
    
    ax2.hist(prob_pos, range=(0, 1), bins=10, label='Logistic regression',
                 histtype="step", lw=2)
    
    ax1.set_ylabel("Fraction of positives / True positive rate")
    ax1.set_xlabel("Mean predicted value / False positive rate")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve and ROC curve)')

    ax2.set_title('Predicted values from test dataset')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    
    cnf_matrix =  metrics.confusion_matrix(y_test,clf.predict(X_test))
    ax3 = _plotConfusionMatrix(cnf_matrix, classes=['Forest','Nonforest'], normalize=True)
        
    plt.tight_layout()
    plt.savefig('%s/%s_quality_assessment.png'%(output_dir,output_name))


def saveModel(clf, output_name, output_dir = _getCfgDir()):
    '''
    Saves the model from Scikit-learn to the deforest configuration directory with pickle.
    
    Args:
        clf: A fitted model from sklearn
        output_name: A string wih the image type (i.e. S1single, S1dual, S2)
        output_dir: Directory to output model coefficients. Defaults to the cfg directory."
    '''

    # Determine name of output file
    filename = '%s/%s_model.pkl'%(output_dir, output_name)
    
    # Pickle
    file_out = joblib.dump(clf, filename)


def main(data, max_samples = 100000, output_dir = _getCfgDir()):
    '''
    Train a random forest model to predict the probability of forest/nonforest given data fextracted from imagery by extract_training_data.py.
    
    Args:
        data: A .npz file from extract_training_data.py
        max_samples: Maximum number of pixels to use in training the classifier
        output_dir: Directory to save the calibrated model. Defaults to deforest/cfg/.
    '''
    
    # Get output_name
    output_name = data.split('/')[-1].split('_')[0]
    
    # Get data
    forest_px, nonforest_px = loadData(data)
    
    # Fit an RF model
    clf = fitModel(forest_px, nonforest_px, output_name, max_pixels = max_samples, output_QA = True, output_dir = output_dir)
    
    # Save the classifier
    saveModel(clf, output_name, output_dir = output_dir)
    

if __name__ == '__main__':
    '''
    Script to train a Random Forest model to classify S2 images into forest/nonforest probabilities.
    Returns a calibrated model and QA graphics.
    
    Requires a .npz file from extract_training_data.py.
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Ingest Sentinel-2 data to train a random forest model to predict the probability of a pixel being forested. Returns a calibrated model and QA graphics.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('data', metavar = 'DATA', type = str, help = 'Path to .npz file containing training data, generated by extract_training_data.py')

    # Optional arguments
    optional.add_argument('-m', '--max_samples', type = int, metavar = 'N', default = 100000, help = "Maximum number of samples to train the classifier with. Smaller sample sizes will run faster and produce a simpler model, possibly at the cost of predictive power.")
    optional.add_argument('-o', '--output_dir', type = str, metavar = 'PATH', default = _getCfgDir(), help = "Directory to save the classifier. Defaults to the deforest/cfg directory.")
    
    # Get arguments
    args = parser.parse_args()
    
    # Execute script
    main(args.data, max_samples = args.max_samples, output_dir = args.output_dir)
    
    #~/anaconda2/bin/python ~/DATA/deforest/deforest/train_logistic_model.py -o ./ S2_training_data.npz