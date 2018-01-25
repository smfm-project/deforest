import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

import pdb

def _getCfgDir():
    '''
    Returns the directory of the cfg directory to output model coefficients.
    '''
    
    return '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]) + '/cfg/'


def loadData(data):
    '''
    '''
    
    data = np.load(data)
    forest_px = data['forest_px']
    nonforest_px = data['nonforest_px']
    
    return forest_px, nonforest_px


def fitModel(forest_px, nonforest_px, image_type, regularisation_strength = 1., regularisation_type = 'l1', output_QA = True, output_dir = _getCfgDir()):
    '''
    '''
    
    assert regularisation_type == 'l1' or regularisation_type == 'l2', "The parameter regularisation_type must be set to 'l1' or 'l2'."
    
    # Balance data by undersampling the larger class
    forest_px, nonforest_px = undersampleArray(forest_px, nonforest_px)
    
    # Make sample size managable
    #forest_px = forest_px[::10]
    #nonforest_px = nonforest_px[::10]
    
    # Prepare data for sklearn
    y = np.array(([1] * forest_px.shape[0]) + ([0] * nonforest_px.shape[0]))
    X = np.vstack((forest_px,nonforest_px))
    
    # Polynomial expansion method (optional). May in some cases improve results, if strong regularisation used.
    #from sklearn.preprocessing import PolynomialFeatures
    #poly = PolynomialFeatures(2)
    #X = poly.fit_transform(X)
    
    # Split into training and test datasets
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    # Normalising the training data. This is necessary for l1/l2 regularisation to function properly.
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # Fit the logistic model. We chose 'l1' regularisation because it hard the property of sparsity, which removes predictors that add little additional information. This is important where features are highly correlated.
    # Parameter C determines the degree of regularisation. This parameter needs to be tuned, such that the model is as simple as it can reasonably be, but no simpler.
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C = regularisation_strength, penalty = regularisation_type, class_weight = 'balanced', solver = 'liblinear')
    lr.fit(X_train_std,y_train)
    
    # Quality assessment:
    # We strongly recommend you carefully consider the quality of your models. These are the metrics that worked for us, but it's perfeclty possible they'll work poorly in your case. Proceed with caution! 

    if output_QA: buildQAPlot(lr, X_test_std, y_test, image_type, output_dir = output_dir)
    
    return lr, sc 


def undersampleArray(array1, array2):
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


def _plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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


def buildQAPlot(lr, X_test_std, y_test, image_type, output_dir = _getCfgDir()):
    '''
    Function to construct a plot with quality assessment metrics for logistic reression fit.
    '''
    
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn import metrics
    
    
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    ax3 = plt.subplot2grid((3, 2), (2, 1))
    
    ax1.plot([0, 1], [0, 1], "b:", label="Perfectly calibrated")
    
    prob_pos = lr.predict_proba(X_test_std)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, lr.predict_proba(X_test_std)[:,1])
    
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
    
    cnf_matrix =  metrics.confusion_matrix(y_test,lr.predict(X_test_std))
    ax3 = _plotConfusionMatrix(cnf_matrix, classes=['Forest','Nonforest'], normalize=True)
    
    plt.tight_layout()
    plt.savefig('%s/%s_quality_assessment.png'%(output_dir,image_type))


def saveCoefficients(lr, sc, image_type, output_dir = _getCfgDir()):
    '''
    Writes the coefficients from a logistic regression to a csv file in the deforest configuration directory.
    
    Args:
        lr: A logistic regression object from the sklearn module
        sc: A standardscaler object from the sklearn module
        image_type: A string wih the image type (i.e. S1single, S1dual, S2)
        output_dir: Directory to output model coefficients. Defaults to the cfg directory."
    '''

    # Determine name of output file
    filename = '%s/%s_coef.csv'%(output_dir, image_type)

    # Write to csv file
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['model_term','value','mean', 'scale'])
        writer.writerow(['intercept', lr.intercept_[0], 1, 1])

        for layer in range(lr.coef_[0].shape[0]):
            writer.writerow([layer,lr.coef_[0][layer], sc.mean_[layer], sc.scale_[layer]])




def main(data, output_dir = _getCfgDir(), regularisation_strength = 1., regularisation_type = 'l1'):
    '''
    
    '''
    
    # Get image_type
    image_type = data.split('/')[-1].split('_')[0]
    
    # Get data
    forest_px, nonforest_px = loadData(data)
    
    # Fit a logistic model
    lr, sc = fitModel(forest_px, nonforest_px, image_type, regularisation_strength = regularisation_strength, regularisation_type = regularisation_type, output_QA = True, output_dir = output_dir)
    
    # Save model coefficients
    saveCoefficients(lr, sc, image_type, output_dir = output_dir)
    



if __name__ == '__main__':
    
    '''
    Script to train a logistic model to classify S1/S2 images into forest/nonforest probabilities. Requires a .npz file from extract_training_data.py.
    '''
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "Ingest Sentinel-1 and Sentinel-2 data to train logistic regression functions.")

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Required arguments
    required.add_argument('data', metavar = 'DATA', type = str, help = 'Path to .npz file containing training data, generated by extract_training_data.py')

    # Optional arguments
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'PATH', default = _getCfgDir(), help="Directory to output model coefficients. Defaults to the cfg directory.")
    optional.add_argument('-c', '--regularisation_strength', type=float, metavar = 'F', default = 1., help="Inverse regularisation strength. Low values result in greater regularisation.")
    optional.add_argument('-r', '--regularisation_type', type=str, metavar = 'T', default = 'l1', help="Regularisation type, either 'l1' or 'l2'. Defaults to 'l1'.")

    # Get arguments
    args = parser.parse_args()
    
    # Execute script
    main(args.data, output_dir = args.output_dir, regularisation_strength = args.regularisation_strength, regularisation_type = args.regularisation_type)
