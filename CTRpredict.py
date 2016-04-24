
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from scipy import interp
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np


def create_submission(learner, train_data, training_set, test_data):
    """ This function does trains the model and create a submission file in csv format.

        Args:
            param1: Training model
            param2: Training data
	    param3: Training set
	    param4: Test data

        Returns:
	    Void

    """
    learner.fit(train_data, training_set['click'])
    test_pred = learner.predict_proba(test_data)[:,1]
    pred_frame = pd.DataFrame(data=test_pred,columns=['Prediction'])
    pred_frame.index += 1
    pred_frame.index.name = 'Id'
    pred_frame.to_csv('out.csv')
    
def prepare_data_sets(data, test_set):
    """ This function transforms the training data set to a one-hot encoding of 
	    the selected features for the training model

        Args:
            param1: Training Data set
            param2: Test Data set

        Returns:
            Training data in transformed format
            Test data in transformed format

    """
    vec = DictVectorizer()
    data['ad area'] = data['ad slot width'] * data['ad slot height']
    test_set['ad area'] = test_set['ad slot width'] * test_set['ad slot height']
    selector = SelectPercentile(f_classif, percentile=10)
    cat_features = ['user-agent', 'ad area', 'ad slot visibility',
                    'ad exchange', 'ad slot format', 'ad slot floor price',
                    'weekday', 'hour','ad slot id','domain',
                    ]
    
    for f in cat_features:
        data[f] = data[f].astype('str')

    train_data = vec.fit_transform(data[cat_features].to_dict('records'))
    test_data = vec.transform(test_set[cat_features].to_dict('records'))
    selector.fit(train_data, training_set['click'])
    return selector.transform(train_data), selector.transform(test_data)

def cross_folds(learner, train_data, training_set):
    """ This function splits the raw training data to train and test sets.
	    The function also generates ROC's for 5 differnt folds based on the training model
	    and plots them with an average ROC.

        Args:
            param1: Learning model
            param2: One hot encoded Training data
            param2: Raw training data set

        Returns:
            Void

    """
    
    folds = StratifiedKFold(training_set['click'], 5)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    for count, (train, test) in enumerate(folds):
        predictions = learner.fit(train_data[train, :], training_set['click'].loc[train]).predict_proba(train_data[test, :])
        fpr, tpr, _ = metrics.roc_curve(training_set['click'].loc[test], predictions[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (count, roc_auc))
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr /= len(folds)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.show()
    


""" Read the training data into pandas data frame format  """
training_set = pd.read_csv('data_train.txt', delimiter='\t') 

""" Read the Test data into pandas data frame format  """
test_set = pd.read_csv('shuffle_data_test.txt', delimiter='\t')

""" Create a learning model based on Logistic regression """
learner = linear_model.LogisticRegression()

""" Transform the data sets to a format that aids 
    leaning model to make a better prediction """
train_data, test_data = prepare_data_sets(training_set, test_set)

""" Split data in to train and test """
cross_folds(learner, train_data, training_set)

""" Train model and create submission file in csv format """
create_submission(learner, train_data, training_set, test_data)
   
