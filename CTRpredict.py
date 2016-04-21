import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from scipy import interp
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np


def create_submission(learner, train_data, training_set, test_data):
    learner.fit(train_data, training_set['click'])
    test_pred = learner.predict_proba(test_data)[:,1]
    pred_frame = pd.DataFrame(data=test_pred,columns=['Prediction'])
    pred_frame.index += 1
    pred_frame.index.name = 'Id'
    pred_frame.to_csv('out.csv')
    
def prepare_data_sets(data, test_set):
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
        test_set[f] = test_set[f].astype('str')
        
    '''
    tags = data['user tags'].str.split(',')
    test_tags = test_set['user tags'].str.split(',')
    unique_tags = pd.Series(data['user tags'].str.split(',',expand=True).values.ravel()).unique()
    unique_tags = unique_tags[unique_tags != np.array(None)]
    unique_tags = 'tag_'+unique_tags

    for tag in unique_tags:
        data[tag] = tags.map(lambda x: 1 if tag in x else 0)
        test_set[tag] = test_tags.map(lambda x: 1 if tag in x else 0)
        
    cat_features.extend(unique_tags)
    '''
    train_data = vec.fit_transform(data[cat_features].to_dict('records'))
    test_data = vec.transform(test_set[cat_features].to_dict('records'))
    selector.fit(train_data, training_set['click'])
    return selector.transform(train_data), selector.transform(test_data)

def cross_folds(learner, train_data, training_set):
    
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
    


training_set = pd.read_csv('data_train.txt', delimiter='\t')
test_set = pd.read_csv('shuffle_data_test.txt', delimiter='\t')
learner = linear_model.LogisticRegression()

train_data, test_data = prepare_data_sets(training_set, test_set)
cross_folds(learner, train_data, training_set)
# create_submission(learner, train_data, training_set, test_data)
   
