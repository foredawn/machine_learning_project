#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_payments','bonus','expenses',
                 'exercised_stock_options','restricted_stock'] 
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# explore_data
'''
len(data_dict)
len(data_dict["SKILLING JEFFREY K"])
# 有多少个人是POI？
n = 0
for i in data_dict:
    if data_dict[i]["poi"]==1:
        n=n+1
print n
'''

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### feature rescaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#try naive_bayes
'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
# Accuracy: 0.85880       Precision: 0.44843      Recall: 0.25650
'''

##try KNN
'''
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(weights='distance')
# clf.fit(features_train,labels_train)
# Accuracy: 0.87993       Precision: 0.64741      Recall: 0.21850
'''


## try decisionTreeClassifier
#'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
parameters = {'criterion':('gini', 'entropy'),
              'splitter':('best', 'random'),
              'min_samples_split':[3,4,5],
              'random_state':[0]}
dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters)
#clf.fit(features_train,labels_train)
# Accuracy: 0.83593       Precision: 0.36189      Recall: 0.30200
#'''



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
'''
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
''' 

# 交叉验证

'''
from sklearn.cross_validation import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
        
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
pred = clf.predict(features_test)
acc = accuracy_score(labels_test,pred)
ave = precision_score(labels_test,pred)
rec = recall_score(labels_test,pred)

'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


