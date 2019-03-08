#%reset -f
#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset1 = pd.read_csv('cancer_data.csv')
dataset2 = pd.read_csv('cancer_data_y.csv')
dataset3 = pd.read_csv('test_cancer_data.csv')
dataset4 = pd.read_csv('test_cancer_data_y.csv')

#making the dependent and independent values
X_train = dataset1.iloc[:,:].values
y_train = dataset2.iloc[:, ].values
X_test  = dataset3.iloc[:, ].values
y_test  = dataset4.iloc[:, ].values
#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting Random forest to Training Set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#96 percentage

# Fitting K-NN to the Training set

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)
#81.4 percentage


# Fitting Kernel SVM to the Training set

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, y_train)
#82.22 percentage


# Fitting Naive Bayes to the Training set

#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)
#75.8 percentage

# Fitting Decision Tree Classification to the Training set

#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)
#91.4 percentage

#making prediction
y_pred = classifier.predict(X_test)


#confusion matrix for accuracy prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
