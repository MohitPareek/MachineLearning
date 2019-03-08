#%reset -f

#IMPORTING LIBRBRY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#FETCHING DATASET
dataset = pd.read_csv('spamDectector.csv')



#CLEANING DATASET


import re
from  nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
#7.repeat each step for all

for i in range(0,5728):
    #1.removing commas and all
    spam=re.sub('[^a-zA-Z]',' ',dataset['text'][i])
    
    #2.convert to lower order
    spam=spam.lower()
    
    #3.split the sentence into word
    spam=spam.split()
    
    #4.removing unuseful keywords
    #5. stem the similar words
    
    
    ps=PorterStemmer()
    spam =[ps.stem(word) for word in spam if not word in set(stopwords.words('english'))]
    
    #6. join the words back
    spam=' '.join(spam)
    
    #8.add to corpus
    corpus.append(spam)
    

#MAKING BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values  


#USING NAVIEW BAYES CLASSIFICATION

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# 94% ACCURUCY


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  
