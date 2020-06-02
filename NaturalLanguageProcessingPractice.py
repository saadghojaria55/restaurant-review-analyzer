# Natural Language Processing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')  
from nltk.corpus import stopwords  # To Remove Stopwords
from nltk.stem.porter import PorterStemmer  # to transform words to root word loved will change to love 
corpus = [] # a new list with perfect tranform reviews 
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # removing extra things except a to z or AtoZ
    review = review.lower()   # transform to lowercase
    review = review.split()   # to split the reviews in different words to detect each word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # To Remove Stopwords
    review = ' '.join(review) #will seprate words be only space same as data set 
    corpus.append(review)   # adding review to corpus

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer # transforms the corpus sentence in words
cv = CountVectorizer(max_features = 1500)  # takes only 1500 words from the corpus
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifie r.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
