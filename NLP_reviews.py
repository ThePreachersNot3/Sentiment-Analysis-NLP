#importing all the needed/necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import re
import nltk
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#importing the data
df = pd.read_csv(r'\\mycode\\chichi\\Ulta_skincare.csv')
#drop the null values
df = df.dropna()
#using regex to remove unwanted characters
df['Review_Text'] = df['Review_Text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
#converting all letters to lowercase for uniformity
df['Review_Text'] = df['Review_Text'].apply(lambda x: x.lower())

#still using the lambda method since it is easier and shorter to tokenize each row content of the specified column 
df['Review_Text'] = df['Review_Text'].apply(lambda x: word_tokenize(x))
tokens = df['Review_Text']

#initializing the wordlemma
lemma = WordNetLemmatizer()

#a function that takes in the tokenized rows
#lemmatizes and joins them
#then removes the stopwords- words that add little to no meaning in the sentence e.g the, he, have
def token_lemma(tokens):
    return ' '.join(lemma.lemmatize(token) for token in tokens if token not in set(stopwords.words('english')))
#then the joined words without stopwords are returned to each row
df['Review_Text'] = df['Review_Text'].apply(token_lemma)

df['id'] = range(0,len(df))

lol = {}
sia = SentimentIntensityAnalyzer()
for index, row in df.iterrows():
    text = row['Review_Text']
    myid = row['id']
    lol[myid] = sia.polarity_scores(text)
vader = pd.DataFrame(lol).T

vaders = vader.drop(['compound', 'neu'], axis=1)
vaders['yes_no'] = vaders.apply(lambda x: 1 if x['pos'] > x['neg'] else 0, axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer().fit_transform(df['Review_Text'])

X = tf
y = vaders['yes_no']

#split the dataset into train-test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

#a dictionary to iterate through for parameters per model
all_params = {
    'logistic_regression':
        {'model': LogisticRegression(),
        'params': {
            'penalty': ['l1', 'l2'],
            'C': [1,2,3,4,5,6,7,8,9,10,20,50]
        }
    }
}

result = []
for model_name, j in all_params.items():
    gsc = GridSearchCV(j['model'], j['params'])
    gsc.fit(X_train, y_train)
    result.append({
        'model_name': model_name,
        'best_params': gsc.best_params_,
        'best_score' : gsc.best_score_
    })
print(pd.DataFrame(result))



