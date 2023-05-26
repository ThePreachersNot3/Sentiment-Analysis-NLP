import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import re
import nltk

df = pd.read_csv(r'\mycode\chichi\Ulta_skincare.csv')
df = df.dropna()
df['Review_Text'] = df['Review_Text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['Review_Text'] = df['Review_Text'].apply(lambda x: x.lower())

from nltk.tokenize import word_tokenize
df['Review_Text'] = df['Review_Text'].apply(lambda x: word_tokenize(x))
tokens = df['Review_Text']

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

def token_lemma(tokens):
    return ' '.join(lemma.lemmatize(token) for token in tokens if token not in set(stopwords.words('english')))
df['Review_Text'] = df['Review_Text'].apply(token_lemma)

df['id'] = range(0,len(df))
from nltk.sentiment import SentimentIntensityAnalyzer
lol = {}
sia = SentimentIntensityAnalyzer()
for index, row in df.iterrows():
    text = row['Review_Text']
    myid = row['id']
    lol[myid] = sia.polarity_scores(text)
vader = pd.DataFrame(lol).T

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

vaders = vader.drop(['compound', 'neu'], axis=1)
vaders['yes_no'] = vaders.apply(lambda x: 1 if x['pos'] > x['neg'] else 0, axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer().fit_transform(df['Review_Text'])

X = tf
y = vaders['yes_no']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

all_params = {
    'logistic_regression':
        {'model': LogisticRegression(),
        'params': {
            'penalty': ['l1', 'l2'],
            'C': [1,2,3,4,5,6,7,8,9,10,20,50]
        }
    }
}

from sklearn.model_selection import GridSearchCV
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



