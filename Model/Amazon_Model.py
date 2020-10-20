#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import glob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk.classify.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.classify import NaiveBayesClassifier
import numpy as np
import re
import string
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import cohen_kappa_score


# In[ ]:


df = pd.read_csv(r'E:\\EDA\\fill.csv')
df.head()


# In[ ]:


def NaiveBaiyes_Sentimental(sentence):
    blob = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
    NaiveBayes_SentimentScore=blob.sentiment.classification
    return NaiveBayes_SentimentScore


# In[ ]:


# VADER sentiment analysis tool for getting Compound score.
def sentimental(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    return score

# VADER sentiment analysis tool for getting pos, neg and neu.
def sentimental_Score(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    if score > 0:
        return 'pos'
    else:
        return 'neg'


# In[ ]:


df['senti']=df['Reviews'].apply(lambda x: sentimental_Score(x))


# In[ ]:


df.head(10)


# In[ ]:


print(df.isnull().sum())


# In[ ]:


senti = df.drop(df.columns[[0]], axis=1)


# In[ ]:


senti.head()


# In[ ]:


senti['senti'].value_counts()


# In[ ]:


senti['senti'].value_counts().plot.bar()


# In[ ]:


from sklearn import preprocessing


# In[ ]:


label_encoder = preprocessing.LabelEncoder()


# In[ ]:


senti['senti']= label_encoder.fit_transform(df['senti'])


# In[ ]:


senti['senti'].unique()


# In[ ]:


import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    i= re.sub("[\W+""]", " ",i)        
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


# In[ ]:


senti['Reviews']=senti.Reviews.apply(cleaning_text)


# In[ ]:


senti.head(10)


# In[ ]:


split = senti[['Reviews', 'senti']]
train = split.sample(frac=0.8,random_state = 200)
test = split.drop(train.index)


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


trainX=train.Reviews
trainy=train.senti
testX=test.Reviews
testy=test.senti


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


stopwords = set(STOPWORDS)
stopwords.remove("not")


# In[ ]:


count_vect = CountVectorizer(min_df=2 ,stop_words=stopwords , ngram_range=(1,2))
tfidf_transformer = TfidfTransformer()


# In[ ]:


X_train_counts = count_vect.fit_transform(train["Reviews"])        
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[ ]:


X_new_counts = count_vect.transform(test["Reviews"])
X_test_tfidf = tfidf_transformer.transform(X_new_counts)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression()
model.fit(X_train_tfidf,trainy)


# In[ ]:


y_pred=model.predict(X_test_tfidf)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[ ]:


print(accuracy_score(testy,y_pred))
print(classification_report(testy,y_pred))
print(confusion_matrix(testy,y_pred))
print(cohen_kappa_score(testy,y_pred))


# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


sm=SMOTE(random_state=444)


# In[ ]:


X_train_res, y_train_res = sm.fit_resample(X_train_tfidf, trainy)


# # Logistic Regression After SMOTE

# In[ ]:


model2=LogisticRegression()
model2.fit(X_train_res,y_train_res)


# In[ ]:


y_pred2=model2.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred2))
print(classification_report(testy,y_pred2))
print(confusion_matrix(testy,y_pred2))
print(cohen_kappa_score(testy,y_pred2))


# # MultinomialNB

# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


model3=MultinomialNB()
model3.fit(X_train_tfidf,trainy)


# In[ ]:


y_pred3=model3.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred3))
print(classification_report(testy,y_pred3))
print(confusion_matrix(testy,y_pred3))
print(cohen_kappa_score(testy,y_pred3))


# # MultinomialNB After SMOTE

# In[ ]:


model4=MultinomialNB()
model4.fit(X_train_res,y_train_res)


# In[ ]:


y_pred4=model4.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred4))
print(classification_report(testy,y_pred4))
print(confusion_matrix(testy,y_pred4))
print(cohen_kappa_score(testy,y_pred4))


# # BernoulliNB

# In[ ]:


from sklearn.naive_bayes import BernoulliNB


# In[ ]:


model5=BernoulliNB()
model5.fit(X_train_tfidf,trainy)


# In[ ]:


y_pred5=model5.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred5))
print(classification_report(testy,y_pred5))
print(confusion_matrix(testy,y_pred5))
print(cohen_kappa_score(testy,y_pred5))


# # BernoulliNB After SMOTE

# In[ ]:


model6=BernoulliNB()
model6.fit(X_train_res,y_train_res)


# In[ ]:


y_pred6=model6.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred6))
print(classification_report(testy,y_pred6))
print(confusion_matrix(testy,y_pred6))
print(cohen_kappa_score(testy,y_pred6))


# # SVM

# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


model7=LinearSVC()
model7.fit(X_train_tfidf,trainy)


# In[ ]:


y_pred7=model7.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred7))
print(classification_report(testy,y_pred7))
print(confusion_matrix(testy,y_pred7))
print(cohen_kappa_score(testy,y_pred7))


# # SVM After Smote

# In[ ]:


model8=LinearSVC()
model8.fit(X_train_res,y_train_res)


# In[ ]:


y_pred8=model8.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred8))
print(classification_report(testy,y_pred8))
print(confusion_matrix(testy,y_pred8))
print(cohen_kappa_score(testy,y_pred8))


# # DecisionTree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model9=DecisionTreeClassifier(criterion='entropy')
model9.fit(X_train_tfidf,trainy)


# In[ ]:


y_pred9=model9.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred9))
print(classification_report(testy,y_pred9))
print(confusion_matrix(testy,y_pred9))
print(cohen_kappa_score(testy,y_pred9))


# # Decision Tree After SMOTE

# In[ ]:


model10=DecisionTreeClassifier(criterion='entropy')
model10.fit(X_train_res,y_train_res)


# In[ ]:


y_pred10=model10.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred10))
print(classification_report(testy,y_pred10))
print(confusion_matrix(testy,y_pred10))
print(cohen_kappa_score(testy,y_pred10))


# # Neural Network

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


model11=MLPClassifier(hidden_layer_sizes=(5,5))
model11.fit(X_train_tfidf,trainy)


# In[ ]:


y_pred11=model11.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred11))
print(classification_report(testy,y_pred11))
print(confusion_matrix(testy,y_pred11))
print(cohen_kappa_score(testy,y_pred11))


# # Neural Network After SMOTE

# In[ ]:


model12=MLPClassifier(hidden_layer_sizes=(5,5))
model12.fit(X_train_res,y_train_res)


# In[ ]:


y_pred12=model12.predict(X_test_tfidf)


# In[ ]:


print(accuracy_score(testy,y_pred12))
print(classification_report(testy,y_pred12))
print(confusion_matrix(testy,y_pred12))
print(cohen_kappa_score(testy,y_pred12))

