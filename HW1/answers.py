# Some print statements have been removed to produce
# a less verbose output

# Read data

import pandas as pd

file_path = './amazon_reviews_us_Office_Products_v1_00.tsv'
full_df = pd.read_csv(file_path, delimiter='\t', on_bad_lines='skip')
full_df.dropna()


# Keep reviews and ratings

cols = ['review_body', 'star_rating']
reviews_ratings_df = full_df[cols]


# Form two classes and select 50000 reviews randomly from each class

class_1 = reviews_ratings_df[reviews_ratings_df['star_rating'].isin([1, 2, 3])].copy()
class_2 = reviews_ratings_df[reviews_ratings_df['star_rating'].isin([4, 5])].copy()

sample_size = 50_000
class_1 = class_1.sample(n=min(len(class_1), sample_size))
class_2 = class_2.sample(n=min(len(class_2), sample_size))
classified_df = pd.concat([class_1, class_2])

classified_df.loc[:, 'class'] = classified_df['star_rating'].apply(lambda x: 1 if x in [1,2,3] else 2)


# Data cleaning

import re
from contractions import fix

cleaned_df = classified_df.dropna(subset=['review_body'])
cleaned_df.loc[:, 'review_body'] = cleaned_df['review_body'].astype(str)
cleaned_df.loc[:, 'review_body'] = cleaned_df['review_body'].str.lower()
cleaned_df.loc[:, 'review_body'] = cleaned_df['review_body'].apply(lambda x: re.sub(r'<.*?>', '', x))
cleaned_df.loc[:, 'review_body'] = cleaned_df['review_body'].apply(lambda x: re.sub(r'http\S+', '', x))
cleaned_df.loc[:, 'review_body'] = cleaned_df['review_body'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
cleaned_df.loc[:, 'review_body'] = cleaned_df['review_body'].apply(lambda x: ' '.join(x.split()))
cleaned_df.loc[:, 'review_body'] = cleaned_df['review_body'].apply(lambda x: fix(x))

ave_len_bef = classified_df['review_body'].str.len().mean()
ave_len_aft = cleaned_df['review_body'].str.len().mean()

print(f'{ave_len_bef:.1f}, {ave_len_aft:.1f}')


# Pre-processing

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stop_words(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

no_stopwords_df = cleaned_df
no_stopwords_df.loc[:, 'review_body'] = no_stopwords_df['review_body'].apply(remove_stop_words)

from nltk.stem import WordNetLemmatizer

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

lemmatized_df = no_stopwords_df
lemmatized_df.loc[:, 'review_body'] = lemmatized_df['review_body'].apply(lemmatize)

ave_len_bef = cleaned_df['review_body'].str.len().mean()
ave_len_aft = classified_df['review_body'].str.len().mean()

print(f'{ave_len_bef:.1f}, {ave_len_aft:.1f}')


# TF-IDF and BoW Feature Extraction

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

processed_df = lemmatized_df
max_features = 5000

bow_vectorizer = CountVectorizer(max_features=max_features)
X_bow = bow_vectorizer.fit_transform(processed_df['review_body'])

tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
X_tfidf = tfidf_vectorizer.fit_transform(processed_df['review_body'])

y = processed_df['class']
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(X_bow, y, test_size=0.20)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y, test_size=0.20)


# Perceptron Using Both Features

from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score, f1_score

max_iters = 10_000

perc_bow = Perceptron(max_iter=max_iters)
perc_bow.fit(X_train_bow, y_train_bow)
y_pred_bow = perc_bow.predict(X_test_bow)

precision_bow = precision_score(y_test_bow, y_pred_bow)
recall_bow = recall_score(y_test_bow, y_pred_bow)
f1_bow = f1_score(y_test_bow, y_pred_bow)

print(f'{precision_bow:.2f} {recall_bow:.2f} {f1_bow:.2f}')

perc_tfidf = Perceptron(max_iter=max_iters)
perc_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_tfidf = perc_tfidf.predict(X_test_tfidf)

precision_tfidf = precision_score(y_test_tfidf, y_pred_tfidf)
recall_tfidf = recall_score(y_test_tfidf, y_pred_tfidf)
f1_tfidf = f1_score(y_test_tfidf, y_pred_tfidf)

print(f'{precision_bow:.2f} {recall_bow:.2f} {f1_bow:.2f}')


# SVM Using Both Features

from sklearn.svm import LinearSVC

svm_bow = LinearSVC(max_iter=max_iters, dual=False)
svm_bow.fit(X_train_bow, y_train_bow)
y_pred_bow = svm_bow.predict(X_test_bow)

precision_bow = precision_score(y_test_bow, y_pred_bow)
recall_bow = recall_score(y_test_bow, y_pred_bow)
f1_bow = f1_score(y_test_bow, y_pred_bow)

print(f'{precision_bow:.2f} {recall_bow:.2f} {f1_bow:.2f}')

svm_tfidf = LinearSVC(max_iter=max_iters, dual=False)
svm_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_tfidf = svm_tfidf.predict(X_test_tfidf)

precision_tfidf = precision_score(y_test_tfidf, y_pred_tfidf)
recall_tfidf = recall_score(y_test_tfidf, y_pred_tfidf)
f1_tfidf = f1_score(y_test_tfidf, y_pred_tfidf)

print(f'{precision_bow:.2f} {recall_bow:.2f} {f1_bow:.2f}')


# Logistic Regression Using Both Features

from sklearn.linear_model import LogisticRegression

logreg_bow = LogisticRegression(max_iter=max_iters)
logreg_bow.fit(X_train_bow, y_train_bow)
y_pred_bow = logreg_bow.predict(X_test_bow)

precision_bow = precision_score(y_test_bow, y_pred_bow)
recall_bow = recall_score(y_test_bow, y_pred_bow)
f1_bow = f1_score(y_test_bow, y_pred_bow)

print(f'{precision_bow:.2f} {recall_bow:.2f} {f1_bow:.2f}')

logreg_tfidf = LogisticRegression(max_iter=max_iters)
logreg_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_tfidf = logreg_tfidf.predict(X_test_tfidf)

precision_tfidf = precision_score(y_test_tfidf, y_pred_tfidf)
recall_bow = recall_score(y_test_tfidf, y_pred_tfidf)
f1_bow = f1_score(y_test_tfidf, y_pred_tfidf)

print(f'{precision_bow:.2f} {recall_bow:.2f} {f1_bow:.2f}')


# Naive Bayes Using Both Features

from sklearn.naive_bayes import MultinomialNB

nb_bow = MultinomialNB()
nb_bow.fit(X_train_bow, y_train_bow)
y_pred_bow = nb_bow.predict(X_test_bow)

precision_bow = precision_score(y_test_bow, y_pred_bow)
recall_bow = recall_score(y_test_bow, y_pred_bow)
f1_bow = f1_score(y_test_bow, y_pred_bow)

print(f'{precision_bow:.2f} {recall_bow:.2f} {f1_bow:.2f}')

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

precision_tfidf = precision_score(y_test_tfidf, y_pred_tfidf)
recall_bow = recall_score(y_test_tfidf, y_pred_tfidf)
f1_bow = f1_score(y_test_tfidf, y_pred_tfidf)

print(f'{precision_bow:.2f} {recall_bow:.2f} {f1_bow:.2f}')

