import os
import random
from time import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint as sp_randint

train_df = pd.read_csv("dataset/old_binary_train_data.csv", encoding='CP949')

corpus_data = train_df['QAContent_1'].values
corpus_lable = train_df['label'].values

from konlpy.tag import Twitter

mecab = Twitter()


def get_normalized_data(sentence):
    # original_sentence = mecab.pos(sentence, norm=True, stem=True)
    original_sentence = mecab.pos(sentence, norm=True)
    inputData = []
    for w, t in original_sentence:
        if t not in ['Number', 'Punctuation', 'KoreanParticle']:
            inputData.append(w)
    return (' '.join(inputData)).strip()


corpus = [get_normalized_data(i) for i in corpus_data]

stopWords = []
with open('./dataset/stopwords.txt', encoding='cp949') as f:
    for i in f:
        stopWords.append(i.replace('\n', ''))
stopWords.extend(['안녕','하세요', '감사','합니다'])

totalsvm = 0
totalNB = 0
totalXGB = 0
totalRF = 0
totalEnsemble = 0

totalMatSvm = np.zeros((2, 2))
totalMatNB = np.zeros((2, 2))
totalMatXGB = np.zeros((2, 2))

X_train, X_test, y_train, y_test = train_test_split(corpus, corpus_lable, test_size=0.3, random_state=42)

# vectorizer = CountVectorizer(ngram_range=(1, 4), min_df=1, max_df=500, stop_words=stopWords)
# tfidf_transformer = TfidfTransformer()
# train_corpus_tf_idf = tfidf_transformer.fit_transform(X_train_counts)
# test_corpus_tf_idf = tfidf_transformer.transform(X_test_counts)
#
vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=1, max_df=0.8, sublinear_tf=True, stop_words=stopWords)
train_corpus_tf_idf = vectorizer.fit_transform(X_train)
test_corpus_tf_idf = vectorizer.transform(X_test)

model_SVM = LinearSVC()
model_NB = MultinomialNB()
model_XGB = XGBClassifier(n_jobs=-1)
model_RF = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
model_Ridge = RidgeClassifier(tol=1e-2)
model_pipeline = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])
# eclf = VotingClassifier(estimators=[('rf', model_RF), ('nb', model_NB), ('xgb', model_XGB),
#                                     ('ridge',model_Ridge), ('svm', model_SVM), ('pipeline', model_pipeline)], voting='hard')
eclf = VotingClassifier(estimators=[('rf', model_RF), ('nb', model_NB), ('xgb', model_XGB),
                                    ('ridge',model_Ridge), ('svm', model_SVM)], voting='soft') # weights=[1,1,1]


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {#"svm__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              "xgb__max_depth": sp_randint(3, 25),
              "xgb__min_child_weight": sp_randint(1, 7),
              "xgb__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
              "xgb__reg_lambda": [0.01, 0.1, 1.0],
              "xgb__reg_alpha": [0, 0.1, 0.5, 1.0],
              "rf__n_estimators": [10, 50, 100, 150, 200, 300, 500],
              "rf__max_depth": [5, 8, 15, 25, 30, None],
              "rf__max_features": sp_randint(1, 11),
              "rf__min_samples_split": sp_randint(2, 100),
              "rf__min_samples_leaf": sp_randint(1, 11),
              "rf__bootstrap": [True, False],
              "rf__criterion": ["gini", "entropy"]}

# run randomized search
from tqdm import tqdm, trange
n_iter_search = 10000
start = time()
for i in trange(n_iter_search):
    random_search = RandomizedSearchCV(estimator=eclf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
    random_search.fit(train_corpus_tf_idf, y_train)


print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)
#
# # use a full grid over all parameters
# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [2, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}
#
# # run grid search
# grid_search = GridSearchCV(eclf, param_grid=param_grid)
# start = time()
# grid_search.fit(train_corpus_tf_idf, y_train)
#
# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)