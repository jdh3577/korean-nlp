import os
import random

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("dataset/binary_train_data_all.csv", encoding='CP949')

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
stopWords.extend(['안녕', '감사'])

# Stratified 10-cross fold validation with SVM and Multinomial NB
num_split = 5
kf = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=42)

totalsvm = 0  # Accuracy measure on 2000 files
totalNB = 0
totalXGB = 0
totalRF = 0
totalEnsemble = 0

totalMatSvm = np.zeros((2, 2))
totalMatNB = np.zeros((2, 2))
totalMatXGB = np.zeros((2, 2))

for train_index, test_index in kf.split(corpus, corpus_lable):
    X_train = [corpus[i] for i in train_index]
    X_test = [corpus[i] for i in test_index]
    y_train = [corpus_lable[i] for i in train_index]
    y_test = [corpus_lable[i] for i in test_index]

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words=stopWords)
    train_corpus_tf_idf = vectorizer.fit_transform(X_train)
    test_corpus_tf_idf = vectorizer.transform(X_test)

    model_SVM = LinearSVC()
    model_NB = MultinomialNB()
    model_XGB = XGBClassifier(n_jobs=-1)
    model_RF = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
    # Ensemble - hard voting
    eclf = VotingClassifier(estimators=[('rf', model_RF), ('nb', model_NB), ('xgb', model_XGB), ('svm', model_SVM)], voting='hard')

    model_SVM.fit(train_corpus_tf_idf, y_train)
    model_NB.fit(train_corpus_tf_idf, y_train)
    model_XGB.fit(train_corpus_tf_idf, y_train)
    model_RF.fit(train_corpus_tf_idf, y_train)
    eclf.fit(train_corpus_tf_idf, y_train)

    result_SVM = model_SVM.predict(test_corpus_tf_idf)
    result_NB = model_NB.predict(test_corpus_tf_idf)
    result_XGB = model_XGB.predict(test_corpus_tf_idf)
    result_RF = model_RF.predict(test_corpus_tf_idf)
    result_Ensemble = eclf.predict(test_corpus_tf_idf)

    totalMatSvm = totalMatSvm + confusion_matrix(y_test, result_SVM)
    totalMatNB = totalMatNB + confusion_matrix(y_test, result_NB)
    totalMatXGB = totalMatXGB + confusion_matrix(y_test, result_XGB)

    totalsvm = totalsvm + np.mean(y_test == result_SVM)
    totalNB = totalNB + np.mean(y_test == result_NB)
    totalXGB = totalXGB + np.mean(y_test == result_XGB)
    totalRF = totalRF + np.mean(y_test == result_RF)
    totalEnsemble = totalEnsemble + np.mean(y_test == result_Ensemble)

vocab = vectorizer.get_feature_names()
print(vocab)

# print("totalMatSVM: ", totalMatSvm)
print("totalSVM: ", totalsvm / num_split)
# print("totalMatNB: ", totalMatNB)
print("totalNB: ", totalNB / num_split)
# print("totalMatXGB: ", totalMatXGB)
print("totalXGB: ", totalXGB / num_split)
print("totalRF: ", totalRF / num_split)
print("Ensemble: ", totalEnsemble / num_split)

print(classification_report(y_test, result_Ensemble))


## inference 테스트
test = [
    '가 나 다 군에 대학교 과 한개씩 넣을수 있는데 만약 나 군에 일반 지원하고 (나 사회적배려대상자) 로 중복 지원이 가능한지 여쭤 보고싶습니다! 만약 된다면 각학교마다 정원외 전형 한개만인가요?',
    '4년재 대학 추가모집기간이 최소 언제부터인가요????  대학홈ㅁ페이지를 둘러봐도 안보이는 대학들은 뭔가요??',
    '2차 4년제 추가모집은 횟수 제한없이 지원할수있나요???',
    '지금 전문대 정시2차에 지원한상태이며 결과를 기다리고있습니다    1. 이상태에서 4년제 추가 접수가 가능한가요?  가능하다면  2.전문대 정시2차에 합격해서 등록한후 나중에 4년체 추가에서 합격한다면    전문대 등록포기후 4년제 등록이 가능한가요??  3. 이중지원은 아니죠?',
    '4년제 추가 예정일이 대략 언제부터 언제인가요? 궁금합니다!',
    '4년제 대학교 추가모집에 합격해서 등록을 한 경우에도   등록금 환불이 가능한가요?  원서 접수할 때 적은 환불 계좌가 어떤 용도인지 궁금합니다.',
    '4년제 추가모집은 원서 개수에 한정이 없는건가요? 가,나,다군 에 포함이 되지 않는거죠? 원하는 만큼 쓸 수 있나요?',
    '지금 진학사를 들어오면 4년제 추가모집 하면서 뜨는데 그게 2차인가요 아님 재외국인이나 외국인 학생에 해당되나요',
    '제 친구가 지원한대학에 모두떨어졌습니다..그래서 말인데 원서접수에 추가지원 이라는게있던데 이것은 새로 원서접수가 가능하단뜻인가요??',
    '안녕하세요    이번 수시1차에 강원도립대학과 충북도립대학 이렇게 두군데 원서접수를 하였습니다    다름이아니라 다른대학(2, 3년제 전문대)에도 원서접수를 해보려고하는데 주변에서 최대지원가능횟수가 정해져있다는말을 들은적있어서 여쭈어봅니다    대학(2, 3년제 전문대)에 몇군대까지 최대지원(원서접수)가능한가요?',

    '한국외국어대학교 수시 접수를 취소하고 환불 요청 드립니다.',
    '원서 환불하려고 하는데 어떻게 진행해야 하나요?',
    '원서접수 잘못해서 취소하고 싶은데  어떻게 해야 하나요?    환불해서 다시 하고싶은데 취소가 않된다네요.',
    '제가 목표로 하는 대학이 바뀌어서 수시를 포기하고 정시를 지원하려고   하는데 이미 결제된 전형료는 환불 받을 수 없나요?  접수는 취소가 안되는 걸로 알고 있는데...',
    '서강대학교 고려대학교   논술전형 입시료를 냈는데   다른대학교 합격이 확정되어   환불 가능한지 궁금합니다 .',
    '서울신학대 실고고사 날짜와 다른대학 실기고사 날짜가 시간대는 겹치지 않았지만 비슷한   시간대라 다른 대학의 실기고사를 보러갔는데.. 그전에취소를 못했어요 전형료 환불 받을수   있나요??있다면 어떻게 하면되나요??'
]

test_text = [get_normalized_data(i) for i in test]
predict_vec = vectorizer.transform(test_text)
print()
print(len(test_text))
print('SVM 분류:', model_SVM.predict(predict_vec))
print('NB 분류:', model_NB.predict(predict_vec))
print('XGBoost 분류:', model_XGB.predict(predict_vec))
print('RF 분류: ', model_RF.predict(predict_vec))
print('앙상블 분류:', eclf.predict(predict_vec))
