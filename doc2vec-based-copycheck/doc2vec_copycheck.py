# -*- coding: utf-8 -*-


import gensim
import collections
import smart_open
import random
import multiprocessing

import os

os.environ["NLS_LANG"] = ".UTF8"
os.chdir("D:\\Oracle\\instantclient_12_2")

import cx_Oracle

'''
dns_tns = cx_Oracle.makedsn('165.132.174.133','1521', service_name ="JEMS1108")
'''

conn = cx_Oracle.connect('-----', '----', '----')

db = conn.cursor()
# db.execute(u"select IpsiYear, IpsiGubun, SuhumNo, SpeArea, TO_CHAR(SpeContents) as Contents from HSBSpecial where IpsiYear='2017' and IpsiGubun='1'")
db.execute(
    """select IpsiYear, IpsiGubun, SuhumNo, SpeArea, TO_CHAR(SpeContents) as Contents from HSBSpecial where length(SpeContents)<2000 and IpsiYear='2017' and IpsiGubun='1'""")

import sys

print(sys.getdefaultencoding())  # 현재 default 문자 코드를 확인하는 부분. -- utf-8

matrix = []
# for record in db:
#     #print(type(db), type(record))
#     #print(record[0], record[1], record[2], record[3], record[4])
#     if record[4] !=" " :
#         matrix.append(record[4])


################## 2016년 비교 대상 #################
conn2 = cx_Oracle.connect('JEMSV3', 'A12345', 'JEMS1108')
db2 = conn2.cursor()
db2.execute(
    """select IpsiYear, IpsiGubun, SuhumNo, SpeArea, TO_CHAR(SpeContents) from HSBSpecial where IpsiYear='2016' and IpsiGubun='1' and length(SpeContents)<2000""")
test_matrix = []

num_matrix = []


# # for record2 in db2 :
#     if record2[4] !=" " :
#         test_matrix.append(record2[4])

def del_duple(train_snum_temp, train_string_temp):
    train_snum = []
    train_string = []

    train_snum_temp.append("aaaaaa")
    train_string_temp.append("aaaaaa")

    temp_string = train_string_temp[0]

    for i in range(len(train_snum_temp) - 1):
        if train_snum_temp[i] != train_snum_temp[i + 1]:
            train_snum.append(train_snum_temp[i])
            train_string.append(temp_string)
            temp_string = train_string_temp[i + 1]
        else:
            temp_string = temp_string + train_string_temp[i + 1]

    train_snum_temp.pop(len(train_snum_temp) - 1)
    train_string_temp.pop(len(train_string_temp) - 1)
    return train_snum, train_string


train_snum_temp = []
train_string_temp = []
for record in db:
    # print(type(db), type(record))
    # print(record[0], record[1], record[2], record[3], record[4])
    if record[4] != " ":
        train_snum_temp.append(record[2])
        train_string_temp.append(record[4])

train_snum, train_string = del_duple(train_snum_temp, train_string_temp)

test_snum_temp = []
test_string_temp = []
for record in db2:
    # print(type(db), type(record))
    # print(record[0], record[1], record[2], record[3], record[4])
    if record[4] != " ":
        train_snum_temp.append(record[2])
        train_string_temp.append(record[4])

test_snum, test_string = del_duple(train_snum_temp, train_string_temp)
a = "학급의 바른생활부원으로 예능교양부 부원으로서 학교축제 학예활동 취미활동 동아리활동에 적극적으로 참여함 청주시립교향악단 실내악 연주 관람 부모산성 연화사 현장답사 자연보호활동 한민구 전임합참의장 꿈과 열정 특별강연 참가 yjsc 양지사커클럽 스포츠에 관심이 많고 열정과 지구력 끈기를 가지고 있음 꾸준한 연습과 노력을 통하여 빠른 성장이 보임 생태탐구반 탐구능력과 창의성이 뛰어나고 적극적으로 활동함 학급회 체육활동부원으로 봉사활동 체육활동 적극적으로 활동함 입시설명회 한국외국어대학교 입학사정관 이석록 충북대 입시컨설팅 ebs 참가하여 진로를 탐색하고 결정하는 도움을 받음 교내 체육대회 줄다리기 경기에 참가해 학급의 단합된 힘을 보여줌 입시설명회 서원대 충북대 한국교통대 청주대 충청대 참가하여 진로를 탐색하고 결정하는 도움을 받음 우암산 자연환경 보호활동 통해 환경의 소중함을 깨달았으며 극기훈련 상당산성 등반 통해 자신을 이기는 희열을 느낌 나라사랑 안보교육 통해 한반도 주변 정세를 파악하고 통일안보 의식을 제고하였으며 독립기념관을 탐방 하여 조상들의 나라사랑 정신을 깨닫게 독서토론반 매달 마지막 계발활동 시간에 합동 독서 토론회를 개최함 계발 활동에 적극적이고 작품의 주제 파악 구조 분석을 하며 자신의 주장을 설득력 있게 발표함"
b = "TEST용 입니다"
#  train_string.append(a)
test_string.append(a)
test_snum.append(b)

db.close()
conn.close()

db2.close()
conn2.close()


# Define a Function to Read and Preprocess Text
def read_corpus(data, tokens_only=False):
    # with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
    for i, line in enumerate(data):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
            # convert a document into a list of tokens - simple_prpeprocess가 하는 것.
            # print(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line),
                                                       [i])  # 새로운 taggedDocument객체를 생성하는것, parameter로 words, tags 가 있음.


# lee_train_file 에 있는걸..
train_corpus = list(read_corpus(train_string))  # train경우에는 doc을 tokenlist로 된거에다가 ID를 달아 taggeddocument로 만듦.

test_corpus = list(read_corpus(test_string, tokens_only=True))  # test_corpus의 경우에는 doc을 token list로 잘라버림
# test_corpus = list(read_corpus(matrix[100:5966], tokens_only=True))



# model = gensim.models.doc2vec.Doc2Vec(dm=1, size=200, min_count=3, iter=50, alpha=0.01, negative=20, dm_mean=0, dm_concat=1, workers=4)
# model = gensim.models.doc2vec.Doc2Vec(dm=1, size=200, min_count=3, iter=50, alpha=0.01, negative=20, dm_concat=1, workers=4)
model = gensim.models.doc2vec.Doc2Vec(dm=0, size=300, min_count=3, iter=30, alpha=0.01, negative=20, dbow_words=1, workers=8, window=15)

# Build a Vocabulary
model.build_vocab(train_corpus)

# Trainining
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
print(model.most_similar(u'독서토론반', topn=20))
# Assessing Model
ranks = []
first_ranks = []
second_ranks = []
third_ranks = []
fourth_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    first_ranks.append(sims[0])
    second_ranks.append(sims[1])
    third_ranks.append(sims[2])
    fourth_ranks.append(sims[3])

# list 내의 요소가 몇 번 반복되었는지 count된걸 알려주는 부분...
# 0 값이 많이 나와야 unique한게 나오는...
print(collections.Counter(ranks))  # Results vary due to random seeding and very small corpus

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('1', 0), ('2', 1), ('3', 2), ('4', 3), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s %s: «%s»\n' % (label, sims[index], train_snum[index], ' '.join(train_corpus[sims[index][0]].words)))

# Pick a random document from the test corpus and infer a vector from the model
for i in range(0, 10):
    doc_id = random.randint(0, len(train_corpus))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Train Document ({}, {}): «{}»\n'.format(doc_id, train_snum[doc_id], ' '.join(train_corpus[doc_id].words)))
    sim_id0 = first_ranks[doc_id]
    sim_id1 = second_ranks[doc_id]
    sim_id2 = third_ranks[doc_id]
    sim_id3 = fourth_ranks[doc_id]
    print('Similar Document_0 {}, {}: «{}»\n'.format(sim_id0, train_snum[sim_id0[0]],
                                                     ' '.join(train_corpus[sim_id0[0]].words)))
    print('Similar Document_1 {}, {}: «{}»\n'.format(sim_id1, train_snum[sim_id1[0]],
                                                     ' '.join(train_corpus[sim_id1[0]].words)))
    print('Similar Document_2 {}, {}: «{}»\n'.format(sim_id2, train_snum[sim_id2[0]],
                                                     ' '.join(train_corpus[sim_id2[0]].words)))
    print('Similar Document_3 {}, {}: «{}»\n'.format(sim_id3, train_snum[sim_id3[0]],
                                                     ' '.join(train_corpus[sim_id3[0]].words)))

# Pick a random document from the test corpus and infer a vector from the model
# doc_id = random.randint(0, len(test_corpus))
doc_id = len(test_corpus) - 1
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('\n')
print('Test Document ({}, {}): «{}»\n'.format(doc_id, test_snum[doc_id], ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('1', 0), ('2', 1), ('3', 2), ('4', 3), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s %s: «%s»\n' % (label, sims[index], test_snum[index], ' '.join(train_corpus[sims[index][0]].words)))

# 테스트용 사용자 입력 추가
print('Document id를 입력해주세요 0~{}(종료 : -1)\n'.format(len(train_corpus)))
test_index = int(input())

while test_index > 0:
    # Compare and print the most/median/least similar documents from the train corpus
    print('Train Document ({}, {}): «{}»\n'.format(test_index, train_snum[test_index], ' '.join(train_corpus[test_index].words)))
    sim_id0 = first_ranks[test_index]
    sim_id1 = second_ranks[test_index]
    sim_id2 = third_ranks[test_index]
    sim_id3 = fourth_ranks[test_index]
    print('Similar Document_0 {}, {}: «{}»\n'.format(sim_id0, train_snum[sim_id0[0]],
                                                     ' '.join(train_corpus[sim_id0[0]].words)))
    print('Similar Document_1 {}, {}: «{}»\n'.format(sim_id1, train_snum[sim_id1[0]],
                                                     ' '.join(train_corpus[sim_id1[0]].words)))
    print('Similar Document_2 {}, {}: «{}»\n'.format(sim_id2, train_snum[sim_id2[0]],
                                                     ' '.join(train_corpus[sim_id2[0]].words)))
    print('Similar Document_3 {}, {}: «{}»\n'.format(sim_id3, train_snum[sim_id3[0]],
                                                     ' '.join(train_corpus[sim_id3[0]].words)))

    print('Document id를 입력해주세요 0~{}(종료 : -1)\n'.format(len(train_corpus)))
    test_index = int(input())