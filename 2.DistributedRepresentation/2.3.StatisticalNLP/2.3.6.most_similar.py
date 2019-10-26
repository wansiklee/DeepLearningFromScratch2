import numpy as np

## 유사 단어의 랭킹 표시
# query: 검색어(단어)
# word_to_id: 단어에서 단어 ID로의 딕셔너리
# id_to_word: 단어 ID에서 단어로의 딕셔너리
# word_matrix: 단어 벡터들을 한데 모은 행렬
# top: 상위 몇 개까지 출력할지 결정

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 검색어 꺼내기
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return

    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarty(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        # argsort() : 넘파이 배열의 원소를 오름차순으로 정렬, 반환값은 배열의 인덱스
        # 내림차순으로 정렬하기 위해 (-1)을 곱함
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


# "you"를 검색어로 지정해 유사한 단어 출력
import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)

'''
[query] you
 goodbye: 0.7071067691154799
 i: 0.7071067691154799
 hello: 0.7071067691154799
 say: 0.0
 and: 0.0
'''