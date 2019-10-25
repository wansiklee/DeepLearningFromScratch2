import sys
sys.path.append('..') # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape) # 입력 데이터 (300, 2)
print('t', t.shape) # 정답 레이블 (300, 3) one_hot