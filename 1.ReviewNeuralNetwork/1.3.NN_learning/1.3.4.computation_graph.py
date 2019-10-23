import numpy as np

## Repeat 노드
D, N = 8, 7
x = np.random.randn(1, D)              # 입력
y = np.repeat(x, N, axis=0)            # 순전파

dy = np.random.randn(N, D)             # 무작위 기울기
dx = np.sum(dy, axis=0, keepdims=True) # 역전파

## Sum 노드
x = np.random.randn(N, D)              # 입력
y = np.sum(x, axis=0, keepdims=True)   # 순전파

dy = np.random.randn(1, D)             # 무작위 기울기
dx = np.repeat(dy, N, axis=0)          # 역전파

## MatMul 노드
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx