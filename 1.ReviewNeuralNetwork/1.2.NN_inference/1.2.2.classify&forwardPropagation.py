import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        return out

# 입력 x -> Affine -> Sigmoid -> Affine -> score
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 모든 가중치를 리스트에 모은다.
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)

print(s)
'''
[[ 2.3431784  -2.34684086 -0.80097002]
 [ 1.99715103 -1.9651116  -1.3245735 ]
 [ 1.82470195 -1.96074492 -0.99976957]
 [ 1.86173787 -1.90981269 -1.18052031]
 [ 2.23945258 -2.19258903 -1.11838787]
 [ 1.93330321 -1.9003263  -1.32385503]
 [ 2.03388569 -2.07775603 -1.25389674]
 [ 2.21978215 -2.75022309 -1.3262811 ]
 [ 1.94309912 -2.05198178 -1.19088488]
 [ 2.21713924 -2.0464563  -1.31840692]]
'''