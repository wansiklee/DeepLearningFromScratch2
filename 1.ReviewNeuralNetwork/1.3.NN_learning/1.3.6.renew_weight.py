import numpy as np

# 확률적경사하강법 Stochastic Gradient Descent
class SGD:
    def __init__(self, lr=0.01):
        # lr : learning rate
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

# model = TwoLayerNet(...)
# optimizer = SGD()

# for i in range(10000):
#   ...
#   x_batch, t_batch = get_mini_batch(...)
#   loss = model.forward(x_batch, t_batch)
#   model.backward()
#   optimizer.update(model.params, model.grads)
#   ...