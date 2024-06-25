"""
We use an optimizer to adjust the parameters of our network based on the gradient computed during
backpropagation
"""
from nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

# TODO: RMS Prop, can implement different optimizers?
class SGD(Optimizer): # Need to study SGD optimization (steepest gradient descent)
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad