"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
import numpy as np
from tensors import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

# TODO: Cross Entropy or another Loss function
class MSE(Loss): # Good loss function
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return  2 * (predicted - actual) # Need to study