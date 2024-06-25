"""
A neural net is a collection of layers.
It behaves a lot like a layer itself although it won't be one here.
"""

from typing import Sequence, Iterator, Tuple
from tensors import Tensor
from layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor: # Need to study
        for layer in self.layers:
            inputs = layer.forward(inputs) 

        return inputs
    
    def backward(self, grad: Tensor) -> Tensor: # Need to study
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad
    
    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]: # Need to study Iterator type, Tuple type
        for layer in self.layers:
            for name, param in layer.params.items(): # What is this doing?
                grad = layer.grads[name]
                yield param, grad
                # Yield returns something without exiting the function