from typing import List
import numpy as np
import random
from rmicrograd.src.engine import Value

class Module:
    def zero_grad(self):
        pass

    def parameters(self):
        pass


class Neuron(Module):
    def __init__(self, nin):
        super().__init__(self)
        self.nin = nin
        self.w = [Value(2*random.random() - 1.0) for x in range(nin)]
        self.b = Value(0.0)
        
    def __call__(self, x):
        out = Value(0.0)
        for i in range(self.nin):
            out += self.w[i] * x[i]
        out += self.b 
        return out.relu()

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout 
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        out = []
        for i in range(self.nout):
            out.append(self.neurons[i](x))
        return out

    def parameters(self):
        return [p for neurons in self.neurons for p in neurons.parameters()]

class MLP(Module):
    def __init__(self, nin: int, nouts: List[int]):
        self.io_dims = [nin] + nouts
        io_tup = [(self.io_dims[i], self.io_dims[i+1]) for i in range(len(self.io_dims)-1)]
        self.layers = [Layer(_nin, _nout) for (_nin, _nout) in io_tup]
        
    def __call__(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        return out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        

 
