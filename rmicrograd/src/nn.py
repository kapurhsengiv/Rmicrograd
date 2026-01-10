import numpy as np
import random
from rmicrograd.src.engine import Value

class Module:
    def zero_grad(self):
        pass

    def parameters(self):
        pass


class Neuron(Module):
    pass

class Layer(Module):
    pass

class MLP(Module):
    pass

 
