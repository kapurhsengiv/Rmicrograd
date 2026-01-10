

class Value:
    def __init__(self, data, children=(), _op=''):
        self.grad =  0.0
        self.data = data
        self._prev = set(children)
        self._op = _op
        self._backward = lambda: None
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op="+")
        def _backward():
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad 
        out._backward = _backward
        return out
    
    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        # other = other if isinstance(other, Value) else Value(other)
        # out = Value(self.data - other.data, (self, other), _op="-")
        # def _backward():
        #     self.grad += 1.0 * out.grad
        #     other.grad -= 1.0 * out.grad 
        # out._backward = _backward

        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op="*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = self * (other ** -1)
        return out

    def __pow__(self, other):
        assert isinstance(other, int) or isinstance(other, float), "exponent needs to be int or float"  
        out = Value(self.data**other, (self,), _op='power')
        def _backward():
            self.grad += other * (self.data ** (other -1)) * out.grad 
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), _op='relu')
        def _backward():
            self.grad += (0.0 if self.data <=0.0 else 1.0) * out.grad
        out._backward = _backward
        return out 

    def backward(self):
        topo = []
        visited = set()
        # DFS, build topological order
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                children = list(node._prev)
                while children != []:
                    c = children.pop(0)
                    build_topo(c)
                topo.append(node)
        build_topo(self)
        # reverse the topo list
        topo = topo[::-1]

        # call ._backward()
        topo[0].grad = 1.0
        for x in topo:
            x._backward()
  
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    

 
