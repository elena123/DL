import graphviz
import numpy as np


class Value:

    def __init__(self, x: float, prev=None, label="", **kwargs):
        self.data = float(x.data) if isinstance(x, Value) else float(x)
        self._prev = set(prev) if prev else set()
        self._grad = 0.0
        self._op = str(kwargs.get("op", ""))
        self.label = label
        self._backward = self.default_backward

    def __str__(self):
        return f"Value(data={int(self.data) if self.data.is_integer() else self.data})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, prev=[self, other], label=f"({self.label}+{other.label})")
        result._op = "+"

        def backward():
            print(f"Backward pass on addition for {self.label}, current grad: {self._grad}")
            self._grad += 1.0 * result._grad
            other._grad += 1.0 * result._grad

        result._backward = backward
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data - other.data, prev=[self, other], label=f"({self.label}-{other.label})")
        result._op = "-"

        def backward():
            self._grad += 1.0 * result._grad
            other._grad -= 1.0 * result._grad

        result._backward = backward
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, prev=[self, other], label=f"({self.label}*{other.label})")
        result._op = "*"

        def backward():
            print(f"Backward pass on multiplication for {self.label}, current grad: {self._grad}")
            self._grad += other.data * result._grad
            other._grad += self.data * result._grad

        result._backward = backward
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data / other.data, prev=[self, other], label=f"({self.label}/{other.label})")
        result._op = "/"
        return result

    def __pow__(self, other):
        assert isinstance(other, (float, int)), "Only supporting int or float powers"
        result = Value(self.data ** other, prev=[self], label=f"({self.label}^{other})")
        result._op = "**"
        return result

    def exp(self):
        result = Value(np.exp(self.data), prev=[self], label=f"e^{self.label}", op="exp")
        return result

    def tanh(self):
        # I tried to break it down on different expressions, but I'm still missing something :(
        exp_x = self.exp()
        exp_neg_x = Value(1) / exp_x

        numerator = exp_x - exp_neg_x
        denominator = exp_x + exp_neg_x

        t = numerator / denominator

        out = Value(t.data, prev=[self, numerator, denominator], label=f"tanh({self.label})", op="tanh")

        def _backward():
            self._grad += (1 - t.data ** 2) * out._grad

        out._backward = _backward
        return out

    def default_backward(self):
        pass

    def backward(self):
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if isinstance(v, Value) and v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                topo.append(v)

        build_topo(self)

        # Start from self (final node)
        self._grad = 1.0

        for node in reversed(topo):
            node._backward()


def trace(argument):
    nodes = set()
    edges = set()

    if isinstance(argument, Value):
        nodes.add(argument)

        if argument._prev:
            for prev_node in argument._prev:
                edges.add((prev_node, argument))
                prev_nodes, prev_edges = trace(prev_node)
                # Recursively trace the previous node dependencies
                nodes.update(prev_nodes)
                edges.update(prev_edges)
    return nodes, edges


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='png', graph_attr={
        'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)

    node_labels = {n: n.label if n.label else f"V{id(n)}" for n in nodes}
    node_labels[root] = "L"

    for n in nodes:
        uid = str(id(n))
        letter = node_labels[n]

        formatted_data = f"{n.data:.5f}"
        formatted_grad = f"{float(n._grad.data) if isinstance(n._grad, Value) else float(n._grad):.5f}"

        label = f'{{ {letter} | data: {formatted_data} | grad: {formatted_grad} }}'
        dot.node(name=uid, label=label, shape='record')

        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def main() -> None:
    x1 = Value(2.0, label="x1")
    w1 = Value(-3.0, label="w1")
    x2 = Value(0.0, label="x2")
    w2 = Value(1.0, label="w2")
    b = Value(6.88137, label="b")

    # Forward pass
    logit = (x1 * w1) + (x2 * w2) + b
    L = logit.tanh()

    # Backward pass
    L.backward()

    # Draw computation graph
    draw_dot(L).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
