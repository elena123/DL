import graphviz
import numpy as np


class Value:

    def __init__(self, x: float, prev=None, label="", **kwargs):
        self.data = float(x)
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

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, prev=[self, other], label=f"({self.label}*{other.label})")
        result._op = "*"

        def backward():
            print(f"Backward pass on multiplication for {self.label}, current grad: {self._grad}")
            self._grad += other.x * result._grad
            other._grad += self.data * result._grad

        result._backward = backward
        return result

    def tanh(self):
        t = np.tanh(self.x)
        out = Value(t, prev=[self], label=f"tanh({self.label})", op="tanh")

        def _backward():
            print(f"Backward pass on tanh for {self.label}, current grad: {self._grad}")
            self._grad += (1 - t ** 2) * out._grad
            print(f"Grad after tanh: {self._grad:.5f}")

        out._backward = _backward
        return out

    def default_backward(self):
        pass

    def backward(self):
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                topo.append(v)

        build_topo(self)

        # Start from self (final node)
        self._grad = 1.0

        for node in reversed(topo):
            print(f"Running backward on {node.label}, current grad: {node._grad:.5f}")
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
        formatted_grad = f"{n._grad:.5f}"

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
    x = Value(10.0, label="x")

    # Forward pass
    L = x + x

    # Backward pass
    L.backward()

    print(f"x grad: {x._grad:.5f}")

    draw_dot(L).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
