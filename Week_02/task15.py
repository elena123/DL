import graphviz
import numpy as np


class Value:

    def __init__(self, x: float, prev=None, grad=0.0, label="", **kwargs):
        self.data = float(x)
        self._prev = prev if prev else {}
        self._grad = float(grad)
        self._op = str(kwargs.get("op", ""))
        self.label = label
        self._backward = self.default_backward

    def __str__(self):
        return f"Value(data={int(self.data) if self.data.is_integer() else self.data})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, prev={self, other}, label=f"({self.label}+{other.label})")
        result._op = "+"

        def backward():
            self._grad += result._grad
            other._grad += result._grad

        result._backward = backward
        return result

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, prev={self, other}, label=f"({self.label}*{other.label})")
        result._op = "*"

        return result

    def tanh(self):
        t = np.tanh(self.data)
        return Value(t, prev={self}, label=f"tanh({self.label})", op="tanh")

    def default_backward(self):
        pass

    def backward(self):
        # Topological sort
        list = []
        visited = set()

        def top_sort(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    top_sort(prev)
                list.append(v)

        top_sort(self)

        # Start from self (final node)
        self._grad = 1.0
        for node in reversed(list):
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
    x = Value(5.0, label="x")
    y = Value(10.0, label="y")

    z = x + y
    z.label = "z"

    # Backward pass
    z.backward()

    # Draw computation graph
    draw_dot(z).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
