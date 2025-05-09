import graphviz
import numpy as np


class Value:

    def __init__(self, x: float, prev=None, grad=0.0, label="", **kwargs):
        self.data = float(x)
        self._prev = prev if prev else {}
        self._grad = float(grad)
        self._op = str(kwargs.get("op", ""))
        self.label = label

    def __str__(self):
        return f"Value(data={int(self.data) if self.data.is_integer() else self.data})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, Value):
            new_label = f"({self.label}+{other.label})" if self.label and other.label else ""
            return Value(self.data + other.data, prev={self, other}, label=new_label, op="+")
        return Value(self.data + other, prev={self, Value(other)}, label=f"({self.label}+{other})", op="+")

    def __mul__(self, other):
        if isinstance(other, Value):
            new_label = f"({self.label}*{other.label})" if self.label and other.label else ""
            return Value(self.data * other.data, prev={self, other}, label=new_label, op="*")
        return Value(self.data * other, prev={self, Value(other)}, label=f"({self.label}*{other})", op="*")

    def tanh(self):
        t = np.tanh(self.data)
        return Value(t, prev={self}, label=f"tanh({self.label})", op="tanh")


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


def manual_backpropagate_gradient():
    x1 = Value(2.0, label="x1")
    w1 = Value(-3.0, label="w1")
    x2 = Value(0.0, label="x2")
    w2 = Value(1.0, label="w2")
    b = Value(6.88137, label="b")

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    sum_part = x1w1 + x2w2
    logit = sum_part + b
    logit.label = "logit"

    L = logit.tanh()

    # Set the gradient of the final output
    L._grad = 1.0

    # calculate back manually gradient
    # dL/dlogit = (1 - tanh^2(logit))
    dL_dlogit = 1 - L.data ** 2
    logit._grad = L._grad * dL_dlogit

    # dL/db = dL/dlogit * 1
    b._grad = logit._grad

    # dL/d(x1w1+x2w2) = dL/dlogit * dlogit/d(x1w1+x2w2)
    sum_part._grad = logit._grad

    # dL/d(x1*w1) = dL/dlogit * 1
    x1w1._grad = logit._grad

    # dL/d(x2*w2) = dL/dlogit * 1
    x2w2._grad = logit._grad

    # d(x1*w1)/dx1 = w1, d(x1*w1)/dw1 = x1
    x1._grad = x1w1._grad * w1.data
    w1._grad = x1w1._grad * x1.data

    # d(x2*w2)/dx2 = w2, d(x2*w2)/dw2 = x2
    x2._grad = x2w2._grad * w2.data
    w2._grad = x2w2._grad * x2.data

    return L


def main() -> None:
    L = manual_backpropagate_gradient()

    # Draw computation graph
    draw_dot(L).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
