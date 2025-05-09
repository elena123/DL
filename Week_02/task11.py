import string

import graphviz


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

    # Create a mapping from nodes to letters (a, b, c, ...)
    node_labels = {n: letter for n, letter in zip(nodes, string.ascii_lowercase)}
    node_labels[root] = "L"

    for n in nodes:
        uid = str(id(n))
        letter = node_labels[n]

        label = f'{{ {letter} | data: {n.data} | grad: {n._grad} }}'
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


def initialize_values(epsilon=0):
    a = Value(2.0)
    a.data += epsilon
    b = Value(-3.0)
    # b.data += epsilon
    c = Value(10.0)
    # c.data += epsilon
    e = a * b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0)
    # f.data += epsilon
    f.label = 'f'
    L = d * f
    L.label = 'L'

    # Backpropagation for result = (x * y) + z
    L._grad = 1.0  # dL/dL = 1

    # Gradient of d, L = d * f, so dL/dd = f
    d._grad = f.data
    # Gradient of f - dL/df = d
    f._grad = d.data
    # Gradient of e, d = e + c, dd/de = 1, dd/dc = 1
    # dL/de = dL/dd * dd/de = f * 1 = f
    e._grad = f.data
    # Gradient of c, dL/dc = dL/dd * dd/dc = f * 1 = f
    c._grad = f.data
    # dL/da = dL/de * de/da, dL/de = f, e = a * b, de/da = b, dL/da = f * b
    a._grad = f.data * b.data
    # dL/db = dL/de * de/db, dL/de = f, e = a * b, de/db = a, dL/db = f * a
    b._grad = f.data * a.data

    return L.data


def manual_der():
    epsilon = 0.01
    L_data = initialize_values()
    L1_data = initialize_values(epsilon)

    # draw_dot(L).render(directory='./graphviz_output', view=True)
    # draw_dot(L1).render(directory='./graphviz_output', view=True)
    print(f"Old L = {L_data}")
    print(f"New L = {L1_data}")


def main() -> None:
    manual_der()


if __name__ == '__main__':
    main()
