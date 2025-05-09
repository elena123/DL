import string

import graphviz


class Value:

    def __init__(self, x: float, prev=None, grad=0.0, label="", **kwargs):
        self.data = float("{:.5f}".format(x))
        self._prev = prev if prev else {}
        self._grad = float("{:.5f}".format(grad))
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

    node_labels = {n: n.label if n.label else f"V{id(n)}" for n in nodes}
    node_labels[root] = "logit"

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


def main() -> None:
    x1 = Value(2.0, label="x1")
    w1 = Value(-3.0, label="w1")

    x2 = Value(0.0, label="x2")
    w2 = Value(1.0, label="w2")

    # Bias
    b = Value(6.7, label="b")

    # Compute logit (perceptron)
    logit = (x1 * w1) + (x2 * w2) + b

    print(f"Logit value: {logit.data}")

    # Draw computation graph
    draw_dot(logit).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
