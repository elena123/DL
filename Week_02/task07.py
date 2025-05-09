import graphviz


class Value:

    def __init__(self, x: float, prev=None, **kwargs):
        self.data = float(x)
        self._prev = prev if prev else {}
        self._op = str(kwargs.get("op", ""))

    def __str__(self):
        return f"Value(data={int(self.data) if self.data.is_integer() else self.data})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, Value):
            return Value(self.data + other.data, prev={self, other}, op="+")
        return Value(self.data + other, prev={self, Value(other)}, op="+")

    def __mul__(self, other):
        if isinstance(other, Value):
            return Value(self.data * other.data, prev={self, other}, op="*")
        return Value(self.data + other, prev={self, Value(other)}, op="*")


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
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ data: {n.data} }}', shape='record')
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
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z

    draw_dot(result).render(directory='./graphviz_output', view=True)


if __name__ == '__main__':
    main()
