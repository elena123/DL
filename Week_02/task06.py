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


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z

    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')

    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')


if __name__ == '__main__':
    main()
