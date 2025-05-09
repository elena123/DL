class Value:

    def __init__(self, x: float, prev=None):
        self.data = float(x)
        self._prev = prev if prev else {}

    def __str__(self):
        return f"Value(data={int(self.data) if self.data.is_integer() else self.data})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, Value):
            return Value(self.data + other.data, prev={self, other})
        return Value(self.data + other, prev={self, Value(other)})

    def __mul__(self, other):
        if isinstance(other, Value):
            return Value(self.data * other.data, prev={self, other})
        return Value(self.data + other, prev={self, Value(other)})


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._prev)


if __name__ == '__main__':
    main()
