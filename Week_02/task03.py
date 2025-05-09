class Value:

    def __init__(self, x: float):
        self.data = float(x)

    def __str__(self):
        return f"Value(data={int(self.data) if self.data.is_integer() else self.data})"

    def __add__(self, other):
        if isinstance(other, Value):
            return Value(self.data + other.data)
        return f"Value(data={self.data + other.data})"

    def __mul__(self, other):
        if isinstance(other, Value):
            return Value(self.data * other.data)
        return f"Value(data={self.data * other.data})"


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result)


if __name__ == '__main__':
    main()
