class Value:

    def __init__(self, x: float):
        self.data = float(x)

    def __str__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return f"Value(data={self.data + other.data})"


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    result = x + y
    print(result)


if __name__ == '__main__':
    main()
