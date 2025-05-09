class Value:

    def __init__(self, x: float):
        self.data = float(x)

    def __str__(self):
        return f"Value(data={int(self.data)})"


def main() -> None:
    value1 = Value(5)
    print(value1)

    value2 = Value(6)
    print(value2)


if __name__ == '__main__':
    main()
