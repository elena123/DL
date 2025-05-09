import numpy as np

if __name__ == '__main__':
    statement = np.array([True, 1, 2]) + np.array([3, 4, False])
    print("statement:", statement)

    if np.array_equal(statement, np.array([True, 1, 2, 3, 4, False])):
        print("Answer: A")
    elif np.array_equal(statement, np.array([4, 3, 0]) + np.array([0, 2, 2])):
        print("Answer: B")
    elif np.array_equal(statement, np.array([1, 1, 2]) + np.array([3, 4, -1])):
        print("Answer: C")
    elif np.array_equal(statement, np.array([0, 1, 2, 3, 4, 5])):
        print("Answer: D")
