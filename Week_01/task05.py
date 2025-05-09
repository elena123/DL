import numpy as np

if __name__ == '__main__':
    baseball = [[180, 78.4], [215, 102.7], [210, 98.5], [188, 75.2]]

    np_baseball = np.array(baseball)
    dimensions = np_baseball.shape
    rows, columns = dimensions

    print(f"Type: {type(np.array(baseball))}")
    print(f"Number of rows and columns: ({rows}, {columns})")
