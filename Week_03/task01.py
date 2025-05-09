import torch


def main():
    temperatures = [[72, 75, 78], [70, 73, 76]]
    tensor = torch.tensor(temperatures)
    temperatures_updated = tensor + 2

    print(f'Temperatures: {tensor}')
    print(f'Shape of temperatures: {tensor.shape}')
    print(f'Data type of temperatures: {tensor.dtype}')
    print(f'Corrected temperatures: {temperatures_updated}')


if __name__ == '__main__':
    main()
