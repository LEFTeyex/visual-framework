r"""
Download datasets_path from torchvision datasets_path
"""

from pathlib import Path
from torchvision import datasets


def path_error(pathname):
    raise Exception(f'The Path:({pathname}) is not exist')


def download_mnist():
    path = Path('f:/datasets')
    if path.exists():
        path = str(path)
        datasets.MNIST(root=path, download=True)
        datasets.KMNIST(root=path, download=True)
        datasets.EMNIST(root=path, download=True, split='balanced')
        datasets.FashionMNIST(root=path, download=True)
        datasets.QMNIST(root=path, download=True)  # This can not download, need manual download
    else:
        path_error(path)


def download_voc_detection():
    path = Path('f:/datasets')
    if path.exists():
        path = str(path)
        datasets.VOCDetection(root=path, download=True)
    else:
        path_error(path)


def download_cifar10():
    path = Path('f:/datasets')
    if path.exists():
        path = str(path)
        datasets.CIFAR10(root=path, download=True)
    else:
        path_error(path)


if __name__ == '__main__':
    pass
