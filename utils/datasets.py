r"""
Custom Dataset module.
Consist of Dataset that designed for different tasks.
"""

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from utils.log import LOGGER

__all__ = ['DatasetDetect']


class DatasetDetect(Dataset):
    # TODO upgrade val_from_train, test_from_train function in the future
    def __init__(self, path, img_size):
        LOGGER.info('Initializing datasets...')
        self.img_size = img_size

        # get the path list of image files
        files = []  # save image files(the type is str) temporarily
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)
            if p.is_dir():
                files += [str(x) for x in p.rglob('*.*')]
            elif p.is_file():
                with open(p, 'r') as f:
                    f = f.read().splitlines()
                    # local to global path
                    parent = p.parent
                    for element in f:
                        element = Path(element.strip())
                        if '\\' in element.parts:  # remove / in the front of it
                            element = parent / Path(*element[1:])
                            files.append(str(element))
                        else:
                            files.append(str(parent / element))
            else:
                raise TypeError(f'Something wrong with {p} in the type of file')

        self.img_files = sorted(files)  # TODO 2022.2.16
        del files

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
