from torch import from_numpy
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import read_csv


class CreditCardDataDatset(Dataset):
    """
    CreditCard datasets 
    """
    def __init__(self, data_dir, filename):
        self.data_dir = data_dir
        self.filename = filename
        # self.data_files = ['creditcard_small.csv', 'creditcard.csv']

        self.datas, self.targets = self._preprocess()

    def _preprocess(self):
        raw = read_csv(self.data_dir + self.filename)

        x_headers = [header for header in raw.columns.tolist() if header not in ['Class', 'Time', 'Amount']]
        y_headers = ['Class']

        return from_numpy(raw[x_headers].values.astype('float32')), from_numpy(raw[y_headers].values.astype('int64')).squeeze(dim=1)

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, idx):
        return self.datas[idx], self.targets[idx]

class CreditCardDataLoader(BaseDataLoader):
    """
    CreditCard data loading using BaseDataLoader
    """
    def __init__(self, data_dir, filename, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.filename = filename
        self.dataset = CreditCardDataDatset(self.data_dir, self.filename)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


