from base import BaseDataLoader
from .datasets import FinanceDataset


class FinanceDataLoader(BaseDataLoader):
    """
    Finance data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, filename, batch_size, shuffle=True, validation_split=0.0, num_workers=1, pin_memory=False, training=True):
        self.data_dir = data_dir
        self.dataset = FinanceDataset(self.data_dir, filename)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory)
