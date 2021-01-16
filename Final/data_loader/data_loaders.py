from base import BaseDataLoader
from .datasets import AnimeDataset


class AnimeDataLoader(BaseDataLoader):
    """
    Anime data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, pin_memory=False):
        self.data_dir = data_dir
        self.dataset = AnimeDataset(data_dir=self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory)

    def get_whole_dataframe(self):
        return self.dataset.get_whole_dataframe()
