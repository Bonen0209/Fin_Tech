from torchvision import datasets, transforms
from base import BaseDataLoader


class FashionMnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, transform_args=None, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if transform_args is not None:
            trsfm = transforms.Compose([
                getattr(transforms, trans)(**transform_args[trans]) for trans in transform_args
            ])
        else:
            trsfm = transforms.Compose([])
        self.data_dir = data_dir
        self.dataset = datasets.FashionMNIST(self.data_dir, train=training, download=True, transform=trsfm)
        print("[INFO] Loading Fashion MNIST...")

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
