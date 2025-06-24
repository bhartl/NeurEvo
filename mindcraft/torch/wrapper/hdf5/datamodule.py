from torch.utils import data
import pytorch_lightning as pl
from typing import Optional, Union
from mindcraft.torch.wrapper.hdf5 import Dataset


class DataModule(pl.LightningDataModule):
    """ A wrapper between a `mindcraft` hdf5-logs, and PyTorchLightning `DataModule`s
        that is based on the `mindcraft.torch.wrapper.hdf5.DataSet` wrapper.

    see https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5

    (c) B. Hartl 2021
    """
    def __init__(self,
                 file_path: str,
                 data_keys: Union[str, tuple, list],
                 label_keys: Union[str, tuple, list],
                 label_filter: Optional[dict] = None,
                 batch_size: int = 32,
                 # load_data: bool = False,
                 transform: object = None,
                 # cache_size: int = 100000,
                 split_val: float = 0.1,
                 shuffle: bool = True,
                 ):
        super().__init__()
        self.file_path = file_path
        self.data_keys = data_keys
        self.label_keys = label_keys
        self.label_filter = label_filter
        self.batch_size = batch_size
        # self.load_data = load_data
        self.transform = transform
        # self.cache_size = cache_size
        self.split_val = split_val
        self.shuffle = shuffle

        self.h5_train, self.h5_val, self.h5_test, self.h5_predict = None, None, None, None

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            h5_dataset = Dataset(file_path=self.file_path,
                                 data_keys=self.data_keys,
                                 label_keys=self.label_keys,
                                 # load_data=self.load_data,
                                 transform=self.transform,
                                 # cache_size=self.cache_size,
                                 label_filter=self.label_filter,
                                 )

            val_size = int(len(h5_dataset) * self.split_val)
            train_size = len(h5_dataset) - val_size
            self.h5_train, self.h5_val = data.random_split(h5_dataset, [train_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.h5_test = Dataset(file_path=self.file_path,
                                   data_keys=self.data_keys,
                                   label_keys=self.label_keys,
                                   # load_data=self.load_data,
                                   transform=self.transform,
                                   # cache_size=self.cache_size,
                                   label_filter=self.label_filter,
                                   )

        if stage == "predict" or stage is None:
            self.h5_predict = Dataset(file_path=self.file_path,
                                      data_keys=self.data_keys,
                                      label_keys=self.label_keys,
                                      # load_data=self.load_data,
                                      transform=self.transform,
                                      # cache_size=self.cache_size,
                                      label_filter=self.label_filter,
                                      )

    def train_dataloader(self):
        return data.DataLoader(self.h5_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return data.DataLoader(self.h5_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.h5_test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return data.DataLoader(self.h5_predict, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass


if __name__ == '__main__':
    ds = Dataset(file_path="test/dat/train/boost_es_method/",
                 data_keys='parameters',
                 label_keys='reward',
                 # load_data=True,
                 )
    print(len(ds))
    print(ds[0])
    print(len(ds))
