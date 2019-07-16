import numpy as np
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset, DataLoader
import LITS_reader


class Lits_DataSet(Dataset):
    def __init__(self, crop_size, batch_size, lits_reader,resize_scale):
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.lits_reader = lits_reader
        self.resize_scale=resize_scale

    def __getitem__(self, index):
        data, target = self.lits_reader.next_train_batch_3d_sub_by_index(train_batch_size=self.batch_size,
                                                                         crop_size=self.crop_size, index=index,
                                                                         resize_scale=self.resize_scale)
        data = data.transpose(0, 4, 1, 2, 3)
        target = target.transpose(0, 4, 1, 2, 3)
        return torch.from_numpy(data), torch.from_numpy(target)

    def __len__(self):
        return 104


class Lits_DataSet_val(Dataset):
    def __init__(self, crop_size, batch_size, lits_reader,resize_scale):
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.lits_reader = lits_reader
        self.resize_scale=resize_scale

    def __getitem__(self, index):
        data, target = self.lits_reader.next_val_batch_3d_sub_by_index(val_batch_size=self.batch_size,
                                                                       crop_size=self.crop_size, index=index,
                                                                       resize_scale=self.resize_scale)
        data = data.transpose(0, 4, 1, 2, 3)
        target = target.transpose(0, 4, 1, 2, 3)
        return torch.from_numpy(data), torch.from_numpy(target)

    def __len__(self):
        return 13


def main():
    reader = LITS_reader.LITS_reader(data_fix=False)
    dataset = Lits_DataSet([32, 64, 64], 4, reader,resize_scale=0.5)
    data_loader=DataLoader(dataset=dataset,shuffle=True,num_workers=2)
    for data, mask in data_loader:
        data=torch.squeeze(data,dim=0)
        mask=torch.squeeze(mask,dim=0)
        print(data.shape, mask.shape)


if __name__ == '__main__':
    main()
