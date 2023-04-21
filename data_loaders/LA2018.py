import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import SimpleITK as sitk
import h5py


class LA2018_train(Dataset):

    def __init__(self, args, transform=None):

        self.args = args
        self.crop_size = args.patch_size
        self.transform = transform
        self.rootpath = str(args.root_path)
        self.train_path = self.rootpath + 'LA2018/train/'
        self.data_names = os.listdir(self.train_path)
        self.data_nums = len(self.data_names)

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, i):

        h5f = h5py.File(os.path.join(self.train_path, self.data_names[i], "la_mri.h5"), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class LA2018_val(Dataset):

    def __init__(self, args, transform=None):

        self.args = args
        self.crop_size = args.patch_size
        self.transform = transform
        self.rootpath = str(args.root_path)
        self.val_path = self.rootpath + 'LA2018/val/'
        self.data_names = os.listdir(self.val_path)
        self.data_nums = len(self.data_names)

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, i):

        h5f = h5py.File(os.path.join(self.val_path, self.data_names[i], "la_mri.h5"), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
