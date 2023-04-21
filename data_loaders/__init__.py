import random
from torchvision import transforms
from torch.utils.data import DataLoader
from .utils import TwoStreamBatchSampler, OneStreamBatchSampler
from .LA2018 import LA2018_train, LA2018_val
from data_augmentation.transforms import NormalizationLabel, Resize, RandomCrop, CenterCrop, RandomRotFlip, ToTensor


def worker_init_fn(args, worker_id):
    random.seed(args.seed + worker_id)

def generate_datasets(args, SSL=True):
    split_percent = args.split
    batch_size = args.batch_size
    labeled_bs = args.labeled_bs
    patch_size = args.patch_size


    if args.dataset == "LA2018":
        total_data = 80
        val_nums = 20
        split_idx = int(split_percent * total_data)
        train_data = LA2018_train(args,
                                  transform=transforms.Compose([
                                      # RandomRotFlip(),
                                      RandomCrop(patch_size),
                                      ToTensor(),
                                  ]))
        val_data = LA2018_val(args,
                              transform=transforms.Compose([
                                  CenterCrop(patch_size),
                                  ToTensor(),
                              ]))


    else:
        raise Exception("No Dataset Error")

    if SSL == True:
        val_idxs = list(range(val_nums))
        labeled_idxs = list(range(split_idx))
        unlabeled_idxs = list(range(split_idx, total_data))
        batch_sampler_train = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        batch_sampler_val = OneStreamBatchSampler(val_idxs, labeled_bs)
        train_loader = DataLoader(train_data, batch_sampler=batch_sampler_train, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_data, batch_sampler=batch_sampler_val, num_workers=4, pin_memory=True)
        train_nums = total_data
        labeled_nums = split_idx
        unlabeled_nums = train_nums - labeled_nums
        print("======DATA HAVE BEEN GENERATED SUCCESSFULLY======")

        print("====train nums:{}====labeled nums:{}====unlabeled nums:{}==== ".format(train_nums, labeled_nums,
                                                                                      unlabeled_nums))
    else:
        train_nums = total_data
        train_loader = DataLoader(train_data, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_data, num_workers=4, pin_memory=True, drop_last=True)
        print("======DATA HAVE BEEN GENERATED SUCCESSFULLY======")
        print("====train nums:{}===val nums:{}==== ".format(train_nums, val_nums))
    return train_loader, val_loader


