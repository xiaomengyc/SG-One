from __future__ import print_function
from __future__ import absolute_import

from torchvision import transforms
from torch.utils.data import DataLoader
from .mydataset import mydataset

def data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    tsfm_train = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])


    img_train = mydataset(args, transform=tsfm_train)

    train_loader = DataLoader(img_train, batch_size=1, shuffle=True, num_workers=1)

    return train_loader

def val_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    tsfm_val = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])


    img_val = mydataset(args, is_train=False, transform=tsfm_val)

    val_loader = DataLoader(img_val, batch_size=1, shuffle=False, num_workers=1)

    return val_loader
