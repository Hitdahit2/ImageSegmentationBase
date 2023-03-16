from os import path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def create_coin_dataloader(hparams, cv):
    def collate_fn(batch):
        x_list = list()
        rgb_list = list()
        for x, rgb in batch:
            x_list.append(x)
            rgb_list.append(rgb)
        x_list = torch.stack(x_list, dim=0)
        rgb_list = torch.stack(rgb_list, dim=0)

        return x_list, rgb_list
    if type(hparams.data.image) !=str:
        DS = MultipleImageDataset(hparams, cv)
    else:
        DS = ImageDataset(hparams, cv)
    if cv == 0:
        return DataLoader(dataset=DS,
                          batch_size=1,  # hparams.train.batch_size,
                          shuffle=True,
                          # collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=DS,
                          batch_size=hparams.train.bastch_size,
                          drop_last=False,
                          shuffle=False,
                          )


class ImageDataset(Dataset):
    def __init__(self, hparams, mode=0):  # mode 0: train, 1: val, 2: test
        self.hparams = hparams
        self.cv = cv
        assert path.isfile(self.hparams.data.image), f"given hparam image path({self.hparams.data.image}) is not a file"
        if mode==0:
            self.imgs = glob.glob(hparams.data.image_tr)
            self.labels = glob.glob(hparams.data.label_tr)
        elif mode==1:
            self.imgs = glob.glob(hparams.data.image_vl)
            self.labels = glob.glob(hparams.data.label_vl)
        else:
            self.imgs = glob.glob(hparams.data.image_ts)
            self.labels = glob.glob(hparams.data.label_ts)
        
        # make sure that your img and corresponding mask has same filename.
        self.imgs = sorted(self.imgs)
        self.labels = sorted(self.labels)
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index])
        label = cv2.imread(self.imgs[index])
        
        img = np.array(img/255.0, dtype=np.float32)
        if self.hparams.data.is_multi==0:
            label = np.where(a==0, 0, 1)
        else:
            '''
            TODO: multi label mask One hot encoding 후 stacking 코드 구현
            '''
        label = np.array(label, dtype=np.float32)
        
        '''
        TODO: Preprocessing code. Where should this code work? online? offline?
        '''
        
        augmented = self.transform(image=img, mask=label)
        img = augmented['image']
        masks = augmented['mask']
        
        return img, label


