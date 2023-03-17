from math import sqrt

import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torch.nn as nn
from models import *

import dataloader
from collections import OrderedDict

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        
        self.model = U_Net(img_ch=3, output_ch=3)
        '''
        TODO:
            1. model __getattr__ 로 소환
            2. transfer learning 구현
            3. transfer learning 모델과 안맞을시 assertion
        '''
        if hparams.transfer_learning:
            try:
                dicts = torch.load(hparams.transfer_learning)
                self.model.load_state_dict(dicts)
            except:
                print("Load error you may loaded another state dict of model. try another state dict")
        
    
    def load_dict(self, target_dict):
        TODO: load checkpoint as specified in hparameter.yaml file.
        
    
    def forward(self, x):
        '''
        INPUT:
            x -> [B, W, H, C] 
            out -> [B, W, H, C]
        '''
        
        out = self.model(x)
        return out
    
    def common_step(self, x, y):
        output = self(x)
        '''
        activation, thersholding here.
        '''
        loss = self.mse(output, y)
        return loss, output

    def training_step(self, batch):
        x, y = batch  # coordinate [B,2], rgb [B,3]
        loss, _ = self.common_step(x, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        
        loss, output = self.common_step(x, y, is_train=False)  
        
        self.log('val_loss', loss)
        self.logger.log_image(output, x*2 - 1, self.current_epoch)
        
        return {'loss': loss, 'output': output}

    def test_step(self, batch):
        x, y = batch
        loss, output = self.common_step(x, y, is_train=False)
        
        self.log('test_loss', loss)
        return {'test_loss': loss, 'output': output}


    def train_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 0)

    def val_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 1)

    def test_dataloader(self):
        return dataloader.create_coin_dataloader(self.hparams, 2)
