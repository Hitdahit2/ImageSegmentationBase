import argparse
import datetime
from glob import glob
import os

from omegaconf import OmegaConf as OC
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning_model import COIN, COINpp
from tblogger import TensorBoardLoggerExpanded

os.environ['CUDA_VISIBLE_DEVICES']='0'
def train():
    hparams = OC.load('hparameter.yaml')
    now = datetime.datetime.now().strftime('%m_%d_%H')
    hparams.name = f"{hparams.log.name}_{now}"
    hparams.log.checkpoint_dir = os.path.join(hparams.log.checkpoint_dir, hparams.name)
    
    os.makedirs(hparams.log.tensorboard_dir, exist_ok=True)
    os.makedirs(hparams.log.checkpoint_dir, exist_ok=True)
    
    model = Model(hparams)
    tblogger = TensorBoardLoggerExpanded(hparams)
    filename = f'{hparams.log.name}_{now}_{{epoch}}'
    checkpoint_callback = ModelCheckpoint(dirpath=hparams.log.checkpoint_dir,
                                          filename=filename,
                                          verbose=True,
                                          save_last=True,
                                          save_top_k=3,
                                          monitor='val_loss',
                                          mode='min',
                                          prefix='')
    
    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        gpus=[hparams.train.device],
        check_val_every_n_epoch=10,
        max_epochs=1000000,
        logger=tblogger,
        progress_bar_refresh_rate=1,
        resume_from_checkpoint=None
    )
    trainer.fit(model)


if __name__ == '__main__':
    '''
    TODO:
        1. hparameter argparser로 받아오기
        2. training log 폴더링 네이밍 정확히 하기. (tblogger.py)
    '''
    train()
