from os import path, makedirs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import torch


class TensorBoardLoggerExpanded(TensorBoardLogger):
    def __init__(self, hparams):
        super().__init__(hparams.log.tensorboard_dir, name=hparams.name,
                         default_hp_metric=False)
        self.hparams = hparams
        self.log_hyperparams(hparams)
        

    def fig2np(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def plot_to_numpy(self, image, epoch):
        fig = plt.figure(figsize=(5, 4))
        plt.title(f'Epoch {epoch}')
        plt.imshow(np.clip(image, 0, 255),
                   aspect='equal',
                   )
        fig.canvas.draw()
        data = self.fig2np(fig)
        plt.close()
        return data

    def log_image(self, output, image, epoch):
        output = output.view([image.shape[0], image.shape[1], 3])
        output = (128 * output + 128).detach().cpu().to(torch.int32).numpy()
        if epoch == 99:
            image = (128 * image + 128).detach().cpu().to(torch.int32).numpy()
            image = self.plot_to_numpy(image, epoch)
            self.experiment.add_image(path.join(self.save_dir, 'image'),
                                      image,
                                      epoch,
                                      dataformats='HWC')

        output = self.plot_to_numpy(output, epoch)
        self.experiment.add_image(path.join(self.save_dir, 'output'),
                                  output,
                                  epoch,
                                  dataformats='HWC')
        self.experiment.flush()
        return
