from abc import *

import pytorch_lightning as pl


class LightningBaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def _shared_eval_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass