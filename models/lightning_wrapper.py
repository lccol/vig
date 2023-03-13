import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch import Tensor
from torch import functional as F

from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection

from .vig import ViG, PyramidViG
from typing import Optional, Union, Tuple, Dict, List, Type

class ViGLT(pl.LightningModule):
    def __init__(self,
                in_channels: int,
                out_channels: List[int],
                heads: int,
                n_classes: int,
                input_resolution: Tuple[int, int],
                reduce_factor: int, act: str='relu',
                k: int=9,
                overlapped_patch_emb: bool=True,
                **kwargs) -> None:
        super(ViGLT, self).__init__()
        self.model = ViG(in_channels,
                        out_channels,
                        heads,
                        n_classes,
                        input_resolution,
                        reduce_factor,
                        act,
                        k,
                        overlapped_patch_emb,
                        **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=n_classes)

        self.train_count = 0
        self.val_count = 0
        return

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.model(x).squeeze(-1).squeeze(-1)
        loss = self.loss(out, y)
        
        self.train_count += 1
        if self.train_count % 10 == 0:
            self.train_count = 0
            acc = self.acc(out, y)
            print(f'Train loss: {loss} - Accuracy: {acc}')
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.model(x).squeeze(-1).squeeze(-1)
        loss = self.loss(out, y)

        self.val_count += 1
        if self.val_count % 10 == 0:
            self.val_count = 0
            acc = self.acc(out, y)
            print(f'Validation loss: {loss} - Acc: {acc}')
        return

    def backward(self, loss: Tensor, optimizer, optimizer_idx) -> None:
        loss.backward()
        return

class PyramidViGLT(pl.LightningModule):
    def __init__(self,
                in_channels: int,
                out_channels: List[int],
                heads: int,
                n_classes: int,
                input_resolution: Tuple[int, int],
                reduce_factor: int,
                pyramid_reduction: int=2,
                act: str = 'relu',
                k: int = 9,
                overlapped_patch_emb: bool = True,
                **kwargs) -> None:
        super(PyramidViGLT, self).__init__()
        self.model = PyramidViG(in_channels,
                                out_channels,
                                heads,
                                n_classes,
                                input_resolution,
                                reduce_factor,
                                pyramid_reduction,
                                act,
                                k,
                                overlapped_patch_emb,
                                **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        return
    
    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.model(x)
        out = out.squeeze(-1).squeeze(-1)
        loss = self.loss(out, y)
        
        self.acc.update(out, y)
        return loss
    
    def training_epoch_end(self):
        acc = self.acc.compute()
        
        self.acc.reset()
        return
        

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.model(x)
        out = out.squeeze(-1).squeeze(-1)
        loss = self.loss(out, y)

        self.val_acc.update(out, y)
        return
    
    def validation_epoch_end(self):
        acc = self.val_acc.compute()        
        self.val_acc.reset()
        return

    def backward(self, loss: Tensor, optimizer, optimizer_idx) -> None:
        loss.backward()
        return