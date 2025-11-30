import os,sys
import torch
from torch import nn
from torch.nn import functional as F
from lightning import pytorch as pl
import torchmetrics

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout = 0.1):
        super(ResidualBlock, self).__init__()

        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Sequential()

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.norm(self.fc2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PromptHeader(nn.Module):
    def __init__(self, n_parts = 3,dropout = 0.1):
        super().__init__()
        self.fcs = nn.Sequential(
            ResidualBlock(n_parts * 768, n_parts * 768, dropout = dropout),
            ResidualBlock(n_parts * 768, 768, dropout = dropout),
            ResidualBlock(768, 768, dropout = dropout),
            ResidualBlock(768, 64, dropout = dropout),
            nn.Linear(64, 1),
        )
    def forward(self, *args):
        for arg in args:
            assert isinstance(arg, torch.Tensor)

        full_emb = torch.cat(args, dim = -1)
        logit = self.fcs(full_emb).squeeze(-1)
        assert len(logit.shape) == 1
        return logit   
        

class ChromBERTPrompt(nn.Module):

    def __init__(self, dropout=0.1):
        super().__init__()
        self.ft_header = PromptHeader(n_parts = 3, dropout = dropout)

    def forward(self, emb_cell, emb_regulator, emb_all):
        header_out = self.ft_header(emb_cell,emb_regulator,emb_all)
        return header_out 


class ImputationModel(pl.LightningModule):
    def __init__(self, dropout=0.1, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = ChromBERTPrompt(dropout = dropout)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task = "binary"),
            torchmetrics.Precision(task = "binary"),
            torchmetrics.Recall(task = "binary"),
            torchmetrics.F1Score(task = "binary"),
        ], prefix = "train/")
        self.val_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task = "binary"),
            torchmetrics.Precision(task = "binary"),
            torchmetrics.Recall(task = "binary"),
            torchmetrics.F1Score(task = "binary"),
            torchmetrics.AUROC(task = "binary"),
            torchmetrics.AveragePrecision(task = "binary"),
        ], prefix = "val/")
        self.test_metrics = self.val_metrics.clone(prefix = "test/")

    def forward(self, batch):
        emb_cell = batch["emb_cell"]
        emb_regulator = batch["emb_regulator"]
        emb_all = batch["emb_all"]
        return self.model(emb_cell, emb_regulator, emb_all)

    def training_step(self, batch):
        pred = self(batch)
        loss = self.loss_fn(pred, batch["label"].float())
        self.train_metrics(pred, batch["label"])
        self.log_dict(self.train_metrics.compute(), on_step = True, on_epoch = True, prog_bar = True)
        self.log("train/cross_entropy_loss", loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch):
        pred = self(batch)
        loss = self.loss_fn(pred, batch["label"].float())
        self.val_metrics(pred, batch["label"])
        self.log_dict(self.val_metrics.compute(), on_step = False, on_epoch = True, prog_bar = True)
        self.log("val/cross_entropy_loss", loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max = self.trainer.estimated_stepping_batches, 
            eta_min = self.lr * 0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

