import os
import sys 
import zarr
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from lightning import pytorch as pl
from torch.utils.data import DataLoader

class EmbHandler():
    def __init__(self, path_emb: str):
        self.path_emb = path_emb
        self.indice_maps = None
        self.handler = None
        self.arrays = {}

        # Eager build in main process to avoid per-worker rebuild
        self.__build_indices__()

    def __build_indices__(self):
        if self.indice_maps is not None:
            return len(self.indice_maps)

        region_path = os.path.join(self.path_emb, "region")
        raw_indices = zarr.open(region_path, mode="r")[:, 3].astype(int)
        self.indice_maps = {index: i for i, index in enumerate(raw_indices)}
        return len(self.indice_maps)

    def get(self, index, part):
        if self.handler is None:
            self.handler = zarr.open(self.path_emb, mode="r")

        index_raw = self.indice_maps[index]
        
        if part not in self.arrays:
            if part is None:
                self.arrays[part] = self.handler
            else:
                self.arrays[part] = self.handler[part]
        
        emb = self.arrays[part][index_raw]
        return emb


class ImputationDataset(Dataset):
    def __init__(self, emb_hander: EmbHandler, peaks_path: str):
        self.emb_hander = emb_hander
        self.peaks_path = peaks_path

        self.dataset = pd.read_parquet(peaks_path)
        self.samples = self.dataset.columns[4:] # exclude the first 4 columns
        self.region_indices = self.dataset.iloc[:,3].values 

    def __len__(self):
        return len(self.samples) * len(self.region_indices)

    def __getitem__(self, index):
        n_col = index // len(self.region_indices)
        n_row = index % len(self.region_indices)

        label = self.dataset.iloc[n_row, n_col + 4] > 0 # bool

        sample_name = self.samples[n_col]
        factor = sample_name.split("|")[0]
        cell_type = sample_name.split("|")[1]

        raw_index = self.region_indices[n_row]
        emb_cell = self.emb_hander.get(raw_index, f"atac/{cell_type}")
        emb_regulator = self.emb_hander.get(raw_index, f"regulator/{factor}")
        emb_all = self.emb_hander.get(raw_index, "whole")
        # return emb_cell, emb_regulator, emb_all, label
        item = {
            "emb_cell": torch.from_numpy(emb_cell).float(),
            "emb_regulator": torch.from_numpy(emb_regulator).float(),
            "emb_all": torch.from_numpy(emb_all).float(),
            "label": label.astype(int),
            "region_index": raw_index,
            "sample_name": sample_name
        }
        return item

class ImputationDataModule(pl.LightningDataModule):
    def __init__(self, emb_path, path_train, path_val, path_test, batch_size=128, num_workers=8):
        super().__init__()
        self.emb_path = emb_path
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.emb_handler = None

    def setup(self, stage: str):
        if self.emb_handler is None:
            self.emb_handler = EmbHandler(self.emb_path)
        self.train_dataset = ImputationDataset(self.emb_handler, peaks_path = self.path_train)
        self.val_dataset = ImputationDataset(self.emb_handler, peaks_path = self.path_val)
        self.test_dataset = ImputationDataset(self.emb_handler, peaks_path = self.path_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            # sampler=sampler, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True,
            prefetch_factor=2,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True,
            prefetch_factor=2,
        )


if __name__ == "__main__":
    path_emb = "/workspace/experiments/1.original/1.prepare_dataset/6.convert_zarr/emb_mm10_original_chr1_v3.zarr"
    path_peaks = "/workspace/experiments/1.original/1.prepare_dataset/4.make_dataset/output/align_minimal_val_chr1.parquet"
    emb_hander = EmbHandler(path_emb)
    dataset = ImputationDataset(emb_hander, path_peaks)
    print(dataset[0])