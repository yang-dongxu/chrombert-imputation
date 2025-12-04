import os
import sys 
from collections import Counter

import zarr
import pandas as pd
import pyarrow.dataset as ds

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from lightning import pytorch as pl
from torch.utils.data import DataLoader

import os
import zarr
import torch
import numpy as np

class EmbHandler:
    """
    Embedding handler with:
    - Fixed cached parts (decided in preload()) stored as shared-memory tensors.
    - Non-cached parts are always loaded on demand from Zarr and NOT stored.

    This prevents the cache from silently growing when new parts appear.
    """

    _indice_maps = None          # dict[raw_index] -> row index
    _handler = None              # zarr.Group
    _tensors = {}                # part -> shared-memory tensor
    _cached_parts = set()        # parts that are allowed to live in _tensors

    def __init__(self, path_emb: str, dtype=torch.float32):
        self.path_emb = path_emb
        self.dtype = dtype

        # Views of class-level state
        self.indice_maps = EmbHandler._indice_maps
        self.handler = EmbHandler._handler
        self.tensors = EmbHandler._tensors
        self.cached_parts = EmbHandler._cached_parts

        self.__build_indices__()

    def __build_indices__(self):
        if EmbHandler._indice_maps is not None:
            self.indice_maps = EmbHandler._indice_maps
            return len(self.indice_maps)

        region_path = os.path.join(self.path_emb, "region")
        raw_indices = zarr.open(region_path, mode="r")[:, 3].astype(int)
        EmbHandler._indice_maps = {idx: i for i, idx in enumerate(raw_indices)}
        self.indice_maps = EmbHandler._indice_maps
        return len(self.indice_maps)

    def _ensure_handler(self):
        if EmbHandler._handler is None:
            EmbHandler._handler = zarr.open(self.path_emb, mode="r")
        self.handler = EmbHandler._handler

    def _load_part_to_shared(self, part: str):
        """Load a single part into a shared-memory tensor (for cached parts only)."""
        self._ensure_handler()

        if part == "whole":
            np_arr = self.handler["whole"][:]      # (n_regions, d)
        else:
            np_arr = self.handler[part][:]         # (n_regions, d)

        t = torch.from_numpy(np_arr).to(self.dtype)
        t.share_memory_()  # shared across workers

        EmbHandler._tensors[part] = t
        self.tensors = EmbHandler._tensors

    def preload(self, parts):
        """
        Define the fixed cache set and load them into shared memory.

        - parts: iterable of part names to be cached (e.g. "whole", "atac/...", "regulator/...").

        After this:
        - Only parts in this set will ever be stored in `_tensors`.
        - Others will always be read from Zarr on-demand.
        """
        # Update the global cached set
        if EmbHandler._cached_parts:
            EmbHandler._cached_parts |= set(parts)
        else:
            EmbHandler._cached_parts = set(parts)

        self.cached_parts = EmbHandler._cached_parts

        # Load each cached part into shared memory if not already loaded
        for part in self.cached_parts:
            if part not in EmbHandler._tensors:
                self._load_part_to_shared(part)

        self.tensors = EmbHandler._tensors

    def _get_from_zarr_on_demand(self, index_new: int, part: str) -> torch.Tensor:
        """
        For non-cached parts: read a single row from Zarr, do NOT store globally.
        """
        self._ensure_handler()

        if part == "whole":
            np_row = self.handler["whole"][index_new]
        else:
            np_row = self.handler[part][index_new]

        return torch.from_numpy(np_row).to(self.dtype)

    def get(self, index, part: str) -> torch.Tensor:
        """
        Get embedding for raw region index and part.

        Behavior:
        - If part in fixed cache set -> served from shared-memory tensor.
        - If part not in cache set -> read that row from Zarr each time, no caching.
        """
        index_new = self.indice_maps[index]

        # 1) If already loaded in cache -> just slice
        if part in self.tensors:
            return self.tensors[part][index_new]

        # 2) If this part belongs to the fixed cache set but wasn't loaded yet
        #    (e.g. preload didn't include it but cached_parts updated later),
        #    we allow loading ONCE into shared cache.
        if part in self.cached_parts:
            self._load_part_to_shared(part)
            return self.tensors[part][index_new]

        # 3) Non-cached part: pure on-demand read, no cache pollution
        return self._get_from_zarr_on_demand(index_new, part)

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
            "emb_cell": emb_cell,
            "emb_regulator": emb_regulator,
            "emb_all": emb_all,
            "label": label.astype(int),
            "region_index": raw_index,
            "sample_name": sample_name
        }
        return item


class ImputationDataModule(pl.LightningDataModule):
    def __init__(self,
                 emb_path,
                 path_train,
                 path_val,
                 path_test,
                 batch_size=128,
                 num_workers=8,
                 max_preload_parts=8,      # <--- your preload nums
                 always_preload_whole=True,
                 dtype = "float32"
                 ):  # if True, "whole" is always preloaded
        super().__init__()
        self.emb_path = emb_path
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_preload_parts = max_preload_parts
        self.always_preload_whole = always_preload_whole

        if dtype == "float32":
            self.emb_dtype = torch.float32
        elif dtype == "float16":
            self.emb_dtype = torch.float16
        elif dtype == "bfloat16":
            self.emb_dtype = torch.bfloat16
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        self.emb_handler = None

    def _collect_part_frequencies(self, paths):
        """
        Count how often each part is used across given parquet files.

        Here we approximate frequency by number of sample columns that
        correspond to each cell_type / factor. That's enough to rank parts.
        """
        freq = Counter()

        for p in paths:
            df = ds.dataset(p).head(3).to_pandas()
            samples = df.columns[4:]  # skip first 4 meta columns
            for s in samples:
                factor, cell_type = s.split("|")[:2]
                freq[f"atac/{cell_type}"] += 1
                freq[f"regulator/{factor}"] += 1

        return freq

    def _select_preload_parts(self, freq: Counter):
        """
        From the part frequency Counter, select which parts to preload:
        - Optionally always include "whole".
        - Sort remaining by frequency desc.
        - Truncate to max_preload_parts.
        """
        parts = []

        # Always preload "whole" if requested and present
        if self.always_preload_whole:
            parts.append("whole")

        # Remove 'whole' from the frequency dict for ranking
        remaining = [(p, c) for p, c in freq.items() if p != "whole"]

        # Sort by frequency (descending)
        remaining.sort(key=lambda x: x[1], reverse=True)

        if self.max_preload_parts is None:
            # Preload everything we know about
            parts.extend([p for p, _ in remaining])
        else:
            # max_preload_parts counts total parts INCLUDING "whole"
            # so we limit how many "remaining" to add.
            remaining_budget = max(self.max_preload_parts - len(parts), 0)
            parts.extend([p for p, _ in remaining[:remaining_budget]])

        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                ordered.append(p)

        return ordered

    def setup(self, stage: str):
        if self.emb_handler is None:
            self.emb_handler = EmbHandler(self.emb_path, dtype=self.emb_dtype)
            # 1) Collect frequencies from all splits
            freq_val = self._collect_part_frequencies(
                [self.path_val]
            )
            freq_train = self._collect_part_frequencies(
                [self.path_train]
            )
            # concat freq_val and freq_train
            freq ={}
            for k, v in freq_val.items():
                freq[k] = v
            for k, v in freq_train.items():
                if k in freq:
                    freq[k] += v
                else:
                    freq[k] = v

            # all validation parts should be cached
            max_num = max(freq.values())
            for k,v in freq_val.items():
                freq[k] = max_num + v

            # 2) Choose which parts to preload
            preload_parts = self._select_preload_parts(freq)

            print(f"[ImputationDataModule] Preloading {len(preload_parts)} parts "
                  f"(max_preload_parts={self.max_preload_parts})", file=sys.stderr)
            print(f"[ImputationDataModule] Preloading parts: {preload_parts}", file=sys.stderr)
            # 3) Preload in main process (before DataLoader workers)
            self.emb_handler.preload(preload_parts)

        # Datasets use the same handler (shared caches)
        self.train_dataset = ImputationDataset(self.emb_handler,
                                               peaks_path=self.path_train)
        self.val_dataset = ImputationDataset(self.emb_handler,
                                             peaks_path=self.path_val)
        self.test_dataset = ImputationDataset(self.emb_handler,
                                              peaks_path=self.path_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


if __name__ == "__main__":
    path_emb = "/workspace/experiments/1.original/1.prepare_dataset/6.convert_zarr/emb_mm10_original_chr1_v3.zarr"
    path_peaks = "/workspace/experiments/1.original/1.prepare_dataset/4.make_dataset/output/align_minimal_val_chr1.parquet"
    emb_hander = EmbHandler(path_emb)
    dataset = ImputationDataset(emb_hander, path_peaks)
    print(dataset[0])