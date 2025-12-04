import os 
import sys 
import h5py 
import zarr
import json 
import argparse

import numpy as np 
import pandas as pd 

import torch
from torch import nn 
from tqdm import tqdm 

import chrombert 
from chrombert import ChromBERTFTConfig, DatasetConfig

DEFAULT_BASEDIR = os.path.expanduser("~/.cache/chrombert/data")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract regulator embeddings from ChromBERT")
    parser.add_argument("supervised_file", type=str, help="Path to the supervised file")

    parser.add_argument("--regulators", type=str, required=True, help="Path to the regulators file")
    parser.add_argument("--atac", type=str, required=True, help="Path to the atac file")
    parser.add_argument("-o", "--oname", type=str, required=True, help="Path to the output hdf5 file")

    parser.add_argument("--basedir", type=str, default = DEFAULT_BASEDIR, help="Base directory for the required files")
    
    parser.add_argument("-g", "--genome", type=str, default = "hg38", help="genome version. For example, hg38 or mm10. ")
    parser.add_argument("-k", "--ckpt", type=str, required=False, default=None, help="Path to the pretrain checkpoint or fine-tuned. Optial if it could infered from other arguments")
    parser.add_argument("--meta", type=str, required=False, default=None, help="Path to the meta file. Optional if it could infered from other arguments")
    parser.add_argument("--mask", type=str, required=False, default=None, help="Path to the mtx mask file. Optional if it could infered from other arguments")

    parser.add_argument("-d","--hdf5-file", type=str, required=False, default=None, help="Path to the hdf5 file that contains the dataset. Optional if it could infered from other arguments")
    parser.add_argument("-hr","--high-resolution", dest = "hr", action = "store_true", help="Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.")

    parser.add_argument("--batch-size", dest="batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--num-workers",dest="num_workers", type=int, required=False, default=8, help="number of workers for dataloader")

    return parser.parse_args()

def validate_args(args):
    assert os.path.exists(args.supervised_file), f"Supervised file does not exist: {args.supervised_file}"
    assert args.genome in ["hg38", "mm10", "mm10-embryo-lack", "mm10-embryo"], f"Genome {args.genome} is not supported. "
    assert args.hr == False, "200-bp resolution is not supported now. "

    assert os.path.exists(args.regulators), f"Regulators file does not exist: {args.regulators}"

    global IDS_regulator, IDS_atac, IDS_atac_name
    with open(args.regulators) as f:
        IDS_regulator = [i.strip() for i in f.read().splitlines()]

    assert len(IDS_regulator) > 0, f"Regulators file is empty: {args.regulators}"
    assert os.path.exists(args.atac), f"ATAC file does not exist: {args.atac}"
    df_atac_info = pd.read_csv(args.atac, sep="\t")
    assert "GSMID" in df_atac_info.columns, f"GSMID column does not exist in the ATAC file: {args.atac}"
    assert "cell" in df_atac_info.columns, f"cell_type column does not exist in the ATAC file: {args.atac}"
    IDS_atac = [i.lower().strip() for i in df_atac_info["GSMID"].tolist()]
    IDS_atac_name = [i.lower().strip() for i in df_atac_info["cell"].tolist()]


def get_model_config(args):
    assert args.genome in ["hg38", "mm10", "mm10-embryo-lack", "mm10-embryo"], f"Genome {args.genome} is not supported. "
    if args.ckpt is not None:
        ckpt = args.ckpt
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        if args.hr:
            res = "200bp"
        else:
            res = "1kb"
        if args.genome == "hg38":
            ckpt = os.path.join(args.basedir, "checkpoint", f"{args.genome}_6k_{res}_pretrain.ckpt")
        elif args.genome == "mm10":
            ckpt = os.path.join(args.basedir, "checkpoint", f"{args.genome}_5k_{res}_pretrain.ckpt")
        else:
            raise ValueError(f"Genome {args.genome} is not supported. ")

    if args.mask is not None:
        mtx_mask = args.mask
    else:
        if args.genome == "hg38":
            mtx_mask = os.path.join(args.basedir, "config", f"{args.genome}_6k_mask_matrix.tsv")
        elif args.genome == "mm10":
            mtx_mask = os.path.join(args.basedir, "config", f"{args.genome}_5k_mask_matrix.tsv")
        else:
            raise ValueError(f"Genome {args.genome} is not supported. ")
    parameters = {
        "genome": args.genome,
        "dropout": 0,
        "preset": "general",
        "mtx_mask": mtx_mask,
    }
    if ChromBERTFTConfig.get_ckpt_type(ckpt) == "pretrain":
        parameters["pretrain_ckpt"] = ckpt
    else:
        parameters["finetune_ckpt"] = ckpt

    config = chrombert.get_preset_model_config(
        basedir = args.basedir,
        **parameters
    )

    return config

def get_meta_file(meta_file,basedir, genome):
    
    if meta_file is None:
        if genome == "hg38":
            meta_file = os.path.join(basedir, "config", f"{genome}_6k_meta.json")
        elif genome == "mm10":
            meta_file = os.path.join(basedir, "config", f"{genome}_5k_meta.json")
        else:
            raise ValueError(f"Genome {genome} is not supported now")
    return meta_file


def get_dataset_config(args):
    if args.hr:
        res = "200bp"
    else:
        res = "1kb"
    if args.hdf5_file is not None:
        hdf5_file = args.hdf5_file
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        if args.genome == "hg38":
            hdf5_file = os.path.join(args.basedir, f"{args.genome}_6k_{res}.hdf5")
        elif args.genome == "mm10":
            hdf5_file = os.path.join(args.basedir, f"{args.genome}_5k_{res}.hdf5")
        else:
            print(f"Genome {args.genome} is not supported. ")
    print(f"Using hdf5 file: {hdf5_file}")
    dataset_config = DatasetConfig(
        kind = "GeneralDataset", 
        supervised_file = args.supervised_file,
        hdf5_file = hdf5_file,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        )
    return dataset_config


def get_regulator_ids(ids, meta_file):

    with open(meta_file) as f:
        meta = json.load(f)

    for i in ids:
        assert i in meta["regulator"], f"Regulator {i} is not in the meta file"
    dict_ids = {i: meta.get(i, i) for i in ids}
    return dict_ids


def get_cistrome_ids(ids, meta_file):

    ids = [i.strip() for i in ids]
    gsm_ids = [i for i in ids if ":" not in i ]
    reg_ids = [i for i in ids if ":" in i]

    with open(meta_file) as f:
        meta = json.load(f)

    dict_ids = {i:i for i in gsm_ids}
    try:
        dict_ids.update({k:meta[k] for k in reg_ids})
    except:
        for k in reg_ids:
            if k not in meta:
                print(f"{k} is not in the meta file!")
        sys.exit(1)

    return dict_ids



class ZarrWriter:
    def __init__(self, output_file, dataset_size, atac_names, regulator_ids, buffer_size=512):
        self.output_file = output_file
        self.dataset_size = dataset_size
        self.atac_names = atac_names
        self.regulator_ids = regulator_ids
        self.buffer_size = buffer_size
        
        self.f = zarr.open(output_file, mode="w")
        self._init_datasets()
        self._init_buffers()
        
        self.current_buffer_rows = 0
        self.n_written = 0

    def _init_datasets(self):
        self.f.create_dataset("region", shape=(self.dataset_size, 4), dtype=np.int64, chunks=(10240, 4))
        self.f.create_dataset("whole", shape=(self.dataset_size, 768), dtype=np.float16, chunks=(10240, 768))
        for cell_type in self.atac_names:
            self.f.create_dataset(f"atac/{cell_type}", shape=(self.dataset_size, 768), dtype=np.float16, chunks=(10240, 768))
        for k in self.regulator_ids:
            self.f.create_dataset(f"regulator/{k}", shape=(self.dataset_size, 768), dtype=np.float16, chunks=(10240, 768))

    def _init_buffers(self):
        self.buffers = {
            "region": [],
            "whole": [],
        }
        for cell_type in self.atac_names:
            self.buffers[f"atac/{cell_type}"] = []
        for k in self.regulator_ids:
            self.buffers[f"regulator/{k}"] = []

    def update_buffer(self, key, value):
        self.buffers[key].append(value)

    def check_and_flush(self, batch_size):
        self.current_buffer_rows += batch_size
        if self.current_buffer_rows >= self.buffer_size:
            self._flush()

    def _flush(self):
        n_end = self.n_written + self.current_buffer_rows
        for key in self.buffers:
            if self.buffers[key]:
                self.f[key][self.n_written:n_end] = np.concatenate(self.buffers[key], axis=0)
                self.buffers[key] = []
        self.n_written = n_end
        self.current_buffer_rows = 0

    def close(self):
        if self.current_buffer_rows > 0:
            self._flush()


def main():
    args = parse_args()
    validate_args(args)
    config = get_model_config(args)
    model = config.init_model().get_embedding_manager().cuda().bfloat16()
    dc = get_dataset_config(args)
    dl = dc.init_dataloader()
    ds = dc.init_dataset()

    meta_file = get_meta_file(args.meta, args.basedir, args.genome)

    dict_ids_atac = get_cistrome_ids(IDS_atac, meta_file)
    dict_ids_regulator = get_regulator_ids(IDS_regulator, meta_file)

    writer = ZarrWriter(
        args.oname, 
        len(ds), 
        IDS_atac_name, 
        dict_ids_regulator.keys()
    )

    with torch.no_grad():
        for batch in tqdm(dl, total=len(dl)):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            model(batch)  # initialize the cache
            
            bs = batch["input_ids"].shape[0]
            
            region = np.concatenate([
                batch["region"].long().cpu().numpy(), 
                batch["build_region_index"].long().cpu().unsqueeze(-1).numpy()
            ], axis=1)
            writer.update_buffer("region", region)

            for cell_type, gsm_id in zip(IDS_atac_name, IDS_atac):
                writer.update_buffer(f"atac/{cell_type}", model.get_cistrome_embedding(gsm_id).float().cpu().detach().numpy())

            for k, v in dict_ids_regulator.items():
                writer.update_buffer(f"regulator/{k}", model.get_cistromes_embedding(v).float().cpu().detach().numpy())

            writer.update_buffer("whole", model.get_region_embedding().float().cpu().detach().numpy())
            
            writer.check_and_flush(bs)

    writer.close()
    return None

    

if __name__ == "__main__":
    main()