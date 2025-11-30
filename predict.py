#!/usr/bin/env python3
"""
Command-line script for running imputation model inference.

This script loads a trained imputation model and runs forward inference
to compute logits for all cell type and regulator combinations.
"""

import argparse
import sys
import pathlib
import zarr
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run imputation model inference to compute logits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-k',
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )

    parser.add_argument(
        "--list_regulators_file",
        type=str,
        required=True,
        help="Path to list regulators file (one regulator per line)"
    )

    parser.add_argument(
        "--list_cells_file",
        type=str,
        required=True,
        help="Path to list cells file (one cell type per line)"
    )

    parser.add_argument(
        "--num_cell_parallel",
        type=int,
        default = 5,
        help="Number of cells to process in parallel"
    )

    parser.add_argument(
        "--atac-dir",
        type=str,
        required=True,
        help="Directory containing ATAC embeddings (zarr format)"
    )

    parser.add_argument(
        "--regulator-dir",
        type=str,
        required=True,
        help="Directory containing regulator embeddings (zarr format)"
    )

    parser.add_argument(
        "--whole-dir",
        type=str,
        required=True,
        help="Directory containing whole embeddings (zarr format)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results"
    )


    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    try:
        from imputation_model import ImputationModel
    except ImportError as e:
        print(f"Error importing ImputationModel: {e}")
        sys.exit(1)

    # Create output directory
    odir = pathlib.Path(args.output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = ImputationModel.load_from_checkpoint(args.checkpoint).model.cuda().bfloat16()
    model.eval()

    # Load dataset to extract cell types and regulators
    with open(args.list_regulators_file, "r") as f:
        regulators_unique = [line.strip() for line in f.readlines() if line.strip() != "zarr.json"]
        regulators_unique = sorted(list(set(regulators_unique)))
    with open(args.list_cells_file, "r") as f:
        cts_unique = [line.strip() for line in f.readlines() if line.strip() != "zarr.json"]
        cts_unique = sorted(list(set(cts_unique)))

    cts_group = []
    for i in range(0, len(cts_unique), args.num_cell_parallel):
        cts_group.append(cts_unique[i:i+args.num_cell_parallel])

    print(f"Found {len(cts_group)} groups of {args.num_cell_parallel} cell types")
    print(f"Found {len(regulators_unique)} unique regulators: {regulators_unique}")

    # Load whole embedding matrix
    path_zarr_whole = pathlib.Path(args.whole_dir)
    print(f"Loading whole embeddings from: {path_zarr_whole}")
    mtx_whole = torch.from_numpy(zarr.load(path_zarr_whole)).to(
        device="cuda", dtype=torch.bfloat16
    )

    # Create output zarr file
    path_zarr_atac = pathlib.Path(args.atac_dir)
    path_zarr_regulator = pathlib.Path(args.regulator_dir)

    output_zarr = zarr.open(odir, mode="a")

    # Run inference for each cell type
    for cts in cts_group:
        dict_cache_ct = {}
        dict_arr_ct = {}
        print(f"\nProcessing cell types: {cts}")
        for ct in cts:
            emb_ct = torch.from_numpy(zarr.load(path_zarr_atac / f"{ct}")).to(
                device="cuda", dtype=torch.bfloat16
            )
            dict_cache_ct[ct] = emb_ct

            dict_arr_ct[ct] = output_zarr.create_array(
                ct,
                shape=(emb_ct.shape[0], len(regulators_unique)),
                dtype=np.float16,
                chunks=(emb_ct.shape[0], 1)
            )
            dict_arr_ct[ct].attrs["regulators"] = regulators_unique

        # Run inference for each regulator
        for i, regulator in enumerate(tqdm(regulators_unique, desc=f"  Regulators")):
            emb_regulator = torch.from_numpy(
                zarr.load(path_zarr_regulator / f"{regulator}")
            ).to(device="cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                for ct in cts:
                    emb_ct = dict_cache_ct[ct]
                    arr = dict_arr_ct[ct]
                    logit = model(emb_ct, emb_regulator, mtx_whole)
                    arr[:, i] = logit.float().cpu().numpy().astype(np.float16)

    print(f"\nInference complete! Results saved to: {odir / 'logits'}")


if __name__ == "__main__":
    main()
