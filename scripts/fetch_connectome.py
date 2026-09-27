"""Downloads the MaleCNS (Drosophila male CNS connectome) flat-connectome files from the Janelia GCS bucket and converts
them to PyTorch tensors."""

import argparse
from pathlib import Path

import pandas as pd
import torch


def list_selectors(list_class: str, feather_dir: Path):
    """Lists the available annodations of the given class."""

    ann_path = feather_dir / ESSENTIAL_FILES["annotations"]
    if not ann_path.exists():
        download_file(f"{BASE_URL}/{ESSENTIAL_FILES['annotations']}", ann_path)

    annotations = pd.read_feather(ann_path)
    annotations.columns = annotations.columns.str.strip().str.lower()

    values = annotations[list_class].dropna().unique()
    values = sorted(values)
    print(f"\nAvailable {list_class} values ({len(values)}):")
    for value in values:
        print(f"{value}")
    return


def main():
    parser = argparse.ArgumentParser(
        description="Download MaleCNS connectome and save as PyTorch tensors."
    )
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--out-file", default="malecns_tensors.pt", help="Name of the output .pt file")
    parser.add_argument(
        "--superclass",
        nargs="+",
        default=None,
        help="Filter to one or more superclasses (or 'labeled' for all) e.g. --superclass descending_neuron visual_projection"
    )
    parser.add_argument(
        "--type",
        nargs="+",
        default=None,
        help="Filter to one or more cell types (or 'labeled' for all) e.g. --type DNp01 DNp02"
    )
    parser.add_argument(
        "--list",
        choices=["superclass", "type", "subclass"],
        default=None,
        help="List available values for a given annotation field and exit"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    feather_dir = out_dir / "feather"
    pt_path = out_dir / args.out_file

    if args.list:
        list_selectors(args.list, feather_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    paths = ensure_files(feather_dir)
    tensors = build_tensors(paths, args.superclass, args.type)

    print(f"\nSaving tensors to {pt_path} ...")
    torch.save(tensors, pt_path)
    print(f"Saved {pt_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
