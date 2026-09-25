"""Downloads the MaleCNS (Drosophila male CNS connectome) flat-connectome files from the Janelia GCS bucket and converts
them to PyTorch tensors."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm


BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns"
    "/v1.0/connectome-data/flat-connectome"
)

ESSENTIAL_FILES = {
    "edges": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "nt": "body-neurotransmitters-male-cns-v1.0.feather",
}

NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,  # inhibitory at most central synapses in Drosophila
    "octopamine": +1.0,
    "serotonin": +1.0,
    "dopamine": +1.0,
}


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download url to dest."""

    print(f"Downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=60) as req:
        req.raise_for_status()
        total_bytes = int(req.headers.get("content-length", 0))
        with open(dest, "wb") as fh:
            with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in req.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    pbar.update(len(chunk))


def ensure_files(feather_dir: Path) -> dict[str, Path]:
    """Return local paths for each essential file, downloading any that are missing."""

    feather_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, file_name in ESSENTIAL_FILES.items():
        dest = feather_dir / file_name
        paths[key] = dest
        if not dest.exists():
            download_file(f"{BASE_URL}/{file_name}", dest)
    return paths


def load_feathers(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three feather files and normalize column names to lowercase.

    Returns:
        edges, annotations, neurotransmitters as DataFrames."""

    edges = pd.read_feather(paths["edges"])
    annotations = pd.read_feather(paths["annotations"])
    neurotransmitters = pd.read_feather(paths["nt"])

    for dataframe in (edges, annotations, neurotransmitters):
        dataframe.columns = dataframe.columns.str.strip().str.lower()

    return edges, annotations, neurotransmitters


def filter_neurons(
        edges: pd.DataFrame,
        annotations: pd.DataFrame,
        superclasses: list[str] | None,
        types: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict annotations and edges to a subset of neurons.

    Filters annotations by superclass and/or type, then drops any edge whose pre or post body ID is not in the surviving
    annotation set. Both filters are applied as OR within each argument and AND between arguments, e.g.
    --superclass descending_neuron visual_projection --type DNp01
    keeps neurons that are (descending_neuron OR visual_projection) AND type DNp01."""

    mask = pd.Series(True, index=annotations.index)

    if superclasses:
        if "labeled" in superclasses:
            mask &= annotations["superclass"].notna()
        else:
            mask &= annotations["superclass"].isin(superclasses)

    if types:
        if "labeled" in types:
            mask &= annotations["type"].notna()
        else:
            mask &= annotations["type"].isin(types)

    annotations = annotations[mask]
    surviving = set(annotations["bodyid"].values)
    edges = edges[
        edges["body_pre"].isin(surviving) & edges["body_post"].isin(surviving)
        ]

    print(f"Filtered to {len(annotations):,} neurons, {len(edges):,} edges")
    return edges, annotations


def build_neuron_index(edges: pd.DataFrame, annotations: pd.DataFrame) -> tuple[pd.Index, pd.Series]:
    """Build a contiguous integer index over all body IDs.

    Takes the union of body IDs from the edge list and the annotation table so that annotated neurons with no
    connections are still represented.

    Returns:
        all_bodies (ordered Index of body IDs) and body_to_idx (body ID to integer position)."""

    chunk_size = 100_000
    unique_bodies = set(annotations["bodyid"].values)

    with tqdm(total=len(edges), desc="Deduplicating body_pre") as pbar:
        for start in range(0, len(edges), chunk_size):
            chunk = edges["body_pre"].values[start: start + chunk_size]
            unique_bodies.update(chunk)
            pbar.update(chunk_size)

    with tqdm(total=len(edges), desc="Deduplicating body_post") as pbar:
        for start in range(0, len(edges), chunk_size):
            chunk = edges["body_post"].values[start: start + chunk_size]
            unique_bodies.update(chunk)
            pbar.update(chunk_size)

    unique_array = np.fromiter(unique_bodies, dtype=np.int64, count=len(unique_bodies))
    del unique_bodies
    unique_array.sort()  # in-place, no copy

    all_bodies = pd.Index(unique_array)
    del unique_array

    body_to_idx = pd.Series(np.arange(len(all_bodies), dtype=np.int64), index=all_bodies)
    print(f"Total neurons: {len(all_bodies):,}")
    return all_bodies, body_to_idx


def build_edge_tensors(edges: pd.DataFrame, body_to_idx: pd.Series) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the edge dataframe to index and weight tensors.

    Returns:
        pre_idx, post_idx, and raw_weights."""

    chunk_size = 100_000
    num_edges = len(edges)
    pre_idx = torch.empty(num_edges, dtype=torch.long)
    post_idx = torch.empty(num_edges, dtype=torch.long)
    raw_weights = torch.tensor(edges["weight"].values, dtype=torch.float32)

    with tqdm(total=num_edges, desc="Building edge tensors") as pbar:
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            chunk = slice(start, end)

            pre_idx[chunk] = torch.tensor(
                body_to_idx.reindex(edges["body_pre"].values[chunk]).values,
                dtype=torch.long
            )
            post_idx[chunk] = torch.tensor(
                body_to_idx.reindex(edges["body_post"].values[chunk]).values,
                dtype=torch.long
            )
            pbar.update(end - start)

    print(f"Total edges: {num_edges:,}")
    return pre_idx, post_idx, raw_weights


def build_sign_vector(neurotransmitters: pd.DataFrame, all_bodies: pd.Index) -> torch.Tensor:
    """Derive a per-neuron excitatory/inhibitory sign from predicted_nt.

    Neurons absent from the neurotransmitter table default to +1 (excitatory).

    Returns:
        A tensor of sign information for neurons."""

    nt_indexed = neurotransmitters.set_index("body")["predicted_nt"]
    nt_aligned = nt_indexed.reindex(all_bodies)

    sign_values = nt_aligned.map(NT_SIGN).fillna(1.0).values.astype(np.float32)
    return torch.tensor(sign_values)


def build_adjacency_matrices(
        pre_idx: torch.Tensor,
        post_idx: torch.Tensor,
        raw_weights: torch.Tensor,
        sign_vec: torch.Tensor,
        num_neurons: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build signed and unsigned sparse CSR adjacency matrices.

    W applies the pre-synaptic neuron's neurotransmitter sign to each edge weight, so inhibitory connections carry
    negative values. W_unsigned holds raw synapse counts. Both are sparse CSR for efficient matrix-vector products.

    Returns:
        W (signed) and W_unsigned matrices."""

    edge_indices = torch.stack([pre_idx, post_idx])
    signed_weights = raw_weights * sign_vec[pre_idx]

    W = (
        torch.sparse_coo_tensor(edge_indices, signed_weights, size=(num_neurons, num_neurons))
        .to_sparse_csr()
    )
    W_unsigned = (
        torch.sparse_coo_tensor(edge_indices, raw_weights, size=(num_neurons, num_neurons))
        .to_sparse_csr()
    )
    return W, W_unsigned


def build_soma_coordinates(annotations: pd.DataFrame, all_bodies: pd.Index) -> torch.Tensor:
    """Parse soma location strings into a float32 coordinate tensor.

    Neurons with no soma location default to [0, 0, 0]. Coordinates are in voxel units at 8nm resolution.

    Returns:
        A 3D float32 tensor of soma coordinates."""

    annotations = annotations.set_index("bodyid")
    aligned = annotations["somalocation"].reindex(all_bodies)

    coords = [
        value if value is not None else [float('nan')] * 3
        for value in aligned
    ]
    return torch.tensor(coords, dtype=torch.float32)


def _encode_categorical(series: pd.Series) -> tuple[torch.Tensor, list[str]]:
    """Encode a string Series as int32 codes. NaN becomes 'unknown'.

    Returns:
        An int32 tensor of codes and a label list for decoding."""

    categorical = series.fillna("unknown").astype("category")
    codes = torch.tensor(categorical.cat.codes.values.astype(np.int32))
    labels = list(categorical.cat.categories)
    return codes, labels


def build_annotation_tensors(annotations: pd.DataFrame, all_bodies: pd.Index) -> dict[str, torch.Tensor | list[str]]:
    """Encode annotation columns as integer tensors aligned to the neuron index.

    Extracts superclass, class, type, subclass, somaside, entrynerve, and exitnerve.
    Each column is encoded as an int32 tensor with a companion label list for decoding. Neurons absent from the
    annotation table receive 'unknown' for every field.

    Returns:
        A dict of tensor and label list pairs, keyed by field name."""

    annotations = annotations.set_index("bodyid")

    # Columns to encode: (output key prefix, source column)
    annotation_columns = [
        ("superclass", "superclass"),
        ("class", "class"),
        ("type", "type"),
        ("subclass", "subclass"),
        ("side", "somaside"),
        ("entrynerve", "entrynerve"),
        ("exitnerve", "exitnerve"),
    ]

    result = {}
    for key, column in annotation_columns:
        if column in annotations.columns:
            aligned_series = annotations[column].reindex(all_bodies)
        else:
            aligned_series = pd.Series("unknown", index=all_bodies)

        codes, labels = _encode_categorical(aligned_series)
        result[f"{key}_ids"] = codes
        result[f"{key}_labels"] = labels
        print(f"{key}: {len(labels)} categories")

    return result


def build_tensors(paths: dict[str, Path], superclasses: list[str] | None = None, types: list[str] | None = None) -> dict:
    """Orchestrate the full feather to tensor conversion pipeline.

    Loads the three source files, builds the neuron index, edge tensors, sign vector, adjacency matrices, and annotation
    tensors, then packs everything into a single dict for torch.save()."""

    print("\nLoading feather files ...")
    edges, annotations, neurotransmitters = load_feathers(paths)

    edges, annotations = filter_neurons(edges, annotations, superclasses=superclasses, types=types)

    print("Building neuron index ...")
    all_bodies, body_to_idx = build_neuron_index(edges, annotations)

    print("Building edge tensors ...")
    pre_idx, post_idx, raw_weights = build_edge_tensors(edges, body_to_idx)

    print("Computing neurotransmitter signs ...")
    sign_vec = build_sign_vector(neurotransmitters, all_bodies)

    print("Building adjacency matrices ...")
    W, W_unsigned = build_adjacency_matrices(
        pre_idx, post_idx, raw_weights, sign_vec, num_neurons=len(all_bodies)
    )

    print("Building soma coordinates ...")
    soma_xyz = build_soma_coordinates(annotations, all_bodies)

    print("Encoding annotations ...")
    annotation_tensors = build_annotation_tensors(annotations, all_bodies)

    return {
        "W": W,
        "W_unsigned": W_unsigned,
        "sign_vec": sign_vec,
        "body_ids": torch.tensor(all_bodies.values, dtype=torch.int64),
        "N": len(all_bodies),
        "soma_xyz": soma_xyz,
        **annotation_tensors,
        "meta": {
            "dataset": "male-cns:v1.0",
            "source": BASE_URL,
            "num_edges": len(edges),
            "nt_sign_map": NT_SIGN,
        },
    }


def list_selectors(list_class: str, feather_dir: Path):
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
