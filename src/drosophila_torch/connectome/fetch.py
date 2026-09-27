"""Downloads the MaleCNS (Drosophila male CNS connectome) flat-connectome files from the Janelia GCS bucket and converts
them to PyTorch tensors."""

from pathlib import Path

import requests
from tqdm import tqdm


_BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns"
    "/v1.0/connectome-data/flat-connectome"
)

_ESSENTIAL_FILES = {
    "edges": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "nt": "body-neurotransmitters-male-cns-v1.0.feather",
}


def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
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


def download(feather_dir: Path) -> dict[str, Path]:
    """Return local paths for each essential file, downloading any that are missing."""

    feather_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, file_name in _ESSENTIAL_FILES.items():
        dest = feather_dir / file_name
        paths[key] = dest
        if not dest.exists():
            _download_file(f"{_BASE_URL}/{file_name}", dest)
    return paths
