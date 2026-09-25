"""Visualise MaleCNS neuron soma locations as a 3D scatter plot.

Each point is a neuron soma, colored by superclass. Neurons without a soma location are skipped. Coordinates are in
voxel units at 8nm resolution."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_tensors(pt_path: Path) -> dict:
    """Load the tensor dict from a .pt file."""

    print(f"Loading {pt_path} ...")
    return torch.load(pt_path, weights_only=True)


def plot_soma_locations(data: dict, modulo_filter: int = 1) -> None:
    """Plot neuron soma locations as a 3D scatter colored by superclass.

    Skips neurons where any coordinate is NaN. Adding a modulo filter filters out all neurons except the Nth."""

    soma_xyz = data["soma_xyz"]  # [N, 3]
    superclass_ids = data["superclass_ids"]  # [N]
    superclass_labels = data["superclass_labels"]  # list[str]

    # Filter out all points except for those modulo the filter
    soma_xyz = soma_xyz[::modulo_filter]
    superclass_ids = superclass_ids[::modulo_filter]
    superclass_labels = superclass_labels[::modulo_filter]

    # Mask out neurons with no soma location
    valid = ~torch.isnan(soma_xyz).any(dim=1)
    soma_xyz = soma_xyz[valid]
    superclass_ids = superclass_ids[valid]

    num_classes = len(superclass_labels)
    colormap = plt.colormaps["tab20"].resampled(num_classes)
    colors = colormap(superclass_ids / max(num_classes - 1, 1))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    max_range = (soma_xyz.max(dim=0).values - soma_xyz.min(dim=0).values).max().item() / 2
    midpoints = soma_xyz.float().mean(dim=0)

    ax.set_xlim(midpoints[0].item() - max_range, midpoints[0].item() + max_range)
    ax.set_ylim(midpoints[1].item() - max_range, midpoints[1].item() + max_range)
    ax.set_zlim(midpoints[2].item() - max_range, midpoints[2].item() + max_range)

    ax.scatter(
        soma_xyz[:, 0],
        soma_xyz[:, 1],
        soma_xyz[:, 2],
        c=colors,
        s=2,
        alpha=0.5,
        linewidths=0,
    )

    # Legend lists one entry per superclass
    legend_handles = [
        plt.Line2D([0], [0],
                   marker="o", color="w",
                   markerfacecolor=colormap(i / max(num_classes - 1, 1)),
                   markersize=6,
                   label=label)
        for i, label in enumerate(superclass_labels)
        if label != "unknown"
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=7, framealpha=0.7, markerscale=1.5)

    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")
    ax.set_zlabel("Z (voxels)")
    ax.set_title(f"MaleCNS soma locations: {valid.sum():,} neurons")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualise MaleCNS soma locations from a .pt tensor file."
    )
    parser.add_argument("pt_file", help="Path to the neuron data tensor")
    parser.add_argument(
        "--mod", "--modulo-filter",
        type=int,
        default=1,
        help="Filters out all data points except for the Nth."
    )
    args = parser.parse_args()

    data = load_tensors(Path(args.pt_file))
    plot_soma_locations(data, modulo_filter=args.mod)


if __name__ == "__main__":
    main()
