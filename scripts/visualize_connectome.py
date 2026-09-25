"""Visualise MaleCNS neuron soma locations as a 3D scatter plot.

Each point is a neuron soma, colored by superclass. Neurons without a soma location are skipped. Coordinates are in
voxel units at 8nm resolution."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def load_tensors(pt_path: Path) -> dict:
    """Load the tensor dict from a .pt file."""

    print(f"Loading {pt_path} ...")
    return torch.load(pt_path, weights_only=True)


def plot_soma_locations(ax: plt.Axes, data: dict, colormap, num_classes: int, modulo_filter: int = 1) -> torch.Tensor:
    """Scatter plot of soma locations colored by superclass.

    Adding a modulo filter filters out all neurons except the Nth.

    Returns:
        The valid soma coordinates after NaN filtering, for use in computing axis limits."""

    print("Plotting soma locations ...")
    soma_xyz = data["soma_xyz"]
    superclass_ids = data["superclass_ids"]

    # Filter out all points except for those modulo the filter
    soma_xyz = soma_xyz[::modulo_filter]
    superclass_ids = superclass_ids[::modulo_filter]

    # Mask out neurons with no soma location
    valid = ~torch.isnan(soma_xyz).any(dim=1)
    soma_xyz = soma_xyz[valid]
    superclass_ids = superclass_ids[valid]

    colors = colormap(superclass_ids.numpy() / max(num_classes - 1, 1))

    ax.scatter(
        soma_xyz[:, 0], soma_xyz[:, 1], soma_xyz[:, 2],
        c=colors, s=1, alpha=0.5, linewidths=0,
    )

    return soma_xyz


def plot_edges(ax: plt.Axes, data: dict, sample_proportion: float, colormap, num_classes: int) -> None:
    """Draw a random sample of edges as translucent gray lines.

    Extracts pre- / post-indices from the sparse CSR adjacency matrix, samples a proportion of them, then looks up soma
    coordinates for each endpoint. Edges where either endpoint has no soma location are skipped."""

    print("Extracting edges from sparse matrix ...")
    W_coo = data["W"].to_sparse_coo().coalesce()
    pre_idx = W_coo.indices()[0]
    post_idx = W_coo.indices()[1]

    num_edges = pre_idx.shape[0]
    num_samples = max(1, int(num_edges * sample_proportion))
    sample_indices = torch.randperm(num_edges)[:num_samples]

    pre_idx = pre_idx[sample_indices]
    post_idx = post_idx[sample_indices]

    soma_xyz = data["soma_xyz"]
    pre_coords = soma_xyz[pre_idx]
    post_coords = soma_xyz[post_idx]

    # Drop edges where either endpoint has no soma location
    valid = (
            ~torch.isnan(pre_coords).any(dim=1) &
            ~torch.isnan(post_coords).any(dim=1)
    )
    pre_coords = pre_coords[valid]
    post_coords = post_coords[valid]

    print(f"Drawing {valid.sum():,} edges (sampled {num_samples:,} from {num_edges:,})")

    # Build segment list for Line3DCollection: each segment is [[x0,y0,z0],[x1,y1,z1]]
    segments = torch.stack([pre_coords, post_coords], dim=1).numpy()

    pre_superclass = data["superclass_ids"][pre_idx[valid]]
    edge_colors = colormap(pre_superclass.numpy() / max(num_classes - 1, 1))

    edge_collection = Line3DCollection(segments, linewidths=0.3, alpha=0.1, colors=edge_colors)
    ax.add_collection3d(edge_collection)


def set_equal_aspect(ax: plt.Axes, soma_xyz: torch.Tensor) -> None:
    """Force equal axis scaling so the volume renders without distortion."""

    max_range = (soma_xyz.max(dim=0).values - soma_xyz.min(dim=0).values).max().item() / 2
    midpoints = soma_xyz.float().mean(dim=0)

    ax.set_xlim(midpoints[0].item() - max_range, midpoints[0].item() + max_range)
    ax.set_ylim(midpoints[1].item() - max_range, midpoints[1].item() + max_range)
    ax.set_zlim(midpoints[2].item() - max_range, midpoints[2].item() + max_range)


def build_legend(colormap, num_classes: int, superclass_labels: list[str]) -> list:
    """Build legend handles, one per superclass, skipping 'unknown'."""

    return [
        plt.Line2D([0], [0],
                   marker="o", color="w",
                   markerfacecolor=colormap(i / max(num_classes - 1, 1)),
                   markersize=6,
                   label=label)
        for i, label in enumerate(superclass_labels)
        if label != "unknown"
    ]


def visualize(data: dict, edge_sample: float | None = None, modulo_filter: int = 1) -> None:
    """Organize the full visualization."""

    superclass_labels = data["superclass_labels"]
    num_classes = len(superclass_labels)
    colormap = plt.colormaps["tab20"].resampled(num_classes)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    soma_xyz = plot_soma_locations(ax, data, colormap, num_classes, modulo_filter=modulo_filter)

    if edge_sample is not None:
        plot_edges(ax, data, edge_sample, colormap, num_classes)

    set_equal_aspect(ax, soma_xyz)

    ax.legend(handles=build_legend(colormap, num_classes, superclass_labels),
              loc="upper left", fontsize=7, framealpha=0.7, markerscale=1.5)

    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Y (voxels)")
    ax.set_zlabel("Z (voxels)")
    ax.set_title(f"MaleCNS soma locations, {len(soma_xyz):,} neurons")

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
    parser.add_argument(
        "--edge-sample", type=float, default=None,
        help="Proportion of edges to draw e.g. 0.0001 for 0.01%% (default: no edges)"
    )
    args = parser.parse_args()

    data = load_tensors(Path(args.pt_file))
    visualize(data, modulo_filter=args.mod, edge_sample=args.edge_sample)


if __name__ == "__main__":
    main()
