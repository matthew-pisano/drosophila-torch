"""Visualise MaleCNS neuron soma locations as a 3D scatter plot.

Each point is a neuron soma, colored by superclass. Neurons without a soma location are skipped. Coordinates are in
voxel units at 8nm resolution."""

import argparse

import torch

from drosophila_torch.connectome import visualize


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

    data = torch.load(args.pt_file, weights_only=True)
    visualize(data, modulo_filter=args.mod, edge_sample=args.edge_sample)


if __name__ == "__main__":
    main()
