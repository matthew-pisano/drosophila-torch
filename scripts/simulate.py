"""Simulates the activity of the Drosophilia neurons based on random sampled inputs."""

import argparse
from pathlib import Path

import torch

from drosophila_torch.simulation.lif_network import LIFConfig, DrosophilaLIF, STDPRule
from drosophila_torch.simulation.simulation import superclass_mask, simulate
from drosophila_torch.simulation.superclass import NeuronSuperclass


def _mean_firing_rate(spikes: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Compute mean firing rate per neuron over a spike train.

    Args:
        spikes: float spike tensor.
        dt: Timestep duration in ms.
    Returns:
        The mean firing rate in Hz."""

    T = spikes.shape[0]
    duration_s = T * dt * 1e-3
    return spikes.sum(dim=0) / duration_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Drosophila male CNS LIF simulation."
    )
    parser.add_argument("data_path", type=Path,
                        help="Path to the connectome .pt file")
    parser.add_argument("--superclass", nargs="+", default=None,
                        help="Filter input to one or more superclasses e.g. --superclass descending_neuron visual_projection")
    parser.add_argument("--duration", type=float, default=1000.0,
                        help="Simulation duration in ms (default: 1000)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Poisson input rate for sensory neurons in Hz (default: 10)")
    parser.add_argument("--frozen", action="store_true",
                        help="Disable STDP weight updates (frozen weights)")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Timestep in ms (default: 1.0)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to save results as a .pt file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string e.g. 'cuda:0' or 'cpu'")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading connectome tensors from {args.data_path}")
    data = torch.load(args.data_path, map_location=device)

    config = LIFConfig(dt=args.dt)

    model = DrosophilaLIF(
        edge_pre=data["edge_pre_idx"].to(device),
        edge_post=data["edge_post_idx"].to(device),
        edge_delay=data["edge_delay_vec"].to(device),
        edge_weights=data["edge_weights"].to(device),
        sign_vec=data["sign_vec"].to(device),
        nt_vec=data["nt_vec"].to(device),
        config=config,
    ).to(device)

    print(f"Neurons: {model.neuron_count():,}  Edges: {model.edge_count():,}")

    stdp = STDPRule(
        n_neurons=model.mem_voltage.shape[0],
        dt=config.dt,
        device=device,
    ) if not args.frozen else None

    superclasses = [NeuronSuperclass.from_string(s) for s in args.superclass]
    s_mask = superclass_mask(data["superclass_ids"], superclasses).to(device)

    spikes, voltages = simulate(model, stdp, s_mask, duration_ms=args.duration, rate_hz=args.rate, config=config, device=device)

    rates = _mean_firing_rate(spikes, config.dt)
    print(
        f"Mean firing rate: {rates.mean().item():.2f} Hz  "
        f"Max: {rates.max().item():.2f} Hz  "
        f"Active neurons: {(rates > 0).sum().item():,}"
    )

    if args.out:
        print(f"Saving results to {args.out}")
        torch.save({"spikes": spikes, "voltages": voltages}, args.out)
        print(f"Saved {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
