"""Execution harness for the Drosophila male CNS LIF simulation.

Input is Poisson spike trains injected into sensory neurons identified from the connectome annotations.
Output is the full spike record across all neurons."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from drosophila_torch.simulation.lif_network import DrosophilaLIF, LIFConfig, STDPRule
from drosophila_torch.simulation.superclass import NeuronSuperclass


logger = logging.getLogger(__name__)


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


def _superclass_mask(superclass_ids: torch.Tensor, superclasses: list[NeuronSuperclass]) -> torch.Tensor:
    """Return a boolean mask over neurons identifying selected superclass neurons.

    Args:
        superclass_ids: A tensor of superclass ids associated with the neurons.
        superclasses: A list of superclasses to select out of the neurons.
    Returns:
        A bool tensor which is True for selected superclass neurons."""

    mask = torch.zeros(len(superclass_ids), dtype=torch.bool)
    for sc in superclasses:
        mask |= superclass_ids == sc.value

    logger.info(f"Selected neurons: {mask.sum().item():,} / {len(superclass_ids):,}")
    return mask


def run(
        data_path: Path,
        duration_ms: float = 1000.0,
        rate_hz: float = 10.0,
        use_stdp: bool = True,
        config: LIFConfig = LIFConfig(),
        device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load connectome data, build model, and run simulation.

    Args:
        data_path: Path to the .pt file produced by the preprocessing script.
        duration_ms: Simulation duration in ms.
        rate_hz: Poisson input rate for sensory neurons in Hz.
        use_stdp: Whether to enable STDP weight updates during the run.
        config: LIF configuration.
        device: Target device.
    Returns:
        A tuple of spike trains and membrane voltages at each time step."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    logger.info(f"Loading connectome tensors from {data_path}")
    data = torch.load(data_path, map_location=device)
    logger.info(f"Neurons: {data['N']:,}  Edges: {data['meta']['num_edges']:,}")

    model = DrosophilaLIF(
        edge_pre=data["edge_pre_idx"].to(device),
        edge_post=data["edge_post_idx"].to(device),
        edge_delay=data["edge_delay_vec"].to(device),
        edge_weights=data["edge_weights"].to(device),
        sign_vec=data["sign_vec"].to(device),
        nt_vec=data["nt_vec"].to(device),
        total_neurons=data["N"],
        config=config,
    ).to(device)

    stdp = STDPRule(
        n_neurons=model.mem_voltage.shape[0],
        dt=config.dt,
        device=device,
    ) if use_stdp else None

    N = data["N"]
    T = int(duration_ms / config.dt)
    s_mask = _superclass_mask(data).to(device)
    n_selected = s_mask.sum().item()

    # Generate Poisson input only for selected neurons, zero elsewhere
    rate_per_step = torch.full((T, n_selected), rate_hz * config.dt * 1e-3, device=device)
    sensory_input = torch.poisson(rate_per_step).clamp(max=1.0)
    current_in = torch.zeros(T, N, device=device)
    current_in[:, s_mask] = sensory_input

    logger.info(
        f"Running {T} timesteps ({duration_ms:.0f} ms) "
        f"with {n_selected:,} sensory neurons at {rate_hz:.1f} Hz"
    )

    spikes, voltages = model.run(current_in, stdp=stdp)

    return spikes, voltages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Drosophila male CNS LIF simulation."
    )
    parser.add_argument("data_path", type=Path,
                        help="Path to the connectome .pt file")
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

    config = LIFConfig(dt=args.dt)
    device = torch.device(args.device) if args.device else None

    spikes, voltages = run(
        data_path=args.data_path,
        duration_ms=args.duration,
        rate_hz=args.rate,
        use_stdp=not args.frozen,
        config=config,
        device=device,
    )

    rates = _mean_firing_rate(spikes, config.dt)
    logger.info(
        f"Mean firing rate: {rates.mean().item():.2f} Hz  "
        f"Max: {rates.max().item():.2f} Hz  "
        f"Active neurons: {(rates > 0).sum().item():,}"
    )

    if args.out:
        logger.info(f"Saving results to {args.out}")
        torch.save({"spikes": spikes, "voltages": voltages}, args.out)
        logger.info(f"Saved {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
