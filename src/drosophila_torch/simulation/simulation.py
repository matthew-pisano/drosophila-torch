"""Execution harness for the Drosophila male CNS LIF simulation.

Input is Poisson spike trains injected into sensory neurons identified from the connectome annotations.
Output is the full spike record across all neurons."""

from __future__ import annotations

import logging
from typing import Callable

import torch

from drosophila_torch.neurons import NeuronSuperclass
from drosophila_torch.simulation.lif_network import DrosophilaLIF, LIFConfig, STDPRule


logger = logging.getLogger(__name__)


def generate_pulses_(input_mask: torch.Tensor, total_timesteps: int, total_neurons: int, pulses_per_step: float, pulse_amplitude: float, device=None) -> torch.Tensor:
    """Generate a poisson spike train for each input neuron.

    Args:
        input_mask: The neurons to spike.
        total_timesteps: The number of timesteps to generate spikes for.
        total_neurons: The total neurons in the network.
        pulses_per_step: The pulses to generate at each time step.
        pulse_amplitude: The voltage amplitude per Poisson event in mV
        device: Target device.
    Returns:
        The external voltage supplied to the selected neurons at each timestep."""

    n_selected = input_mask.sum().item()
    # Generate Poisson input only for selected neurons, zero elsewhere
    rate_per_step = torch.full((total_timesteps, n_selected), pulses_per_step, device=device)
    spike_voltage = torch.poisson(rate_per_step).clamp(max=1.0) * pulse_amplitude
    external_voltage = torch.zeros(total_timesteps, total_neurons, device=device)
    external_voltage[:, input_mask] = spike_voltage

    return external_voltage


def superclass_mask(superclass_ids: torch.Tensor, superclasses: list[NeuronSuperclass]) -> torch.Tensor:
    """Return a boolean mask over neurons identifying selected superclass neurons.

    Args:
        superclass_ids: A tensor of superclass ids associated with the neurons.
        superclasses: A list of superclasses to select out of the neurons.
    Returns:
        A bool tensor which is True for selected superclass neurons."""

    mask = torch.zeros(len(superclass_ids), dtype=torch.bool)
    for sc in superclasses:
        mask |= superclass_ids == sc.value

    return mask


def simulate(
        model: DrosophilaLIF,
        stdp: STDPRule | None = None,
        input_mask: torch.Tensor | None = None,
        duration_ms: float = 1000.0,
        rate_hz: float = 10.0,
        amplitude_mv: float = 15.0,
        config: LIFConfig = LIFConfig(),
        device: torch.device | None = None,
        on_step: Callable | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load connectome data, build model, and run simulation.

    Args:
        model: The LIF neural network to model.
        stdp: The updating rule for the neurons in the network.
        input_mask: A mask for which neurons to omit from artificial stimulation.
        duration_ms: Simulation duration in ms.
        rate_hz: Poisson input rate for sensory neurons in Hz.
        amplitude_mv: Voltage amplitude per Poisson event in mV.
        config: LIF configuration.
        device: Target device.
        on_step: An optional callback which fires at the beginning of a simulation timestep.
    Returns:
        A tuple of input voltages, spike trains, and membrane voltages at each time step."""

    if input_mask is None:
        input_mask = torch.ones(model.neuron_count(), dtype=torch.bool)

    total_timesteps = int(duration_ms / config.dt)
    n_selected = input_mask.sum().item()

    logger.info(f"Selected neurons for stimulation: {n_selected:,} / {model.neuron_count():,}")

    pulses_per_step = rate_hz * config.dt * 1e-3
    external_voltage = generate_pulses_(input_mask, total_timesteps, model.neuron_count(), pulses_per_step, amplitude_mv, device)

    logger.info(
        f"Running {total_timesteps} timesteps ({duration_ms:.0f} ms) "
        f"with {n_selected:,} sensory neurons at {rate_hz:.1f} Hz"
    )

    spikes, voltages = model.run(external_voltage, stdp=stdp, on_step=on_step)

    return external_voltage, spikes, voltages
