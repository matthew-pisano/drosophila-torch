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


def generate_pulses_(input_mask: torch.Tensor, pulse_vector: torch.Tensor,
                     noise_std: float = 0.0, device=None) -> torch.Tensor:
    """Generate a constant voltage stimulus for a targeted subset of neurons.

    Delivers pulse_amplitude mV to each selected neuron at every timestep,
    simulating sustained targeted excitatory input.

    Args:
        input_mask: Boolean mask over neurons to stimulate.
        pulse_vector: A vector spanning the time span of the simulation with elements representing pulse amplitude.
        noise_std: The standard deviation of gaussian noise to add to pulses.
        device: Target device.
    Returns:
        The external voltage supplied to the selected neurons at each timestep."""

    pulse_mask = pulse_vector > 0
    jitter = torch.randn(len(pulse_vector), device=device) * noise_std
    jitter[~pulse_mask] = 0
    bounded_pulse_vector = (pulse_vector + jitter).abs()
    return torch.outer(bounded_pulse_vector, input_mask.float())


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
        duration_ms: float,
        rate_hz: float,
        amplitude_mv: float,
        stdp: STDPRule | None = None,
        input_mask: torch.Tensor | None = None,
        config: LIFConfig = LIFConfig(),
        device: torch.device | None = None,
        on_step: Callable | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load connectome data, build model, and run simulation.

    Args:
        model: The LIF neural network to model.
        duration_ms: Simulation duration in ms.
        rate_hz: Poisson input rate for sensory neurons in Hz.
        amplitude_mv: Voltage amplitude per Poisson event in mV.
        stdp: The updating rule for the neurons in the network.
        input_mask: A mask for which neurons to omit from artificial stimulation.
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

    pulse_vector = torch.zeros(total_timesteps, device=device)
    pulse_vector[:int(total_timesteps * 0.6)] = amplitude_mv

    external_voltage = generate_pulses_(input_mask, pulse_vector, device=device)

    logger.info(
        f"Running {total_timesteps} timesteps ({duration_ms:.0f} ms) "
        f"with {n_selected:,} sensory neurons at {rate_hz:.1f} Hz"
    )

    spikes, voltages = model.run(external_voltage, stdp=stdp, on_step=on_step)

    return external_voltage, spikes, voltages
