"""Leaky Integrate-and-Fire (LIF) simulation of the Drosophila male CNS connectome.

This model integrates the known parameters from the dataset as a ground truth while leaving missing information as an
approximation. With these fixed as constants, the neuron optimizes the synaptic weights as learnable parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from drosophila_torch.neurons import NTType


logger = logging.getLogger(__name__)


@dataclass
class LIFConfig:
    """Configuration for the Drosophila male CNS LIF simulation."""

    # Estimated hyperparameters based on biophysical measurements #

    tau_membrane: float = 20.0
    """Membrane time constant in ms. Governs how long the neuron takes to leak potential back its resting voltage."""

    tau_synapse: float = 5.0
    """Synapse time constant in ms. Governs how long it takes for synaptic voltage to absorb into the neuron."""

    v_rest: float = -70.0
    """Resting membrane potential in mV."""

    v_thresh: float = -55.0
    """Spike threshold in mV."""

    v_spike: float = +40
    """Spike potential in mV."""

    v_reset: float = -75.0
    """Post-spike reset potential in mV. Approximates after-hyperpolarization following a spike."""

    # Simulation hyperparameters #

    dt: float = 1.0
    """Simulation timestep in ms."""

    min_delay: int = 1
    """Minimum synaptic delay in timesteps. Applied to axons where soma coordinates are missing or where the 
    distance-derived delay rounds to zero."""

    weight_scale: float = 1.0
    """Global scale factor applied to synapse counts. Synapse count is a structural proxy for connection strength 
    instead of a conductance measurement."""


class STDPRule:
    """Spike-timing dependent plasticity (STDP).

    Weight updates depend only on the relative timing of pre- and post-synaptic spikes instead of a global error signal.
    As this model does not have a direct optimization objective, there is no global signal to learn from.

    LTP (long-term potentiation): A post-synaptic spike following a pre-synaptic spike strengthens the synapse.
    LTD (long-term depression): A post-synaptic spike preceding a pre-synaptic spike weakens the synapse.

    Traces decay exponentially between spikes; the update at each step is proportional to the current trace of the
    opposing neuron."""

    def __init__(
            self,
            n_neurons: int,
            a_plus: float = 0.010,
            a_minus: float = 0.012,
            tau: float = 20.0,
            w_min: float = 0.0,
            w_max: float | None = None,
            dt: float = 1.0,
            device: torch.device | None = None,
    ) -> None:
        """
        Args:
            n_neurons: Total number of neurons.
            a_plus: LTP learning rate.
            a_minus: LTD learning rate. Slightly larger than a_plus by convention to ensure weight stability.
            tau: Trace decay time constant in ms.
            w_min: Hard lower bound on synaptic weights.
            w_max: Hard upper bound on synaptic weights. None for unbounded.
            dt: Simulation timestep in ms."""

        self.a_plus = a_plus
        self.a_minus = a_minus
        self.alpha = dt / tau  # Learning rate
        self.w_min = w_min
        self.w_max = w_max

        # Track when each connecting neuron last spiked
        self.trace = torch.zeros(n_neurons, device=device)

    def reset(self) -> None:
        """Reset eligibility trace. Call at the start of each trial."""

        self.trace.zero_()

    @torch.no_grad()
    def step(self, spikes: torch.Tensor, axon_pre: torch.Tensor, axon_post: torch.Tensor, nt_vec: torch.Tensor, W: nn.Parameter) -> None:
        """Update trace and apply weight changes for one timestep.

        Args:
            spikes: Spike vector for network neurons.
            axon_pre: Pre-synaptic neuron index for each axon.
            axon_post: Post-synaptic neuron index for each axon.
            nt_vec: The neurotransmitter vector for each neuron.
            W: Synaptic weight parameter to update in-place."""

        # Decay traces
        self.trace = self.trace * (1.0 - self.alpha) + spikes

        # Modulatory signal: mean activity of dopaminergic neurons
        # Acts as a global or region-specific learning rate scale
        da_mask = nt_vec == NTType.DOPAMINE
        # Measures how active dopaminergic are
        da_signal = spikes[da_mask].mean()

        # Only ionotropic axons undergo classic STDP
        axon_nt = nt_vec[axon_pre]
        ionotropic = (
                (axon_nt == NTType.ACETYLCHOLINE) |
                (axon_nt == NTType.GABA) |
                (axon_nt == NTType.GLUTAMATE)
        )

        # Per-axon: LTP when post fires (pre trace captures recent pre activity)
        dW_plus = self.a_plus * self.trace[axon_post] * self.trace[axon_pre]

        # Per-axon: LTD when pre fires (post trace captures recent post activity)
        dW_minus = self.a_minus * self.trace[axon_pre] * self.trace[axon_post]

        # Dopamine gates the magnitude of plasticity
        dW = (dW_plus - dW_minus) * ionotropic.float() * (1.0 + da_signal)

        W.data.add_(dW)
        W.data.clamp_(min=self.w_min, max=self.w_max)


class DrosophilaLIF(nn.Module):
    """Structurally constrained LIF simulation of the Drosophila male CNS.

    Connectivity, neurotransmitter identity, synapse counts, and synaptic delays are all fixed from the connectome.
    Synaptic weights are initialized from synapse counts and updated via STDP.
    Membrane biophysics (tau_mem, v_thresh, v_rest, v_reset) are uniform hyperparameters in LIFConfig.

    Spike generation is instant and immediately goes into the refractory period.
    No surrogate gradients are used; STDP is a local rule that does not require backprop.

    Args:
        axon_pre: A tensor of pre-synaptic, axon-originating neurons.
        axon_post: A tensor of post-synaptic, axon-terminating neurons.
        axon_delay: The delay in timesteps of information between neurons.
        sign_vec: A vector of signal signs for each neuron, excitatory or inhibitory.
        nt_vec: A vector of neuron types for each neuron.
        config: LIFConfig instance controlling biophysical and simulation  parameters. Defaults to LIFConfig()."""

    def __init__(self,
                 axon_pre: torch.Tensor,
                 axon_post: torch.Tensor,
                 axon_delay: torch.Tensor,
                 sign_vec: torch.Tensor,
                 nt_vec: torch.Tensor,
                 config: LIFConfig = LIFConfig()) -> None:
        super().__init__()
        self.config = config
        n_neurons = len(nt_vec)

        self.register_buffer("axon_pre", axon_pre)
        self.register_buffer("axon_post", axon_post)
        self.register_buffer("axon_delay", axon_delay)
        self.register_buffer("sign_vec", sign_vec)
        self.register_buffer("nt_vec", nt_vec)

        # Weights are randomly initialized and scaled by weight_scale.
        # Kept non-negative; sign is applied at runtime via sign_vec[axon_pre].
        # Updated by STDPRule, not by gradient descent.
        min_init, max_init = 0.5, 2.0
        weight_range = (max_init - min_init) * torch.rand(axon_post.shape) + min_init
        self.W = nn.Parameter(weight_range * config.weight_scale, requires_grad=False)

        max_delay = int(axon_delay.max().item())
        self.max_delay = max_delay

        # The cross membrane voltage of each neuron.
        self.register_buffer("mem_voltage", torch.full((n_neurons,), config.v_rest))

        # A circular buffer of past spike trains.
        self.register_buffer("spike_buf", torch.zeros(max_delay + 1, n_neurons))

        self.register_buffer("spiking", torch.zeros(n_neurons))

    def reset_state(self) -> None:
        """Reset all dynamic state to initial conditions. Call at the start of each trial or epoch."""

        self.mem_voltage.fill_(self.config.v_rest)
        self.spike_buf.zero_()
        self.spiking.zero_()

    def forward(self, timestep: int, external_voltage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the simulation by one timestep.

        Args:
            timestep: Current timestep index (used to index spike buffer).
            external_voltage: External input voltage per neuron.
        Returns:
            spikes: A tensor representing neuron spikes, 1.0 where a neuron fired, else 0.0.
            voltage: Membrane voltage per neuron after this step."""

        cfg = self.config

        # Retrieve delayed pre-synaptic spikes for each axon
        # buf_idx wraps around the circular buffer for each axon's delay
        buf_idx = (timestep - self.axon_delay) % (self.max_delay + 1)
        # The spikes from pre-synaptic neurons
        pre_spikes = self.spike_buf[buf_idx, self.axon_pre]

        # Signed synaptic voltage per axon
        # Sign comes from the pre-synaptic neuron's NT type
        per_synapse_voltages = pre_spikes * self.W * self.sign_vec[self.axon_pre]

        # Scatter-add axon voltages to post-synaptic neurons
        synaptic_voltages = torch.zeros_like(self.mem_voltage)
        synaptic_voltages.scatter_add_(0, self.axon_post, per_synapse_voltages)

        # Neurons that spiked last timestep hyperpolarize to v_reset
        self.mem_voltage = torch.where(
            self.spiking.bool(),
            torch.full_like(self.mem_voltage, cfg.v_reset),
            self.mem_voltage,
        )

        # Membrane voltage update
        # The leak naturally pulls v_reset back toward v_rest over time
        membrane_leak = (cfg.dt / cfg.tau_membrane) * (self.mem_voltage - cfg.v_rest)
        synaptic_drive = (cfg.dt / cfg.tau_synapse) * (synaptic_voltages + external_voltage)
        self.mem_voltage += -membrane_leak + synaptic_drive

        # Spike detection
        spikes = (self.mem_voltage >= cfg.v_thresh).float()

        # Spiking neurons jump to v_spike this timestep
        self.mem_voltage = torch.where(
            spikes.bool(),
            torch.full_like(self.mem_voltage, cfg.v_spike),
            self.mem_voltage,
        )

        self.spiking = spikes
        self.spike_buf[timestep % (self.max_delay + 1)] = spikes

        return spikes, self.mem_voltage

    def run(self, external_voltage: torch.Tensor, stdp: STDPRule | None = None, on_step: Callable | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the simulation for total_timesteps.

        Args:
            external_voltage: External input voltage per neuron.
            stdp: If provided, weights are updated at each timestep using local spike timing. If None, weights are frozen.
            on_step: An optional callback which fires at the beginning of a timestep.
        Returns:
            A tuple of spike trains and membrane voltages at each time step."""

        total_timesteps = external_voltage.shape[0]
        self.reset_state()
        if stdp is not None:
            stdp.reset()

        spike_record = torch.zeros(total_timesteps, self.mem_voltage.shape[0], device=self.mem_voltage.device)
        voltage_record = torch.zeros(total_timesteps, self.mem_voltage.shape[0], device=self.mem_voltage.device)

        for t in range(total_timesteps):
            if on_step is not None:
                on_step()  # Fire callback

            spikes, voltage = self.forward(t, external_voltage[t])
            spike_record[t] = spikes

            voltage_record[t] = voltage

            if stdp is not None:
                stdp.step(
                    spikes=spikes,
                    axon_pre=self.axon_pre,
                    axon_post=self.axon_post,
                    nt_vec=self.nt_vec,
                    W=self.W,
                )

        return spike_record, voltage_record

    def neuron_count(self):
        """The number of neurons in the network."""

        return len(self.nt_vec)

    def synapse_count(self):
        """The number of synapses in the network."""

        return len(self.axon_pre)
