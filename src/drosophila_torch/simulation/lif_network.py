"""Leaky Integrate-and-Fire (LIF) simulation of the Drosophila male CNS connectome.

This model integrates the known parameters from the dataset as a ground truth while leaving missing information as an
approximation. With these fixed as constants, the neuron optimizes the synaptic weights as learnable parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from drosophila_torch.simulation.neurotransmitters import NTType


logger = logging.getLogger(__name__)


@dataclass
class LIFConfig:
    """Configuration for the Drosophila male CNS LIF simulation."""

    # Estimated hyperparameters based on biophysical measurements #

    tau_mem: float = 20.0
    """Membrane time constant in ms. Governs how long the neuron takes to leak potential back its resting voltage."""

    v_rest: float = -70.0
    """Resting membrane potential in mV."""

    v_thresh: float = -55.0
    """Spike threshold in mV."""

    v_reset: float = -75.0
    """Post-spike reset potential in mV. Approximates after-hyperpolarization following a spike."""

    # Simulation hyperparameters #

    dt: float = 1.0
    """Simulation timestep in ms."""

    min_delay: int = 1
    """Minimum synaptic delay in timesteps. Applied to edges where soma coordinates are missing or where the 
    distance-derived delay rounds to zero."""

    refractory_steps: int = 2
    """Absolute refractory period in timesteps. During this window the neuron cannot spike regardless of input."""

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
    def step(self, spikes: torch.Tensor, edge_pre: torch.Tensor, edge_post: torch.Tensor, nt_vec: torch.Tensor, W: nn.Parameter) -> None:
        """Update trace and apply weight changes for one timestep.

        Args:
            spikes: Spike vector for network neurons.
            edge_pre: Pre-synaptic neuron index for each edge.
            edge_post: Post-synaptic neuron index for each edge.
            nt_vec: The neurotransmitter vector for each neuron.
            W: Synaptic weight parameter to update in-place."""

        # Decay traces
        self.trace = self.trace * (1.0 - self.alpha) + spikes

        # Modulatory signal: mean activity of dopaminergic neurons
        # Acts as a global or region-specific learning rate scale
        da_mask = nt_vec == NTType.DOPAMINE
        # Measures how active dopaminergic are
        da_signal = spikes[da_mask].mean()

        # Only ionotropic edges undergo classic STDP
        edge_nt = nt_vec[edge_pre]
        ionotropic = (
                (edge_nt == NTType.ACETYLCHOLINE) |
                (edge_nt == NTType.GABA) |
                (edge_nt == NTType.GLUTAMATE)
        )

        # Per-edge: LTP when post fires (pre trace captures recent pre activity)
        dW_plus = self.a_plus * self.trace[edge_post] * self.trace[edge_pre]

        # Per-edge: LTD when pre fires (post trace captures recent post activity)
        dW_minus = self.a_minus * self.trace[edge_pre] * self.trace[edge_post]

        # Dopamine gates the magnitude of plasticity
        dW = (dW_plus - dW_minus) * ionotropic.float() * (1.0 + da_signal)

        W.data.add_(dW)
        W.data.clamp_(min=self.w_min, max=self.w_max)


class DrosophilaLIF(nn.Module):
    """Structurally constrained LIF simulation of the Drosophila male CNS.

    Connectivity, neurotransmitter identity, synapse counts, and synaptic delays are all fixed from the connectome.
    Synaptic weights are initialized from synapse counts and updated via STDP.
    Membrane biophysics (tau_mem, v_thresh, v_rest, v_reset) are uniform hyperparameters in LIFConfig.

    Spike generation uses a hard Heaviside threshold and hard reset.
    No surrogate gradients are used; STDP is a local rule that does not require backprop.

    Args:
        edge_pre: A tensor of pre-synaptic, edge-originating neurons.
        edge_post: A tensor of post-synaptic, edge-terminating neurons.
        edge_delay: The delay in timesteps of information between neurons.
        edge_weights: The strength of edges.
        sign_vec: A vector of signal signs for each neuron, excitatory or inhibitory.
        nt_vec: A vector of neuron types for each neuron.
        total_neurons: The total number of neurons in the network.
        config: LIFConfig instance controlling biophysical and simulation  parameters. Defaults to LIFConfig()."""

    def __init__(self,
                 edge_pre: torch.Tensor,
                 edge_post: torch.Tensor,
                 edge_delay: torch.Tensor,
                 edge_weights: torch.Tensor,
                 sign_vec: torch.Tensor,
                 nt_vec: torch.Tensor,
                 total_neurons: int,
                 config: LIFConfig = LIFConfig()) -> None:
        super().__init__()
        self.config = config
        N = total_neurons

        self.register_buffer("edge_pre", edge_pre)
        self.register_buffer("edge_post", edge_post)
        self.register_buffer("edge_delay", edge_delay)
        self.register_buffer("sign_vec", sign_vec)
        self.register_buffer("nt_vec", nt_vec)

        # Initialized from synapse counts scaled by weight_scale.
        # Kept non-negative; sign is applied at runtime via sign_vec[edge_pre].
        # Updated by STDPRule, not by gradient descent.
        self.W = nn.Parameter(edge_weights.float().clamp(min=0.0) * config.weight_scale, requires_grad=False)

        max_delay = int(edge_delay.max().item())
        self.max_delay = max_delay

        # The cross membrane voltage of each neuron.
        self.register_buffer("mem_voltage", torch.full((N,), config.v_rest))

        # The remaining refractory steps per neuron.
        self.register_buffer("refractory_remaining", torch.zeros(N, dtype=torch.long))

        # A circular buffer of past spike trains.
        self.register_buffer("spike_buf", torch.zeros(max_delay + 1, N))

    def reset_state(self) -> None:
        """Reset all dynamic state to initial conditions. Call at the start of each trial or epoch."""

        self.mem_voltage.fill_(self.config.v_rest)
        self.refractory_remaining.zero_()
        self.spike_buf.zero_()

    def forward(self, timestep: int, current_in: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the simulation by one timestep.

        Args:
            timestep: Current timestep index (used to index spike buffer).
            current_in: External input current per neuron.
        Returns:
            spikes: A tensor representing neuron spikes, 1.0 where a neuron fired, else 0.0.
            voltage: Membrane voltage per neuron after this step."""

        cfg = self.config

        # Retrieve delayed pre-synaptic spikes for each edge
        # buf_idx wraps around the circular buffer for each edge's delay
        buf_idx = (timestep - self.edge_delay) % (self.max_delay + 1)
        # The spikes from pre-synaptic neurons
        pre_spikes = self.spike_buf[buf_idx, self.edge_pre]

        # Signed synaptic current per edge
        # Sign comes from the pre-synaptic neuron's NT type
        edge_current = pre_spikes * self.W * self.sign_vec[self.edge_pre]

        # Scatter-add edge currents to post-synaptic neurons
        synaptic_currents = torch.zeros_like(self.mem_voltage)
        synaptic_currents.scatter_add_(0, self.edge_post, edge_current)

        # Membrane voltage update
        # Euler discretization of: tau_mem * dV/dt = -(V - V_rest) + I
        # Blocked for neurons currently in their refractory period.
        not_refractory = (self.refractory_remaining == 0).float()
        alpha = cfg.dt / cfg.tau_mem
        self.mem_voltage = self.mem_voltage + not_refractory * alpha * (
                -(self.mem_voltage - cfg.v_rest) + synaptic_currents + current_in
        )

        # Spike detection (hard threshold)
        spikes = (self.mem_voltage >= cfg.v_thresh).float()

        # Hard reset and refractory counter
        self.mem_voltage = torch.where(
            spikes.bool(),
            torch.full_like(self.mem_voltage, cfg.v_reset),
            self.mem_voltage,
        )
        self.refractory_remaining = torch.where(
            spikes.bool(),
            torch.full_like(self.refractory_remaining, cfg.refractory_steps),
            (self.refractory_remaining - 1).clamp(min=0),
        )

        # Write spikes into circular buffer
        self.spike_buf[timestep % (self.max_delay + 1)] = spikes

        return spikes, self.mem_voltage

    def run(self, current_in: torch.Tensor, stdp: STDPRule | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the simulation for T timesteps.

        Args:
            current_in: External input current per neuron.
            stdp: If provided, weights are updated at each timestep using local spike timing. If None, weights are frozen.
        Returns:
            A tuple of float spike trains and float membrane voltages"""

        T = current_in.shape[0]
        self.reset_state()
        if stdp is not None:
            stdp.reset()

        spike_record = torch.zeros(T, self.mem_voltage.shape[0], device=self.mem_voltage.device)
        voltage_record = torch.zeros(T, self.mem_voltage.shape[0], device=self.mem_voltage.device)

        for t in range(T):
            spikes, voltage = self.forward(t, current_in[t])
            spike_record[t] = spikes

            voltage_record[t] = voltage

            if stdp is not None:
                # Retrieve the pre-synaptic spikes that were active this step
                buf_idx = (t - self.edge_delay) % (self.max_delay + 1)
                pre_spikes = self.spike_buf[buf_idx, self.edge_pre]

                # Aggregate pre-synaptic activity back to neuron level for traces
                pre_neuron_spikes = torch.zeros_like(spikes)
                pre_neuron_spikes.scatter_add_(0, self.edge_pre, pre_spikes)
                pre_neuron_spikes = pre_neuron_spikes.clamp(max=1.0)

                stdp.step(
                    pre_spikes=pre_neuron_spikes,
                    post_spikes=spikes,
                    edge_pre=self.edge_pre,
                    edge_post=self.edge_post,
                    W=self.W,
                )

        return spike_record, voltage_record
