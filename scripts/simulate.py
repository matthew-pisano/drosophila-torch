"""Simulates the activity of the Drosophilia neurons based on random sampled inputs."""

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import drosophila_torch
from drosophila_torch.neurons.superclass import NeuronSuperclass
from drosophila_torch.simulation.lif_network import LIFConfig, DrosophilaLIF, STDPRule
from drosophila_torch.simulation.simulation import simulate, superclass_mask


drosophila_torch.enable_logging()


def plot_simulation_(external_voltage: torch.Tensor, spikes: torch.Tensor, voltages: torch.Tensor,
                     superclass_ids: torch.Tensor, config: LIFConfig) -> None:
    """Plot mean firing rate and mean membrane voltage over time.

    Args:
        external_voltage: External input voltage to selected neurons.
        spikes: Spike tensor over time.
        voltages: Voltage tensor over time.
        superclass_ids: The superclass ids of simulation neurons.
        config: The LIF config used for simulation."""

    T = spikes.shape[0]
    time_ms = torch.arange(T).float() * config.dt

    # General activity metrics
    external_mask = (external_voltage > 0).any(dim=0)  # Only select neurons which have been externally stimulated
    mean_input_voltage = external_voltage[:, external_mask].mean(dim=1).cpu()
    mean_spike_rate = (spikes.mean(dim=1) / (config.dt * 1e-3)).cpu()
    mean_internal_voltage = voltages.mean(dim=1).cpu()
    max_internal_voltage = voltages.max(dim=1).values.cpu()
    active_neuron_count = (spikes > 0).sum(dim=1).cpu()

    # Per-region activity metrics
    cb_mask = superclass_mask(superclass_ids, NeuronSuperclass.central_brain())
    mean_cb_spike_rate = spikes[:, cb_mask].mean(dim=1).cpu()
    ol_mask = superclass_mask(superclass_ids, NeuronSuperclass.optic_lobe())
    mean_ol_spike_rate = spikes[:, ol_mask].mean(dim=1).cpu()
    vnc_mask = superclass_mask(superclass_ids, NeuronSuperclass.ventral_nerve_cord())
    mean_vnc_spike_rate = spikes[:, vnc_mask].mean(dim=1).cpu()
    sensory_mask = superclass_mask(superclass_ids, NeuronSuperclass.sensory())
    mean_sensory_spike_rate = spikes[:, sensory_mask].mean(dim=1).cpu()
    motor_mask = superclass_mask(superclass_ids, NeuronSuperclass.motor())
    mean_motor_spike_rate = spikes[:, motor_mask].mean(dim=1).cpu()

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 5, hspace=0.6, wspace=0.3)

    # Input voltage
    ax_input_volt = fig.add_subplot(gs[0, :])
    ax_input_volt.plot(time_ms, mean_input_voltage, linewidth=0.8, color="green")
    ax_input_volt.set_ylabel("Mean input voltage (mV)")
    ax_input_volt.set_xlabel("Time (ms)")
    ax_input_volt.set_title("Input neuron mean external voltage")
    ax_input_volt.set_xlim(0, time_ms[-1].item())

    # Spike rate
    ax_rate = fig.add_subplot(gs[1, :])
    plot_rate, = ax_rate.plot(time_ms, mean_spike_rate, linewidth=0.8, color="steelblue", label="Spike rate")
    ax_rate.set_ylabel("Mean firing rate (Hz)")
    ax_rate.set_xlabel("Time (ms)")
    ax_rate.set_title("Neuron Activity")
    ax_rate.set_xlim(0, time_ms[-1].item())

    # Active neuron count
    ax_active = ax_rate.twinx()
    plot_active, = ax_active.plot(time_ms, active_neuron_count, linewidth=0.8, color="mediumpurple", label="Active neurons")
    ax_active.set_ylabel("Active neurons")

    ax_rate.legend(handles=[plot_rate, plot_active], fontsize=8)

    # Membrane voltage
    ax_internal_volt = fig.add_subplot(gs[2, :])
    ax_internal_volt.plot(time_ms, mean_internal_voltage, linewidth=0.8, color="darkorange")
    ax_internal_volt.plot(time_ms, max_internal_voltage, linestyle="--", linewidth=0.8, color="darkorange")
    ax_internal_volt.axhline(config.v_thresh, linestyle="--", linewidth=0.8,
                             color="red", label=f"Threshold ({config.v_thresh} mV)")
    ax_internal_volt.axhline(config.v_rest, linestyle="--", linewidth=0.8,
                             color="gray", label=f"Rest ({config.v_rest} mV)")
    ax_internal_volt.set_ylabel("Mean membrane voltage (mV)")
    ax_internal_volt.set_xlabel("Time (ms)")
    ax_internal_volt.set_title("Population mean membrane voltage")
    ax_internal_volt.set_xlim(0, time_ms[-1].item())
    ax_internal_volt.legend(fontsize=8)

    # CB spike rate
    ax_cb_rate = fig.add_subplot(gs[3, 0])
    ax_cb_rate.plot(time_ms, mean_cb_spike_rate, linewidth=0.8, color="steelblue")
    ax_cb_rate.set_ylabel("Mean firing rate (Hz)")
    ax_cb_rate.set_xlabel("Time (ms)")
    ax_cb_rate.set_title("Central brain mean firing rate")
    ax_cb_rate.set_xlim(0, time_ms[-1].item())

    # OL spike rate
    ax_ol_rate = fig.add_subplot(gs[3, 1])
    ax_ol_rate.plot(time_ms, mean_ol_spike_rate, linewidth=0.8, color="steelblue")
    ax_ol_rate.set_ylabel("Mean firing rate (Hz)")
    ax_ol_rate.set_xlabel("Time (ms)")
    ax_ol_rate.set_title("Optic lobe mean firing rate")
    ax_ol_rate.set_xlim(0, time_ms[-1].item())

    # VNC spike rate
    ax_vnc_rate = fig.add_subplot(gs[3, 2])
    ax_vnc_rate.plot(time_ms, mean_vnc_spike_rate, linewidth=0.8, color="steelblue")
    ax_vnc_rate.set_ylabel("Mean firing rate (Hz)")
    ax_vnc_rate.set_xlabel("Time (ms)")
    ax_vnc_rate.set_title("Ventral nerve cord mean firing rate")
    ax_vnc_rate.set_xlim(0, time_ms[-1].item())

    # Sensory spike rate
    ax_sensory_rate = fig.add_subplot(gs[3, 3])
    ax_sensory_rate.plot(time_ms, mean_sensory_spike_rate, linewidth=0.8, color="steelblue")
    ax_sensory_rate.set_ylabel("Mean firing rate (Hz)")
    ax_sensory_rate.set_xlabel("Time (ms)")
    ax_sensory_rate.set_title("Sensory neuron mean firing rate")
    ax_sensory_rate.set_xlim(0, time_ms[-1].item())

    # Motor spike rate
    ax_motor_rate = fig.add_subplot(gs[3, 4])
    ax_motor_rate.plot(time_ms, mean_motor_spike_rate, linewidth=0.8, color="steelblue")
    ax_motor_rate.set_ylabel("Mean firing rate (Hz)")
    ax_motor_rate.set_xlabel("Time (ms)")
    ax_motor_rate.set_title("Motor neuron mean firing rate")
    ax_motor_rate.set_xlim(0, time_ms[-1].item())

    plt.show()
    plt.close(fig)


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
    parser.add_argument("--amplitude", type=float, default=5.0,
                        help="Poisson input voltage for sensory neurons in mV (default: 5)")
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
        axon_pre=data["axon_pre_idx"].to(device),
        axon_post=data["axon_post_idx"].to(device),
        axon_delay=data["axon_delay_vec"].to(device),
        sign_vec=data["sign_vec"].to(device),
        nt_vec=data["nt_vec"].to(device),
        config=config,
    ).to(device)

    print(f"Neurons: {model.neuron_count():,}  Synapses: {model.synapse_count():,}")

    stdp = STDPRule(
        n_neurons=model.mem_voltage.shape[0],
        dt=config.dt,
        device=device,
    ) if not args.frozen else None

    if args.superclass:
        superclasses = [NeuronSuperclass.from_string(s) for s in args.superclass]
    else:
        superclasses = NeuronSuperclass.sensory()

    superclass_ids = data["superclass_ids"]
    s_mask = superclass_mask(superclass_ids, superclasses).to(device)

    pbar = tqdm(desc="Simulation time", total=args.duration, unit="ms")
    on_step = lambda: pbar.update(config.dt)
    external_voltage, spikes, voltages = simulate(model, args.duration, args.rate, args.amplitude, stdp=stdp,
                                                  input_mask=s_mask, config=config, device=device, on_step=on_step)

    plot_simulation_(external_voltage, spikes, voltages, superclass_ids, config)

    rates = _mean_firing_rate(spikes, config.dt)
    print(
        f"\nMean membrane voltage: {voltages.mean().item():.2f} mV  "
        f"Max: {voltages.max().item():.2f} mV  "
        f"\nMean firing rate: {rates.mean().item():.2f} Hz  "
        f"Max: {rates.max().item():.2f} Hz  "
        f"\nActive neurons: {(rates > 0).sum().item():,}"
    )

    if args.out:
        print(f"Saving results to {args.out}")
        torch.save({"spikes": spikes, "voltages": voltages}, args.out)
        print(f"Saved {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
