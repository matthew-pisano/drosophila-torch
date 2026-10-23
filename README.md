# Drosophila Torch Simulation

`drosophila_torch` builds and simulates a pure PyTorch model of the *Drosophila* male central nervous system (MaleCNS) connectome. It utilizes the v1.0 connectome tables published by Janelia, converts neuron annotations, neurotransmitter identities, connections, and soma locations into tensors, and
uses those tensors to run a leaky integrate-and-fire (LIF) network simulation. The model also supports local spike-timing-dependent synaptic weight updates (STDP).

This repository contains the `drosophila_torch` package itself and a series of utility scripts for fetching, simulating, and visualizing connectome data.

## Requirements and setup

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Internet access to download the connectome source files when they are not already present locally

Clone the repository, then install its dependencies:

```bash
git clone https://github.com/matthew-pisano/drosophila-torch.git
cd drosophila-torch
uv sync
```

Run project commands from the repository root with `uv run`. Alternatively, install the project and its dependencies into your active Python environment:

```bash
python -m pip install -e .
```

The scripts can then be run with `python`. PyTorch uses CUDA when available for the simulation by default, otherwise it runs on the CPU; a device can also be selected explicitly.

## Usage

### Build connectome tensors

Download the MaleCNS data and convert it to a PyTorch tensor bundle:

```bash
uv run python scripts/fetch_connectome.py data
```

This writes `data/malecns_tensors.pt` and downloads the Feather source files to `data/feather/`. The three source files are fetched from Janelia's public MaleCNS v1.0 connectome-data bucket. Existing Feather files are reused.

You can restrict the generated bundle by annotation, or list available annotation values before choosing filters:

```bash
uv run python scripts/fetch_connectome.py data --list superclass
uv run python scripts/fetch_connectome.py data --list type
uv run python scripts/fetch_connectome.py data --superclass descending_neuron visual_projection
uv run python scripts/fetch_connectome.py data --type DNp01 DNp02
```

Multiple values within a filter are combined as alternatives; superclass and type filters are combined. Use `labeled` to retain neurons that have an annotation in that field. Filtering keeps connections only when both endpoint neurons survive.

### Visualize soma locations

Open an interactive 3D plot of neuron soma locations, colored by superclass:

```bash
uv run python scripts/visualize_connectome.py data/malecns_tensors.pt
```

To draw a small random sample of connections as well as neuron locations, set an edge sample proportion:

```bash
uv run python scripts/visualize_connectome.py data/malecns_tensors.pt --edge-sample 0.0001
```

The plot skips neurons without a valid soma location. Coordinates are in voxel units at 8 nm resolution. Use `--help` on the script for all options.

### Run a simulation

Run a 1-second simulation using sensory neurons as the default stimulated population, display activity plots, and print summary statistics:

```bash
uv run python scripts/simulate.py data/malecns_tensors.pt
```

Choose an input population, simulation duration, and device, and save spike and voltage records:

```bash
uv run python scripts/simulate.py data/malecns_tensors.pt \
  --superclass sensory_ascending cb_sensory ol_sensory vnc_sensory \
  --duration 500 \
  --device cpu \
  --out results/simulation.pt
```

The current simulation applies a constant external voltage to the selected neurons for the first 60% of the run. Synaptic weights are updated with STDP by default; pass `--frozen` to disable weight updates. The saved result contains `spikes` and `voltages` tensors indexed by time step and neuron. Use `--help` for all simulation options.

## Project layout

- `src/drosophila_torch/` contains the `drosophila_torch` package for simulating neural activity within the connectome.
- `scripts/` contains the command-line workflows for fetching data,
  visualization, and simulation.
- `data/` contains the generated tensor bundle and downloaded source data.
