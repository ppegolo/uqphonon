# uqphonon

Minimal Python library for computing phonon band structures with uncertainty quantification (UQ) from machine learning interatomic potentials (MLIPs).

**Force backends:**
- an ASE calculator with a `get_forces_ensemble(atoms, non_conservative)` method, such as [`upet`](https://github.com/lab-cosmo/upet)'s `UPETCalculator`. The ensemble comes either from gradients of an energy ensemble (`non_conservative=False`) or from the direct-force head (`non_conservative=True`).
- i-PI with [metatomic](https://github.com/lab-cosmo/metatomic) models that predict ensembles, i.e. multiple energy/force predictions per snapshot.

## Installation

```bash
pip install -e .          # i-PI backend
pip install -e ".[upet]"  # with upet for the calculator backend
```

## Quick Start

```python
from uqphonon import PhononEnsemble
from ase.build import bulk

# Pre-relaxed primitive cell
atoms = bulk("BeO", crystalstructure="wurtzite")

# Initialize with supercell matrix and a calculator returning force ensembles
from upet.calculator import UPETCalculator

calc = UPETCalculator(model="pet-mad-s", version="1.6.0", device="cpu")
ph = PhononEnsemble(
    atoms,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    calculator=calc,
    non_conservative=True,  # direct-force ensemble, centred on conservative forces
)
# or, with the i-PI backend:
# ph = PhononEnsemble(atoms, supercell_matrix=..., model="path/to/model.pt")

# Step 1: Generate displaced supercells
ph.compute_displacements(distance=0.03)

# Step 2: Get ensemble forces (i-PI backend: ph.run_forces(workdir="./results"))
ph.run_forces()

# Step 3: Compute band structures
ph.compute_bands()  # auto-detect path via seekpath
# or: ph.compute_bands("GMKGA")  # explicit path

# Step 4: Plot with UQ
fig, ax = ph.plot(mode="mean+std", cmap="viridis")
fig.savefig("bands.pdf")
```

## Plot Modes

- `"mean"` – mean band structure only
- `"mean+std"` – mean ± std as shaded region
- `"ensemble"` – all ensemble members + mean

## Requirements

- ASE
- phonopy
- numpy
- matplotlib
- upet (calculator backend) or i-PI and metatomic (i-PI backend)
