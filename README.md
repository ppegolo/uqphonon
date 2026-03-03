# uqphonon

Minimal Python library for computing phonon band structures with uncertainty quantification (UQ) from machine learning interatomic potentials (MLIPs).

**Force backend:** i-PI with [metatomic](https://github.com/lab-cosmo/metatomic) models that can predict ensembles that return multiple energy/force predictions per snapshot, enabling UQ through ensemble variance.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from uqphonon import PhononEnsemble
from ase.build import bulk

# Pre-relaxed primitive cell
atoms = bulk("BeO", crystalstructure="wurtzite")

# Initialize with supercell matrix and MLIP model
ph = PhononEnsemble(
    atoms,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    model="path/to/model.pt",
    device="cpu", # or cuda, if available
)

# Step 1: Generate displaced supercells
ph.compute_displacements(distance=0.03)

# Step 2: Run i-PI to get ensemble forces
ph.run_forces(workdir="./results")

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
- i-PI (for force evaluation)
- metatomic (for MLIP models)
