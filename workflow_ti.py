# %% Imports and configuration
# Run from the workspace root after: pip install -e petphonon/

import ase
import ase.io
from ase.build import bulk
import matplotlib.pyplot as plt
import numpy as np
import phono3py
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGSLineSearch
from matplotlib.lines import Line2D
from pathlib import Path
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from upet.calculator import UPETCalculator

from petphonon import PhononEnsemble

REFERENCE_DIR = Path("reference")
RESULTS_DIR = Path("results")
MODEL = "pet-mad-s-v1.0.2.pt"
DEVICE = "cuda"

# %% Load structures from phono3py reference files
N = 7
name = "Ti"
data = {
    "atoms": bulk("Ti", "hcp"),
    "supercell_matrix": [[N, 0, 0], [0, N, 0], [0, 0, N]],
    "bandpath": "GMKG|ALHA",
    "primitive_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
}

# %% (Optional) Relax structures with PET-MAD
# Skip this cell if your structures are already well-relaxed.


relax_calc = UPETCalculator(
    checkpoint_path=MODEL.replace(".pt", ".ckpt"), device=DEVICE
)

atoms = data["atoms"].copy()
atoms.calc = relax_calc
atoms.set_constraint(FixSymmetry(atoms))
opt = BFGSLineSearch(
    FrechetCellFilter(atoms, mask=[True] * 3 + [False] * 3), logfile="-"
)
opt.run(fmax=1e-6, steps=100)
atoms.calc = None
atoms.constraints = None
data["atoms"] = atoms
print(f"{name}: relaxed in {opt.nsteps} steps")

# %% Set up PhononEnsemble objects and generate displacements


ensemble = PhononEnsemble(
    data["atoms"],
    supercell_matrix=data["supercell_matrix"],
    model=MODEL,
    device=DEVICE,
    primitive_matrix=data["primitive_matrix"],
)
ensemble.compute_displacements(distance=0.03)

# %% Run i-PI to compute ensemble forces

workdir = RESULTS_DIR / name
print(f"Running i-PI for {name} ...")
ensemble.run_forces(workdir=workdir)
n_disp, n_ens, n_atoms, _ = ensemble.forces.shape
print(f"  → {n_ens} ensemble members, {n_disp} displacements, {n_atoms} atoms")

# %% Compute phonon band structures (ensemble + mean)

ensemble.compute_bands(bandpath=data["bandpath"])
print(f"{name}: bands computed")

# %% Plot: MLIP ensemble + DFT reference

THZ_TO_CM = 33.35641


fig, ax = plt.subplots()

ensemble.plot(ax=ax, mode="mean+std", unit="cm-1", color="tab:red")
ax.set_title(name)

ax.set_ylabel("Frequency (cm⁻¹)")
ax.set_ylabel("")

legend_handles = [
    Line2D([0], [0], color="tab:blue", ls="--", label="DFT"),
    Line2D([0], [0], color="tab:red", ls="-", label="PET-MAD (ensemble)"),
]
ax.legend(handles=legend_handles, loc="upper center", ncol=2)

fig.tight_layout()
fig.savefig("phonons.pdf", dpi=300, bbox_inches="tight")
fig.savefig("phonons.svg", dpi=300, bbox_inches="tight", transparent=True)
plt.show()

# %%
