"""i-PI force evaluation for ensemble committee models.

This module handles:
- Building the i-PI input XML for a metatomic committee model in replay mode
- Running the simulation via i-PI's scripting interface
- Parsing the committee force output into a numpy array
"""

from __future__ import annotations

from pathlib import Path

import ase
import ase.io
import ase.units
import numpy as np

_IPI_OUTPUT_PREFIX = "ipi_forces"


def _make_forcefield_xml(model: str, device: str, template_path: str) -> str:
    params = {
        "template": template_path,
        "model": model,
        "device": device,
        "force_virial_ensemble": True,
    }
    from ipi.utils.scripting import forcefield_xml

    return forcefield_xml(
        name="metatomic", mode="direct", pes="metatomic", parameters=params
    )


def _make_output_xml(workdir: Path, prefix: str) -> str:
    return f"""<output prefix="{workdir}/{prefix}">
<properties stride="1" filename="out">
    [step, cell_abcABC]
</properties>
<trajectory filename="committee_pot" stride="1" extra_type="committee_pot">extras</trajectory>
<trajectory filename="committee_force" stride="1" extra_type="committee_force">extras</trajectory>
<checkpoint stride="1"/>
</output>"""


def _make_simulation_xml(
    supercell: ase.Atoms,
    displacements_path: str,
    model: str,
    device: str,
    workdir: Path,
    prefix: str = _IPI_OUTPUT_PREFIX,
) -> str:
    from ipi.utils.scripting import simulation_xml

    template_path = str(workdir / "relaxed_supercell.xyz")
    forcefield = _make_forcefield_xml(model, device, template_path)
    output = _make_output_xml(workdir, prefix)
    motion = f'<motion mode="replay">\n<file mode="ase"> {displacements_path} </file>\n</motion>'

    return simulation_xml(
        structures=supercell,
        forcefield=forcefield,
        motion=motion,
        prefix=prefix,
        output=output,
    )


def _parse_committee_forces(
    force_file: Path, n_disp: int, n_atoms: int
) -> np.ndarray:
    """Parse the i-PI committee force trajectory file.

    The file has one row per simulation step (i.e. one displaced supercell),
    and each row contains the flattened forces for all ensemble members:
    shape of a row = (n_ensemble * n_atoms * 3,).

    Returns
    -------
    forces : ndarray, shape (n_disp, n_ensemble, n_atoms, 3)
        Forces in eV/Å.
    """
    raw = np.loadtxt(force_file)
    # Take only the last n_disp rows (replay mode may have an extra header step)
    raw = raw[-n_disp:]
    n_cols = raw.shape[1]
    if n_cols % (n_atoms * 3) != 0:
        raise ValueError(
            f"Committee force file has {n_cols} columns, which is not divisible by "
            f"n_atoms * 3 = {n_atoms * 3}. Cannot auto-detect ensemble size."
        )
    n_ensemble = n_cols // (n_atoms * 3)
    forces = raw.reshape(n_disp, n_ensemble, n_atoms, 3)
    # Convert from Hartree/Bohr to eV/Å
    forces = forces * (ase.units.Hartree / ase.units.Bohr)
    return forces


def run_ipi_forces(
    supercell: ase.Atoms,
    displacements: list[ase.Atoms],
    model: str,
    device: str,
    workdir: Path | str,
    prefix: str = _IPI_OUTPUT_PREFIX,
) -> np.ndarray:
    """Run i-PI with a metatomic committee model and return ensemble forces.

    Writes input files to `workdir`, runs i-PI in replay mode over the
    displaced supercells, and returns parsed forces.

    Parameters
    ----------
    supercell : ase.Atoms
        The equilibrium (undisplaced) supercell.
    displacements : list of ase.Atoms
        The displaced supercells (one per phonon displacement).
    model : str
        Path to the metatomic `.pt` model file.
    device : str
        Compute device, e.g. ``"cpu"`` or ``"cuda"``.
    workdir : Path or str
        Directory where i-PI writes its output files. Will be created if needed.
    prefix : str
        Prefix for i-PI output files.

    Returns
    -------
    forces : ndarray, shape (n_disp, n_ensemble, n_atoms, 3)
        Ensemble forces in eV/Å.
    """
    from ipi.utils.scripting import InteractiveSimulation

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    supercell_path = workdir / "relaxed_supercell.xyz"
    displacements_path = workdir / "supercells.xyz"

    ase.io.write(str(supercell_path), supercell)
    ase.io.write(str(displacements_path), displacements)

    xml = _make_simulation_xml(
        supercell=supercell,
        displacements_path=str(displacements_path),
        model=model,
        device=device,
        workdir=workdir,
        prefix=prefix,
    )

    sim = InteractiveSimulation(xml)
    sim.run(len(displacements))

    force_file = workdir / f"{prefix}.committee_force_0"
    n_atoms = len(supercell)
    return _parse_committee_forces(force_file, len(displacements), n_atoms)
