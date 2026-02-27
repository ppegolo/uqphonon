"""i-PI force evaluation for ensemble committee models.

This module handles:
- Building the i-PI input XML for a metatomic committee model in replay mode
- Running the simulation via i-PI's scripting interface
- Parsing the committee force output into a numpy array
"""

from __future__ import annotations

import os
from pathlib import Path

import ase
import ase.io
import ase.units
import numpy as np

_IPI_OUTPUT_PREFIX = "ipi_forces"


def _rotate_to_upper_triangular(atoms: ase.Atoms) -> tuple[np.ndarray, ase.Atoms]:
    """Rotate atoms so its cell matrix is upper triangular (required by i-PI).

    i-PI requires h = cell.T to be upper triangular (h[1,0] = h[2,0] = h[2,1] = 0).
    Returns (Q, rotated_atoms) where Q is the rotation matrix such that
    new_positions = old_positions @ Q and new_cell = old_cell @ Q.
    Forces rotate the same way, so to invert: f_old = f_new @ Q.T.
    """
    cell = atoms.cell.array
    Q, R = np.linalg.qr(cell.T)
    # Ensure positive diagonal so the cell vectors point in canonical directions
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs  # Q @ diag(signs)
    R = (R.T * signs).T  # diag(signs) @ R

    rotated = atoms.copy()
    rotated.set_cell(R.T, scale_atoms=False)
    rotated.set_positions(atoms.get_positions() @ Q)
    return Q, rotated


def _apply_rotation(Q: np.ndarray, atoms: ase.Atoms) -> ase.Atoms:
    """Apply rotation Q to atoms positions and cell."""
    rotated = atoms.copy()
    rotated.set_cell(atoms.cell.array @ Q, scale_atoms=False)
    rotated.set_positions(atoms.get_positions() @ Q)
    return rotated


def _make_forcefield_xml(model: str, device: str, template_path: str) -> str:
    params = {
        "template": template_path,
        "model": model,
        "device": device,
        "energy_ensemble": True,
        "force_virial_ensemble": True,
    }
    from ipi.utils.scripting import forcefield_xml

    return forcefield_xml(
        name="metatomic", mode="direct", pes="metatomic", parameters=params
    )


def _make_output_xml(prefix: str) -> str:
    # Paths are relative to CWD at simulation time (i.e. workdir).
    return f"""<output prefix="{prefix}">
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
    template_path: str,
    prefix: str = _IPI_OUTPUT_PREFIX,
) -> str:
    from ipi.utils.scripting import simulation_xml

    forcefield = _make_forcefield_xml(model, device, template_path)
    output = _make_output_xml(prefix)
    motion = f'<motion mode="replay">\n<file mode="ase"> {displacements_path} </file>\n</motion>'

    return simulation_xml(
        structures=supercell,
        forcefield=forcefield,
        motion=motion,
        prefix=prefix,
        output=output,
    )


def _parse_committee_forces(force_file: Path, n_disp: int, n_atoms: int) -> np.ndarray:
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

    workdir = Path(workdir).resolve()
    model = str(Path(model).resolve())  # must be absolute before chdir
    workdir.mkdir(parents=True, exist_ok=True)

    supercell_path = workdir / "relaxed_supercell.xyz"
    displacements_path = workdir / "supercells.xyz"

    # i-PI requires the cell to be upper triangular. Rotate all structures,
    # then rotate the returned forces back to the original frame.
    Q, supercell_rot = _rotate_to_upper_triangular(supercell)
    displacements_rot = [_apply_rotation(Q, d) for d in displacements]

    ase.io.write(str(supercell_path), supercell_rot)
    ase.io.write(str(displacements_path), displacements_rot)

    # i-PI writes all output relative to CWD, so we run from inside workdir.
    # All paths passed into the XML must be absolute.
    xml = _make_simulation_xml(
        supercell=supercell_rot,
        displacements_path=str(displacements_path),
        model=model,
        device=device,
        template_path=str(supercell_path),
        prefix=prefix,
    )

    old_cwd = Path.cwd()
    try:
        os.chdir(workdir)
        sim = InteractiveSimulation(xml)
        sim.run(len(displacements))
    finally:
        os.chdir(old_cwd)

    force_file = workdir / f"{prefix}.committee_force_0"
    n_atoms = len(supercell)
    forces = _parse_committee_forces(force_file, len(displacements), n_atoms)

    # Clean up i-PI output files and our xyz inputs.
    for f in workdir.glob(f"{prefix}.*"):
        f.unlink()
    supercell_path.unlink(missing_ok=True)
    displacements_path.unlink(missing_ok=True)
    try:
        workdir.rmdir()  # removes the directory only if now empty
    except OSError:
        pass

    # Rotate forces back from the upper-triangular frame to the original frame.
    # Forces transform as vectors: f_orig = f_rot @ Q.T
    return forces @ Q.T
