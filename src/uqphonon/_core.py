"""Core PhononEnsemble class."""

from __future__ import annotations

from copy import copy
from pathlib import Path

import ase
import numpy as np
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms

from ._ipi import run_ipi_forces
from ._plot import plot_bands


def _ase_to_phonopy(atoms: ase.Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell().array,
    )


def _phonopy_to_ase(ph_atoms: PhonopyAtoms) -> ase.Atoms:
    return ase.Atoms(
        symbols=ph_atoms.symbols,
        positions=ph_atoms.positions,
        cell=ph_atoms.cell,
        pbc=True,
    )


def _parse_path_string(path_str: str) -> list[list[str]]:
    """Convert an ASE band path string to a list of segments.

    Handles three formats:
    - Compact user: 'GMKG|ALHA'  (| = segment break, each char = point)
    - Explicit user: 'G,M,K|G,A' (| = segment break, , = point separator)
    - ASE auto-detect: 'GXWK,GLUWLK' (, = segment break, each char = point)
    """
    if "|" in path_str:
        raw_segments = path_str.split("|")
        if "," in path_str:
            return [seg.split(",") for seg in raw_segments]
        return [list(seg) for seg in raw_segments]
    elif "," in path_str:
        # ASE format: comma separates disconnected segments
        return [list(seg) for seg in path_str.split(",")]
    else:
        return [list(path_str)]


def _make_labels(segments: list[list[str]]) -> list[str]:
    """Build phonopy-style label list, substituting G → Γ."""
    labels = []
    for seg in segments:
        for name in seg:
            labels.append("$\\Gamma$" if name == "G" else name)
    return labels


def _resolve_bandpath(
    prim_atoms: ase.Atoms,
    bandpath_spec,
    npoints: int,
) -> tuple:
    """Resolve a band path spec to (qpoints, connections, labels).

    Parameters
    ----------
    prim_atoms : ase.Atoms
        Primitive cell atoms (used for auto-detection).
    bandpath_spec : None, str, or ase.dft.kpoints.BandPath
        None  → auto-detect via ASE/seekpath
        str   → compact path like "MKGAL"
        BandPath → use as-is
    npoints : int
        Number of q-points per segment.
    """
    if bandpath_spec is None:
        bp = prim_atoms.cell.bandpath(npoints=npoints)
    elif isinstance(bandpath_spec, str):
        bp = prim_atoms.cell.bandpath(bandpath_spec, npoints=npoints)
    else:
        bp = bandpath_spec  # already ASE BandPath

    segments = _parse_path_string(bp.path)
    bpath = [[bp.special_points[p] for p in seg] for seg in segments]
    labels = _make_labels(segments)
    qpoints, connections = get_band_qpoints_and_path_connections(bpath, npoints=npoints)
    return qpoints, connections, labels


class PhononEnsemble:
    """Compute phonon band structures with uncertainty quantification.

    Uses i-PI with a metatomic committee model to obtain an ensemble of
    force predictions, then computes phonon bands for each ensemble member.

    Parameters
    ----------
    atoms : ase.Atoms
        Pre-relaxed primitive cell.
    supercell_matrix : array-like, shape (3, 3)
        Supercell matrix passed to phonopy (e.g. ``[[2,0,0],[0,2,0],[0,0,2]]``).
    model : str
        Path to the metatomic ``.pt`` committee model file.
    device : str
        Compute device for the model (``"cpu"`` or ``"cuda"``).
    primitive_matrix : array-like or "auto"
        Primitive matrix for phonopy. ``"auto"`` lets phonopy detect it.
    displacement_distance : float
        Default displacement distance in Å (used if not overridden in
        :meth:`compute_displacements`).

    Examples
    --------
    >>> ph = PhononEnsemble(atoms, [[2,0,0],[0,2,0],[0,0,2]], "pet-mad.pt")
    >>> ph.compute_displacements()
    >>> ph.run_forces("results/BeO")
    >>> ph.compute_bands()
    >>> fig, ax = ph.plot(mode="mean+std")
    """

    def __init__(
        self,
        atoms: ase.Atoms,
        supercell_matrix,
        model: str,
        device: str = "cpu",
        primitive_matrix="auto",
        displacement_distance: float = 0.03,
    ):
        self._atoms = atoms
        self._supercell_matrix = supercell_matrix
        self._model = model
        self._device = device
        self._primitive_matrix = primitive_matrix
        self._displacement_distance = displacement_distance

        # State set by compute_displacements
        self._phonon: Phonopy | None = None
        self._displaced_supercells: list[ase.Atoms] | None = None

        # State set by run_forces
        self._forces: np.ndarray | None = None  # (n_disp, n_ensemble, n_atoms, 3)

        # State set by compute_bands
        self._phonons: list[Phonopy] | None = None
        self._mean_phonon: Phonopy | None = None

    # ------------------------------------------------------------------
    # Step 1: displacements
    # ------------------------------------------------------------------

    def compute_displacements(self, distance: float | None = None) -> None:
        """Generate displaced supercells for the finite-difference force calculation.

        Sets :attr:`phonons` and internal displaced supercell list.

        Parameters
        ----------
        distance : float, optional
            Displacement distance in Å. Defaults to the value set at construction.
        """
        if distance is None:
            distance = self._displacement_distance

        phonopy_atoms = _ase_to_phonopy(self._atoms)
        phonon = Phonopy(
            phonopy_atoms,
            self._supercell_matrix,
            primitive_matrix=self._primitive_matrix,
        )
        phonon.generate_displacements(distance=distance)

        self._phonon = phonon
        self._displaced_supercells = [
            _phonopy_to_ase(sc) for sc in phonon.supercells_with_displacements
        ]

    # ------------------------------------------------------------------
    # Step 2: force evaluation
    # ------------------------------------------------------------------

    def run_forces(self, workdir: Path | str) -> None:
        """Run i-PI to compute ensemble forces on all displaced supercells.

        Writes input files to *workdir* and parses the committee force output.

        Parameters
        ----------
        workdir : Path or str
            Directory for i-PI files (created if needed, files are kept).
        """
        if self._phonon is None:
            raise RuntimeError("Call compute_displacements() first.")

        supercell = _phonopy_to_ase(self._phonon.supercell)
        self._forces = run_ipi_forces(
            supercell=supercell,
            displacements=self._displaced_supercells,
            model=self._model,
            device=self._device,
            workdir=workdir,
        )

    # ------------------------------------------------------------------
    # Step 3: band structure
    # ------------------------------------------------------------------

    def compute_bands(self, bandpath=None, npoints: int = 151) -> None:
        """Compute phonon band structures for each ensemble member.

        Parameters
        ----------
        bandpath : None, str, or ase.dft.kpoints.BandPath
            Band path specification:
            - ``None``: auto-detect from the primitive cell via seekpath.
            - ``str``: compact path string, e.g. ``"MKGAL"``.
            - ``BandPath``: an existing ASE BandPath object.
        npoints : int
            Number of q-points per path segment.
        """
        if self._forces is None:
            raise RuntimeError("Call run_forces() first.")

        # Build primitive-cell atoms for path detection
        prim_atoms = _phonopy_to_ase(self._phonon.primitive)

        qpoints, connections, labels = _resolve_bandpath(prim_atoms, bandpath, npoints)

        force_sets = self._forces  # (n_disp, n_ensemble, n_atoms, 3)
        n_ensemble = force_sets.shape[1]

        def _compute_single(forces_for_member):
            ph = copy(self._phonon)
            ph.forces = forces_for_member
            ph.produce_force_constants()
            ph.symmetrize_force_constants()
            ph.run_band_structure(qpoints, path_connections=connections, labels=labels)
            return ph

        self._phonons = [_compute_single(force_sets[:, i]) for i in range(n_ensemble)]

        # Mean phonon: average forces across ensemble members
        mean_forces = force_sets.mean(axis=1)  # (n_disp, n_atoms, 3)
        self._mean_phonon = _compute_single(mean_forces)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax=None,
        mode: str = "ensemble",
        unit: str = "cm-1",
        **kwargs,
    ):
        """Plot phonon bands with uncertainty quantification.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on. A new figure is created if not given.
        mode : str
            ``"mean"``      – mean band structure only.
            ``"mean+std"``  – mean ± std (shaded region).
            ``"ensemble"``  – all ensemble members + mean.
        unit : str
            Frequency unit: ``"THz"``, ``"cm-1"``, or ``"meV"``.
        **kwargs
            Extra keyword arguments forwarded to the plotting function
            (e.g. ``color``, ``ensemble_alpha``, ``std_alpha``).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        if self._phonons is None:
            raise RuntimeError("Call compute_bands() first.")

        return plot_bands(
            self._phonons,
            self._mean_phonon,
            mode=mode,
            unit=unit,
            ax=ax,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phonons(self) -> list[Phonopy]:
        """List of Phonopy objects, one per ensemble member."""
        if self._phonons is None:
            raise RuntimeError("Call compute_bands() first.")
        return self._phonons

    @property
    def mean_phonon(self) -> Phonopy:
        """Phonopy object computed from the mean ensemble forces."""
        if self._mean_phonon is None:
            raise RuntimeError("Call compute_bands() first.")
        return self._mean_phonon

    @property
    def forces(self) -> np.ndarray:
        """Ensemble forces, shape (n_displacements, n_ensemble, n_atoms, 3), in eV/Å."""
        if self._forces is None:
            raise RuntimeError("Call run_forces() first.")
        return self._forces
