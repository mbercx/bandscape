# Encoding UTF-8

import pdb

import numpy as np
import itertools as iter
import matplotlib.pyplot as plt

from math import pi

from scipy import interpolate

from cage.core import Facet
from pymatgen.core import PeriodicSite
from pymatgen.util.coord import pbc_diff
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.outputs import Vasprun


class BandScape(object):

    def __init__(self, vasprun):
        """
        Initialize the BandScape from a pymatgen.io.vasp.outputs.Vasprun
        instance.

        :param vasprun:
        """

        self._out = vasprun
        self._kpoints = self._out.actual_kpoints
        self._eigenvals = list(self._out.eigenvalues.values())[0]
        self._lu_energies = self._find_lu_energies()
        self._all_kpoints = None
        self._all_lu_energies = None

    @property
    def kpoints(self):
        return self._kpoints

    @property
    def eigenvals(self):
        return self._eigenvals

    @property
    def lu_energies(self):
        return self._lu_energies

    def _find_lu_energies(self):
        """

        :return:
        """
        lu_energies = []

        for kpoint in self.eigenvals:
            i = 0
            while kpoint[i, 1] == 1:
                i += 1
            lu_energies.append(kpoint[i, 0])

        return lu_energies

    def reconstruct_bz_kpoints(self, tol=1e-2):
        """
        Reconstruct all the kpoints in the 1st Brillouin zone from the
        irreducible points chosen by VASP.

        :return:
        """
        bz = set_up_brillouin(self._out.lattice)

        kpts_bz = []

        for k in self.kpoints:
            kpts_bz.append(return_to_brillouin(k, bz, cartesian=False))

        all_kpts = np.array(kpts_bz[0])
        all_lu_energies = np.array([self.lu_energies[0], ])

        sg = SpacegroupAnalyzer(self._out.structures[-1])
        symmops = sg.get_point_group_operations(cartesian=False)

        print("spaceman!")

        # Reconstruct all the kpoints with the corresponding energies
        for k, energy in zip(kpts_bz[1:], self.lu_energies[1:]):

            # Get all symmetry equivalent kpoints of the kpoint in the IBZ
            sym_kpoints = np.dot(k, [m.rotation_matrix for m in symmops])

            rm_list = []

            # Identify and remove duplicates from the list of equivalent
            # k-points:
            for i in range(len(sym_kpoints) - 1):
                for j in range(i + 1, len(sym_kpoints)):
                    if np.allclose(pbc_diff(sym_kpoints[i], sym_kpoints[j]), [0, 0, 0],
                                   tol):
                        rm_list.append(i)
                        break
            sym_kpoints = np.delete(sym_kpoints, rm_list, axis=0)

            # Add the list of kpoints to the total array
            all_kpts = np.vstack([all_kpts, sym_kpoints])

            # Also add the energy value of the corresponding kpoint #sympoints
            # times
            all_lu_energies = np.vstack(
                [all_lu_energies, np.ones([len(sym_kpoints), 1]) * energy]
            )

        self._all_kpoints = all_kpts
        self._all_lu_energies = all_lu_energies

    @classmethod
    def from_file(cls, filename):
        """
        Initialize the BandScape from a vasprun.xml file.

        :param vasprun_file:
        :return:
        """
        return cls(Vasprun(filename))

    def plot_map(self, kpoint, cartesian=False):
        """
        Plot the lowest energy transition map for each q vector in the 1st
        Brillouin zone of a given initial state corresponding the kpoint.

        Currently: Only possible for 110 map.

        :param kpoint:
        :return:
        """


        kpts_110 = self._all_kpoints[0]
        lu_110 = np.array([self._all_lu_energies[0], ])

        # Extract the kpoints in the 110 plane
        for k, energy in zip(self._all_kpoints[1:], self._all_lu_energies[1:]):
            if abs(k[2]) < 1e-5:
                kpts_110 = np.vstack([kpts_110, k])
                lu_110 = np.vstack([lu_110, energy])

        # Set up a new axis system to find suitable coordinates
        b1 = self._out.lattice_rec.matrix[0, :]
        b2 = self._out.lattice_rec.matrix[1, :]

        vx = b1 - b2
        vx = vx / np.linalg.norm(vx)
        vy = b1 + b2
        vy = vy / np.linalg.norm(vy)
        vz = np.cross(vx, vy)

        v_mat = np.vstack([vx, vy, vz])

        # Find the coordinates of the kpoints in this new axis system
        kpts_110_v = np.dot(np.dot(kpts_110, self._out.lattice_rec.matrix),
                            np.linalg.inv(v_mat))

        # Interpolate
        x, y = np.mgrid[np.min(kpts_110_v[:, 0]):np.max(kpts_110_v[:, 0]):0.01,
               np.min(kpts_110_v[:, 1]):np.max(kpts_110_v[:, 1]):0.01]

        e = interpolate.griddata(kpts_110_v[:, 0:2], lu_110, (x, y),
                                 method='cubic')

        # Plot
        plt.pcolor(x, y, e[:, :, 0], vmin=np.nanmin(e), vmax=np.nanmax(e),
                   cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('Energy (eV)', size='x-large')

        plt.show()


def plot_map(vasprun_file):
    out = Vasprun(vasprun_file)

    kpoints = out.actual_kpoints
    eigenvals = list(out.eigenvalues.values())[0]

    # Find the energies of the lowest unoccupied states

    lu_energies = []

    for kpoint in eigenvals:
        i = 1
        while kpoint[i, 1] == 1:
            i += 1
        lu_energies.append(kpoint[i, 0])

    bz = set_up_brillouin(out.lattice)
    kpts_bz = []

    for k in kpoints:
        kpts_bz.append(return_to_brillouin(k, bz, cartesian=False))

    print("Returned all VASP kpoints to the 1st BZ.")

    all_kpts = np.array(kpts_bz[0])
    all_lu_energies = np.array([lu_energies[0], ])

    bs = out.get_band_structure()

    # Substract the maximum valence energy
    lu_energies -= bs.get_vbm()["energy"]

    # Reconstruct all the kpoints with the corresponding energies
    for k, energy in zip(kpts_bz[1:], lu_energies[1:]):
        # Get all symmetry equivalent kpoints of the kpoint in the IBZ
        sym_kpoints = bs.get_sym_eq_kpoints(k)

        # Add the list of kpoints to the total array
        all_kpts = np.vstack([all_kpts, sym_kpoints])

        # Also add the energy value of the corresponding kpoint #sympoints
        # times
        all_lu_energies = np.vstack(
            [all_lu_energies, np.ones([len(sym_kpoints), 1]) * energy]
        )

    print("Reconstructed complete 1st BZ.")

    kpts_110 = all_kpts[0]
    lu_110 = np.array([all_lu_energies[0], ])

    # Extract the kpoints in the 110 plane
    for k, energy in zip(all_kpts[1:], all_lu_energies[1:]):
        if abs(k[2]) < 1e-5:
            kpts_110 = np.vstack([kpts_110, k])
            lu_110 = np.vstack([lu_110, energy])

    # Set up a new axis system to find suitable coordinates
    b1 = out.lattice_rec.matrix[0, :]
    b2 = out.lattice_rec.matrix[1, :]

    vx = b1 - b2
    vx = vx / np.linalg.norm(vx)
    vy = b1 + b2
    vy = vy / np.linalg.norm(vy)
    vz = np.cross(vx, vy)

    v_mat = np.vstack([vx, vy, vz])

    # Find the coordinates of the kpoints in this new axis system
    kpts_110_v = np.dot(np.dot(kpts_110, out.lattice_rec.matrix),
                        np.linalg.inv(v_mat))

    # Interpolate
    x, y = np.mgrid[np.min(kpts_110_v[:, 0]):np.max(kpts_110_v[:, 0]):0.01,
           np.min(kpts_110_v[:, 1]):np.max(kpts_110_v[:, 1]):0.01]

    e = interpolate.griddata(kpts_110_v[:, 0:2], lu_110, (x, y),
                             method='cubic')

    # Plot
    plt.pcolor(x, y, e[:, :, 0], vmin=np.nanmin(e), vmax=np.nanmax(e),
               cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label('Energy (eV)', size='x-large')

    plt.show()

    return None


def set_up_brillouin(lattice):
    """

    :param lattice:
    :return:
    """
    rec_lattice = lattice.reciprocal_lattice
    bz_facet_coords = lattice.get_brillouin_zone()
    bz_facet_sites = []

    for facet in bz_facet_coords:
        # The coordinates are transferred into sites to be able to use the
        # Facet object more easily.
        bz_facet_sites.append(
            [PeriodicSite("H", coords, rec_lattice, coords_are_cartesian=True)
             for coords in facet]
        )

    bz_facets = [Facet(facet_sites) for facet_sites in bz_facet_sites]

    return bz_facets

def is_in_brillouin(kpoint, brillouin, cartesian=True, tol=1e-3):
    """

    :param kpoint:
    :param brillouin:
    :return:
    """
    if not cartesian:
        rec_lattice = brillouin[0][0].lattice
        kpoint = np.dot(kpoint, rec_lattice.matrix)

    in_brillouin = True

    facet_number = 0

    while in_brillouin and facet_number < len(brillouin):

        if brillouin[facet_number].angle_to_normal(kpoint) < pi/2 - tol:
            in_brillouin = False

        facet_number += 1

    return in_brillouin


def return_to_brillouin(kpoint, brillouin, cartesian=True,
                        max_combinations=2):
    """
    Return a kpoint to the first Brillouin zone.

    :param kpoint: Reciprocal coordinates of the k-point
    :param lattice: Lattice

    :return:
    """
    if is_in_brillouin(kpoint, brillouin, cartesian=cartesian):
        return kpoint
    else:
        rec_lattice = brillouin[0][0].lattice

        if not cartesian:
            kpoint = np.dot(kpoint, rec_lattice.matrix)

        # Get all combinations of the reciprocal lattice vectors and the zero
        # vector, up to the maximum number of combination length
        combinations = iter.combinations_with_replacement(
            list(np.vstack([np.array([0, 0, 0]), rec_lattice.matrix,
                            - rec_lattice.matrix])), max_combinations
        )

        # Add the vectors together for each combination of reciprocal lattice
        # vectors
        test_vectors = [sum(combination) for combination in combinations]

        vector_number = 1 # Ignore the first test vector, i.e. the zero vector
        # TODO remove all duplicate vectors

        new_kpoint = None
        in_brillouin = False

        while not in_brillouin and vector_number < len(test_vectors):

            new_kpoint = kpoint + test_vectors[vector_number]
            in_brillouin = is_in_brillouin(new_kpoint, brillouin)

            vector_number += 1

        if not cartesian:
            new_kpoint = np.dot(new_kpoint, np.linalg.inv(rec_lattice.matrix))

        if in_brillouin:
            return new_kpoint
        else:
            raise ValueError("Could not find corresponding kpoint in 1st BZ.")




