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
        self._energy_maps = [None]*len(self.kpoints)
        self._eigenvals = list(self._out.eigenvalues.values())[0]
        self._lu_energies = self._find_lu_energies()
        self._ho_energies = self._find_ho_energies()
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

    @property
    def ho_energies(self):
        return self._ho_energies

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

    def _find_ho_energies(self):
        """

        :return:
        """
        ho_energies = []

        for kpoint in self.eigenvals:
            i = 0
            while kpoint[i, 1] == 1:
                i += 1
            ho_energies.append(kpoint[i - 1, 0])

        return ho_energies

    def reconstruct_bz_kpoints(self, tol=1e-2):
        """
        Reconstruct all the kpoints in the 1st Brillouin zone from the
        irreducible points chosen by VASP.

        :return:
        """
        neighbors = set_up_neighbors(self._out.lattice)

        kpts_bz = []

        for k in self.kpoints:
            kpts_bz.append(return_to_brillouin(k, self._out.lattice,
                                               neighbors=neighbors,
                                               cartesian=False))

        all_kpts = np.array(kpts_bz[0])
        all_lu_energies = np.array([self.lu_energies[0], ])

        sg = SpacegroupAnalyzer(self._out.structures[-1])
        symmops = sg.get_point_group_operations(cartesian=False)

        # Reconstruct all the kpoints with the corresponding energies
        for k, energy in zip(kpts_bz[1:], self.lu_energies[1:]):

            # Get all symmetry equivalent kpoints of the kpoint in the IBZ
            sym_kpoints = np.dot(k, [m.rotation_matrix for m in symmops])

            add_list = [sym_kpoints[0], ]

            for point in list(sym_kpoints[1:]):

                if not any(np.linalg.norm(x - point) < 1e-6 for x in
                           add_list):
                    add_list.append(point)

            # Add the list of kpoints to the total array
            all_kpts = np.vstack([all_kpts, add_list])

            # Also add the energy value of the corresponding kpoint #sympoints
            # times
            all_lu_energies = np.vstack(
                [all_lu_energies, np.ones([len(add_list), 1]) * energy]
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

    def find_energy_map(self, kpoint, cartesian=False):
        """
        Find the minimum energy map for transitions to the lowest unoccupied
        states starting from a chosen initial kpoint in the IBZ.

        Args:
            kpoint: Initial kpoint of the transition.

        Returns:

        """
        if self._all_kpoints is None:
            self.reconstruct_bz_kpoints()

        print("setting up index")

        kpoint_index = [i for i, x in enumerate(list(self.kpoints)) if
                        np.linalg.norm(x - np.array(kpoint)) < 1e-4]

        if len(kpoint_index) == 1:
            kpoint_index = kpoint_index[0]
        else:
            raise ValueError("Kpoint not found in list of irreducible "
                             "kpoints.")

        if self._energy_maps[kpoint_index]:

            return self._energy_maps[kpoint_index]

        else:

            print("setting up neighbors")

            neighbors = set_up_neighbors(self._out.lattice)

            ho_energy = self.ho_energies[kpoint_index]

            print("setting up q vectors")

            q_vectors = self._all_kpoints - np.vstack(
                len(self._all_kpoints) * [kpoint])
            lu_energies = self._all_lu_energies - ho_energy

            q_vectors_c = np.dot(q_vectors, self._out.lattice_rec.matrix)

            q_vectors_bz_c = []

            print("mapping q vectors to 1st BZ")

            for i, q in enumerate(q_vectors_c):
                try:
                    new_q = return_to_brillouin(q, self._out.lattice,
                                                neighbors=neighbors,
                                                cartesian=True)
                    q_vectors_bz_c.append(new_q)
                except ValueError:
                    print("Found a kpoint that could not be returned to 1st BZ, "
                          "ignoring...")
                    np.delete(lu_energies, i, axis=0)

            print("done!")

            print("Calculating coordinates in fractional coords.")

            q_vectors_bz = np.dot(q_vectors_bz_c, np.linalg.inv(
                self._out.lattice_rec.matrix))

            q_110 = q_vectors_bz[0]
            lu_110 = np.array([lu_energies[0], ])

            print("extracting q vectors in 110 plane.")

            # Extract the kpoints in the 110 plane
            for k, energy in zip(q_vectors_bz[1:], lu_energies[1:]):
                if abs(k[2]) < 1e-5:
                    q_110 = np.vstack([q_110, k])
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
            q_110_v = np.dot(np.dot(q_110, self._out.lattice_rec.matrix),
                             np.linalg.inv(v_mat))

            # Interpolate
            x, y = np.mgrid[-2:2:0.02, -2:2:0.02]

            energies = interpolate.griddata(q_110_v[:, 0:2], lu_110, (x, y),
                                            method='linear')

            self._energy_maps[kpoint_index] = (x, y, energies)

            return (x, y, energies)

    def plot_map(self, kpoint, cartesian=False):
        """
        Plot the lowest energy transition map for each q vector in the 1st
        Brillouin zone of a given initial state corresponding the kpoint.

        Currently: Only possible for 110 map.

        Args:
            kpoint:
            cartesian:

        Returns:

        """

        x, y, energies = self.find_energy_map(kpoint, cartesian)

        # Plot
        plt.pcolor(x, y, energies[:, :, 0], vmin=np.nanmin(energies),
                   vmax=np.nanmax(energies),
                   cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('Energy (eV)', size='x-large')

        plt.show()

    def plot_min_map(self, indices, cartesian=False):
        """
        Plot the lowest energy transition map for each q vector in the 1st
        Brillouin zone of a given initial state corresponding the range of
        kpoint indices.

        Currently: Only possible for 110 map.

        :param kpoint:
        :return:
        """
        bz = set_up_brillouin(self._out.lattice)

        energy_maps = []
        x, y = np.mgrid[-2:2:0.02, -2:2:0.02]

        for kpoint_index in indices:

            kpoint = self.kpoints[kpoint_index]

            ho_energy = self.ho_energies[kpoint_index]

            q_vectors = self._all_kpoints - np.vstack(
                len(self._all_kpoints) * [kpoint])
            lu_energies = self._all_lu_energies - ho_energy

            q_vectors_c = np.dot(q_vectors, self._out.lattice_rec.matrix)

            q_vectors_bz_c = []

            for i, q in enumerate(q_vectors_c):
                try:
                    new_q = return_to_brillouin(q, bz, cartesian=True)
                    q_vectors_bz_c.append(new_q)
                except ValueError:
                    print(
                        "Found a kpoint that could not be returned to 1st BZ, "
                        "ignoring...")
                    np.delete(lu_energies, i, axis=0)

            q_vectors_bz = np.dot(q_vectors_bz_c, np.linalg.inv(
                self._out.lattice_rec.matrix))

            q_110 = q_vectors_bz[0]
            lu_110 = np.array([lu_energies[0], ])

            # Extract the kpoints in the 110 plane
            for k, energy in zip(q_vectors_bz[1:], lu_energies[1:]):
                if abs(k[2]) < 1e-5:
                    q_110 = np.vstack([q_110, k])
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
            q_110_v = np.dot(np.dot(q_110, self._out.lattice_rec.matrix),
                             np.linalg.inv(v_mat))

            # Interpolate
            e = interpolate.griddata(q_110_v[:, 0:2], lu_110, (x, y),
                                     method='linear')

            energy_maps.append(e)

        energy_maps = np.array(energy_maps)

        min_energy_map = np.nanmin(energy_maps, axis=0)

        # Plot
        plt.pcolor(x, y, min_energy_map[:, :, 0],
                   vmin=np.nanmin(min_energy_map),
                   vmax=10.5,
                   cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('Energy (eV)', size='x-large')

        plt.show()

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

def set_up_neighbors(lattice):
    """

    :param lattice:
    :return:
    """
    bz_facet_coords = lattice.get_brillouin_zone()
    neighbors = []

    for facet in bz_facet_coords:
        # The coordinates are transferred into sites to be able to use the
        # Facet object more easily.
        neighbors.append(np.mean(facet, axis=0)*2)

    return neighbors

def is_in_brillouin(kpoint, lattice, neighbors=None, cartesian=True):
    """

    :param kpoint:
    :param brillouin:
    :return:
    """
    if not cartesian:
        kpoint = np.dot(kpoint, lattice.reciprocal_lattice.matrix)

    in_brillouin = True

    # Get all combinations of the reciprocal lattice vectors and the zero
    # vector, up to the maximum number of combination length
    if neighbors is None:
        neighbors = set_up_neighbors(lattice)

    closest_point = np.array([0, 0, 0])
    dist = np.linalg.norm(kpoint)

    for point in neighbors:

        if np.linalg.norm(point - kpoint) < dist:

            in_brillouin = False
            dist = np.linalg.norm(point - kpoint)
            closest_point = point

    return in_brillouin, closest_point


def return_to_brillouin(kpoint, lattice, neighbors=None, cartesian=True):
    """
    Return a kpoint to the first Brillouin zone.

    Currently only works for points that are not too far away from the 1st BZ!

    :param kpoint: Reciprocal coordinates of the k-point
    :param lattice: Lattice

    :return:
    """
    in_brillouin, point = is_in_brillouin(kpoint, lattice, cartesian=cartesian)

    if in_brillouin:
        return kpoint
    else:
        if not cartesian:
            kpoint = np.dot(kpoint, lattice.reciprocal_lattice.matrix)

        new_kpoint = kpoint - point

        if not cartesian:
            new_kpoint = np.dot(new_kpoint, np.linalg.inv(
                lattice.reciprocal_lattice.matrix))

        if is_in_brillouin(new_kpoint, lattice, neighbors, cartesian):
            return new_kpoint
        else:
            raise ValueError("Could not find corresponding kpoint in 1st BZ.")
