# Encoding UTF-8

import pdb
import json

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import interpolate
from monty.io import zopen
from cage.core import Facet
from monty.json import MSONable
from pymatgen.core import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.outputs import Vasprun


class Bandscape(MSONable):
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
        self._ho_energies = self._find_ho_energies()
        self.reconstruct_bz_kpoints()
        self._energy_maps = [None] * len(self._all_kpoints)

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

    def reconstruct_bz_kpoints(self):
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
        all_ho_energies = np.array([self.ho_energies[0], ])

        sg = SpacegroupAnalyzer(self._out.structures[-1])
        symmops = sg.get_point_group_operations(cartesian=False)

        # Reconstruct all the kpoints with the corresponding energies
        for k, lu_energy, ho_energy in zip(kpts_bz[1:], self.lu_energies[1:],
                                           self.ho_energies[1:]):

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
                [all_lu_energies, np.ones([len(add_list), 1]) * lu_energy]
            )
            all_ho_energies = np.vstack(
                [all_ho_energies, np.ones([len(add_list), 1]) * ho_energy]
            )

        self._all_kpoints = all_kpts
        self._all_lu_energies = all_lu_energies
        self._all_ho_energies = all_ho_energies

    def as_dict(self):
        """""
        Json-serialization dict representation of the Bandscape.

        Args:
            verbosity (int): Verbosity level. Default of 0 only includes the
                matrix representation. Set to 1 for more details.
        """

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "vasprun": self._out.as_dict(),
             "energy_maps": self._energy_maps}

        return d

    @classmethod
    def from_dict(cls, d):
        """
        Create a Bandscape from a dictionary containing the vasprun and
        energy_maps.

        """
        bandscape = cls(vasprun=Vasprun.from_dict(d["vasprun"]))
        bandscape._energy_maps = d["energy_maps"]

        return bandscape

    @classmethod
    def from_str(cls, input_string, fmt="json"):
        """
        Initialize a Bandscape from a string.

        Currently only supports 'json' format.

        Args:
            input_string (str): String from which the Bandscape is initialized.
            fmt (str): Format of the string representation.

        Returns:
            (bandscape.Bandscape)
        """
        if fmt == "json":
            d = json.loads(input_string)
            return cls.from_dict(d)
        else:
            raise NotImplementedError('Only json formats have been '
                                      'implemented.')

    @classmethod
    def from_file(cls, fmt, filename):
        """
        Initialize a Bandscape from a file.

        Args:
            filename (str): File in which the Bandscape is stored.

        Returns:
            (bandscape.Bandscape)
        """
        if fmt == "vasprun":
            return cls(Vasprun(filename))
        else:
            with zopen(filename) as file:
                contents = file.read()

            return cls.from_str(contents)

    def to(self, filename):
            s = json.dumps(self.as_dict())
            if filename:
                with zopen(filename, "wt") as f:
                    f.write("%s" % s)
                return
            else:
                return s

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

        kpoint_index = [i for i, x in enumerate(list(self._all_kpoints)) if
                        np.linalg.norm(x - np.array(kpoint)) < 1e-4]

        if len(kpoint_index) == 1:
            kpoint_index = kpoint_index[0]
        else:
            raise ValueError("Kpoint not found in list of kpoints.")

        if self._energy_maps[kpoint_index]:

            return self._energy_maps[kpoint_index]

        else:

            print("setting up neighbors")

            neighbors = set_up_neighbors(self._out.lattice)

            ho_energy = self._all_ho_energies[kpoint_index]

            print("setting up q vectors")

            all_kpts_c = np.dot(self._all_kpoints,
                                self._out.lattice_rec.matrix)

            if not cartesian:
                kpoint = np.dot(kpoint, self._out.lattice_rec.matrix)

            q_vectors_c = all_kpts_c - np.vstack(len(all_kpts_c) * [kpoint])
            lu_energies = self._all_lu_energies - ho_energy

            q_vectors_bz_c = []

            print("mapping q vectors to 1st BZ")

            for i, q in enumerate(q_vectors_c):
                try:
                    new_q = return_to_brillouin(q, self._out.lattice,
                                                neighbors=neighbors,
                                                cartesian=True)
                    q_vectors_bz_c.append(new_q)
                except ValueError:
                    print(
                        "Found a kpoint that could not be returned to 1st BZ, "
                        "ignoring...")
                    np.delete(lu_energies, i, axis=0)

            print("done!")

            print("Calculating coordinates in fractional coords.")

            q_vectors_bz = np.dot(q_vectors_bz_c, np.linalg.inv(
                self._out.lattice_rec.matrix))

            # q_110 = q_vectors_bz[0]
            # lu_110 = np.array([lu_energies[0], ])

            q_110 = []
            lu_110 = []

            print("extracting q vectors in 110 plane.")

            # Extract the kpoints in the 110 plane
            for k, energy in zip(q_vectors_bz, lu_energies):
                if abs(k[2]) < 1e-5:

                    if q_110 == []:
                        q_110 = k
                        lu_110 = np.array([energy, ])
                    else:
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
        energy_maps = []
        x, y = np.mgrid[-2:2:0.02, -2:2:0.02]

        for kpoint_index in indices:
            kpoint = self._all_kpoints[kpoint_index]

            x, y, energies = self.find_energy_map(kpoint, cartesian)

            energy_maps.append(energies)

        energy_maps = np.array(energy_maps)

        min_energy_map = np.nanmin(energy_maps, axis=0)

        # Plot
        plt.pcolor(x, y, min_energy_map[:, :, 0],
                   vmin=np.nanmin(min_energy_map),
                   vmax=np.nanmax(min_energy_map),
                   cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('Energy (eV)', size='x-large')

        plt.show()


def set_up_brillouin_facets(lattice):
    """
    Set up the facets of the 1st Brillouin zone as a list of cage.core.Facets.

    Args:
        lattice (pymatgen.core.lattice.Lattice): Lattice of the structure.

    Returns:
        (list) List of cage.core.Facet objects.

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
    Set up the nearest neighbors for the gamma point of the reciprocal lattice.

    Args:
        lattice (pymatgen.core.lattice.Lattice): Lattice of the structure.

    Returns:
        (list): List of coordinates (np.array) of the nearest reciprocal
            lattice points.

    """
    bz_facet_coords = lattice.get_brillouin_zone()
    neighbors = []

    for facet in bz_facet_coords:

        neighbors.append(np.mean(facet, axis=0) * 2)

    return neighbors


def is_in_brillouin(kpoint, lattice, neighbors=None, cartesian=True):
    """
    Check whether a k-point is in the 1st Brillouin zone of a lattice.

    Args:
        kpoint (3x1 numpy.array): Coordinates of the k-point.
        lattice (pymatgen.core.lattice.Lattice): Lattice of the structure.
        neighbors (list): List of coordinates (np.array) of the nearest
            reciprocal lattice points.
        cartesian (bool): Flag to indicate whether the coordinates are given in
            cartesian or direct coordinates. Defaults to True.

    Returns:
        (tuple) [0]: (bool) Indicates whether or not the kpoint is in the
                        1st Brillouin zone
                [1]: (3x1 numpy.array) Closest neighbor.
    """
    if not cartesian:
        kpoint = np.dot(kpoint, lattice.reciprocal_lattice.matrix)

    in_brillouin = True

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
    Return a k-point to the first Brillouin zone of a lattice.

    Args:
        kpoint (3x1 numpy.array): Coordinates of the k-point.
        lattice (pymatgen.core.lattice.Lattice): Lattice of the structure.
        neighbors (list): List of coordinates (np.array) of the nearest
            reciprocal lattice points.
        cartesian (bool): Flag to indicate whether the coordinates are given in
            cartesian or direct coordinates. Defaults to True.

    Returns:
        (3x1 numpy.array) Corresponding k-point in the 1st Brillouin zone.

    """
    in_brillouin, point = is_in_brillouin(kpoint, lattice,
                                          neighbors=neighbors,
                                          cartesian=cartesian)

    if in_brillouin:
        return kpoint
    else:
        if not cartesian:
            kpoint = np.dot(kpoint, lattice.reciprocal_lattice.matrix)

        new_kpoint = kpoint - point

        if not cartesian:
            new_kpoint = np.dot(new_kpoint, np.linalg.inv(
                lattice.reciprocal_lattice.matrix))

        if is_in_brillouin(new_kpoint, lattice, neighbors, cartesian)[0]:
            return new_kpoint
        else:
            return return_to_brillouin(new_kpoint, lattice, neighbors,
                                       cartesian)


def plot_scatter(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()
