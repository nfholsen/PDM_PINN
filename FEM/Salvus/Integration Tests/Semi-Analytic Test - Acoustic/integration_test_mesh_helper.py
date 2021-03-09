import abc
import pathlib
import textwrap
from enum import Enum

import numpy as np
import obspy
from salvus.mesh.structured_grid_2D import StructuredGrid2D


class Physics(Enum):
    """Types of physics available for analytic tests."""

    ACOUSTIC = 1
    ELASTIC = 2
    COUPLED = 3


class AnalyticCode(Enum):
    """Collection of codes with which to compare analytic solutions."""

    Gar6more2D = 1
    Gar6more3D = 2
    EX2DIR = 3


def read_gar6more(filename: pathlib.Path) -> obspy.Stream:
    stream = obspy.Stream()
    data = np.loadtxt(filename)
    for _i in range(1, data.shape[1]):
        stream.append(
            obspy.Trace(
                data=data[:, _i],
                header={
                    "delta": 1e-4,
                    "channel": "XDX",
                    "network": "XX",
                    "station": "%03i" % _i,
                },
            )
        )
    return stream


def read_ex2ddir(filename: pathlib.Path, n_receiver: int) -> obspy.Stream:
    """Read the semi-analytic solution output by a run from EX2DDIR.

    :param filename: EX2DDIR output file.
    :param n_receiver: Number of receivers expected.
    :return: Stream of traces from EX2DDIR receivers.
    """
    stream = obspy.Stream()
    data = np.loadtxt(filename)
    n_step = int(len(data) / n_receiver)
    for _i in range(n_receiver):
        stream.append(
            obspy.Trace(
                data=data[_i * n_step : (_i + 1) * n_step],
                header={
                    "delta": 1e-4,
                    "channel": "XDX",
                    "network": "XX",
                    "station": "%03i" % _i,
                },
            )
        )
    return stream


def get_gll_points(n=5, eps=1e-15):
    """
    compute Gauss-Lobatto-Legendre (GLL) quadrature weights and points [-1, 1]
    using Newton-Raphson optimization.

    :param n: number of integration points (order + 1)
    :type n: integer
    :param eps: accuracy of the nodes
    :type eps: float

    :returns: tuple of two numpy arrays of floats containing the points and
        weights
    """

    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess

    x = np.cos(np.pi * np.arange(n) / (n - 1))

    P = np.zeros((n, n))

    xold = 0

    while np.max(np.abs(x - xold)) > eps:

        xold = x

        # Compute P_(N) using the recursion relation
        P[:, 0] = 1
        P[:, 1] = x

        for k in np.arange(2, n):
            P[:, k] = (
                (2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]
            ) / k

        # Compute its first and second derivatives and
        # update x using the Newton-Raphson method.
        x = xold - (x * P[:, n - 1] - P[:, n - 2]) / (n * P[:, n - 1])

    w = 2.0 / ((n - 1) * n * P[:, n - 1] ** 2)

    return x[::-1], w


class IntegrationTestMesh(metaclass=abc.ABCMeta):
    """Encapsulates a mesh for a standard Salvus integration test.

    By their nature, the integration test meshes will (mostly) consist of homogeneous models which can be compared with
    an analytic solution. Depending on the code used to generate the comparable analytic solution, some parameters
    (i.e. coordiantes, locations) must be set differently. This base class should be able to handle all these different
    cases. The overridden functions should primarily be dimension-based.
    """

    def __init__(
        self,
        physics: Physics,
        analytic_code: AnalyticCode,
        polynomial_order: int,
        shape_order: int,
        n_elem_per_wavelength: int,
    ):
        """Set basic mesh parameters.

        The source for all integration tests is a Ricker wavelet with a center frequency of 100 Hz. Within the seismic
        velocity range, this results in a domain size of approximately 1km x 1km. In truth, this is a bit big, but we
        may as well fit a few more wavelengths into the analytic tests. Depending on the specific analytical code with
        which we are comparing solutions, some internal parameters may be set differently.

        :param physics: Physics of the test (i.e. acoustic, elastic).
        :param analytic_code: Which code generated the analytic solution (i.e. EX2DIR, Gar6more).
        :param polynomial_order: Interpolating order on each element.
        :param n_elem_per_wavelength: Number of elements per wavelength.
        """
        self._x0 = 0.0
        self._x1 = 1000.0

        self._vp = 5800.0
        self._vs = 4000.0
        self._rho = 2600.0
        self._peak_frequency = 151.0

        self._physics = physics
        self._shape_order = shape_order
        self._analytic_code = analytic_code
        self._polynomial_order = polynomial_order
        self._n_elem_per_wavelength = n_elem_per_wavelength

        if physics == Physics.ACOUSTIC:
            self._minimum_speed = self._vp
        else:
            self._minimum_speed = self._vs
        self.element_size = (
            self._minimum_speed / self._peak_frequency / n_elem_per_wavelength
        )
        self.n_elem_per_dim = np.ceil(
            (self._x1 - self._x0) / self.element_size
        ).astype(int)

    def attach_fields(self, um):
        """Attach parameter fields to the mesh, which depend on the class physics.

        :param um: Unstructured mesh object.
        :return: Unstructured mesh object with parameters attached.
        """

        template = np.ones_like(um.get_element_nodes()[:, :, 0])
        if self._physics == Physics.ELASTIC:
            um.attach_field(f"VP", template * self._vp)
            um.attach_field(f"VS", template * self._vs)
            um.attach_field(f"RHO", template * self._rho)
        elif self._physics == Physics.ACOUSTIC:
            um.attach_field(f"VP", template * self._vp)
            um.attach_field(f"RHO", template * self._rho)
        elif self._physics == Physics.COUPLED:
            um.attach_field(f"VS", template * self._vs)
            um.attach_field(f"VP", template * self._vp)
            um.attach_field(f"RHO", template * self._rho)

        if self._physics == Physics.ELASTIC:
            um.attach_field("fluid", np.zeros(um.nelem))
        elif self._physics == Physics.ACOUSTIC:
            um.attach_field("fluid", np.ones(um.nelem))
        elif self._physics == Physics.COUPLED:
            um.attach_field(
                "fluid",
                np.append(np.zeros(um.nelem // 2), np.ones(um.nelem // 2)),
            )
        return um

    def generate_receiver_toml(self, coordinates):
        """Generate a list of dictionaries describing the Salvus receiver toml.

        :param coordinates: List of tuples of receiver coordinates.
        :return: List of dictionaries.
        """
        if self._physics == Physics.ELASTIC:
            medium = ["solid"] * len(coordinates)
        elif self._physics == Physics.ACOUSTIC:
            medium = ["fluid"] * len(coordinates)
        elif self._physics == Physics.COUPLED:
            medium = ["fluid"] * (len(coordinates) // 2) + ["solid"] * (
                len(coordinates) // 2
            )
        else:
            raise NotImplementedError(
                "Only elastic, acoustic, and coupled currently implemented."
            )

        receivers = ""
        for _i, (x, m) in enumerate(zip(coordinates, medium)):
            receivers += textwrap.dedent(
                """
            [[receiver]]
            network = "XX"
            station = "{num:>03d}"
            medium = "{medium}"
            location-code = ""
            location = [{x0:f}, {x1:f}]
            """.format(
                    num=_i, medium=m, x0=x[0], x1=x[1]
                )
            )

        return receivers

    @property
    def mu(self):
        """Return elastic mu derived from velocities."""

        return self._vs ** 2 * self._rho

    @property
    def lamb(self):
        """Return elastic lambda derived from velocities."""

        return self._vp ** 2 * self._rho - 2 * self.mu

    @property
    @abc.abstractmethod
    def generate(
        self, time_step: float, center_frequency: float, amplitude: float
    ):
        """Lazy generate the mesh and command line options for Salvus.

        This method is overridden by a child class, which needs to generate a mesh, populate it with parameters,
        and fill out the rest of the command line parameters.

        :return: Dictionary of command line parameters.
        """
        pass


def get_mesh(
    dimension: int,
    analytic_code: AnalyticCode,
    physics: Physics,
    n_elem_per_wavelength: int,
    polynomial_order: int,
    shape_order: int,
):
    if dimension == 2:
        m = Mesh2D(
            analytic_code=analytic_code,
            physics=physics,
            n_elem_per_wavelength=n_elem_per_wavelength,
            polynomial_order=polynomial_order,
            shape_order=shape_order,
        )
        return m.get_unstructured_mesh_with_parameters()
    else:
        raise NotImplementedError


class Mesh2D(IntegrationTestMesh):
    """Encapsulates a 2-D for a standard Salvus integration test.

    By their nature, the integration test meshes will (mostly) consist of homogeneous models which can be compared with
    an analytic solution. Depending on the code used to generate the comparable analytic solution, some parameters
    (i.e. coordiantes, locations) must be set differently. This base class should be able to handle all these different
    cases. The overridden functions should primarily be dimension-based.
    """

    def __init__(
        self,
        physics: Physics,
        analytic_code: AnalyticCode,
        polynomial_order: int,
        shape_order: int,
        n_elem_per_wavelength: int,
    ):
        """Set basic mesh parameters.

        The source for all integration tests is a Ricker wavelet with a center frequency of 100 Hz. Within the seismic
        velocity range, this results in a domain size of approximately 1km x 1km. In truth, this is a bit big, but we
        may as well fit a few more wavelengths into the analytic tests. Depending on the specific analytical code with
        which we are comparing solutions, some internal parameters may be set differently.

        :param physics: Physics of the test (i.e. acoustic, elastic).
        :param analytic_code: Which code generated the analytic solution (i.e. EX2DIR, Gar6more).
        :param polynomial_order: Interpolating order on each element.
        :param n_elem_per_wavelength: Number of elements per wavelength.
        """
        super().__init__(
            physics,
            analytic_code,
            polynomial_order,
            shape_order,
            n_elem_per_wavelength,
        )

    def get_unstructured_mesh_with_parameters(self):
        """Generate the 2-D unstructured mesh, and populate it with some material parameters based on the physics."""
        x0, x1 = 0, self._x1
        y0, y1 = 0, self._x1 // 2
        nx, ny = self.n_elem_per_dim, self.n_elem_per_dim // 2

        um = StructuredGrid2D.rectangle(
            nelem_x=nx, nelem_y=ny, min_x=x0, max_x=x1, min_y=y0, max_y=y1
        ).get_unstructured_mesh()
        um.find_side_sets()
        um.change_tensor_order(self._shape_order)

        if self._physics != Physics.COUPLED:
            kx = 6.0 * 2 * np.pi / self._x1
            ky = 3.0 * 2 * np.pi / self._x1
            amplitude = self._x1 / self.n_elem_per_dim / 3.0
            um.points[:, 0] += (
                amplitude
                * np.sin(kx * um.points[:, 0])
                * np.sin(ky * um.points[:, 1])
            )
            um.points[:, 1] += (
                amplitude
                * np.sin(kx * um.points[:, 0])
                * np.sin(ky * um.points[:, 1])
            )

        um = self.attach_fields(um)
        return um

    def generate(
        self,
        time_step: float,
        center_frequency: float,
        amplitude: float,
        mesh_name="IntegrationTestMesh.e",
        start_time=-2e-2,
        end_time=8e-2,
        receiver_file="Receivers.h5",
        wavefield_file="Wavefield.h5",
    ):
        """Lazy generate the mesh and command line options for Salvus.

        This method is overridden by a child class, which needs to generate a mesh, populate it with parameters,
        and fill out the rest of the command line parameters.
        :return: Dictionary of command line parameters.
        """

        # Get mesh and model.
        um = self.get_unstructured_mesh_with_parameters()

        # Setup receivers.
        if self._analytic_code == AnalyticCode.EX2DIR:
            receivers = [
                (600, 950, "solid"),
                (650, 950, "solid"),
                (700, 950, "solid"),
                (750, 950, "solid"),
                (800, 950, "solid"),
            ]
        elif self._analytic_code == AnalyticCode.Gar6more2D:
            if self._physics == Physics.COUPLED:
                receivers = [
                    (300, 250.1, "phi"),
                    (400, 250.1, "phi"),
                    (500, 250.1, "phi"),
                    (600, 250.1, "phi"),
                    (700, 250.1, "phi"),
                    (300, 249.9, "displacement"),
                    (400, 249.9, "displacement"),
                    (500, 249.9, "displacement"),
                    (600, 249.9, "displacement"),
                    (700, 249.9, "displacement"),
                ]
            elif self._physics == Physics.ACOUSTIC:
                receivers = [
                    (300, 50, "phi"),
                    (400, 50, "phi"),
                    (500, 50, "phi"),
                    (600, 50, "phi"),
                    (700, 50, "phi"),
                ]
            elif self._physics == Physics.ELASTIC:
                receivers = [
                    (300, 50, "displacement"),
                    (400, 50, "displacement"),
                    (500, 50, "displacement"),
                    (600, 50, "displacement"),
                    (700, 50, "displacement"),
                ]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        receivers = {
            "receiver": [
                {
                    "network-code": "XX",
                    "station-code": f"{i:03d}",
                    "fields": [m],
                    "location": [float(x), float(y)],
                }
                for i, (x, y, m) in enumerate(receivers)
            ]
        }

        # Setup source.
        center_frequency = center_frequency
        if self._analytic_code == AnalyticCode.Gar6more2D:
            if self._physics == Physics.ELASTIC:
                field = ["u"]
                scale = [1e9, 1e9, 0.0]
                spatial_type = "moment_tensor"
                location = [500.0, 250.0]
                boundaries = {
                    "boundaries": [
                        {"type": "homogeneous-dirichlet", "side-sets": ["y0"]}
                    ]
                }
            elif self._physics == Physics.ACOUSTIC:
                field = ["phi"]
                scale = [amplitude]
                spatial_type = "scalar"
                location = [500.0, 250.0]
                boundaries = {
                    "boundaries": [
                        {"type": "homogeneous-dirichlet", "side-sets": ["y0"]}
                    ]
                }
            elif self._physics == Physics.COUPLED:
                scale = [amplitude]
                spatial_type = "scalar"
                location = [500.0, 375.0]
                boundaries = {
                    "boundaries": [
                        {
                            "type": "absorbing",
                            "side-sets": ["y1"],
                            "taper-amplitude": 0.0,
                            "width-in-meters": 0.0,
                        }
                    ]
                }
            else:
                raise NotImplementedError
        elif self._analytic_code == AnalyticCode.EX2DIR:
            if self._physics == Physics.ELASTIC:
                field = ["displacement"]
                scale = [0.0, -1e9]
                spatial_type = "vector"
                location = [500.0, 750.0]
                boundaries = {}
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        source = {
            "point-source": [
                {
                    "location": location,
                    "spatial-type": spatial_type,
                    "spatial-weights": scale,
                    "source-time-function": {
                        "wavelet": "ricker",
                        "center-frequency": center_frequency,
                    },
                }
            ]
        }

        # Base parameter file.
        input_toml = {
            "domain": {
                "dimension": 2,
                "polynomial-order": 4,
                "mesh": {"filename": mesh_name, "format": "hdf5"},
                "model": {"filename": mesh_name, "format": "hdf5"},
                "geometry": {"filename": mesh_name, "format": "hdf5"},
            },
            "physics": {
                "wave-equation": {
                    "time-stepping-scheme": "newmark",
                    "start-time-in-seconds": start_time,
                    "time-step-in-seconds": time_step,
                    "end-time-in-seconds": end_time,
                    "attenuation": False,
                }
            },
            "output": {
                "point-data": {
                    "sampling-interval-in-time-steps": 1,
                    "filename": receiver_file,
                    "format": "asdf",
                }
            },
        }

        # Insert our sources and receivers.
        input_toml["output"]["point-data"].update(receivers)
        input_toml["physics"]["wave-equation"].update(source)
        input_toml["physics"]["wave-equation"].update(boundaries)
        return input_toml, um
