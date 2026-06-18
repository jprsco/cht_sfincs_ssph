"""SFINCS XMI (Basic Model Interface) wrapper and RTC utilities.

Provides SfincsXmi (extends xmipy.XmiWrapper) for higher-level access to
the SFINCS shared library, together with RTCCollection and RTC classes for
real-time control coupling (weirs and pumps).
"""

import pathlib as pl
from ctypes import POINTER, byref, c_char_p, c_double, c_int

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from xmipy import XmiWrapper


class SfincsXmi(XmiWrapper):
    """SFINCS XMI (Basic Model Interface) wrapper.

    Extends :class:`xmipy.XmiWrapper` to provide higher-level access to
    the SFINCS shared library, including domain retrieval, RTC coupling,
    and cell-index/area queries.

    Parameters
    ----------
    dll_path : str or pathlib.Path
        Path to the SFINCS shared library (``sfincs.dll`` / ``sfincs.so``).
    working_directory : str
        Working directory for the SFINCS instance.
    """

    def __init__(self, dll_path, working_directory: str) -> None:
        # if dll_path is a string, convert it to a pathlib.Path object
        if isinstance(dll_path, str):
            dll_path = pl.Path(dll_path)
        super().__init__(dll_path, working_directory=working_directory)
        self.rtc_collection = RTCCollection(self)

    def get_domain(self) -> None:
        """Retrieve the SFINCS domain arrays (coordinates, water levels, bed levels).

        For subgrid runs ``self.subgrid_z_zmin`` is populated (live pointer
        into the kernel's subgrid minimum-bed array) and ``self.subgrid`` is
        set to ``True``; for non-subgrid runs ``self.zb`` is populated and
        ``self.subgrid`` is ``False``.

        Returns
        -------
        None
        """
        self.get_xz_yz()
        self.get_zs()
        # Detect subgrid vs non-subgrid by checking which bed-level array the
        # kernel exposes with a non-zero shape.  In subgrid mode `zb` is not
        # allocated; in non-subgrid mode `subgrid_z_zmin` is not allocated.
        self.subgrid = False
        try:
            zmin_shape = self.get_var_shape("subgrid_z_zmin")
            if zmin_shape is not None and int(zmin_shape[0]) > 0:
                self.get_subgrid_z_zmin()
                self.subgrid = True
        except Exception:
            self.subgrid = False
        if self.subgrid:
            # In subgrid mode there is no `zb` array; expose
            # `subgrid_z_zmin` under the `zb` name as well so external
            # scripts can read "the bed level" without branching on mode.
            self.zb = self.subgrid_z_zmin
        else:
            self.get_zb()
        self.get_qext()

    def reset_qext(self) -> None:
        """Reset all external flux values to zero.

        Returns
        -------
        None
        """
        self.qext[:] = 0.0

    def read(self) -> None:
        """No-op placeholder required by the BMI interface.

        Returns
        -------
        None
        """

    def write(self) -> None:
        """No-op placeholder required by the BMI interface.

        Returns
        -------
        None
        """

    def find_cell(self, x: float, y: float) -> int:
        """Find the index of the cell that contains the point (x, y).

        Parameters
        ----------
        x : float
            X-coordinate of the query point.
        y : float
            Y-coordinate of the query point.

        Returns
        -------
        int
            Zero-based cell index.
        """
        indx = self.get_sfincs_cell_index(x, y)
        return indx

    def find_cell_area(self, index: int) -> float:
        """Find the area of the cell with the given index.

        Parameters
        ----------
        index : int
            Zero-based cell index.

        Returns
        -------
        float
            Cell area in the model coordinate units squared.
        """
        # Get the area of the cell with the given index
        area = self.get_sfincs_cell_area(index)
        return area

    def get_xz_yz(self) -> None:
        """Retrieve pointers to the cell x/y coordinate arrays.

        Returns
        -------
        None
        """
        self.xz = self.get_value_ptr("z_xz")
        self.yz = self.get_value_ptr("z_yz")

    def get_zb(self) -> None:
        """Retrieve a pointer to the bed level array.

        Returns
        -------
        None
        """
        self.zb = self.get_value_ptr("zb")

    def get_subgrid_z_zmin(self) -> None:
        """Retrieve a pointer to the subgrid minimum cell-centre bed level array.

        Only valid for subgrid runs.  The pointer stays in sync with the
        kernel's ``subgrid_z_zmin``, which is mutated in place by
        ``update_bed_level`` when an external delta is applied via
        ``dzbext``.

        Returns
        -------
        None
        """
        self.subgrid_z_zmin = self.get_value_ptr("subgrid_z_zmin")

    def get_zs(self) -> None:
        """Retrieve a pointer to the water level array at the current time step.

        Returns
        -------
        None
        """
        self.zs = self.get_value_ptr("zs")

    def get_qext(self) -> None:
        """Retrieve a pointer to the external flux array.

        Returns
        -------
        None
        """
        self.qext = self.get_value_ptr("qext")

    def get_uorb(self) -> None:
        """Retrieve a pointer to the orbital velocity array.

        Returns
        -------
        None
        """
        self.uorb = self.get_value_ptr("uorb")

    def _set_logical(self, flag_name: str, value: bool) -> None:
        """Toggle a boolean flag in the SFINCS kernel via the BMI.

        Parameters
        ----------
        flag_name : str
            Kernel-side flag name (e.g. ``"qext"`` or ``"dzbext"``).
        value : bool
            ``True`` to enable the flag, ``False`` to disable it.
        """
        ival = (c_int * 1)(1 if value else 0)
        self._execute_function(
            self.lib.set_logical, c_char_p(flag_name.encode("utf-8")), ival
        )

    def get_dzbext(self) -> None:
        """Enable and retrieve a pointer to the external delta-bed-level array.

        Toggles ``use_dzbext`` on in the kernel (which lazily allocates the
        array on first enable) and stores a NumPy view of it in
        ``self.dzbext``.

        Returns
        -------
        None
        """
        self._set_logical("dzbext", True)
        self.dzbext = self.get_value_ptr("dzbext")

    def set_bed_level(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        z: np.ndarray | None = None,
        zb0: np.ndarray | None = None,
        update_water_level: bool = False,
    ) -> None:
        """Set the bed level by interpolating scattered (x, y, z) data to the grid.

        The new absolute bed is interpolated from the supplied scatter, the
        delta against the live kernel-side baseline is written into
        ``self.dzbext``, and the kernel applies it via ``update_bed_level``.

        For non-subgrid models the baseline is ``self.zb``; for subgrid
        models it is ``self.subgrid_z_zmin``.  Both are live pointers into
        the kernel and are mutated in place by ``update_bed_level``, so no
        Python-side cache of the previous bed is needed.

        Parameters
        ----------
        x : numpy.ndarray, optional
            X-coordinates of the new bed level data.
        y : numpy.ndarray, optional
            Y-coordinates of the new bed level data.
        z : numpy.ndarray, optional
            New bed level values (m) at each (x, y) point.
        zb0 : numpy.ndarray, optional
            If provided, *z* is added to *zb0* rather than used directly.
        update_water_level : bool, optional
            If ``True``, the water surface is adjusted by the same amount as
            the bed level change.  Defaults to ``False``.

        Returns
        -------
        None
        """

        if x is None or y is None or z is None:
            return

        if not hasattr(self, "dzbext"):
            self.get_dzbext()

        # New absolute bed level
        zb = interp2(x, y, z, self.xz, self.yz)

        # Replace NaNs with zeros
        zb[np.isnan(zb)] = 0.0

        if zb0 is not None:
            zb = zb0[:].copy() + zb

        # Baseline absolute bed: live kernel pointer, mutated in place by
        # update_bed_level.  Subgrid mode: subgrid_z_zmin (cell-centre min);
        # non-subgrid: zb.
        if getattr(self, "subgrid", False):
            zb_prev = np.asarray(self.subgrid_z_zmin[:])
        else:
            zb_prev = np.asarray(self.zb[:])

        # Delta to push to the kernel
        dzb = (zb - zb_prev).astype(np.float32, copy=False)

        # Stash the delta in the kernel-side dzbext array, then apply
        self.dzbext[:] = dzb
        self.update_bed_level()

        if update_water_level:
            self.zs += dzb

        # Reset dzbext so subsequent calls start from a clean slate
        self.dzbext[:] = 0.0

    def set_bed_level_change(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        dz: np.ndarray | None = None,
        update_water_level: bool = False,
    ) -> None:
        """Apply a bed level change relative to the initial bed level.

        Useful for simulating dynamic faulting or landslides.  The supplied
        change is interpolated to the grid and pushed to the kernel via
        ``self.dzbext`` and ``update_bed_level``; this routine no longer
        touches ``self.zb``, ``self.subgrid_z_zmin``, or
        ``self.subgrid_z_zmax`` directly.

        Parameters
        ----------
        x : numpy.ndarray, optional
            X-coordinates of the bed level change data.
        y : numpy.ndarray, optional
            Y-coordinates of the bed level change data.
        dz : numpy.ndarray, optional
            Bed level change (m) at each (x, y) point relative to the
            previous time step (i.e. an incremental delta).
        update_water_level : bool, optional
            If ``True``, the water surface is adjusted by the same delta.
            Defaults to ``False``.

        Returns
        -------
        None
        """

        if x is None or y is None or dz is None:
            return

        if not hasattr(self, "dzbext"):
            self.get_dzbext()

        # Interpolate the delta to the SFINCS grid
        dzb = interp2(x, y, dz, self.xz, self.yz)
        dzb[np.isnan(dzb)] = 0.0
        dzb = dzb.astype(np.float32, copy=False)

        # Push to kernel and apply
        self.dzbext[:] = dzb
        self.update_bed_level()

        if update_water_level:
            self.zs += dzb

        # Reset dzbext so subsequent calls start from a clean slate
        self.dzbext[:] = 0.0

    def update_bed_level(self) -> None:
        """Apply ``self.dzbext`` to the SFINCS kernel's bed-level arrays.

        For non-subgrid models this updates ``zb``; for subgrid models it
        updates ``subgrid_z_zmin``/``subgrid_z_zmax`` (cell centres) and
        ``subgrid_uv_zmin``/``subgrid_uv_zmax`` (uv points), with the uv zmin
        clamped from below by the cell-centre zmin of the two neighbours.
        ``zbuvmx`` is also recomputed for non-subgrid runs.

        Returns
        -------
        None
        """
        self._execute_function(self.lib.update_bed_level)

    def update_apparent_roughness(self, uorb: np.ndarray) -> None:
        """Update the apparent roughness using the orbital velocity.

        Parameters
        ----------
        uorb : numpy.ndarray
            Orbital velocity array (m/s) at each cell.

        Returns
        -------
        None
        """
        #        self.tp[:] = tp
        self.uorb[:] = uorb
        self._execute_function(self.lib.update_apparent_roughness)

    def get_sfincs_cell_index(self, x, y):
        indx = c_int(0)
        self._execute_function(
            self.lib.get_sfincs_cell_index,
            byref(c_double(x)),
            byref(c_double(y)),
            byref(indx),
        )
        # Index is 1-based in sfincs, so we need to subtract 1 to get the 0-based index
        return indx.value - 1

    def get_cell_indices(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Find the cell indices for a set of (x, y) query points.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinates of the query points.
        y : numpy.ndarray
            Y-coordinates of the query points.

        Returns
        -------
        numpy.ndarray
            Zero-based cell index array (int32).
        """
        # Convert x and y to double arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = x.shape[0]
        indx = np.empty(n, dtype=np.int32)
        # Convert to pointers
        x_ptr = x.ctypes.data_as(POINTER(c_double))
        y_ptr = y.ctypes.data_as(POINTER(c_double))
        indx_ptr = indx.ctypes.data_as(POINTER(c_int))

        self._execute_function(
            self.lib.get_sfincs_cell_indices, x_ptr, y_ptr, indx_ptr, c_int(n)
        )

        # Index is 1-based in sfincs, so we need to subtract 1 to get the 0-based index
        return indx - 1

    def get_sfincs_cell_area(self, index):
        area = c_double(0.0)
        self._execute_function(
            self.lib.get_sfincs_cell_area, byref(c_int(index + 1)), byref(area)
        )
        return area.value

    def update_water_level(self, t: float) -> None:
        """No-op placeholder for water level update hook.

        Parameters
        ----------
        t : float
            Current simulation time (s).

        Returns
        -------
        None
        """

    def run_timestep(self) -> float:
        """Advance the model by one time step and return the new simulation time.

        Returns
        -------
        float
            Updated simulation time (s).
        """
        self.update()
        return self.get_current_time()


class RTCCollection:
    """Collection of RTC (real-time control) objects.

    Manages a set of weir and pump RTC objects and applies their computed
    fluxes to the SFINCS model via the XMI interface.

    Parameters
    ----------
    sfx : SfincsXmi
        The parent SfincsXmi instance.
    """

    def __init__(self, sfx: "SfincsXmi") -> None:
        self.sfx = sfx

    def initialize(self, rtc_dict_list: list) -> None:
        """Initialise the RTC collection from a list of configuration dicts.

        Parameters
        ----------
        rtc_dict_list : list[dict]
            Each dict must contain at minimum ``"name"`` and ``"type"`` keys
            and may optionally include ``"p1"``, ``"p2"``, ``"elevation"``,
            ``"width"``, ``"alpha"``, ``"zmin"``, ``"qpump"``, ``"dzpump"``,
            and ``"t_relax"``.

        Returns
        -------
        None
        """
        self.rtc_list = []
        # Loop through all RTC objects and create them
        for rtc_dict in rtc_dict_list:
            name = rtc_dict["name"]
            type = rtc_dict["type"]
            p1 = rtc_dict.get("p1", (0.0, 0.0))
            p2 = rtc_dict.get(
                "p2", None
            )  # p2 is optional, if not given, we assume a single point source
            elevation = rtc_dict.get("elevation", 0.0)
            width = rtc_dict.get("width", 0.0)
            alpha = rtc_dict.get("alpha", 0.5)
            zmin = rtc_dict.get("zmin", 0.0)
            qpump = rtc_dict.get("qpump", 0.0)
            dzpump = rtc_dict.get("dzpump", 0.02)
            zmin = rtc_dict.get("zmin", 0.0)
            t_relax = rtc_dict.get("t_relax", 10.0)
            # Create the RTC object
            rtc = RTC(
                self.sfx,
                name,
                type,
                p1,
                p2,
                elevation,
                width,
                alpha,
                qpump,
                dzpump,
                zmin,
                t_relax,
            )
            self.rtc_list.append(rtc)

    def compute_fluxes(self, dt: float = 1.0e6) -> None:
        """Compute and apply fluxes for all RTC objects.

        Parameters
        ----------
        dt : float, optional
            Time step (s) used to compute the relaxation factor.
            Defaults to ``1.0e6`` (effectively no relaxation).

        Returns
        -------
        None
        """
        # First set qext to zero
        self.sfx.reset_qext()

        # Now get water levels
        self.sfx.get_zs()

        # Loop through all RTC objects and compute the fluxes
        for rtc in self.rtc_list:
            # Compute the flux for this RTC object
            q = rtc.compute_flux()

            # Apply relaxation factor if t_relax > 0.0
            if rtc.t_relax > 0.0:
                # Compute the relaxation factor
                relaxation_factor = min(dt / rtc.t_relax, 1.0)
                # Compute the flux based on the relaxation factor
                # q0 = q
                q = (1.0 - relaxation_factor) * rtc.q + relaxation_factor * q
                # if rtc.type == "pump":
                #     print(f"RTC {rtc.name}: q0 = {q0:.4f} m3/s, q = {q:.4f}, rtc.q = {rtc.q:.4f}, relaxation_factor = {relaxation_factor:.2f}")

            # Fluxes in qext are in m/s, so we need to convert m3/s to m/s by dividing by the area
            # Determine the direction of the flux
            if q > 0.0:
                # If the flux is positive, we are discharging from the first point to the second point
                self.sfx.qext[rtc.index1] = -q / rtc.area1
                if rtc.index2 is not None:
                    # If p2 is given, we assume a two point source
                    # Set the flux for the second point
                    self.sfx.qext[rtc.index2] = q / rtc.area2
            else:
                # If the flux is negative, we are discharging from the second point to the first point
                self.sfx.qext[rtc.index1] = q / rtc.area1
                if rtc.index2 is not None:
                    # If p2 is given, we assume a two point source
                    self.sfx.qext[rtc.index2] = -q / rtc.area2
            rtc.q = q

    def set_fluxes(self, qlist: list) -> None:
        """Set the fluxes for the RTC objects directly.

        Parameters
        ----------
        qlist : list[float]
            List of fluxes (m3/s) for each RTC object in the collection.

        Returns
        -------
        None
        """
        # q is a list of fluxes for each RTC object
        self.sfx.reset_qext()
        # Loop through all RTC objects and set the fluxes
        for irtc, rtc in enumerate(self.rtc_list):
            # Set the flux for this RTC object
            # Fluxes in qext are in m/s, so we need to convert m3/s to m/s by dividing by the area
            # Determine the direction of the flux
            if qlist[irtc] > 0.0:
                # If the flux is positive, we are discharging from the first point to the second point
                self.sfx.qext[rtc.index1] = -qlist[irtc] / rtc.area1
                self.sfx.qext[rtc.index2] = qlist[irtc] / rtc.area2
            else:
                # If the flux is negative, we are discharging from the second point to the first point
                self.sfx.qext[rtc.index1] = qlist[irtc] / rtc.area1
                self.sfx.qext[rtc.index2] = -qlist[irtc] / rtc.area2

    def get_zs(self) -> list:
        """Get the water levels for all RTC objects.

        Returns
        -------
        list[tuple[float, float]]
            A list of ``(z1, z2)`` water level tuples for each RTC object.
        """
        self.sfx.get_zs()  # Update water levels in sfx
        zs = []
        for rtc in self.rtc_list:
            # Get the water levels for this RTC object
            z1 = self.sfx.zs[rtc.index1]
            z2 = self.sfx.zs[rtc.index2]
            zs.append((z1, z2))
        return zs


class RTC:
    """Single real-time control structure (weir or pump).

    Computes the flux exchanged between one or two SFINCS cells based on
    simple hydraulic equations for weirs or pumps.

    Parameters
    ----------
    sfx : SfincsXmi
        Parent XMI wrapper.
    name : str
        Name of the RTC structure.
    type : str
        Structure type: ``"weir"`` or ``"pump"``.
    p1 : tuple[float, float]
        (x, y) coordinates of the primary cell.
    p2 : tuple[float, float] or None
        (x, y) coordinates of the secondary cell, or ``None`` for a single
        point source.
    elevation : float
        Crest elevation (m) of the weir.
    width : float
        Crest width (m) of the weir.
    alpha : float
        Discharge coefficient for the weir equation.
    qpump : float
        Design pump discharge (m3/s).
    dzpump : float
        Minimum water depth above bed level required to activate the pump.
    zmin : float
        Minimum water level (m) required to activate the pump.
    t_relax : float
        Relaxation time (s); set to ``0.0`` to disable relaxation.
    """

    def __init__(
        self,
        sfx: "SfincsXmi",
        name: str,
        type: str,
        p1: tuple,
        p2: tuple | None,
        elevation: float,
        width: float,
        alpha: float,
        qpump: float,
        dzpump: float,
        zmin: float,
        t_relax: float,
    ) -> None:
        self.sfx = sfx
        self.name = name
        self.type = type
        self.index1 = self.sfx.find_cell(p1[0], p1[1])
        self.area1 = self.sfx.find_cell_area(self.index1)
        if p2 is None:
            # If p2 is not given, we assume a single point source
            self.index2 = None
            self.area2 = None
        else:
            self.index2 = self.sfx.find_cell(p2[0], p2[1])
            self.area2 = self.sfx.find_cell_area(self.index2)
        self.zb1 = self.sfx.zb[self.index1]
        self.qmax = 1.0e6
        self.qh = []  # Q-H curve?
        self.qpump = 0.0
        self.elevation = elevation
        self.width = width
        self.alpha = alpha
        self.qpump = qpump
        self.dzpump = dzpump
        self.zmin = zmin
        self.q = 0.0
        self.t_relax = t_relax  # Relaxation time (s)

    def compute_flux(self) -> float:
        """Compute the flux for this RTC structure at the current time step.

        Returns
        -------
        float
            Flux (m3/s); positive means flow from cell 1 to cell 2.
        """
        # Get the water levels at the two points
        # zs = self.get_value_ptr("zs")
        z1 = self.sfx.zs[self.index1]
        if self.index2 is None:
            # If p2 is not given, we assume a single point source
            z2 = None
        else:
            z2 = self.sfx.zs[self.index2]

        if self.type == "weir":
            q = self.compute_flux_weir(z1, z2)
        if self.type == "pump":
            # water level z2 not used for pump
            q = self.compute_flux_pump(z1)

        return q

    def compute_flux_weir(self, z1: float, z2: float) -> float:
        """Compute the weir flux using a simple broad-crested weir equation.

        Parameters
        ----------
        z1 : float
            Water level (m) at the upstream/primary cell.
        z2 : float
            Water level (m) at the downstream/secondary cell.

        Returns
        -------
        float
            Flux (m3/s); positive means flow from cell 1 to cell 2.
        """

        if z1 < self.elevation and z2 < self.elevation:
            # No flow, both levels below the weir
            return 0.0
        else:
            if z1 > z2:
                zup = z1
                zdown = z2
                direction = 1
            else:
                zup = z2
                zdown = z1
                direction = -1

            # Compute the flux based on very simple weir equation
            zmx = max(zdown, self.elevation)  # max of zdown and elevation
            q = (
                direction
                * self.alpha
                * self.width
                * (zup - self.elevation)
                * (zup - zmx) ** (2 / 3)
            )
            # print(f"zup: {zup}, zdown: {zdown}, q: {q}")

            return q

    def compute_flux_pump(self, z1: float) -> float:
        """Compute the pump flux.

        Parameters
        ----------
        z1 : float
            Water level (m) at the primary cell.

        Returns
        -------
        float
            Pump discharge (m3/s); zero if the water level is below the
            activation threshold.
        """
        if z1 < self.zmin or z1 < self.zb1 + self.dzpump:
            # No flow, level below the minimum
            return 0.0
        else:
            # Compute the flux based on very simple pump equation. Let pumping gradually increase with water level.
            # fac = min((z1 - self.zmin) / self.dzpump, 1.0)
            fac = 1.0
            return fac * self.qpump


def interp2(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a 2-D field from a regular grid to scattered points.

    Parameters
    ----------
    x0 : numpy.ndarray
        1-D array of x-coordinates of the source grid.
    y0 : numpy.ndarray
        1-D array of y-coordinates of the source grid.
    z0 : numpy.ndarray
        2-D array of values on the source grid, shape ``(len(y0), len(x0))``.
    x1 : numpy.ndarray
        X-coordinates of the target points (1-D or 2-D).
    y1 : numpy.ndarray
        Y-coordinates of the target points (same shape as *x1*).
    method : str, optional
        Interpolation method passed to
        :class:`~scipy.interpolate.RegularGridInterpolator`.
        Defaults to ``"linear"``.

    Returns
    -------
    numpy.ndarray
        Interpolated values at the target points; same shape as *x1*.
        Points outside the source grid are filled with ``NaN``.
    """

    # meanx = np.mean(x0)
    # meany = np.mean(y0)
    # x0 -= meanx
    # y0 -= meany
    # x1 -= meanx
    # y1 -= meany

    f = RegularGridInterpolator(
        (y0, x0), z0, bounds_error=False, fill_value=np.nan, method=method
    )
    # reshape x1 and y1
    if x1.ndim > 1:
        sz = x1.shape
        x1 = x1.reshape(sz[0] * sz[1])
        y1 = y1.reshape(sz[0] * sz[1])
        # interpolate
        z1 = f((y1, x1)).reshape(sz)
    else:
        z1 = f((y1, x1))

    return z1
