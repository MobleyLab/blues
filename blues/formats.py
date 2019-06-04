import json
import logging
import subprocess

import mdtraj.version
import netCDF4 as nc
import numpy as np
import parmed
import simtk.openmm.version
import yaml
from mdtraj.formats.hdf5 import HDF5TrajectoryFile
from mdtraj.utils import ensure_type, in_units_of
from parmed.amber.netcdffiles import NetCDFTraj

from blues import storage


######################
#  REPORTER FORMATS  #
######################
class LoggerFormatter(logging.Formatter):
    """
    Formats the output of the `logger.Logger` object. Allows customization
    for customized logging levels. This will add a custom level 'REPORT'
    to all custom BLUES reporters from the `blues.reporters` module.

    Examples
    --------
    Below we add a custom level 'REPORT' and have the logger module stream the
    message to `sys.stdout` without any additional information to our custom
    reporters from the `blues.reporters` module

    >>> from blues import reporters
    >>> from blues.formats import LoggerFormatter
    >>> import logging, sys
    >>> logger = logging.getLogger(__name__)
    >>> reporters.addLoggingLevel('REPORT', logging.WARNING - 5)
    >>> fmt = LoggerFormatter(fmt="%(message)s")
    >>> stdout_handler = logging.StreamHandler(stream=sys.stdout)
    >>> stdout_handler.setFormatter(fmt)
    >>> logger.addHandler(stdout_handler)
    >>> logger.report('This is a REPORT call')
        This is a REPORT call
    >>> logger.info('This is an INFO call')
        INFO: This is an INFO call
    """

    dbg_fmt = "%(levelname)s: [%(module)s.%(funcName)s] %(message)s"
    info_fmt = "%(levelname)s: %(message)s"
    rep_fmt = "%(message)s"

    def __init__(self):
        super().__init__(fmt="%(levelname)s: %(msg)s", datefmt="%H:%M:%S", style='%')
        storage.addLoggingLevel('REPORT', logging.WARNING - 5)

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = LoggerFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = LoggerFormatter.info_fmt

        elif record.levelno == logging.WARNING:
            self._style._fmt = LoggerFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = LoggerFormatter.dbg_fmt

        elif record.levelno == logging.REPORT:
            self._style._fmt = LoggerFormatter.rep_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result


class NetCDF4Traj(NetCDFTraj):
    """Extension of `parmed.amber.netcdffiles.NetCDFTraj` to allow proper file
    flushing. Requires the netcdf4 library (not scipy), install with
    `conda install -c conda-forge netcdf4` .

    Parameters
    ----------
    fname : str
        File name for the trajectory file
    mode : str, default='r'
        The mode to open the file in.

    """

    def __init__(self, fname, mode='r'):
        super(NetCDF4Traj, self).__init__(fname, mode)

    def flush(self):
        """
        Flush buffered data to disc.
        """
        if nc is None:
            # netCDF4.Dataset does not have a flush method
            self._ncfile.flush()
        if nc:
            self._ncfile.sync()

    @classmethod
    def open_new(cls,
                 fname,
                 natom,
                 box,
                 crds=True,
                 vels=False,
                 frcs=False,
                 remd=None,
                 remd_dimension=None,
                 title='',
                 protocolWork=False,
                 alchemicalLambda=False):
        """Opens a new NetCDF file and sets the attributes

        Parameters
        ----------
        fname : str
            Name of the new file to open (overwritten)
        natom : int
            Number of atoms in the restart
        box : bool
            Indicates if cell lengths and angles are written to the NetCDF file
        crds : bool, default=True
            Indicates if coordinates are written to the NetCDF file
        vels : bool, default=False
            Indicates if velocities are written to the NetCDF file
        frcs : bool, default=False
            Indicates if forces are written to the NetCDF file
        remd : str, default=None
            'T[emperature]' if replica temperature is written
            'M[ulti]' if Multi-D REMD information is written
            None if no REMD information is written
        remd_dimension : int, default=None
            If remd above is 'M[ulti]', this is how many REMD dimensions exist
        title : str, default=''
            The title of the NetCDF trajectory file
        protocolWork : bool, default=False
            Indicates if protocolWork from the NCMC simulation should be written
            to the NetCDF file
        alchemicalLambda : bool, default=False
            Indicates if alchemicalLambda from the NCMC simulation should be written
            to the NetCDF file

        """
        inst = cls(fname, 'w')
        ncfile = inst._ncfile
        if remd is not None:
            if remd[0] in 'Tt':
                inst.remd = 'TEMPERATURE'
            elif remd[0] in 'Mm':
                inst.remd = 'MULTI'
                if remd_dimension is None:
                    raise ValueError('remd_dimension must be given ' 'for multi-D REMD')
                inst.remd_dimension = int(remd_dimension)
            else:
                raise ValueError('remd must be T[emperature] or M[ultiD]')
        else:
            inst.remd = None
        inst.hasbox = bool(box)
        inst.hasvels = bool(vels)
        inst.hascrds = bool(crds)
        inst.hasfrcs = bool(frcs)

        inst.hasprotocolWork = bool(protocolWork)
        inst.hasalchemicalLambda = bool(alchemicalLambda)

        # Assign the main attributes
        ncfile.Conventions = "AMBER"
        ncfile.ConventionVersion = "1.0"
        ncfile.application = "AmberTools"
        ncfile.program = "ParmEd"
        ncfile.programVersion = parmed.__version__
        ncfile.title = "ParmEd-created trajectory"
        inst.Conventions = "AMBER"
        inst.ConventionVersion = "1.0"
        inst.application = "AmberTools"
        inst.program = "ParmEd"
        inst.programVersion = parmed.__version__
        inst.title = ncfile.title
        # Create the dimensions
        ncfile.createDimension('frame', None)
        ncfile.createDimension('spatial', 3)
        ncfile.createDimension('atom', natom)
        if inst.remd == 'MULTI':
            ncfile.createDimension('remd_dimension', inst.remd_dimension)
        inst.frame, inst.spatial, inst.atom = None, 3, natom
        if inst.hasbox:
            ncfile.createDimension('cell_spatial', 3)
            ncfile.createDimension('cell_angular', 3)
            ncfile.createDimension('label', 5)
            inst.cell_spatial, inst.cell_angular, inst.label = 3, 3, 5
        # Create the variables and assign units and scaling factors
        v = ncfile.createVariable('spatial', 'c', ('spatial',))
        v[:] = np.asarray(list('xyz'))
        if inst.hasbox:
            v = ncfile.createVariable('cell_spatial', 'c', ('cell_spatial',))
            v[:] = np.asarray(list('abc'))
            v = ncfile.createVariable('cell_angular', 'c', (
                'cell_angular',
                'label',
            ))
            v[:] = np.asarray([list('alpha'), list('beta '), list('gamma')])
        v = ncfile.createVariable('time', 'f', ('frame',))
        v.units = 'picosecond'
        if inst.hascrds:
            v = ncfile.createVariable('coordinates', 'f', ('frame', 'atom', 'spatial'))
            v.units = 'angstrom'
            inst._last_crd_frame = 0
        if inst.hasvels:
            v = ncfile.createVariable('velocities', 'f', ('frame', 'atom', 'spatial'))
            v.units = 'angstrom/picosecond'
            inst.velocity_scale = v.scale_factor = 20.455
            inst._last_vel_frame = 0
            if nc is not None:
                v.set_auto_maskandscale(False)
        if inst.hasfrcs:
            v = ncfile.createVariable('forces', 'f', ('frame', 'atom', 'spatial'))
            v.units = 'kilocalorie/mole/angstrom'
            inst._last_frc_frame = 0
        if inst.hasbox:
            v = ncfile.createVariable('cell_lengths', 'd', ('frame', 'cell_spatial'))
            v.units = 'angstrom'
            v = ncfile.createVariable('cell_angles', 'd', ('frame', 'cell_angular'))
            v.units = 'degree'
            inst._last_box_frame = 0
        if inst.remd == 'TEMPERATURE':
            v = ncfile.createVariable('temp0', 'd', ('frame',))
            v.units = 'kelvin'
            inst._last_remd_frame = 0
        elif inst.remd == 'MULTI':
            ncfile.createVariable('remd_indices', 'i', ('frame', 'remd_dimension'))
            ncfile.createVariable('remd_dimtype', 'i', ('remd_dimension',))
            inst._last_remd_frame = 0

        inst._last_time_frame = 0

        if inst.hasprotocolWork:
            v = ncfile.createVariable('protocolWork', 'f', ('frame',))
            v.units = 'kT'
            inst._last_protocolWork_frame = 0

        if inst.hasalchemicalLambda:
            v = ncfile.createVariable('alchemicalLambda', 'f', ('frame',))
            v.units = 'unitless'
            inst._last_alchemicalLambda_frame = 0

        return inst

    @property
    def protocolWork(self):
        """
        Store the accumulated protocolWork from the NCMC simulation as property.
        """
        return self._ncfile.variables['protocolWork'][:]

    def add_protocolWork(self, stuff):
        """ Adds the time to the current frame of the NetCDF file

        Parameters
        ----------
        stuff : float or time-dimension Quantity
            The time to add to the current frame
        """
        #if u.is_quantity(stuff): stuff = stuff.value_in_unit(u.picoseconds)
        self._ncfile.variables['protocolWork'][self._last_protocolWork_frame] = float(stuff)
        self._last_protocolWork_frame += 1
        self.flush()

    @property
    def alchemicalLambda(self):
        """
        Store the current alchemicalLambda (0->1.0) from the NCMC simulation as property.
        """
        return self._ncfile.variables['alchemicalLambda'][:]

    def add_alchemicalLambda(self, stuff):
        """ Adds the time to the current frame of the NetCDF file

        Parameters
        ----------
        stuff : float or time-dimension Quantity
            The time to add to the current frame
        """
        #if u.is_quantity(stuff): stuff = stuff.value_in_unit(u.picoseconds)
        self._ncfile.variables['alchemicalLambda'][self._last_alchemicalLambda_frame] = float(stuff)
        self._last_alchemicalLambda_frame += 1
        self.flush()
