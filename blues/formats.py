from mdtraj.formats.hdf5 import HDF5TrajectoryFile
import json, yaml
import subprocess
import numpy as np
from mdtraj.utils import in_units_of, ensure_type
import mdtraj.version
import simtk.openmm.version
import blues.version
import logging
import parmed
import netCDF4 as nc
from blues import reporters
from parmed.amber.netcdffiles import NetCDFTraj


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
        super().__init__(
            fmt="%(levelname)s: %(msg)s", datefmt="%H:%M:%S", style='%')
        reporters.addLoggingLevel('REPORT', logging.WARNING - 5)

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


class BLUESHDF5TrajectoryFile(HDF5TrajectoryFile):
    """
    Extension of the `mdtraj.formats.hdf5.HDF5TrajectoryFile` class which
    handles the writing of the trajectory data to the HDF5 file format.
    Additional features include writing NCMC related data to the HDF5 file.

    Parameters
    ----------
    filename : str
        The filename for the HDF5 file.
    mode : str, default='r'
        The mode to open the HDF5 file in.
    force_overwrite : bool, default=True
        If True, overwrite the file if it already exists
    compression : str, default='zlib'
        Valid choices are ['zlib', 'lzo', 'bzip2', 'blosc']

    """

    def __init__(self,
                 filename,
                 mode='r',
                 force_overwrite=True,
                 compression='zlib'):
        super(BLUESHDF5TrajectoryFile,
              self).__init__(filename, mode, force_overwrite, compression)

    def write(self,
              coordinates,
              parameters=None,
              environment=None,
              time=None,
              cell_lengths=None,
              cell_angles=None,
              velocities=None,
              kineticEnergy=None,
              potentialEnergy=None,
              temperature=None,
              alchemicalLambda=None,
              protocolWork=None,
              title=None):
        """Write one or more frames of data to the file
        This method saves data that is associated with one or more simulation
        frames. Note that all of the arguments can either be raw numpy arrays
        or unitted arrays (with simtk.unit.Quantity). If the arrays are unittted,
        a unit conversion will be automatically done from the supplied units
        into the proper units for saving on disk. You won't have to worry about
        it.

        Furthermore, if you wish to save a single frame of simulation data, you
        can do so naturally, for instance by supplying a 2d array for the
        coordinates and a single float for the time. This "shape deficiency"
        will be recognized, and handled appropriately.

        Parameters
        ----------
        coordinates : np.ndarray, shape=(n_frames, n_atoms, 3)
            The cartesian coordinates of the atoms to write. By convention, the
            lengths should be in units of nanometers.
        time : np.ndarray, shape=(n_frames,), optional
            You may optionally specify the simulation time, in picoseconds
            corresponding to each frame.
        cell_lengths : np.ndarray, shape=(n_frames, 3), dtype=float32, optional
            You may optionally specify the unitcell lengths.
            The length of the periodic box in each frame, in each direction,
            `a`, `b`, `c`. By convention the lengths should be in units
            of angstroms.
        cell_angles : np.ndarray, shape=(n_frames, 3), dtype=float32, optional
            You may optionally specify the unitcell angles in each frame.
            Organized analogously to cell_lengths. Gives the alpha, beta and
            gamma angles respectively. By convention, the angles should be
            in units of degrees.
        velocities :  np.ndarray, shape=(n_frames, n_atoms, 3), optional
            You may optionally specify the cartesian components of the velocity
            for each atom in each frame. By convention, the velocities
            should be in units of nanometers / picosecond.
        kineticEnergy : np.ndarray, shape=(n_frames,), optional
            You may optionally specify the kinetic energy in each frame. By
            convention the kinetic energies should b in units of kilojoules per
            mole.
        potentialEnergy : np.ndarray, shape=(n_frames,), optional
            You may optionally specify the potential energy in each frame. By
            convention the kinetic energies should b in units of kilojoules per
            mole.
        temperature : np.ndarray, shape=(n_frames,), optional
            You may optionally specify the temperature in each frame. By
            convention the temperatures should b in units of Kelvin.
        alchemicalLambda : np.ndarray, shape=(n_frames,), optional
            You may optionally specify the alchemicalLambda in each frame. These
            have no units, but are generally between zero and one.
        protocolWork : np.ndarray, shape=(n_frames,), optional
            You may optionally specify the protocolWork in each frame. These
            are in reduced units of kT but are stored dimensionless
        title : str
            Title of the HDF5 trajectory file
        """
        _check_mode(self.mode, ('w', 'a'))

        # these must be either both present or both absent. since
        # we're going to throw an error if one is present w/o the other,
        # lets do it now.
        if cell_lengths is None and cell_angles is not None:
            raise ValueError('cell_lengths were given, but no cell_angles')
        if cell_lengths is not None and cell_angles is None:
            raise ValueError('cell_angles were given, but no cell_lengths')

        # if the input arrays are simtk.unit.Quantities, convert them
        # into md units. Note that this acts as a no-op if the user doesn't
        # have simtk.unit installed (e.g. they didn't install OpenMM)
        coordinates = in_units_of(coordinates, None, 'nanometers')
        time = in_units_of(time, None, 'picoseconds')
        cell_lengths = in_units_of(cell_lengths, None, 'nanometers')
        cell_angles = in_units_of(cell_angles, None, 'degrees')
        velocities = in_units_of(velocities, None, 'nanometers/picosecond')
        kineticEnergy = in_units_of(kineticEnergy, None, 'kilojoules_per_mole')
        potentialEnergy = in_units_of(potentialEnergy, None,
                                      'kilojoules_per_mole')
        temperature = in_units_of(temperature, None, 'kelvin')
        alchemicalLambda = in_units_of(alchemicalLambda, None, 'dimensionless')
        protocolWork = in_units_of(protocolWork, None, 'kT')

        # do typechecking and shapechecking on the arrays
        # this ensure_type method has a lot of options, but basically it lets
        # us validate most aspects of the array. Also, we can upconvert
        # on defficent ndim, which means that if the user sends in a single
        # frame of data (i.e. coordinates is shape=(n_atoms, 3)), we can
        # realize that. obviously the default mode is that they want to
        # write multiple frames at a time, so the coordinate shape is
        # (n_frames, n_atoms, 3)
        coordinates = ensure_type(
            coordinates,
            dtype=np.float32,
            ndim=3,
            name='coordinates',
            shape=(None, None, 3),
            can_be_none=False,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        n_frames, n_atoms, = coordinates.shape[0:2]
        time = ensure_type(
            time,
            dtype=np.float32,
            ndim=1,
            name='time',
            shape=(n_frames, ),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        cell_lengths = ensure_type(
            cell_lengths,
            dtype=np.float32,
            ndim=2,
            name='cell_lengths',
            shape=(n_frames, 3),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        cell_angles = ensure_type(
            cell_angles,
            dtype=np.float32,
            ndim=2,
            name='cell_angles',
            shape=(n_frames, 3),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        velocities = ensure_type(
            velocities,
            dtype=np.float32,
            ndim=3,
            name='velocoties',
            shape=(n_frames, n_atoms, 3),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        kineticEnergy = ensure_type(
            kineticEnergy,
            dtype=np.float32,
            ndim=1,
            name='kineticEnergy',
            shape=(n_frames, ),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        potentialEnergy = ensure_type(
            potentialEnergy,
            dtype=np.float32,
            ndim=1,
            name='potentialEnergy',
            shape=(n_frames, ),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        temperature = ensure_type(
            temperature,
            dtype=np.float32,
            ndim=1,
            name='temperature',
            shape=(n_frames, ),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        alchemicalLambda = ensure_type(
            alchemicalLambda,
            dtype=np.float32,
            ndim=1,
            name='alchemicalLambda',
            shape=(n_frames, ),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)
        protocolWork = ensure_type(
            protocolWork,
            dtype=np.float32,
            ndim=1,
            name='protocolWork',
            shape=(n_frames, ),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True)

        # if this is our first call to write(), we need to create the headers
        # and the arrays in the underlying HDF5 file
        if self._needs_initialization:
            self._initialize_headers(
                n_atoms=n_atoms,
                title=title,
                parameters=parameters,
                set_environment=(environment is not None),
                set_coordinates=True,
                set_time=(time is not None),
                set_cell=(cell_lengths is not None or cell_angles is not None),
                set_velocities=(velocities is not None),
                set_kineticEnergy=(kineticEnergy is not None),
                set_potentialEnergy=(potentialEnergy is not None),
                set_temperature=(temperature is not None),
                set_alchemicalLambda=(alchemicalLambda is not None),
                set_protocolWork=(protocolWork is not None))
            self._needs_initialization = False

            # we need to check that that the entries that the user is trying
            # to save are actually fields in OUR file

        try:
            # try to get the nodes for all of the fields that we have
            # which are not None
            for name in [
                    'coordinates', 'time', 'cell_angles', 'cell_lengths',
                    'velocities', 'kineticEnergy', 'potentialEnergy',
                    'temperature', 'protocolWork', 'alchemicalLambda'
            ]:
                contents = locals()[name]
                if contents is not None:
                    self._get_node(where='/', name=name).append(contents)
                if contents is None:
                    # for each attribute that they're not saving, we want
                    # to make sure the file doesn't explect it
                    try:
                        self._get_node(where='/', name=name)
                        raise AssertionError()
                    except self.tables.NoSuchNodeError:
                        pass

        except self.tables.NoSuchNodeError:
            raise ValueError(
                "The file that you're trying to save to doesn't "
                "contain the field %s. You can always save a new trajectory "
                "and have it contain this information, but I don't allow 'ragged' "
                "arrays. If one frame is going to have %s information, then I expect "
                "all of them to. So I can't save it for just these frames. Sorry "
                "about that :)" % (name, name))
        except AssertionError:
            raise ValueError(
                "The file that you're saving to expects each frame "
                "to contain %s information, but you did not supply it."
                "I don't allow 'ragged' arrays. If one frame is going "
                "to have %s information, then I expect all of them to. " %
                (name, name))

        self._frame_index += n_frames
        self.flush()

    def _encodeStringForPyTables(self,
                                 string,
                                 name,
                                 where='/',
                                 complevel=1,
                                 complib='zlib',
                                 shuffle=True):
        """
        Encode a given string into a character array (PyTables)

        Parameters:
        -----------
        string : str, input string to be encoded
        name : str, title or name of table for character array
        where : str, filepath where character array will be stored
            By default ('/') the character array will be stored at the root level.
        complevel : int, compression level
        complib : str, default='zlib'
            Valid choices are ['zlib', 'lzo', 'bzip2', 'blosc']
        shuffle : bool, default=True
            Whether or not to use the Shuffle filter in the HDF5 library. This is normally used to improve the compression ratio. A false value disables shuffling and a true one enables it. The default value depends on whether compression is enabled or not; if compression is enabled, shuffling defaults to be enabled, else shuffling is disabled. Shuffling can only be used when compression is enabled.

        """
        bytestring = np.fromstring(string.encode('utf-8'), np.uint8)
        atom = self.tables.UInt8Atom()
        filters = self.tables.Filters(complevel, complib, shuffle)
        if self.tables.__version__ >= '3.0.0':
            self._handle.create_carray(
                where=where,
                name=name,
                obj=bytestring,
                atom=atom,
                filters=filters)
        else:
            self._handle.createCArray(
                where=where,
                name=name,
                obj=bytestring,
                atom=atom,
                filters=filters)

    def _initialize_headers(self, n_atoms, title, parameters, set_environment,
                            set_coordinates, set_time, set_cell,
                            set_velocities, set_kineticEnergy,
                            set_potentialEnergy, set_temperature,
                            set_alchemicalLambda, set_protocolWork):
        """
        Function that initializes the tables for storing data from the simulation
        and writes metadata at the root level of the HDF5 file.

        Parameters
        -----------
        n_atoms : int
            Number of atoms in system
        title : str
            Title for the root level data table
        parameters : dict
            Arguments/parameters used for the simulation.
        set_* : bool
            Parameters that begin with set_*. If True, the corresponding data
            will be written to the HDF5 file.

        """
        self._n_atoms = n_atoms
        self._parameters = parameters
        self._handle.root._v_attrs.title = str(title)
        self._handle.root._v_attrs.conventions = str('Pande')
        self._handle.root._v_attrs.conventionVersion = str('1.1')
        self._handle.root._v_attrs.program = str('MDTraj')
        self._handle.root._v_attrs.programVersion = str(
            mdtraj.version.full_version)
        self._handle.root._v_attrs.method = str('BLUES')
        self._handle.root._v_attrs.methodVersion = str(
            blues.version.full_version)
        self._handle.root._v_attrs.reference = str(
            'DOI: 10.1021/acs.jpcb.7b11820')

        if not hasattr(self._handle.root._v_attrs, 'application'):
            self._handle.root._v_attrs.application = str('OpenMM')
            self._handle.root._v_attrs.applicationVersion = str(
                simtk.openmm.version.full_version)

        # create arrays that store frame level informat
        if set_coordinates:
            self._create_earray(
                where='/',
                name='coordinates',
                atom=self.tables.Float32Atom(),
                shape=(0, self._n_atoms, 3))
            self._handle.root.coordinates.attrs['units'] = str('nanometers')

        if set_time:
            self._create_earray(
                where='/',
                name='time',
                atom=self.tables.Float32Atom(),
                shape=(0, ))
            self._handle.root.time.attrs['units'] = str('picoseconds')

        if set_cell:
            self._create_earray(
                where='/',
                name='cell_lengths',
                atom=self.tables.Float32Atom(),
                shape=(0, 3))
            self._create_earray(
                where='/',
                name='cell_angles',
                atom=self.tables.Float32Atom(),
                shape=(0, 3))
            self._handle.root.cell_lengths.attrs['units'] = str('nanometers')
            self._handle.root.cell_angles.attrs['units'] = str('degrees')

        if set_velocities:
            self._create_earray(
                where='/',
                name='velocities',
                atom=self.tables.Float32Atom(),
                shape=(0, self._n_atoms, 3))
            self._handle.root.velocities.attrs['units'] = str(
                'nanometers/picosecond')

        if set_kineticEnergy:
            self._create_earray(
                where='/',
                name='kineticEnergy',
                atom=self.tables.Float32Atom(),
                shape=(0, ))
            self._handle.root.kineticEnergy.attrs['units'] = str(
                'kilojoules_per_mole')

        if set_potentialEnergy:
            self._create_earray(
                where='/',
                name='potentialEnergy',
                atom=self.tables.Float32Atom(),
                shape=(0, ))
            self._handle.root.potentialEnergy.attrs['units'] = str(
                'kilojoules_per_mole')

        if set_temperature:
            self._create_earray(
                where='/',
                name='temperature',
                atom=self.tables.Float32Atom(),
                shape=(0, ))
            self._handle.root.temperature.attrs['units'] = str('kelvin')

        #Add another portion akin to this if you want to store more data in the h5 file
        if set_alchemicalLambda:
            self._create_earray(
                where='/',
                name='alchemicalLambda',
                atom=self.tables.Float32Atom(),
                shape=(0, ))
            self._handle.root.alchemicalLambda.attrs['units'] = str(
                'dimensionless')

        if set_protocolWork:
            self._create_earray(
                where='/',
                name='protocolWork',
                atom=self.tables.Float32Atom(),
                shape=(0, ))
            self._handle.root.protocolWork.attrs['units'] = str('kT')

        if parameters:
            if 'Logger' in self._parameters: self._parameters.pop('Logger')
            paramjson = json.dumps(self._parameters)
            self._encodeStringForPyTables(string=paramjson, name='parameters')

        if set_environment:
            try:
                envout = subprocess.check_output(
                    'conda env export --no-builds',
                    shell=True,
                    stderr=subprocess.STDOUT)
                envjson = json.dumps(
                    yaml.load(envout), sort_keys=True, indent=2)
                self._encodeStringForPyTables(envjson, name='environment')
            except Exception as e:
                print(e)
                pass


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
                    raise ValueError('remd_dimension must be given '
                                     'for multi-D REMD')
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
        v = ncfile.createVariable('spatial', 'c', ('spatial', ))
        v[:] = np.asarray(list('xyz'))
        if inst.hasbox:
            v = ncfile.createVariable('cell_spatial', 'c', ('cell_spatial', ))
            v[:] = np.asarray(list('abc'))
            v = ncfile.createVariable('cell_angular', 'c', (
                'cell_angular',
                'label',
            ))
            v[:] = np.asarray([list('alpha'), list('beta '), list('gamma')])
        v = ncfile.createVariable('time', 'f', ('frame', ))
        v.units = 'picosecond'
        if inst.hascrds:
            v = ncfile.createVariable('coordinates', 'f',
                                      ('frame', 'atom', 'spatial'))
            v.units = 'angstrom'
            inst._last_crd_frame = 0
        if inst.hasvels:
            v = ncfile.createVariable('velocities', 'f',
                                      ('frame', 'atom', 'spatial'))
            v.units = 'angstrom/picosecond'
            inst.velocity_scale = v.scale_factor = 20.455
            inst._last_vel_frame = 0
            if nc is not None:
                v.set_auto_maskandscale(False)
        if inst.hasfrcs:
            v = ncfile.createVariable('forces', 'f',
                                      ('frame', 'atom', 'spatial'))
            v.units = 'kilocalorie/mole/angstrom'
            inst._last_frc_frame = 0
        if inst.hasbox:
            v = ncfile.createVariable('cell_lengths', 'd',
                                      ('frame', 'cell_spatial'))
            v.units = 'angstrom'
            v = ncfile.createVariable('cell_angles', 'd',
                                      ('frame', 'cell_angular'))
            v.units = 'degree'
            inst._last_box_frame = 0
        if inst.remd == 'TEMPERATURE':
            v = ncfile.createVariable('temp0', 'd', ('frame', ))
            v.units = 'kelvin'
            inst._last_remd_frame = 0
        elif inst.remd == 'MULTI':
            ncfile.createVariable('remd_indices', 'i',
                                  ('frame', 'remd_dimension'))
            ncfile.createVariable('remd_dimtype', 'i', ('remd_dimension', ))
            inst._last_remd_frame = 0

        inst._last_time_frame = 0

        if inst.hasprotocolWork:
            v = ncfile.createVariable('protocolWork', 'f', ('frame', ))
            v.units = 'kT'
            inst._last_protocolWork_frame = 0

        if inst.hasalchemicalLambda:
            v = ncfile.createVariable('alchemicalLambda', 'f', ('frame', ))
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
        self._ncfile.variables['protocolWork'][
            self._last_protocolWork_frame] = float(stuff)
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
        self._ncfile.variables['alchemicalLambda'][
            self._last_alchemicalLambda_frame] = float(stuff)
        self._last_alchemicalLambda_frame += 1
        self.flush()
