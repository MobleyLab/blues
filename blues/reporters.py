from mdtraj.formats.hdf5 import HDF5TrajectoryFile
from mdtraj.reporters import HDF5Reporter
from simtk.openmm.app import StateDataReporter
import simtk.unit as units
import json, yaml
import subprocess
import numpy as np
from mdtraj.utils import unitcell
from mdtraj.utils import in_units_of, ensure_type

import mdtraj.version
import simtk.openmm.version
import blues.version
import logging
import sys, time

def _check_mode(m, modes):
    if m not in modes:
        raise ValueError('This operation is only available when a file '
                         'is open in mode="%s".' % m)

def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

def init_logger(level=logging.INFO, modname=None, outfname=None):
    if not modname: modname = '__name__'
    logger = logging.getLogger(modname)

    addLoggingLevel('REPORT', logging.INFO - 5)
    fmt = MyFormatter()

    # Stream to terminal
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    # Write to File
    if outfname:
        fh = logging.FileHandler(outfname+'.log')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.addHandler(logging.NullHandler())
    logger.setLevel(level)

    return logger

class BLUESHDF5TrajectoryFile(HDF5TrajectoryFile):
    #This is a subclass of the HDF5TrajectoryFile class from mdtraj that handles the writing of
    #the trajectory information to the HDF5 file format.
    def __init__(self, filename, mode='r', force_overwrite=True, compression='zlib'):
        super(BLUESHDF5TrajectoryFile, self).__init__(filename, mode, force_overwrite, compression)


    def write(self, coordinates, parameters=None, environment=None,
                    time=None, cell_lengths=None, cell_angles=None,
                    velocities=None, kineticEnergy=None, potentialEnergy=None,
                    temperature=None, alchemicalLambda=None,
                    protocolWork=None, title=None):
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
            You may optionally specify the alchemical lambda in each frame. These
            have no units, but are generally between zero and one.
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
        potentialEnergy = in_units_of(potentialEnergy, None, 'kilojoules_per_mole')
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
        coordinates = ensure_type(coordinates, dtype=np.float32, ndim=3,
            name='coordinates', shape=(None, None, 3), can_be_none=False,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        n_frames, n_atoms, = coordinates.shape[0:2]
        time = ensure_type(time, dtype=np.float32, ndim=1,
            name='time', shape=(n_frames,), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        cell_lengths = ensure_type(cell_lengths, dtype=np.float32, ndim=2,
            name='cell_lengths', shape=(n_frames, 3), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        cell_angles = ensure_type(cell_angles, dtype=np.float32, ndim=2,
            name='cell_angles', shape=(n_frames, 3), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        velocities = ensure_type(velocities, dtype=np.float32, ndim=3,
            name='velocoties', shape=(n_frames, n_atoms, 3), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        kineticEnergy = ensure_type(kineticEnergy, dtype=np.float32, ndim=1,
            name='kineticEnergy', shape=(n_frames,), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        potentialEnergy = ensure_type(potentialEnergy, dtype=np.float32, ndim=1,
            name='potentialEnergy', shape=(n_frames,), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        temperature = ensure_type(temperature, dtype=np.float32, ndim=1,
            name='temperature', shape=(n_frames,), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        alchemicalLambda = ensure_type(alchemicalLambda, dtype=np.float32, ndim=1,
            name='alchemicalLambda', shape=(n_frames,), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        protocolWork = ensure_type(protocolWork, dtype=np.float32, ndim=1,
            name='protocolWork', shape=(n_frames,), can_be_none=True,
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)

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
            for name in ['coordinates', 'time', 'cell_angles', 'cell_lengths',
                         'velocities', 'kineticEnergy', 'potentialEnergy', 'temperature', 'protocolWork', 'alchemicalLambda']:
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
            raise ValueError("The file that you're trying to save to doesn't "
                "contain the field %s. You can always save a new trajectory "
                "and have it contain this information, but I don't allow 'ragged' "
                "arrays. If one frame is going to have %s information, then I expect "
                "all of them to. So I can't save it for just these frames. Sorry "
                "about that :)" % (name, name))
        except AssertionError:
            raise ValueError("The file that you're saving to expects each frame "
                            "to contain %s information, but you did not supply it."
                            "I don't allow 'ragged' arrays. If one frame is going "
                            "to have %s information, then I expect all of them to. "
                            % (name, name))

        self._frame_index += n_frames
        self.flush()

    def _encodeStringForPyTables(self, string, name, where='/', complevel=1, complib='zlib', shuffle=True):
        bytestring = np.fromstring(string.encode('utf-8'),np.uint8)
        atom = self.tables.UInt8Atom()
        filters = self.tables.Filters(complevel,complib, shuffle)
        if self.tables.__version__ >= '3.0.0':
            self._handle.create_carray(where=where, name=name, obj=bytestring,
                                       atom=atom, filters=filters)
        else:
            self._handle.createCArray(where=where, name=name, obj=bytestring,
                                       atom=atom, filters=filters)

    def _initialize_headers(self, n_atoms, title, parameters, set_environment,
                            set_coordinates, set_time, set_cell,
                            set_velocities, set_kineticEnergy, set_potentialEnergy,
                            set_temperature, set_alchemicalLambda, set_protocolWork):
        self._n_atoms = n_atoms
        self._parameters = parameters
        self._handle.root._v_attrs.title = str(title)
        self._handle.root._v_attrs.conventions = str('Pande')
        self._handle.root._v_attrs.conventionVersion = str('1.1')
        self._handle.root._v_attrs.program = str('MDTraj')
        self._handle.root._v_attrs.programVersion = str(mdtraj.version.full_version)
        self._handle.root._v_attrs.method = str('BLUES')
        self._handle.root._v_attrs.methodVersion = str(blues.version.full_version)
        self._handle.root._v_attrs.reference = str('DOI: 10.1021/acs.jpcb.7b11820')

        if not hasattr(self._handle.root._v_attrs, 'application'):
            self._handle.root._v_attrs.application = str('OpenMM')
            self._handle.root._v_attrs.applicationVersion = str(simtk.openmm.version.full_version)

        # create arrays that store frame level informat
        if set_coordinates:
            self._create_earray(where='/', name='coordinates',
                atom=self.tables.Float32Atom(), shape=(0, self._n_atoms, 3))
            self._handle.root.coordinates.attrs['units'] = str('nanometers')

        if set_time:
            self._create_earray(where='/', name='time',
                atom=self.tables.Float32Atom(), shape=(0,))
            self._handle.root.time.attrs['units'] = str('picoseconds')

        if set_cell:
            self._create_earray(where='/', name='cell_lengths',
                atom=self.tables.Float32Atom(), shape=(0, 3))
            self._create_earray(where='/', name='cell_angles',
                atom=self.tables.Float32Atom(), shape=(0, 3))
            self._handle.root.cell_lengths.attrs['units'] = str('nanometers')
            self._handle.root.cell_angles.attrs['units'] = str('degrees')

        if set_velocities:
            self._create_earray(where='/', name='velocities',
                atom=self.tables.Float32Atom(), shape=(0, self._n_atoms, 3))
            self._handle.root.velocities.attrs['units'] = str('nanometers/picosecond')

        if set_kineticEnergy:
            self._create_earray(where='/', name='kineticEnergy',
                atom=self.tables.Float32Atom(), shape=(0,))
            self._handle.root.kineticEnergy.attrs['units'] = str('kilojoules_per_mole')

        if set_potentialEnergy:
            self._create_earray(where='/', name='potentialEnergy',
                atom=self.tables.Float32Atom(), shape=(0,))
            self._handle.root.potentialEnergy.attrs['units'] = str('kilojoules_per_mole')

        if set_temperature:
            self._create_earray(where='/', name='temperature',
                atom=self.tables.Float32Atom(), shape=(0,))
            self._handle.root.temperature.attrs['units'] = str('kelvin')

        #Add another portion akin to this if you want to store more data in the h5 file
        if set_alchemicalLambda:
            self._create_earray(where='/', name='alchemicalLambda',
                atom=self.tables.Float32Atom(), shape=(0,))
            self._handle.root.alchemicalLambda.attrs['units'] = str('dimensionless')

        if set_protocolWork:
            self._create_earray(where='/', name='protocolWork',
                atom=self.tables.Float32Atom(), shape=(0,))
            self._handle.root.protocolWork.attrs['units'] = str('kT')

        if parameters:
            if 'Logger' in self._parameters: self._parameters.pop('Logger')
            paramjson = json.dumps(self._parameters)
            self._encodeStringForPyTables(string=paramjson, name='parameters')

        if set_environment:
            try:
                envout = subprocess.check_output('conda env export --no-builds', shell=True, stderr=subprocess.STDOUT)
                envjson = json.dumps(yaml.load(envout), sort_keys=True, indent=2)
                self._encodeStringForPyTables(envjson, name='environment')
            except Exception as e:
                print(e)
                pass

class BLUESHDF5Reporter(HDF5Reporter):
    #This is a subclass of the HDF5 class from mdtraj that handles the reporting of
    #the trajectory.
    @property
    def backend(self):
        return BLUESHDF5TrajectoryFile

    def __init__(self, file, reportInterval,
                 title='NCMC Trajectory',
                 coordinates=True, frame_indices=None,
                 time=False, cell=True, temperature=False,
                 potentialEnergy=False, kineticEnergy=False,
                 velocities=False, atomSubset=None,
                 protocolWork=True, alchemicalLambda=True,
                 parameters=None, environment=True):

        super(BLUESHDF5Reporter, self).__init__(file, reportInterval,
            coordinates, time, cell, potentialEnergy, kineticEnergy,
            temperature, velocities, atomSubset)

        self._protocolWork = bool(protocolWork)
        self._alchemicalLambda = bool(alchemicalLambda)
        self._frame_indices = frame_indices
        self._environment = bool(environment)
        self._title = title
        self._parameters = parameters

    def report(self, simulation, state):
        """Generate a report.
        Parameters
        ----------
        simulation : simtk.openmm.app.Simulation
            The Simulation to generate a report for
        state : simtk.openmm.State
            The current state of the simulation
        """
        if not self._is_intialized:
            self._initialize(simulation)
            self._is_intialized = True

        self._checkForErrors(simulation, state)

        args = ()
        kwargs = {}
        if self._coordinates:
            coordinates = state.getPositions(asNumpy=True)[self._atomSlice]
            coordinates = coordinates.value_in_unit(getattr(units, self._traj_file.distance_unit))
            if self._frame_indices:
                if simulation.currentStep not in self._frame_indices:
                    coordinates = np.zeros(coordinates.shape)
            args = (coordinates,)
        if self._time:
            kwargs['time'] = state.getTime()
        if self._cell:
            vectors = state.getPeriodicBoxVectors(asNumpy=True)
            vectors = vectors.value_in_unit(getattr(units, self._traj_file.distance_unit))
            a, b, c, alpha, beta, gamma = unitcell.box_vectors_to_lengths_and_angles(*vectors)
            kwargs['cell_lengths'] = np.array([a, b, c])
            kwargs['cell_angles'] = np.array([alpha, beta, gamma])
        if self._potentialEnergy:
            kwargs['potentialEnergy'] = state.getPotentialEnergy()
        if self._kineticEnergy:
            kwargs['kineticEnergy'] = state.getKineticEnergy()
        if self._temperature:
            kwargs['temperature'] = 2*state.getKineticEnergy()/(self._dof*units.MOLAR_GAS_CONSTANT_R)
        if self._velocities:
            kwargs['velocities'] = state.getVelocities(asNumpy=True)[self._atomSlice, :]

        #add a portion like this to store things other than the protocol work
        if self._protocolWork:
            protocol_work = simulation.integrator.get_protocol_work(dimensionless=True)
            kwargs['protocolWork'] = np.array([protocol_work])
        if self._alchemicalLambda:
            kwargs['alchemicalLambda'] = np.array([simulation.integrator.getGlobalVariableByName('lambda')])
        if self._title:
            kwargs['title'] = self._title
        if self._parameters:
            kwargs['parameters'] = self._parameters
        if self._environment:
            kwargs['environment'] = self._environment

        self._traj_file.write(*args, **kwargs)
        # flush the file to disk. it might not be necessary to do this every
        # report, but this is the most proactive solution. We don't want to
        # accumulate a lot of data in memory only to find out, at the very
        # end of the run, that there wasn't enough space on disk to hold the
        # data.
        if hasattr(self._traj_file, 'flush'):
            self._traj_file.flush()

# Custom formatter
class MyFormatter(logging.Formatter):

    err_fmt  = "(%(asctime)s) %(levelname)s: [%(module)s.%(funcName)s] %(message)s"
    dbg_fmt  = "%(levelname)s: [%(module)s.%(funcName)s] %(message)s"
    info_fmt = "%(levelname)s: %(message)s"
    rep_fmt = "%(message)s"

    def __init__(self):
        super().__init__(fmt="%(levelname)s: %(msg)s", datefmt="%H:%M:%S", style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = MyFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.WARNING:
            self._style._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = MyFormatter.err_fmt

        elif record.levelno == logging.REPORT:
            self._style._fmt = MyFormatter.rep_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result

class BLUESStateDataReporter(StateDataReporter):
    def __init__(self, file,  reportInterval, title='', step=False, time=False, potentialEnergy=False, kineticEnergy=False, totalEnergy=False,   temperature=False, volume=False, density=False,
    progress=False, remainingTime=False, speed=False, elapsedTime=False, separator=',', systemMass=None, totalSteps=None):
        super(BLUESStateDataReporter, self).__init__(file, reportInterval, step, time,
            potentialEnergy, kineticEnergy, totalEnergy, temperature, volume, density,
            progress, remainingTime, speed, elapsedTime, separator, systemMass, totalSteps)
        self.log = self._out
        self.log.setLevel(logging.REPORT)
        self.title = title

    def report(self, simulation, state):
        """Generate a report.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._hasInitialized:
            self._initializeConstants(simulation)
            headers = self._constructHeaders()
            self.log.report('#"%s"' % ('"'+self._separator+'"').join(headers))
            try:
                self._out.flush()
            except AttributeError:
                pass
            self._initialClockTime = time.time()
            self._initialSimulationTime = state.getTime()
            self._initialSteps = simulation.currentStep
            self._hasInitialized = True

        # Check for errors.
        self._checkForErrors(simulation, state)

        # Query for the values
        values = self._constructReportValues(simulation, state)

        # Write the values.
        self.log.report('%s: %s' % (self.title, self._separator.join(str(v) for v in values)))
        try:
            self._out.flush()
        except AttributeError:
            pass
