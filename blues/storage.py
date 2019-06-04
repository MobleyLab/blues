import logging
import sys
import time

import numpy as np
import parmed
import simtk.unit as unit
from mdtraj.reporters import HDF5Reporter
from mdtraj.utils import unitcell
from parmed import unit as u
from parmed.geometry import box_vectors_to_lengths_and_angles
from simtk.openmm import app
from simtk import openmm
import math

import blues._version
from blues.formats import *
VELUNIT = u.angstrom / u.picosecond
FRCUNIT = u.kilocalorie_per_mole / u.angstrom

def _check_mode(m, modes):
    """
    Check if the file has a read or write mode, otherwise throw an error.
    """
    if m not in modes:
        raise ValueError('This operation is only available when a file ' 'is open in mode="%s".' % m)


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

    Parameters
    ----------
    levelName : str
        The new level name to be added to the `logging` module.
    levelNum : int
        The level number indicated for the logging module.
    methodName : str, default=None
        The method to call on the logging module for the new level name.
        For example if provided 'trace', you would call `logging.trace()`.

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
        logging.warning('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        logging.warning('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        logging.warning('{} already defined in logger class'.format(methodName))

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


def init_logger(logger, level=logging.INFO, stream=True, outfname=time.strftime("blues-%Y%m%d-%H%M%S")):
    """Initialize the Logger module with the given logger_level and outfname.

    Parameters
    ----------
    logger : logging.getLogger()
        The root logger object if it has been created already.
    level : logging.<LEVEL>
        Valid options for <LEVEL> would be DEBUG, INFO, warningING, ERROR, CRITICAL.
    stream : bool, default = True
        If True, the logger will also stream information to sys.stdout as well
        as the output file.
    outfname : str, default = time.strftime("blues-%Y%m%d-%H%M%S")
        The output file path prefix to store the logged data. This will always
        write to a file with the extension `.log`.

    Returns
    -------
    logger : logging.getLogger()
        The logging object with additional Handlers added.
    """
    fmt = LoggerFormatter()

    if stream:
        # Stream to terminal
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(fmt)
        logger.addHandler(stdout_handler)

    # Write to File
    if outfname:
        fh = logging.FileHandler(outfname + '.log')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.addHandler(logging.NullHandler())
    logger.setLevel(level)

    return logger

######################
#     REPORTERS      #
######################


class NetCDF4Storage(parmed.openmm.reporters.NetCDFReporter):
    """
    Class to read or write NetCDF trajectory files
    Inherited from `parmed.openmm.reporters.NetCDFReporter`

    Parameters
    ----------
    file : str
        Name of the file to write the trajectory to
    reportInterval : int
        How frequently to write a frame to the trajectory
    frame_indices : list, frame numbers for writing the trajectory
        If this reporter is used for the NCMC simulation,
        0.5 will report at the moveStep and -1 will record at the last frame.
    crds : bool=True
        Should we write coordinates to this trajectory? (Default True)
    vels : bool=False
        Should we write velocities to this trajectory? (Default False)
    frcs : bool=False
        Should we write forces to this trajectory? (Default False)
    protocolWork : bool=False,
        Write the protocolWork for the alchemical process in the NCMC simulation
    alchemicalLambda : bool=False,
        Write the alchemicalLambda step for the alchemical process in the NCMC simulation.
    """

    def __init__(self,
                 file,
                 reportInterval=1,
                 frame_indices=[],
                 crds=True,
                 vels=False,
                 frcs=False,
                 protocolWork=False,
                 alchemicalLambda=False):
        """
        Create a NetCDFReporter instance.
        """
        super(NetCDF4Storage, self).__init__(file, reportInterval, crds, vels, frcs)
        self.crds, self.vels, self.frcs, self.protocolWork, self.alchemicalLambda = crds, vels, frcs, protocolWork, alchemicalLambda
        self.frame_indices = frame_indices
        if self.frame_indices:
            #If simulation.currentStep = 1, store the frame from the previous step.
            # i.e. frame_indices=[1,100] will store the first and frame 100
            self.frame_indices = [x - 1 for x in frame_indices]

    def describeNextReport(self, context_state):
        """
        Get information about the next report this object will generate.

        Parameters
        ----------
        context_state : :class:`openmm.State`
            The current state of the context

        Returns
        -------
        nsteps, pos, vel, frc, ene : int, bool, bool, bool, bool
            nsteps is the number of steps until the next report
            pos, vel, frc, and ene are flags indicating whether positions,
            velocities, forces, and/or energies are needed from the Context
        """
        #Monkeypatch to report at certain frame indices
        if self.frame_indices:
            if context_state.currentStep in self.frame_indices:
                steps = 1
            else:
                steps = -1
        if not self.frame_indices:
            steps_left = context_state.currentStep % self._reportInterval
            steps = self._reportInterval - steps_left
        return (steps, self.crds, self.vels, self.frcs, False)

    def report(self, context_state, integrator):
        """Generate a report.

        Parameters
        ----------
        context_state : :class:`openmm.State`
            The current state of the context
        integrator : :class:`openmm.Integrator`
            The integrator belonging to the given context

        """
        global VELUNIT, FRCUNIT

        if self.crds:
            crds = context_state.getPositions().value_in_unit(u.angstrom)
        if self.vels:
            vels = context_state.getVelocities().value_in_unit(VELUNIT)
        if self.frcs:
            frcs = context_state.getForces().value_in_unit(FRCUNIT)
        if self.protocolWork:
            protocolWork = integrator.get_protocol_work(dimensionless=True)
        if self.alchemicalLambda:
            alchemicalLambda = integrator.getGlobalVariableByName('lambda')
        if self._out is None:
            # This must be the first frame, so set up the trajectory now
            if self.crds:
                atom = len(crds)
            elif self.vels:
                atom = len(vels)
            elif self.frcs:
                atom = len(frcs)
            self.uses_pbc = context_state.getPeriodicBoxVectors() is not None
            self._out = NetCDF4Traj.open_new(
                self.fname,
                atom,
                self.uses_pbc,
                self.crds,
                self.vels,
                self.frcs,
                title="ParmEd-created trajectory using OpenMM",
                protocolWork=self.protocolWork,
                alchemicalLambda=self.alchemicalLambda,
            )

        if self.uses_pbc:
            vecs = context_state.getPeriodicBoxVectors()
            lengths, angles = box_vectors_to_lengths_and_angles(*vecs)
            self._out.add_cell_lengths_angles(lengths.value_in_unit(u.angstrom), angles.value_in_unit(u.degree))

        # Add the coordinates, velocities, and/or forces as needed
        if self.crds:
            self._out.add_coordinates(crds)
        if self.vels:
            # The velocities get scaled right before writing
            self._out.add_velocities(vels)
        if self.frcs:
            self._out.add_forces(frcs)
        if self.protocolWork:
            self._out.add_protocolWork(protocolWork)
        if self.alchemicalLambda:
            self._out.add_alchemicalLambda(alchemicalLambda)
        # Now it's time to add the time.
        self._out.add_time(context_state.getTime().value_in_unit(u.picosecond))


class BLUESStateDataStorage(app.StateDataReporter):
    """StateDataReporter outputs information about a simulation, such as energy and temperature, to a file. To use it, create a StateDataReporter, then add it to the Simulation's list of reporters.  The set of data to write is configurable using boolean flags passed to the constructor.  By default the data is written in comma-separated-value (CSV) format, but you can specify a different separator to use. Inherited from `openmm.app.StateDataReporter`

    Parameters
    ----------
    file : string or file
        The file to write to, specified as a file name or file-like object (Logger)
    reportInterval : int
        The interval (in time steps) at which to write frames
    frame_indices : list, frame numbers for writing the trajectory
    title : str,
        Text prefix for each line of the report. Used to distinguish
        between the NCMC and MD simulation reports.
    step : bool=False
        Whether to write the current step index to the file
    time : bool=False
        Whether to write the current time to the file
    potentialEnergy : bool=False
        Whether to write the potential energy to the file
    kineticEnergy : bool=False
        Whether to write the kinetic energy to the file
    totalEnergy : bool=False
        Whether to write the total energy to the file
    temperature : bool=False
        Whether to write the instantaneous temperature to the file
    volume : bool=False
        Whether to write the periodic box volume to the file
    density : bool=False
        Whether to write the system density to the file
    progress : bool=False
        Whether to write current progress (percent completion) to the file.
        If this is True, you must also specify totalSteps.
    remainingTime : bool=False
        Whether to write an estimate of the remaining clock time until
        completion to the file.  If this is True, you must also specify
        totalSteps.
    speed : bool=False
        Whether to write an estimate of the simulation speed in ns/day to
        the file
    elapsedTime : bool=False
        Whether to write the elapsed time of the simulation in seconds to
        the file.
    separator : string=','
        The separator to use between columns in the file
    systemMass : mass=None
        The total mass to use for the system when reporting density.  If
        this is None (the default), the system mass is computed by summing
        the masses of all particles.  This parameter is useful when the
        particle masses do not reflect their actual physical mass, such as
        when some particles have had their masses set to 0 to immobilize
        them.
    totalSteps : int=None
        The total number of steps that will be included in the simulation.
        This is required if either progress or remainingTime is set to True,
        and defines how many steps will indicate 100% completion.
    protocolWork : bool=False,
        Write the protocolWork for the alchemical process in the NCMC simulation
    alchemicalLambda : bool=False,
        Write the alchemicalLambda step for the alchemical process in the NCMC simulation.

    """

    def __init__(self,
                 file,
                 reportInterval=1,
                 frame_indices=[],
                 title='',
                 step=False,
                 time=False,
                 potentialEnergy=False,
                 kineticEnergy=False,
                 totalEnergy=False,
                 temperature=False,
                 volume=False,
                 density=False,
                 progress=False,
                 remainingTime=False,
                 speed=False,
                 elapsedTime=False,
                 separator='\t',
                 systemMass=None,
                 totalSteps=None,
                 protocolWork=False,
                 alchemicalLambda=False,
                 currentIter=False):
        super(BLUESStateDataStorage, self).__init__(file, reportInterval, step, time, potentialEnergy, kineticEnergy,
                                                    totalEnergy, temperature, volume, density, progress, remainingTime,
                                                    speed, elapsedTime, separator, systemMass, totalSteps)
        self.log = self._out
        self.title = title

        self.frame_indices = frame_indices
        self._protocolWork, self._alchemicalLambda, self._currentIter = protocolWork, alchemicalLambda, currentIter
        if self.frame_indices:
            #If simulation.currentStep = 1, store the frame from the previous step.
            # i.e. frame_indices=[1,100] will store the first and frame 100
            self.frame_indices = [x - 1 for x in frame_indices]

    def describeNextReport(self, context_state):
        """
        Get information about the next report this object will generate.

        Parameters
        ----------
        context_state : :class:`openmm.State`
            The current state of the context

        Returns
        -------
        nsteps, pos, vel, frc, ene : int, bool, bool, bool, bool
            nsteps is the number of steps until the next report
            pos, vel, frc, and ene are flags indicating whether positions,
            velocities, forces, and/or energies are needed from the Context

        """
        #Monkeypatch to report at certain frame indices
        if self.frame_indices:
            if context_state.currentStep in self.frame_indices:
                steps = 1
            else:
                steps = -1
        if not self.frame_indices:
            steps_left = context_state.currentStep % self._reportInterval
            steps = self._reportInterval - steps_left

        return (steps, self._needsPositions, self._needsVelocities, self._needsForces, self._needEnergy)

    def _initializeConstants(self, context_state):
        """Initialize a set of constants required for the reports

        Parameters
        ----------
        context_state : :class:`openmm.State`
            The current state of the context
        """
        system = context_state.system
        if self._temperature:
            # Compute the number of degrees of freedom.
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0 * unit.dalton:
                    dof += 3
            dof -= system.getNumConstraints()
            if any(type(system.getForce(i)) == openmm.CMMotionRemover for i in range(system.getNumForces())):
                dof -= 3
            self._dof = dof
        if self._density:
            if self._totalMass is None:
                # Compute the total system mass.
                self._totalMass = 0 * unit.dalton
                for i in range(system.getNumParticles()):
                    self._totalMass += system.getParticleMass(i)
            elif not unit.is_quantity(self._totalMass):
                self._totalMass = self._totalMass * unit.dalton

    def report(self, context_state, integrator):
        """Generate a report.

        Parameters
        ----------
        context_state : :class:`openmm.State`
            The current state of the context
        integrator : :class:`openmm.Integrator`
            The integrator belonging to the given context
        """
        if not self._hasInitialized:
            self._initializeConstants(context_state)
            headers = self._constructHeaders()
            if hasattr(self.log, 'report'):
                self.log.info = self.log.report
            self.log.info('#"%s"' % ('"' + self._separator + '"').join(headers))
            try:
                self._out.flush()
            except AttributeError:
                pass
            self._initialClockTime = time.time()
            self._initialSimulationTime = context_state.getTime()
            self._initialSteps = context_state.currentStep
            self._hasInitialized = True

        # Check for errors.
        self._checkForErrors(context_state, integrator)
        # Query for the values
        values = self._constructReportValues(context_state, integrator)

        # Write the values.
        if hasattr(self.log, 'report'):
            self.log.info = self.log.report
        self.log.info('%s: %s' % (self.title, self._separator.join(str(v) for v in values)))
        try:
            self._out.flush()
        except AttributeError:
            pass

    def _checkForErrors(self, context_state, integrator):
        """Check for errors in the current state of the context

        Parameters
        ----------
        context_state : :class:`openmm.State`
            The current state of the context
        integrator : :class:`openmm.Integrator`
            The integrator belonging to the given context
        """
        if self._needEnergy:
            energy = (context_state.getKineticEnergy() + context_state.getPotentialEnergy()).value_in_unit(
                unit.kilojoules_per_mole)
            if math.isnan(energy):
                raise ValueError('Energy is NaN')
            if math.isinf(energy):
                raise ValueError('Energy is infinite')

    def _constructReportValues(self, context_state, integrator):
        """Query the contextfor the current state of our observables of interest.

        Parameters
        ----------
        context_state : :class:`openmm.State`
            The current state of the context
        integrator : :class:`openmm.Integrator`
            The integrator belonging to the given context

        Returns
        -------
        values : list
            A list of values summarizing the current state of the simulation,
            to be printed or saved. Each element in the list corresponds to one
            of the columns in the resulting CSV file.
        """
        values = []
        box = context_state.getPeriodicBoxVectors()
        volume = box[0][0] * box[1][1] * box[2][2]
        clockTime = time.time()
        # if self._currentIter:
        #     if not hasattr(simulation, 'currentIter'):
        #         simulation.currentIter = 0
        #     values.append(simulation.currentIter)
        if self._progress:
            values.append('%.1f%%' % (100.0 * context_state.currentStep / self._totalSteps))
        if self._step:
            values.append(context_state.currentStep)
        if self._time:
            values.append(context_state.getTime().value_in_unit(unit.picosecond))
        #add a portion like this to store things other than the protocol work
        if self._alchemicalLambda:
            alchemicalLambda = integrator.getGlobalVariableByName('lambda')
            values.append(alchemicalLambda)
        if self._protocolWork:
            protocolWork = integrator.get_protocol_work(dimensionless=True)
            values.append(protocolWork)
        if self._potentialEnergy:
            values.append(context_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._kineticEnergy:
            values.append(context_state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._totalEnergy:
            values.append((context_state.getKineticEnergy() + context_state.getPotentialEnergy()).value_in_unit(
                unit.kilojoules_per_mole))
        if self._temperature:
            values.append(
                (2 * context_state.getKineticEnergy() / (self._dof * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(
                    unit.kelvin))
        if self._volume:
            values.append(volume.value_in_unit(unit.nanometer**3))
        if self._density:
            values.append((self._totalMass / volume).value_in_unit(unit.gram / unit.item / unit.milliliter))

        if self._speed:
            elapsedDays = (clockTime - self._initialClockTime) / 86400.0
            elapsedNs = (context_state.getTime() - self._initialSimulationTime).value_in_unit(unit.nanosecond)
            if elapsedDays > 0.0:
                values.append('%.3g' % (elapsedNs / elapsedDays))
            else:
                values.append('--')
        if self._elapsedTime:
            values.append(time.time() - self._initialClockTime)
        if self._remainingTime:
            elapsedSeconds = clockTime - self._initialClockTime
            elapsedSteps = context_state.currentStep - self._initialSteps
            if elapsedSteps == 0:
                value = '--'
            else:
                estimatedTotalSeconds = (self._totalSteps - self._initialSteps) * elapsedSeconds / elapsedSteps
                remainingSeconds = int(estimatedTotalSeconds - elapsedSeconds)
                remainingDays = remainingSeconds // 86400
                remainingSeconds -= remainingDays * 86400
                remainingHours = remainingSeconds // 3600
                remainingSeconds -= remainingHours * 3600
                remainingMinutes = remainingSeconds // 60
                remainingSeconds -= remainingMinutes * 60
                if remainingDays > 0:
                    value = "%d:%d:%02d:%02d" % (remainingDays, remainingHours, remainingMinutes, remainingSeconds)
                elif remainingHours > 0:
                    value = "%d:%02d:%02d" % (remainingHours, remainingMinutes, remainingSeconds)
                elif remainingMinutes > 0:
                    value = "%d:%02d" % (remainingMinutes, remainingSeconds)
                else:
                    value = "0:%02d" % remainingSeconds
            values.append(value)
        return values

    def _constructHeaders(self):
        """Construct the headers for the CSV output

        Returns
        -------
        headers : list
            a list of strings giving the title of each observable being reported on.
        """
        headers = []
        # if self._currentIter:
        #     headers.append('Iter')
        if self._progress:
            headers.append('Progress (%)')
        if self._step:
            headers.append('Step')
        if self._time:
            headers.append('Time (ps)')
        if self._alchemicalLambda:
            headers.append('alchemicalLambda')
        if self._protocolWork:
            headers.append('protocolWork')
        if self._potentialEnergy:
            headers.append('Potential Energy (kJ/mole)')
        if self._kineticEnergy:
            headers.append('Kinetic Energy (kJ/mole)')
        if self._totalEnergy:
            headers.append('Total Energy (kJ/mole)')
        if self._temperature:
            headers.append('Temperature (K)')
        if self._volume:
            headers.append('Box Volume (nm^3)')
        if self._density:
            headers.append('Density (g/mL)')
        if self._speed:
            headers.append('Speed (ns/day)')
        if self._elapsedTime:
            headers.append('Elapsed Time (s)')
        if self._remainingTime:
            headers.append('Time Remaining')
        return headers
