"""
Looping through batches of jobs
(former submit_job.py and buildali loop)

Authors:
  Benjamin Schubert, Thomas A. Hopf
"""
import abc
import inspect
import os
import signal
import subprocess
import traceback
import uuid
import re
import time
import psutil
import queue

import ruamel.yaml as yaml
import billiard as mp

from tempfile import NamedTemporaryFile
from enum import Enum

from evcouplings.utils import PersistentDict


EStatus = (lambda **enums: type('Enum', (), enums))(
    RUN="run",
    PEND="pend",
    SUSP="susp",
    EXIT="exit",
    DONE="done"
)

EResource = (lambda **enums: type('Enum', (), enums))(
    time="time",
    mem="mem",
    nodes="nodes",
    queue="queue",
    error="error",
    out="done"
)


class EJob(Enum): # Internal messages for broker
    SUBMIT = 0
    MONITOR = 1
    CANCEL = 2
    STOP = 3
    UPDATE = 4
    PID = 5


class Command(object):
    """
    Wrapper around the command parameters needed
    to execute a script
    """

    def __init__(self, command, name=None, environment=None, workdir=None, resources=None):
        """
        Class represents a command to be submitted and contains all relevant information

        Parameters
        ----------
        command: str/list(str)
            A single string or a list of strings representing the fully configured commands to be executed
        name: str
            The name of the command.
        environment: str/list(str)
            A string representing all calls that should be executed before command that configure the environment
            (e.g., export or source commands).
        workdir: str
            Full path or relational path to working dir in which command is executed
        resources: dict(Resource:str)
            A dictionary defining resources that can be used by the job (time, memory,
        """

        self.command_id = "c"+str(uuid.uuid4())
        self.name = name

        self.command = [command] if isinstance(command, str) else command
        if environment is None:
            self.environment = []
        else:
            self.environment = [environment] if isinstance(environment, str) else environment
        self.workdir = workdir
        self.resources = resources

    def __eq__(self, other):
        if not isinstance(other, Command):
            return False
        return self.command_id == other.command_id

    def __str__(self):
        return "Command:{id}:\n\t{commands}".format(id=self.command_id, commands="&".join(self.command)[:16])

    def __repr__(self):
        return "Command({id})".format(id=self.command_id)

    def __hash__(self):
        return hash(self.command_id)


# Metaclass for Plugins
class APluginRegister(abc.ABCMeta):
    """
    This class allows automatic registration of new plugins.
    """

    def __init__(cls, name, bases, nmspc):
        super(APluginRegister, cls).__init__(name, bases, nmspc)

        if not hasattr(cls, 'registry'):
            cls.registry = dict()
        if not inspect.isabstract(cls):
            cls.registry[str(cls().name).lower()] = cls

    def __getitem__(cls, args):
        name = args
        return cls.registry[name]

    def __iter__(cls):
        return iter(cls.registry.values())

    def __str__(cls):
        if cls in cls.registry:
            return cls.__name__
        return cls.__name__ + ": " + ", ".join([sc.__name__ for sc in cls])


class ASubmitter(object, metaclass=APluginRegister):
    """
    Interface for all submitters
    """

    @property
    @abc.abstractmethod
    def isBlocking(self):
        """
        Indicator whether the submitter is blocking or not

        Returns
        -------
        bool
            whether submitter blocks by calling join or not
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self):
        """
        The name of the submitter

        Returns
        -------
        str
            The name of the submitter
        """
        raise NotImplementedError

    @abc.abstractmethod
    def submit(self, command, dependent=None):
        """
        Consumes job objects and starts them

        Parameters
        ----------
         jobs: Command
            A list of Job objects that should be submitted
         dependent: list(Command)
            A list of command objects the Command depend on

        Returns
        -------
        list(str)
            A list of jobIDs
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cancel(self, command):
        """
        Consumes a list of jobIDs and trys to cancel them


        Parameters
        ----------
        command: Command
            The Command jobejct to cancel

        Returns
        -------
        bool
            If job was canceled
        """
        raise NotImplementedError

    @abc.abstractmethod
    def monitor(self, command):
        """
        Returns the status of the consumed command

        Parameters
        ----------
        command: Command
            The command object whose status is inquired

        Returns
        -------
        Enum(Status)
            The status of the Command
        """
        raise NotImplementedError

    @abc.abstractmethod
    def join(self):
        """
        Blocks script if so desired until all jobs have be finished canceled or died
        """
        raise NotImplementedError


class AClusterSubmitter(ASubmitter):
    """
    Abstract subclass of a cluster submitter
    """

    def submit_command(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def monitor_command(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cancel_command(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def resource_flags(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def db(self):
        """
        The persistent DB to keep track of all submitted jobs and their status

        Returns
        -------
        PersistentDict
                The Persistent DB
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def job_id_pattern(self):
        raise NotImplementedError

    def _get_job_id(self, input):
        return self.job_id_pattern.match(input).group(1)

    @abc.abstractmethod
    def _get_status(self, stdo):
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_resources(self, resources):
        """
        prepares the submitter dependent resource string

        Returns
        -------
        str
            The resource string specific for a submitter
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_dependencies(self, resources):
        """
        prepares the submitter dependent job-dependency string

        Returns
        -------
        str
            The job-dependency string specific for a submitter
        """
        raise NotImplementedError

    def _internal_monitor(self, command_id):
        try:
            job_id = yaml.load(self.db[command_id], yaml.RoundTripLoader)["job_id"]
        except KeyError:
            raise ValueError(
                "Command " + repr(command_id) + " has not been submitted yet."
            )

        submit = self.monitor_command.format(job_id=job_id)

        try:
            p = subprocess.Popen(submit, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError(
                    "Unsuccessful monitoring of " + repr(command_id) +
                    " (EXIT!=0) with error: " + stde
                )

        except Exception as e:
            raise RuntimeError(e)

        status = self._get_status(stdo)

        entry = yaml.load(self.db[command_id], yaml.RoundTripLoader)
        entry["status"] = status
        self.db[command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        self.db.sync()

        return status

    def _update(self, unfinished):
        """
        internally monitor submitted jobs
        only needed if blocking
        """
        if not unfinished:
            # initial call
            for k, v in self.db.items():
                status = self._internal_monitor(k)
                if status in [EStatus.PEND, EStatus.RUN]:
                    unfinished.append(k)
            return unfinished

        else:
            tmp = []
            for k in unfinished:
                status = self._internal_monitor(k)
                if status in [EStatus.PEND, EStatus.RUN]:
                    tmp.append(k)
            return tmp

    def monitor(self, command):
        return self._internal_monitor(command.command_id)

    def join(self):
        if self.isBlocking:
            unfinished = []
            while True:
                unfinished = self._update(unfinished)
                if not unfinished:
                    break
                else:
                    time.sleep(1)

    def cancel(self, command):
        try:
           job = yaml.load(self.db[command.command_id], yaml.RoundTripLoader)
           job_id = job["job_id"]
           if job["status"] in [EStatus.DONE, EStatus.EXIT]:
              return True
        except KeyError:
            raise ValueError(
                "Command " + repr(command) + " has not been submitted yet."
            )

        submit = self.cancel_command.format(job_id=job_id)
        try:
            p = subprocess.Popen(submit, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError(
                    "Unsuccessful cancellation of " + repr(command) +
                    " (EXIT!=0) with error: " + stde
                )
        except Exception as e:
            raise RuntimeError(e)

        entry = yaml.load(self.db[command.command_id], yaml.RoundTripLoader)
        entry["status"] = EStatus.EXIT
        self.db[command.command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        self.db.sync()

        return True

    def submit(self, command, dependent=None):
        dep = self._prepare_dependencies(dependent)

        combine = " && " if command.environment else ""
        cmd = " && ".join(command.environment) + combine + " && ".join(command.command)
        resources = self._prepare_resources(command.resources)
        submit = self.submit_command.format(
            cmd=cmd,
            resources=resources,
            dependent=dep,
            name=command.command_id
        )
        try:
            p = subprocess.Popen(
                submit, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, universal_newlines=True, cwd=command.workdir
            )
            stdo, stde = p.communicate()
            stdr = p.returncode

            if stdr > 0:
                raise RuntimeError(
                    "Unsuccessful execution of " + repr(command) + " (EXIT!=0) with error: " + stde
                )
        except Exception as e:
            raise RuntimeError(e)
        # get job id and submit to db
        job_id = self._get_job_id(stdo)
        try:
            # update entry if existing:
            entry = yaml.load(self.db[command.command_id], yaml.RoundTripLoader)
            entry["name"] = command.name
            entry["tries"] += 1
            entry["job_id"] = job_id
            entry["status"] = EStatus.PEND
            entry["command"] = command.command
            entry["resources"] = command.resources
            entry["workdir"] = command.workdir
            entry["environment"] = command.environment
            self.db[command.command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        except KeyError:
            # add new entry
            entry = {
                "name": command.name,
                "job_id": job_id,
                "tries": 1,
                "status": EStatus.PEND,
                "command": command.command,
                "resources": command.resources,
                "workdir": command.workdir,
                "environment": command.environment
            }
            self.db[command.command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
            self.db.sync()

        return job_id


class LSFSubmitter(AClusterSubmitter):
    """
    Implements an LSF submitter
    """
    __name = "lsf"
    __submit = "bsub -J {name} {dependent} {resources} '{cmd}'"
    __monitor = "bjobs {job_id}"
    __cancel = "bkill {job_id}"
    __resources = ""
    __resources_flag = {EResource.queue: "-q",
                        EResource.time: "-W",
                        EResource.mem: "-R",
                        EResource.nodes: "-n",
                        EResource.error: "-e",
                        EResource.out: "-o"}
    __job_id_pattern = re.compile(r"^Job <([0-9]*)>")

    def __init__(self, blocking=False, db_path=None):
        """
        Init function

        Parameters
        ----------
        blocking: bool
            determines whether join() blocks or not
        db_path: str
            the string to a LevelDB for command persistence
        """
        self.__blocking = blocking
        if db_path is None:
            tmp_db = NamedTemporaryFile(delete=False, dir=os.getcwd(), suffix=".db")
            tmp_db.close()
            self.__is_temp_db = True
            self.__db_path = tmp_db.name
        else:
            self.__is_temp_db = False
            self.__db_path = db_path

        self.__db = PersistentDict(self.__db_path)

    def __del__(self):
        try:
            self.__db.close()
            if self.__is_temp_db:
                os.remove(self.__db_path)
        except AttributeError:
            pass

    @property
    def isBlocking(self):
        return self.__blocking

    @property
    def name(self):
        return self.__name

    @property
    def monitor_command(self):
        return self.__monitor

    @property
    def resource_flags(self):
        return self.__resources_flag

    @property
    def submit_command(self):
        return self.__submit

    @property
    def db(self):
        return self.__db

    @property
    def cancel_command(self):
        return self.__cancel

    @property
    def job_id_pattern(self):
        return self.__job_id_pattern

    def _get_status(self, stdo):
        def status_map(st):
            if st == "PEND":
                return EStatus.PEND
            elif st == "RUN":
                return EStatus.RUN
            elif st == "DONE":
                return EStatus.DONE
            elif st == "EXIT":
                return EStatus.EXIT
            else:
                return EStatus.SUSP
        return status_map(stdo.split("\n")[1].split()[2].strip())

    def _prepare_dependencies(self, dependent):
        dep = ""
        if dependent is not None:
            try:
                if isinstance(dependent, Command):
                    d_info = yaml.load(self.__db[dependent.command_id], yaml.RoundTripLoader)
                    dep = "-w {}".format(d_info["job_id"])
                else:
                    dep_jobs = []
                    for d in dependent:
                        d_info = yaml.load(self.__db[d.command_id], yaml.RoundTripLoader)
                        dep_jobs.append(d_info["job_id"])
                    # not sure if comma-separated is correct
                    dep = "-w {}".format(" && ".join("ended({})".format(d) for d in dep_jobs))
            except KeyError:
                raise ValueError("Specified depended jobs have not been submitted yet.")
        return dep

    def _prepare_resources(self, resources):
        return " ".join(
            "{} {}".format(self.resource_flags[k], v) if k != EResource.mem else "{} 'rusage[mem={}]'".format(
                self.resource_flags[k], v) for k, v in resources.items())

########################################################################################################################
#
#  Slurm submitter
#
########################################################################################################################


class SlurmSubmitter(AClusterSubmitter):
    """
    Implements an LSF submitter
    """
    __name = "slurm"
    __submit = "sbatch --job-name={name} {dependent} {resources} --wrap 'srun {cmd}'"
    __monitor = "squeue -t all -j {job_id}"
    __cancel = "scancel {job_id}"
    __resources = ""
    __resources_flag = {EResource.queue: "-p",
                        EResource.time: "-t",
                        EResource.mem: "--mem-per-cpu",
                        EResource.nodes: "-c",
                        EResource.error: "-e",
                        EResource.out: "-o"}
    __job_id_pattern = re.compile(r"Submitted batch job ([0-9]*)")

    def __init__(self, blocking=False, db_path=None):
        """
        Init function

        Parameters
        ----------
        blocking: bool
            determines whether join() blocks or not
        db_path: str
            the string to a LevelDB for command persistence
        """
        self.__blocking = blocking
        if db_path is None:
            tmp_db = NamedTemporaryFile(delete=False, dir=os.getcwd(), suffix=".db")
            tmp_db.close()
            self.__is_temp_db = True
            self.__db_path = tmp_db.name
        else:
            self.__is_temp_db = False
            self.__db_path = db_path

        self.__db = PersistentDict(self.__db_path)

    def __del__(self):
        try:
            self.__db.close()
            if self.__is_temp_db:
                os.remove(self.__db_path)
        except AttributeError:
            pass
          
    @property
    def isBlocking(self):
        return self.__blocking

    @property
    def name(self):
        return self.__name

    @property
    def monitor_command(self):
        return self.__monitor

    @property
    def resource_flags(self):
        return self.__resources_flag

    @property
    def submit_command(self):
        return self.__submit

    @property
    def db(self):
        return self.__db

    @property
    def cancel_command(self):
        return self.__cancel

    @property
    def job_id_pattern(self):
        return self.__job_id_pattern

    def _prepare_dependencies(self, dependent):
        dep = ""
        if dependent is not None:
            try:
                if isinstance(dependent, Command):
                    d_info = yaml.load(self.__db[dependent.command_id], yaml.RoundTripLoader)
                    dep = "--kill-on-invalid-dep=yes --dependency=afterok:{}".format(d_info["job_id"])
                else:
                    dep_jobs = []
                    for d in dependent:
                        d_info = yaml.load(self.__db[d.command_id], yaml.RoundTripLoader)
                        dep_jobs.append(d_info["job_id"])
                    # not sure if comma-separated is correct
                    dep = "--kill-on-invalid-dep=yes --dependency=afterok:{}".format(":".join("ended({})".format(d)
                                                                                              for d in dep_jobs))
            except KeyError:
                raise ValueError("Specified depended jobs have not been submitted yet.")
        return dep

    def _prepare_resources(self, resources):
        return " ".join("{} {}".format(self.resource_flags[k], v) for k, v in resources.items())

    def _get_status(self, stdo):
        def status_map(st):
            if st in ["PD", "CF"]:
                return EStatus.PEND
            elif st in ["R", "CG"]:
                return EStatus.RUN
            elif st == "CD":
                return EStatus.DONE
            elif st in ["BF", "PR", "TO", "NF", "F", "CA"]:
                return EStatus.EXIT
            else:
                return EStatus.SUSP
        return status_map(stdo.split("\n")[1].split()[4].strip())

########################################################################################################################
#
# Sun Grid Engine Submitter
#
########################################################################################################################


class SGESubmitter(AClusterSubmitter):
    """
    Implements an LSF submitter
    """
    __name = "sge"
    __submit = "echo '{cmd}' | qsub -N {name} {dependent} {resources}"
    __monitor = "qstat"
    __cancel = "qdel {job_id}"
    __resources = ""
    __resources_flag = {EResource.queue: "-q",
                        EResource.time: '-l h_rt=',
                        EResource.mem: '-l h_vmem=',
                        EResource.nodes: "-pe smp",
                        EResource.error: "-e",
                        EResource.out: "-o"}
    __job_id_pattern = re.compile(r'Your job ([0-9]+) .*')

    def __init__(self, blocking=False, db_path=None):
        """
        Init function

        Parameters
        ----------
        blocking: bool
            determines whether join() blocks or not
        db_path: str
            the string to a LevelDB for command persistence
        """
        self.__blocking = blocking
        if db_path is None:
            tmp_db = NamedTemporaryFile(delete=False, dir=os.getcwd(), suffix=".db")
            tmp_db.close()
            self.__is_temp_db = True
            self.__db_path = tmp_db.name
        else:
            self.__is_temp_db = False
            self.__db_path = db_path

        self.__db = PersistentDict(self.__db_path)

    def __del__(self):
        try:
            self.__db.close()
            if self.__is_temp_db:
                os.remove(self.__db_path)
        except AttributeError:
            pass


    @property
    def isBlocking(self):
        return self.__blocking


    @property
    def name(self):
        return self.__name


    @property
    def monitor_command(self):
        return self.__monitor


    @property
    def resource_flags(self):
        return self.__resources_flag


    @property
    def submit_command(self):
        return self.__submit

    @property
    def db(self):
        return self.__db

    @property
    def cancel_command(self):
        return self.__cancel


    @property
    def job_id_pattern(self):
        return self.__job_id_pattern

    def _prepare_resources(self, resources):
        special_res = {EResource.mem, EResource.time}
        return " ".join(
            "{} {}".format(self.resource_flags[k], v) if k not in special_res else "{}{}".format(
                self.resource_flags[k], v) for k, v in resources.items())

    def _prepare_dependencies(self, dependent):
        dep = ""
        if dependent is not None:
            try:
                if isinstance(dependent, Command):
                    d_info = yaml.load(self.__db[dependent.command_id], yaml.RoundTripLoader)
                    dep = "{}".format(d_info["job_id"])
                else:
                    dep_jobs = []
                    for d in dependent:
                        d_info = yaml.load(self.__db[d.command_id], yaml.RoundTripLoader)
                        dep_jobs.append(d_info["job_id"])
                    dep = ",".join(dep_jobs)
                dep = "-hold_jid "+dep
            except KeyError:
                raise ValueError("Specified depended jobs have not been submitted yet.")
        return dep

    def _internal_monitor(self, command_id):
        try:
            job_id = yaml.load(self.db[command_id], yaml.RoundTripLoader)["job_id"]
        except KeyError:
            raise ValueError(
                "Command " + repr(command_id) + " has not been submitted yet."
            )

        submit = self.monitor_command.format(job_id=job_id)

        try:
            p = subprocess.Popen(submit, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError(
                    "Unsuccessful monitoring of " + repr(command_id) +
                    " (EXIT!=0) with error: " + stde
                )

        except Exception as e:
            raise RuntimeError(e)

        status = self._get_status(stdo, job_id)

        entry = yaml.load(self.db[command_id], yaml.RoundTripLoader)
        entry["status"] = status
        self.db[command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        self.db.sync()

        return status

    def _get_status(self, stdo, job_id):
        def status_map(status):
            if status == "r":
                return EStatus.RUN
            elif status == "qw":
                return EStatus.PEND
            elif status in ["Ewq", "e", "E"]:
                return EStatus.SUSP
            else:
                return EStatus.EXIT

        # search in list for command_id and extract status
        for l in stdo.split("\n"):
                if "" == l.strip():
                    continue
                splits = l.split()
                if job_id == splits[0]:
                    return status_map(splits[4])
        return EStatus.DONE


########################################################################################################################
#
# Local submitter
#
########################################################################################################################

class _Worker(mp.Process):
    """
    _Worker class that runs commands

    """

    def __init__(self, broker_queue, input_queue, results_queue):
        mp.Process.__init__(self)
        self.__broker_queue = broker_queue
        self.__input_queue = input_queue
        self.__results_queue = results_queue
        self.daemon = True

    def run(self):
        for job in iter(self.__input_queue.get, EJob.CANCEL):
            # send RUN status to broker for received command
            self.__broker_queue.put((EJob.UPDATE, (job.command_id, EStatus.RUN)))
            try:
                # run command and send DONE status to broker one success
                self.__submit(job)
                self.__broker_queue.put((EJob.UPDATE, (job.command_id, EStatus.DONE)))
            except Exception as e:
                # else send EXIT status
                self.__broker_queue.put((EJob.UPDATE, (job.command_id, EStatus.EXIT)))
            finally:
                # finally update semaphore counter of queue
                self.__input_queue.task_done()

    def __submit(self, command):
        """
        local function to submit a job

        Parameters
        ---------
        command: Command
            The Command object to execute

        Returns
        -------
        str
            The process ID of the command
        """
        try:
            combine = " && " if command.environment else ""
            cmd = " && ".join(command.environment) + combine + " && ".join(command.command)
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True, cwd=command.workdir,
                                 preexec_fn=os.setsid)
            self.__broker_queue.put((EJob.PID, (command.command_id, p.pid)))
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr != 0:
                raise RuntimeError("Unsuccessful execution of " + repr(command) + " (EXIT!=0) with error: " + stde)
        except Exception as e:
            raise RuntimeError(e)


class _Broker(mp.Process):
    """
    _Broker process handling dependencies and
    submission of jobs
    """

    def __init__(self, broker_queue, worker_queue, results_queue, results_queue_worker, pending_dict, db_path=None):
        mp.Process.__init__(self)
        self.__input_queue = broker_queue
        self.__results_queue_master = results_queue
        self.__results_queue_worker = results_queue_worker
        self.__worker_queue = worker_queue
        self.__pending_dict = pending_dict
        self.__db = PersistentDict(db_path)

    def __del__(self):
        try:
            # kill remaining runing commands
            for k, v in self.__db.RangeIter():
                if v["status"] not in [EStatus.EXIT, EStatus.DONE]:
                    os.killpg(os.getpgid(v["job_id"]), signal.SIGKILL)

            self.__db.close()
        except AttributeError:
            pass

    def run(self):
        while True:
            try:
                args = self.__input_queue.get(True, 0.1)
                ejob, args = args
                if ejob == EJob.STOP:
                    # terminate broker
                    self.terminate()

                elif ejob == EJob.MONITOR:
                    # monitor request
                    self.__results_queue_master.put(self.__monitor(args))

                elif ejob == EJob.CANCEL:
                    # cancel job
                    status = self.__cancel(args)
                    self.__results_queue_master.put(True)
                    self.__update_status(args.command_id, status)

                elif ejob == EJob.UPDATE:
                    # updating command status (e.g, RUN, EXIT, DONE)
                    # some command has started message is coming from worker process
                    c_id, status = args
                    self.__update_status(c_id, status)

                elif ejob == EJob.PID:
                    # updating process ID of submitted command
                    job_id, p_id = args
                    entry = yaml.load(self.__db[job_id], yaml.RoundTripLoader)
                    entry["job_id"] = p_id
                    self.__db[job_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
                    self.__db.sync()

                else:
                    # submitting job to worker
                    job, dependent = args
                    self.__add_command(job)
                    if dependent is not None:
                        self.__pending_dict[job] = dependent
                    else:
                        self.__worker_queue.put(job)

            except queue.Empty:
                # go through pending jobs and update status
                # if one of the dependent jobs terminated with an error pending job is also terminated with error
                for job, dependent in list(self.__pending_dict.items()):
                    status = self.__condition_fulfilled(dependent)
                    if job is not None:
                        if status == EStatus.EXIT:
                            # job cannot be called due to termination of dependent jobs
                            self.__update_status(job.command_id, EStatus.EXIT)
                            del self.__pending_dict[job]
                        elif status == EStatus.RUN:
                            self.__worker_queue.put(job)
                            del self.__pending_dict[job]
                    else:
                        # makes sure that a submitted job is properly registered and join works as intended
                        if status == EStatus.RUN:
                            del self.__pending_dict[job]

            except Exception as e:
                tb = traceback.format_exc()
                self.__results_queue_master.put((e, tb))

    def __condition_fulfilled(self, dependent):
        for d in dependent:
            status = self.__monitor(d)
            if status == EStatus.EXIT:
                return EStatus.EXIT
            if status != EStatus.DONE:
                return EStatus.PEND
        return EStatus.RUN

    def __add_command(self, command):
        try:
            # update entry if existing:
            entry = yaml.load(self.__db[command.command_id], yaml.RoundTripLoader)
            entry["name"] = command.name
            entry["tries"] += 1
            entry["job_id"] = None
            entry["status"] = EStatus.PEND
            entry["command"] = command.command
            entry["resources"] = command.resources
            entry["workdir"] = command.workdir
            entry["environment"] = command.environment
            self.__db[command.command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        except KeyError:
            # add new entry
            entry = {
                "name": command.name,
                "job_id": None,
                "tries": 1,
                "status": EStatus.PEND,
                "command": command.command,
                "resources": command.resources,
                "workdir": command.workdir,
                "environment": command.environment
            }
            self.__db[command.command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
            self.__db.sync()

    def __cancel(self, command):
        try:
            entry = yaml.load(self.__db[command.command_id], yaml.RoundTripLoader)
        except KeyError:
            raise ValueError("Command " + repr(command.command_id) + " has not been submitted yet.")
        p_id = entry["job_id"]
        status = self.__monitor(command)
        if status == EStatus.PEND:
            del self.__pending_dict[command]
        elif status in [EStatus.SUSP, EStatus.RUN]:
            os.killpg(os.getpgid(p_id), signal.SIGKILL)
        elif status == EStatus.DONE:
            return status
        return EStatus.EXIT

    def __monitor(self, command):
        """
        local function to monitor a command via assigned pID

        Parameters
        ----------
        command: Command
            The Command object

        Returns
        -------
        EStatus
            The status of the command
        """
        if command in self.__pending_dict:
            return EStatus.PEND

        try:
            entry = yaml.load(self.__db[command.command_id], yaml.RoundTripLoader)
        except KeyError:
            raise ValueError("Command " + repr(command.command_id) + " has not been submitted yet.")

        p_id = entry["job_id"]
        cmd = entry["command"][0]

        try:
            # I think if PID is completed this should through an error ....
            p = psutil.Process(pid=p_id)
            status = p.status()
            p_cmd = " ".join(p.cmdline())

            # test status types # if status is ZOMBIE also kill the job
            # pid can be already in use by other process... one has to check the command as well....
            # if p_cmd is different than the original process is probably completed
            if cmd not in p_cmd:
                c_stat = EStatus.DONE

            elif status in [psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE]:
                p.kill()
                c_stat = EStatus.EXIT

            elif status == psutil.STATUS_RUNNING:
                c_stat = EStatus.SUSP
            else:
                c_stat = EStatus.SUSP
        except psutil.NoSuchProcess:
            c_stat = EStatus.DONE
        except psutil.AccessDenied:
            c_stat = EStatus.RUN

        entry["status"] = c_stat
        self.__db[command.command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        self.__db.sync()
        return c_stat

    def __update_status(self, c_id, status):
        """
        updates the status of a command

        Parameters
        ----------
        c_id: str
            The Command id
        status: EStatus
            The  new status
        """
        try:
            entry = yaml.load(self.__db[c_id], yaml.RoundTripLoader)
        except KeyError:
            raise ValueError("Command " + repr(c_id) + " has not been submitted yet.")
        entry["status"] = status
        self.__db[c_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        self.__db.sync()


class LocalSubmitter(ASubmitter):
    __name = "local"

    def __init__(self, blocking=True, db_path=None, ncpu=1):
        """
        Init function

        Parameter
        ---------
        blocking: bool
            determines whether join() blocks or not
        db_path: str
            the string to a LevelDB for command persistence
        """
        self.__blocking = blocking
        self.__broker_queue = mp.Queue()
        self.__job_queue = mp.JoinableQueue()
        self.__pending_dict = mp.Manager().dict()
        self.__results_queue = mp.Queue()
        self.__results_queue_worker = mp.Queue()

        if db_path is None:
            tmp_db = NamedTemporaryFile(delete=False, dir=os.getcwd(), suffix=".db")
            tmp_db.close()
            self.__is_temp_db = True
            self.__db_path = tmp_db.name
        else:
            self.__is_temp_db = False
            self.__db_path = db_path

        self.__broker = _Broker(self.__broker_queue, self.__job_queue, self.__results_queue, self.__results_queue_worker,
                                self.__pending_dict, db_path=self.__db_path)
        self.__broker.daemon = False
        self.__broker.start()

        self.__worker = []
        for i in range(ncpu):
            p = _Worker(self.__broker_queue, self.__job_queue, self.__results_queue_worker)
            p.daemon = False
            self.__worker.append(p)
            p.start()

    def __del__(self):
        try:
            self.__broker_queue.close()
            self.__job_queue.close()
            self.__results_queue.close()
            self.__results_queue_worker.close()

            self.__broker_queue.join_thread()
            self.__job_queue.join_thread()
            self.__results_queue.join_thread()
            self.__results_queue_worker.join_thread()

            self.__broker.terminate()
            for w in self.__worker:
                w.terminate()

            if self.__is_temp_db:
                os.remove(self.__db_path)
        except AttributeError:
            pass

    @property
    def name(self):
        return self.__name

    @property
    def isBlocking(self):
        return self.__blocking

    def submit(self, command, dependent=None):
        if isinstance(dependent, Command) and dependent is not None:
            dependent = [dependent]
        self.__pending_dict.setdefault(None, []).append(command)
        self.__broker_queue.put((EJob.SUBMIT, (command, dependent)))
        return command.command_id

    def cancel(self, command):
        self.__broker_queue.put((EJob.CANCEL, command))
        return self.__results_queue.get()

    def monitor(self, command):
        self.__broker_queue.put((EJob.MONITOR, command))
        return self.__results_queue.get()

    def join(self):
        if self.isBlocking:
            while self.__pending_dict:
                time.sleep(1)
            self.__job_queue.join()
        else:
            pass
