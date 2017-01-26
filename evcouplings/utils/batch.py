"""
Looping through batches of jobs
(former submit_job.py and buildali loop)

Authors:
  Benjamin Schubert, Thomas A. Hopf
"""
import abc
import inspect
import os
import subprocess
import uuid
import re
import time
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml

from tempfile import NamedTemporaryFile

# ENUMS
# (PEND, RUN, USUSP, PSUSP, SSUSP, DONE, and EXIT statuses)


from evcouplings.utils import PersistentDict

EStatus = (lambda **enums: type('Enum', (), enums))(RUN=0, PEND=1, SUSP=2, EXIT=3, DONE=4)
EResource = (lambda **enums: type('Enum', (), enums))(time=0, mem=1, nodes=2, queue=3, error=4, out=5)


#Metaclass for Plugins
class APluginRegister(abc.ABCMeta):
    """
        This class allows automatic registration of new plugins.
    """
    def __init__(cls, name, bases, nmspc):
        super(APluginRegister, cls).__init__(name, bases, nmspc)

        if not hasattr(cls, 'registry'):
            cls.registry = dict()
        if not inspect.isabstract(cls):
            cls.registry[str(cls().name).lower()]= cls

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
    Iterface for all submitters

    """

    @abc.abstractproperty
    def isBlocking(self):
        """
        Inidactor whether the submitter is blocking or not

        Retruns:
        bool - whether submitter blocks by calling join or not
        """
        raise NotImplementedError

    @abc.abstractproperty
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

        Parameters:
        ----------
         jobs: Command
            A list of Job objects that should be submitted
        dependent: list(Command)
            A list of command objects the Command depend on

        Returns:
        --------
        list(str)
            A list of jobIDs
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cancel(self, command):
        """
        Consumes a list of jobIDs and trys to cancel them


        Parameters:
        -----------
        command: Command
            The Command jobejct to cancel

        Returns:
        --------
            bool - If job was canceled
        """
        raise NotImplementedError

    @abc.abstractmethod
    def monitor(self, command):
        """
        Returns the status of the consumed command

        Parameters:
        -----------
        command: Command
            The command object whos status is inquired

        Returns:
        --------
        Enum(Status) - The status of the Command
        """
        raise NotImplementedError

    @abc.abstractmethod
    def join(self):
        """
        Blocks script if so desired until all jobs have be finished canceled or died
        """
        raise NotImplementedError


class LSFSubmitter(ASubmitter):
    """
    Implements an LSF submitter

    """
    __name = "lsf"
    __submit = "bsub {resources}-J {name} {dependent}-q {queue} {cmd}"
    __monitor = "bjobs {job_id}"
    __cancle = "bkill {job_id}"
    __resources = ""
    __resources_flag = {EResource.time: "-W",
                        EResource.mem: "-M",
                        EResource.nodes: "-n",
                        EResource.error: "-e",
                        EResource.out: "-o"}
    __job_id_pattern = re.compile(r"^Job <([0-9]*)>")

    def __init__(self, blocking=False, db_path=None):
        """
        Init function

        Parameters:
        blocking: bool
            determines whether join() blocks or not
        db_path: str
            the string to a LevelDB for command persistence
        """
        print("Init is called")
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

    def __get_job_id(self, stdo):
        return self.__job_id_pattern.match(stdo).group(1)

    def __get_status(self, stdo):
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
        return status_map(stdo.split("\n")[1].split()[2])

    def __internal_monitor(self, command_id):
        try:
            job_id = yaml.load(self.__db[command_id], yaml.RoundTripLoader)["job_id"]
        except KeyError:
            raise ValueError("Command "+repr(command_id)+" has not been submitted yet.")

        submit = self.__monitor.format(job_id=job_id)

        try:
            p = subprocess.Popen(submit, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful monitoring of " + repr(command_id) + " (EXIT!=0) with error: " + stde)
        except Exception as e:
            raise RuntimeError(e)

        status = self.__get_status(stdo)

        entry = yaml.load(self.__db[command_id], yaml.RoundTripLoader)
        entry["status"] = status
        self.__db[command_id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        self.__db.sync()

        return status

    def __update(self, unfinished):
        """
        internally monitor submitted jobs
        only needed if blocking
        """
        if not unfinished:
            #initial call
            for k,v in self.__db.RangeIter():
                status = self.__internal_monitor(k)
                if status in [EStatus.PEND, EStatus.RUN]:
                    unfinished.append(k)
            return unfinished

        else:
            tmp = []
            for k in unfinished:
                status = self.__internal_monitor(k)
                if status in [EStatus.PEND, EStatus.RUN]:
                    tmp.append(k)
            return tmp

    def submit(self, command, dependent=None):
        dep = ""
        if dependent is not None:
            try:
                if isinstance(dependent , Command):
                    d_info = yaml.load(self.__db[dependent.id], yaml.RoundTripLoader)
                    dep = "-w {}".format(d_info["job_id"])
                else:
                    dep_jobs = []
                    for d in dependent:
                        d_info = yaml.load(self.__db[d.id], yaml.RoundTripLoader)
                        dep_jobs.append(d_info["job_id"])
                    # not sure if comma-separated is correct
                    dep = "-w {}".format(",".join(dep_jobs))
            except KeyError:
                raise ValueError("Specified depended jobs have not been submitted yet.")

        cmd = " && ".join(command.environment)+ " && " + " && ".join(command.command)
        resources =" ".join("{} {}".format(self.__resources_flag[k], v) for k, v in command.resources.items())
        submit = self.__submit.format(
            cmd=cmd,
            queue=command.resources[EResource.queue],
            resoruces=resources,
            dependent=dep,
            name=command.id
        )

        try:
            p = subprocess.Popen(submit, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful execution of " + repr(command) + " (EXIT!=0) with error: " + stde)
        except Exception as e:
            raise RuntimeError(e)

        # get job id and submit to db
        job_id = self.__get_job_id(stdo)
        try:
            # update entry if existing:
            entry = yaml.load(self.__db[command.id], yaml.RoundTripLoader)
            entry["name"] = command.name
            entry["tries"] += 1
            entry["job_id"] = job_id
            entry["status"] = EStatus.PEND
            entry["command"] = command.command
            entry["resources"] = command.resources
            entry["workdir"] = command.workdir
            entry["environment"] = command.environment
            self.__db[command.id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
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
            self.__db[command.id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
            self.__db.sync()

        return job_id

    def cancel(self, command):
        try:
            job_id = yaml.load(self.__db[command.id], yaml.RoundTripLoader)["job_id"]
        except KeyError:
            raise ValueError("Command "+repr(command)+" has not been submitted yet.")

        submit = self.__cancle.format(job_id=job_id)

        try:
            p = subprocess.Popen(submit, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            stdo, stde = p.communicate()
            stdr = p.returncode
            if stdr > 0:
                raise RuntimeError("Unsuccessful cancelation of " + repr(command) + " (EXIT!=0) with error: " + stde)
        except Exception as e:
            raise RuntimeError(e)

        # TODO: update db - Delete entry or just update?
        entry = yaml.load(self.__db[command.id], yaml.RoundTripLoader)
        entry["status"] = EStatus.EXIT
        self.__db[command.id] = yaml.dump(entry, Dumper=yaml.RoundTripDumper)
        self.__db.sync()

        return True

    def monitor(self, command):
        return self.__internal_monitor(command.id)

    def join(self):
        if self.isBlocking:
            unfinished = []
            while True:
                unfinished = self.__update(unfinished)
                if not unfinished:
                    break
                else:
                    time.sleep(5)


class Command(object):
    """
    Wrapper around the command parameters needed
    to execute a script
    """

    def __init__(self, command, name=None, environment=None, workdir=None, resources=None):
        """
        Class represents a command to be submitted and contains all relevant information

        Parameters:
        -----------
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

        self.id = uuid.uuid4().bytes
        self.name = name

        self.command = [command] if isinstance(command, str) else command
        self.environment = [environment] if isinstance(environment, str) else environment
        self.workdir = workdir
        self.resources = resources

    def __eq__(self, other):
        if isinstance(other, Command):
            return False
        return self.id == other.id

    def __str__(self):
        return "Command:{id}:\n\t{commands}".format(id=str(uuid.UUID(bytes=self.id)),
                                                    commands="&".join(self.command)[:16])

    def __repr__(self):
        return "Command({id}}".format(id=str(uuid.UUID(bytes=self.id)))