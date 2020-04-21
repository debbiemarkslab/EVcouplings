from evcouplings.utils.calculations import *
from evcouplings.utils.config import *
from evcouplings.utils.helpers import *
from evcouplings.utils.system import *
from evcouplings.utils.batch import *
from evcouplings.utils.constants import *
from evcouplings.utils.tracker import *


class ASubmitterFactory(abc.ABCMeta):
    def __init__(cls, name, bases, nmspc):
        type.__init__(cls, name, bases, nmspc)

    def __call__(cls, _name, *args, **kwargs):
        """
        If a third person wants to write a new Submitter. He/She has to inherit from ASubmitter.
        That's it nothing more.
        """

        try:
            return ASubmitter[str(_name).lower()](**kwargs)
        except KeyError as e:
            raise ValueError("This submitter is currently not supported")


class SubmitterFactory(object, metaclass=ASubmitterFactory):

    @staticmethod
    def available_methods():
        """
        Returns a dictionary of available epitope predictors and their supported versions

        Returns
        -------
        dict(str,list(str)
            dictionary of supported submitter
        """
        return [ASubmitter.registry.keys()]


class BailoutException(Exception):
    """
    Exception for pipeline stopping itself (e.g. if no sequences found)
    """
