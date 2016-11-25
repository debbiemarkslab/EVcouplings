"""
Mapping indices for complexes / multi-domain sequences to
internal model numbering.

Authors:
  Thomas A. Hopf
"""

from collections import defaultdict, Iterable
import numpy as np


class Segment:
    """
    Represents a continuous stretch of sequence in a sequence
    alignment to infer evolutionary couplings (e.g. multiple domains,
    or monomers in a concatenated complex alignment)
    """
    def __init__(self, segment_type, sequence_id,
                 region_start, region_end, positions,
                 segment_id="A"):
        """
        Create a new sequence segment

        Parameters
        ----------
        segment_id : str
            Identifier of segment (must be unique)
        segment_type : {"aa", "dna", "rna"}
            Type of sequence
        sequence_id : str
            Identifier of sequence
        region_start : int
            Start index of sequence segment
        region_end : int
            End index of sequence segment (position
            is inclusive)
        position_list : list(int)
            Positions in the sequence alignment that
            will be used for EC calculation
            (all positions corresponding to uppercase
            residues)
        """
        self.segment_type = segment_type
        self.sequence_id = sequence_id
        self.region_start = region_start
        self.region_end = region_end
        self.positions = list(map(int, positions))
        self.segment_id = segment_id

    @classmethod
    def from_list(cls, segment):
        """
        Create a segment object from list representation
        (e.g. from config).

        Parameters
        ----------
        segment : list
            List representation of segment, with the following items:
            segment_id (str), segment_type (str), sequence_id (str),
            region_start (int), region_end (int), positions (list(int))

        Returns
        -------
        Segment
            New Segment instance from list
        """
        segment_id, segment_type, sequence_id, region_start, region_end, positions = segment
        return cls(segment_type, sequence_id, region_start, region_end, positions, segment_id)

    def to_list(self):
        """
        Represent segment as list (for storing in configs)

        Returns
        -------
        list
            List representation of segment, with the following items:
            segment_id (str), segment_type (str), sequence_id (str),
            region_start (int), region_end (int), positions (list(int))
        """
        return [
            self.segment_id,
            self.segment_type,
            self.sequence_id,
            self.region_start,
            self.region_end,
            self.positions
        ]


class ComplexIndexMapper:
    """
    Map indices of sequences into concatenated EVcouplings
    object numbering space. Can in principle also be used
    to remap indices for a single sequence.
    """
    def __init__(self, couplings, couplings_range, *monomer_ranges):
        """
        Ranges are tuples of form (start: int, end: int)
        couplings_range must match the range of EVcouplings object
        Example: ComplexIndexMapper(c, (1, 196), (1, 103), (1, 93))

        Parameters
        ----------
        couplings : EVcouplings
            Couplings object of complex
        couplings_range : (int, int)
            Numbering range in couplings that monomers will be
            mapped to
        *monomer_ranges: (int, int):
            Tuples containing numbering range of each monomer
        """
        if len(monomer_ranges) < 1:
            raise ValueError("Give at least one monomer range")

        self.couplings_range = couplings_range
        self.monomer_ranges = monomer_ranges
        self.monomer_to_full_range = {}

        # create a list of positions per region that directly
        # aligns against the full complex range in c_range
        r_map = []
        for i, (r_start, r_end) in enumerate(monomer_ranges):
            m_range = range(r_start, r_end + 1)
            r_map += zip([i] * len(m_range), m_range)
            self.monomer_to_full_range[i] = m_range

        c_range = range(couplings_range[0], couplings_range[1] + 1)
        if len(r_map) != len(c_range):
            raise ValueError(
                "Complex range and monomer ranges do not have equivalent lengths "
                "(complex: {}, sum of monomers: {}).".format(len(c_range), len(r_map))
            )

        # These dicts might contain indices not contained in
        # couplings object because they are lowercase in alignment
        self.monomer_to_couplings = dict(zip(r_map, c_range))
        self.couplings_to_monomer = dict(zip(c_range, r_map))

        # store all indices per subunit that are actually
        # contained in couplings object
        self.monomer_indices = defaultdict(list)
        for (monomer, m_res), c_res in sorted(self.monomer_to_couplings.items()):
            if c_res in couplings.tn():
                self.monomer_indices[monomer].append(m_res)

    def __map(self, indices, mapping_dict):
        """
        Applies a mapping either to a single index, or to a list of indices

        Parameters
        ----------
        indices: int, or (int, int), or lists thereof
            Indices in input numbering space

        mapping_dict : dict(int->(int, int)) or dict((int, int): int)
            Mapping from one numbering space into the other

        Returns
        -------
        list of int, or list of (int, int)
            Mapped indices
        """
        if isinstance(indices, Iterable) and not isinstance(indices, tuple):
            return np.array([mapping_dict[x] for x in indices])
        else:
            return mapping_dict[indices]

    def __call__(self, monomer, res):
        """
        Function-style syntax for single residue to be mapped
        (calls toc method)

        Parameters
        ----------
        monomer : int
            Number of monomer
        res : int
            Position in monomer numbering

        Returns
        -------
        int
            Index in coupling object numbering space
        """
        return self.toc((monomer, res))

    def tom(self, x):
        """
        Map couplings TO *M*onomer

        Parameters
        ----------
        x : int, or list of ints
            Indices in coupling object

        Returns
        -------
        (int, int), or list of (int, int)
            Indices mapped into monomer numbering. Tuples are
            (monomer, index in monomer sequence)
        """
        return self.__map(x, self.couplings_to_monomer)

    def toc(self, x):
        """
        Map monomer TO *C*ouplings / complex

        Parameters
        ----------
        x : (int, int), or list of (int, int)
            Indices in momnomers (monomer, index in monomer sequence)

        Returns
        -------
        int, or list of int
            Monomer indices mapped into couplings object numbering
        """
        return self.__map(x, self.monomer_to_couplings)
