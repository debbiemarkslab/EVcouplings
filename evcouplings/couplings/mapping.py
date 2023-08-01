"""
Mapping indices for complexes / multi-domain sequences to
internal model numbering.

Authors:
  Thomas A. Hopf
  Anna G. Green (MultiSegmentCouplingsModel)
"""

from collections.abc import Iterable
from copy import deepcopy
from evcouplings.couplings.model import CouplingsModel
import pandas as pd
import numpy as np


class Segment:
    """
    Represents a continuous stretch of sequence in a sequence
    alignment to infer evolutionary couplings (e.g. multiple domains,
    or monomers in a concatenated complex alignment)
    """
    def __init__(self, segment_type, sequence_id,
                 region_start, region_end, positions=None,
                 segment_id="A"):
        """
        Create a new sequence segment

        Parameters
        ----------
        segment_type : {"aa", "dna", "rna"}
            Type of sequence
        sequence_id : str
            Identifier of sequence
        region_start : int
            Start index of sequence segment
        region_end : int
            End index of sequence segment (position
            is inclusive)
        positions : list(int), optional (default: None)
            Positions in the sequence alignment that
            will be used for EC calculation
            (all positions corresponding to uppercase
            residues). Compulsory parameter when using
            non-focus mode.
        segment_id : str
            Identifier of segment (must be unique)
        """
        self.segment_type = segment_type
        self.sequence_id = sequence_id
        self.region_start = region_start
        self.region_end = region_end
        if positions is not None:
            self.positions = list(map(int, positions))
        else:
            self.positions = None

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

    def default_chain_name(self):
        """
        Retrieve default PDB chain identifier the segment will
        be mapped to in 3D structures (by convention, segments in
        the pipeline are named A_1, A_2, ..., B_1, B_2, ...; the default
        chain identifier is anything before the underscore).

        Returns
        -------
        chain : str
            Default PDB chain identifier the segment maps to
        """
        return self.segment_id.split("_")[0]


class SegmentIndexMapper:
    """
    Map indices of one or more sequence segments into
    CouplingsModel internal numbering space. Can also
    be used to (trivially) remap indices for a single sequence.
    """
    def __init__(self, focus_mode, first_index, *segments):
        """
        Create index mapping from individual segments

        Parameters
        ----------
        focus_mode : bool
            Set to true if model was inferred in focus mode,
            False otherwise.
        first_index : int
            Index of first position in model/sequence.
            For nonfocus mode, should always be one. For focus
            mode, corresponds to index given in sequence header
            (1 if not in alignment)
        *segments: evcouplings.couplings.mapping.Segment:
            Segments containing numberings for each
            individual segment
        """
        # store segments so we retain full information
        self.segments = deepcopy(segments)

        # build up target indices by going through all segments
        self.target_pos = []
        for s in segments:
            if focus_mode:
                # in focus mode, we simply assemble the
                # ranges of continuous indices, because
                # numbering in model is also continuous
                cur_target = range(
                    s.region_start, s.region_end + 1
                )
            else:
                # in non-focus mode, we need to assemble
                # the indices of actual model positions,
                # since the numbering in model may be
                # discontinuous
                cur_target = s.positions

            # create tuples of (segment_id, target_pos)
            self.target_pos += list(zip(
                [s.segment_id] * len(cur_target),
                cur_target
            ))

        # create correspond list of model positions;
        # note that in focus mode, not all of these
        # positions might actually be in the model
        # (if they correspond to lowercase columns)
        self.model_pos = list(range(
            first_index, first_index + len(self.target_pos)
        ))

        # mapping from target sequences (segments) into
        # model numbering space (continuous numbering)
        self.target_to_model = dict(
            zip(self.target_pos, self.model_pos)
        )

        # inverse mapping from model numbering into target
        # numbering
        self.model_to_target = dict(
            zip(self.model_pos, self.target_pos)
        )

    def patch_model(self, model, inplace=True):
        """
        Change numbering of CouplingModel object
        so that it uses segment-based numbering

        Parameters
        ----------
        model : CouplingsModel
            Model that will be updated to segment-
            based numbering
        inplace : bool, optional (default: True)
            If True, change passed model; otherwise
            returnnew object

        Returns
        -------
        CouplingsModel
            Model with updated numbering
            (if inplace is False, this will
            point to original model)

        Raises
        ------
        ValueError
            If segment mapping does not match
            internal model numbering
        """
        if not inplace:
            model = deepcopy(model)

        try:
            mapped = [
                self.model_to_target[pos]
                for pos in model.index_list
            ]
        except KeyError:
            raise ValueError(
                "Mapping from target to model positions does "
                "not contain all positions of internal model numbering"
            )

        # update model mapping
        model.index_list = mapped

        # return updated model
        return model

    def __map(self, indices, mapping_dict):
        """
        Applies index mapping either to a single index,
        or to a list of indices

        Parameters
        ----------
        indices: int, or (str, int), or lists thereof
            Indices in input numbering space

        mapping_dict : dict(int->(str, int)) or dict((str, int)-> int)
            Mapping from one indexing space into the other

        Returns
        -------
        list of int, or list of (str, int)
            Mapped indices
        """
        if isinstance(indices, Iterable) and not isinstance(indices, tuple):
            return [mapping_dict[x] for x in indices]
        else:
            return mapping_dict[indices]

    def __call__(self, segment_id, pos):
        """
        Function-style syntax for single position to be mapped
        (calls to_model method)

        Parameters
        ----------
        segment_id : str
            Identifier of segment
        pos : int
            Position in segment numbering

        Returns
        -------
        int
            Index in coupling object numbering space
        """
        return self.to_model((segment_id, pos))

    def to_target(self, x):
        """
        Map model index to target index

        Parameters
        ----------
        x : int, or list of ints
            Indices in model numbering

        Returns
        -------
        (str, int), or list of (str, int)
            Indices mapped into target numbering.
            Tuples are (segment_id, index_in_segment)
        """
        return self.__map(x, self.model_to_target)

    def to_model(self, x):
        """
        Map target index to model index

        Parameters
        ----------
        x : (str, int), or list of (str, int)
            Indices in target indexing
            (segment_id, index_in_segment)

        Returns
        -------
        int, or list of int
            Monomer indices mapped into couplings object numbering
        """
        return self.__map(x, self.target_to_model)


def segment_map_ecs(ecs, mapper):
    """
    Map EC dataframe in model numbering into
    segment numbering

    Parameters
    ----------
    ecs : pandas.DataFrame
        EC table (with columns i and j)

    Returns
    -------
    pandas.DataFrame
        Mapped EC table (with columns i and j
        mapped, and additional columns
        segment_i and segment_j)
    """
    ecs = deepcopy(ecs)

    def _map_column(col):
        seg_col = "segment_" + col
        # create new dataframe with two columns
        # 1) mapped segment, 2) mapped position
        col_m = pd.DataFrame(
            mapper.to_target(ecs.loc[:, col]),
            columns=[seg_col, col]
        )
        # assign values instead of Series, because otherwise
        # wrong values end up in column
        ecs.loc[:, col] = col_m.loc[:, col].values
        ecs.loc[:, seg_col] = col_m.loc[:, seg_col].values

    # map both position columns (and add segment id)
    _map_column("i")
    _map_column("j")

    return ecs


class MultiSegmentCouplingsModel(CouplingsModel):
    """
    Complex specific Couplings Model that handles
    segments and provides the option to convert model
    into inter-segment only.
    """

    def __init__(self, filename, *segments,
                 precision="float32", file_format="plmc_v2", **kwargs):

        """
        filename : str
            Binary Jij file containing model parameters from plmc software
        precision : {"float32", "float64"}, default: "float32"
            Sets if input file has single (float32) or double precision (float64)
        }
        file_format : {"plmc_v2", "plmc_v1"}, default: "plmc_v2"
            File format of parameter file.
        segments: list of evcouplings.couplings.Segment

        TODO: have a additional constructor/class method that can take an existing
        loaded instance of a CouplingsModel and turn it into a MultiSegmentCouplingsModel
        """

        super().__init__(filename, precision, file_format, **kwargs)

        # initialize the segment index mapper to update model numbering
        if len(segments) == 0:
            raise ValueError("Must provide at least one segment for MultiSegmentCouplingsModel")

        first_segment = segments[0]
        index_start = first_segment.region_start
        r = SegmentIndexMapper(
            True,  # use focus mode
            index_start,  # first index of first segment
            *segments
        )

        # update model numbering
        r.patch_model(model=self)

    def to_inter_segment_model(self):
        """
        Convert model to inter-segment only
        parameters, ie the J_ijs that correspond
        to inter-protein or inter-domain residue pairs.
        All other parameters are set to 0.

        Returns
        -------
        CouplingsModel
            Copy of object turned into inter-only Epistatic model
        """

        h_i = np.zeros((self.L, self.num_symbols))
        J_ij = np.zeros(self.J_ij.shape)

        for idx_i, i in enumerate(self.index_list):
            for idx_j, j in enumerate(self.index_list):
                if i[0] != j[0]:  # if the segment identifier is different
                    J_ij[idx_i, idx_j] = self.J_ij[idx_i, idx_j]

        ci = deepcopy(self)
        ci.h_i = h_i
        ci.J_ij = J_ij
        ci._reset_precomputed()
        return ci
