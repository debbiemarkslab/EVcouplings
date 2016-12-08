#!/usr/bin/env python

"""
Distance map calculation code, faster scipy cdist version
Thomas A. Hopf, 08.02.2015

Now handles PDB insertion codes.
Now also handles PDB files with more than one Uniprot sequence per PDB chain.
"""

from math import fabs
from os import system, path
from sys import argv, stderr, exit
from collections import defaultdict
from tempfile import mkstemp
from copy import deepcopy

import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_LARGE_DISTANCE = 100000.0
_NEWLINE = "\n"
_NUM_DIGITS = 3


class ContactMap:

    def __init__(self, filename=None, uniprot_index_transform=lambda x: x, pdb_index_transform=lambda x: x,
                 distance_dict=None):

        self.contact_map = defaultdict(defaultdict)
        self.uniprot_to_pdb_index = {}
        self.pdb_to_uniprot_index = {}
        self.index_to_residue = {}

        # Not loading from file is solely to allow pdb_maps.py compatibility layer
        if filename is not None:
            # read all pairwise distances into dict
            with open(filename) as f:
                for line in f:
                    # store distance information
                    up1, pdb1, res1, up2, pdb2, res2, dist = line.rstrip().split(" ")
                    # apply transformations to Uniprot and PDB indices (e.g. to shift by offset)
                    up1, up2 = uniprot_index_transform(int(up1)), uniprot_index_transform(int(up2))
                    pdb1, pdb2 = pdb_index_transform(pdb1), pdb_index_transform(pdb2)

                    self.contact_map[up1][up2] = self.contact_map[up2][up1] = float(dist)

                    # also store information about sequence indeces and the residue types for convenience
                    self.uniprot_to_pdb_index[up1] = pdb1
                    self.uniprot_to_pdb_index[up2] = pdb2
                    self.pdb_to_uniprot_index[pdb1] = up1
                    self.pdb_to_uniprot_index[pdb2] = up2
                    self.index_to_residue[up1] = res1
                    self.index_to_residue[up2] = res2
        elif distance_dict is not None:
            self.contact_map = deepcopy(distance_dict)
        else:
            raise ValueError("Must give filename or distance matrix to initialize object")

        # also create a matrix representation
        self.first_residue, self.last_residue = min(self.contact_map), max(self.contact_map)
        self.distance_matrix = numpy.ones((self.last_residue+1, self.last_residue+1)) * _LARGE_DISTANCE

        for i in self.contact_map:
            self.distance_matrix[i, i] = 0
            for j in self.contact_map[i]:
                self.distance_matrix[i, j] = self.contact_map[i][j]

    def plot_interactive(
            self, pair_list, num_pairs, couplings=None, ax=None,
            distance_threshold=5.0,
            pdb_contact_thresholds=((5.0, "0.75"), (8.0, "0.92")),
            min_sequence_distance=6, bounded_by_structure=False, cm_range=None,
            show_missing_coverage=False, invert_y_axis=True,
            secondary_structure=None, secondary_structure_to_labels=True,
            ss_vertical_dist=7, ss_horizontal_dist=6,
            ec_point_size=50, pdb_point_size=150,
            monochrome=True, monochrome_color="k",
            helix_turn_length=2, min_sse_length=2, sse_width=1):
        """
        Plots a contact map together with predicted ECs.

        # FIXED: make y-axis invertable
        # FIXED: remove integer-based parsing of residues
        # FIXED: keep MSE residues
        # FIXED: flip axis to top
        # FIXED: draw secondary structure
        # FIXED: fix secondary structure not covering complete sequence
        # FIXED: secondary structure location as parameter
        # FIXED: remove dependence on EVcouplings object
        # FIXED: plotting without ECs on
        # FIXED: take off missing segments (make False)
        # FIXED: tick labels outwards? remove on other axis?
        # FIXED: selection of distance ranges
        # FIXED: overlay different colors for diff. distances
        # FIXED: change point size        
        # FIXED: modify colors (only one) -- make parameter

        # TODO: colorgradient by EC strengths
        """
        # import plotting_tools

        if ax is None:
            ax = plt.gca()
        ax.set_aspect('equal')

        # assemble predicted contacts to be shown in plot

        x, y, scores, labels, colors = [], [], [], [], []
        num_shown = 0

        if pair_list is not None:
            for i, j, score in pair_list:
                if fabs(i-j) >= min_sequence_distance:
                    # one half of symmetric contact matrix
                    x.append(i)
                    y.append(j)
                    # other symmetric case too
                    x.append(j)
                    y.append(i)
                    scores += [score]*2

                    if i in self.contact_map and j in self.contact_map:
                        if self.contact_map[i][j] <= distance_threshold:
                            colors += ["DarkBlue"]*2
                        else:
                            colors += ["FireBrick"]*2
                        dist_str = " - {:.2f} A".format(self.contact_map[i][j])
                    else:  # couplings residue not in structure
                        colors += ["Gold"]*2
                        dist_str = ""

                    # generate label string for interactive plots

                    res_i, res_j = "", ""
                    if couplings is not None:
                        res_i, res_j = couplings.seq(i), couplings.seq(j)
                    else:
                        # in this case, residues missing in the structure can not be labeled with a residue name
                        if i in self.index_to_residue:
                            res_i = self.index_to_residue[i]
                        if j in self.index_to_residue:
                            res_j = self.index_to_residue[j]

                    labels.append("{}{}, {}{} ({:.2f}){}".format(res_i, i, res_j, j, score, dist_str))
                    labels.append("{}{}, {}{} ({:.2f}){}".format(res_j, j, res_i, i, score, dist_str))

                    num_shown += 1

                if num_shown >= num_pairs:
                    break

        # draw rectangles for parts missing alignment coverage
        if show_missing_coverage and couplings is not None:
            missing_alignment_res = sorted([r for r in self.contact_map if r not in couplings.map_uniprot_index])
            missing_segments = self._find_segments(missing_alignment_res)
            rectangle_size = self.last_residue - self.first_residue + 1
            for (s_s, s_e) in missing_segments:
                ax.add_patch(Rectangle((self.first_residue, s_s), rectangle_size, s_e - s_s,
                                       facecolor="AliceBlue", edgecolor="none", linewidth=0, zorder=0))
                ax.add_patch(Rectangle((s_s, self.first_residue), s_e - s_s, rectangle_size,
                                       facecolor="AliceBlue", edgecolor="none", linewidth=0, zorder=0))

        # plot structure contacts and ECs

        for (dt, color) in sorted(pdb_contact_thresholds, reverse=True):
            x_xtal, y_xtal = numpy.nonzero(self.distance_matrix <= dt)
            ax.scatter(x_xtal, y_xtal, color=color, marker="o", s=pdb_point_size, edgecolors="none")

        if monochrome:
            scatter_colors = monochrome_color
        else:
            scatter_colors = colors

        ax_ec = ax.scatter(x, y, c=scatter_colors, marker="o", s=ec_point_size, edgecolors="none", cmap=plt.cm.Blues)

        # infer displayed residue range

        if bounded_by_structure or pair_list is None or len(pair_list) == 0:
            plot_range = (self.first_residue, self.last_residue)
        else:
            plot_range = (min(x), max(x))

        # ... or override with user-defined setting
        if cm_range is not None:
            plot_range = cm_range

        # plot secondary structure

        if secondary_structure is not None:
            first_res, last_res = plot_range[0], plot_range[1]

            # check if secondary structure goes together with axis labels
            # or on oppposite side of plot
            if secondary_structure_to_labels:
                vertical_loc = first_res - ss_vertical_dist
                horizontal_loc = first_res - ss_horizontal_dist
            else:
                vertical_loc = last_res + ss_vertical_dist
                horizontal_loc = last_res + ss_horizontal_dist

            """
            # disabled for now to reduce dependencies
            secondary_structure_str = "".join([secondary_structure.get(i, "-") for i in range(first_res, last_res)])
            start, end, segments = plotting_tools.find_secondary_structure_segments(secondary_structure_str, offset=first_res)

            plotting_tools.plot_secondary_structure(
                segments, center=horizontal_loc, sequence_start=first_res, sequence_end=end,
                strand_width_factor=0.5, helix_turn_length=helix_turn_length, min_sse_length=min_sse_length, width=sse_width,
            )
            plotting_tools.plot_secondary_structure(
                segments, center=vertical_loc, sequence_start=first_res, sequence_end=end,
                strand_width_factor=0.5, helix_turn_length=helix_turn_length, min_sse_length=min_sse_length, width=sse_width,
                horizontal=False
            )
            """

        # set axis ranges and tick locations

        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)

        ax.yaxis.set_ticks_position("left")
        if invert_y_axis:
            ax.invert_yaxis()
            ax.xaxis.set_ticks_position("top")
        else:
            ax.xaxis.set_ticks_position("bottom")

        # interactive mpld3 features

        try:
            from mpld3 import plugins
            html_temp = '<div style="background-color:rgba(255,255,255,0.75);"><b>&nbsp;{}&nbsp;</b></div>'
            labels_html = [html_temp.format(l) for l in labels]
            plugins.connect(plt.gcf(), plugins.PointHTMLTooltip(ax_ec, labels_html, voffset=-35, hoffset=-5))
        except ImportError:
            print("mpld3 library for interactive features could not be imported", file=stderr)

    def plot(self, pair_list, num_pairs, out_file=None, distance_threshold=5.0, min_sequence_distance=6, new_figure=False):
        """
        Deprecated, use plot_interactive instead
        """
        pred_contacts_matrix_tp = numpy.zeros((self.last_residue+1, self.last_residue+1))
        pred_contacts_matrix_fp = numpy.zeros((self.last_residue+1, self.last_residue+1))

        num_shown = 0
        for i, j, score in pair_list:
            if i <= self.last_residue and j <= self.last_residue and i >= 0 and j >= 0:
                if fabs(i-j) >= min_sequence_distance:
                    if i in self.contact_map and j in self.contact_map and self.contact_map[i][j] <= distance_threshold:
                        pred_contacts_matrix_tp[i, j] = pred_contacts_matrix_tp[j, i] = 1
                    else:
                        pred_contacts_matrix_fp[i, j] = pred_contacts_matrix_fp[j, i] = 1
                    num_shown += 1

                if num_shown >= num_pairs:
                    break
        if new_figure:
            plt.figure()
        # else:
        #   fig = plt.gcf()

        plt.spy(self.distance_matrix <= distance_threshold, marker='o', color='0.7', markeredgewidth=0)
            
        plt.spy(pred_contacts_matrix_tp, marker='o', color='g', markersize=8, markeredgewidth=0)
        plt.spy(pred_contacts_matrix_fp, marker='x', color='r', markersize=8)

        plt.xlim(self.first_residue, self.last_residue)
        plt.ylim(self.first_residue, self.last_residue)
        plt.gca().invert_yaxis()

        if out_file is None:
            plt.draw()
        else:
            plt.savefig(out_file, bbox=0)
            plt.close()

    def _find_segments(self, index_range):
        if len(index_range) == 0:
            return []

        # segment end is non-inclusive
        segment_start = index_range[0]
        segment_end = segment_start + 1 
        segments = []
        i = 0
        while i < len(index_range) - 1:
            if index_range[i] + 1 == index_range[i + 1]:
                segment_end += 1
            else:
                segments.append([segment_start, segment_end])
                segment_start = index_range[i + 1]
                segment_end = segment_start + 1     
            i += 1
        segments.append([segment_start, segment_end])
        return segments

    def plot_interactive_old(self, couplings, pair_list, num_pairs, distance_threshold=5.0,
            min_sequence_distance=6, bound_by_structure=False, cm_range=None,
            show_missing_coverage=True):
        """
        Plots a contact map together with predicted ECs.
        """
        x, y, labels, colors = [], [], [], []
        x_xtal, y_xtal = numpy.nonzero(self.distance_matrix <= distance_threshold)

        num_shown = 0
        for i, j, score in pair_list:
            if fabs(i-j) >= min_sequence_distance:
                x.append(i); x.append(j);
                y.append(j); y.append(i);

                if i in self.contact_map and j in self.contact_map:
                    if self.contact_map[i][j] <= distance_threshold:
                        colors += ["DarkBlue"]*2
                    else:
                        colors += ["FireBrick"]*2
                    dist_str = " - {:.2f} A".format(self.contact_map[i][j])
                else:  # couplings residue not in structure
                    colors += ["Gold"]*2
                    dist_str = ""

                labels.append("{}{}, {}{} ({:.2f}){}".format(couplings.seq(i), i, couplings.seq(j), j, score, dist_str))
                labels.append("{}{}, {}{} ({:.2f}){}".format(couplings.seq(j), j, couplings.seq(i), i, score, dist_str))
                num_shown += 1

            if num_shown >= num_pairs:
                break

        fig = plt.gcf()
        plt.gca().set_aspect('equal')
        
        # draw rectangles for parts missing alignment coverage
        if show_missing_coverage:
            missing_alignment_res = sorted([r for r in self.contact_map if r not in couplings.map_uniprot_index])
            missing_segments = self._find_segments(missing_alignment_res)
            rectangle_size = self.last_residue - self.first_residue + 1
            for (s_s, s_e) in missing_segments:
                plt.gca().add_patch(Rectangle((self.first_residue, s_s), rectangle_size, s_e - s_s, 
                                               facecolor="AliceBlue", edgecolor="none", linewidth=0, zorder=0))
                plt.gca().add_patch(Rectangle((s_s, self.first_residue), s_e - s_s, rectangle_size, 
                                               facecolor="AliceBlue", edgecolor="none", linewidth=0, zorder=0))

        # plot structure contacts and ECs
        ax_xtal = plt.scatter(x_xtal, y_xtal, color="0.7", marker=".", s=200, edgecolors="none")
        ax_ec = plt.scatter(x, y, color=colors, marker=".", s=200)

        # set displayed residue range
        if bound_by_structure:
            plot_range = (self.first_residue, self.last_residue)
        else:
            plot_range = (min(x), max(x))
        if cm_range is not None:
            plot_range = cm_range

        plt.gca().set_xlim(plot_range)
        plt.gca().set_ylim(plot_range)
        plt.gca().invert_yaxis()
        
        # interactive features
        try:
            from mpld3 import plugins
            html_temp = '<div style="background-color:rgba(255,255,255,0.75);"><b>&nbsp;{}&nbsp;</b></div>' 
            labels_html = [html_temp.format(l) for l in labels]
            plugins.connect(fig, plugins.PointHTMLTooltip(ax_ec, labels_html, voffset=-35, hoffset=-5))
        except ImportError:
            print("mpld3 library for interactive features could not be imported", file=stderr)
    
    def contact_matrix(self, couplings, distance_threshold=5, na_value=_LARGE_DISTANCE):
        contact_mat = numpy.ones((couplings.seq_len, couplings.seq_len)) * na_value
        for i in xrange(0, couplings.seq_len):
            for j in xrange(i+1, couplings.seq_len):
                m_i = couplings.index_to_uniprot_offset[i]
                m_j = couplings.index_to_uniprot_offset[j]
                if m_i in self.contact_map and m_j in self.contact_map[m_i]:
                    contact_mat[i][j] = contact_mat[j][i] = self.contact_map[m_i][m_j]

        return contact_mat

    def calculate_tp_rate(self, predicted_contacts, max_num_pairs, distance_threshold=5.0, min_sequence_distance=6):
        num_tp = 0
        num_counted = 0
        tp_rate = []
        for i, j, score in predicted_contacts:
            if i in self.contact_map and j in self.contact_map:
                if fabs(i-j) >= min_sequence_distance:
                    if self.contact_map[i][j] <= distance_threshold:
                        num_tp += 1
                    num_counted += 1
                    tp_rate.append(num_tp/float(num_counted))
            if num_counted >= max_num_pairs:
                break

        return tp_rate


# methods for providing easy access to contact maps

def load_contact_map(pdb_id, pdb_chain, output_directory=".", ca_dist=False, 
                     force_recalculate=False, uniprot_ac_filter=None, **args):
    """
    loads, and if necessary generates, a distance map of a PDB structure
    """
    contact_map_file = path.join(output_directory, "{}{}_contact_map.txt".format(pdb_id, pdb_chain))
    if not path.exists(contact_map_file) or force_recalculate:
        make_contact_map(pdb_id, pdb_chain, contact_map_file, ca_dist, uniprot_ac_filter)
    conmap = ContactMap(contact_map_file, **args)

    return conmap


# methods for calculating distance map

def calculate_distance_list(pdb_file, chain, sifts_file, model_number=0, ca_dist=False, uniprot_ac_filter=None):
    """
    Calculates a list of minimum residue distances for all residue in a PDB chain
    """
    from xml.dom import minidom
    from Bio.PDB.PDBParser import PDBParser
    from scipy.spatial.distance import cdist

    # first, read SIFTS mapping
    xmldoc = minidom.parse(sifts_file)
    itemlist = xmldoc.getElementsByTagName('residue')
    pdb_to_up, index_to_res_name = {}, {}
    uniprot_ac_numbers = set()

    for s in itemlist:
        cross_refs = s.getElementsByTagName('crossRefDb')
        pdb_res_num, pdb_res_name, pdb_chain, up_res_num, up_res_name = None, None, None, None, None

        for r in cross_refs:
            if r.attributes['dbSource'].value == "PDB":
                pdb_res_num = r.attributes['dbResNum'].value
                pdb_res_name = r.attributes['dbResName'].value
                pdb_chain = r.attributes['dbChainId'].value
            elif r.attributes['dbSource'].value == "UniProt":
                up_res_num = r.attributes['dbResNum'].value
                up_res_name = r.attributes['dbResName'].value
                up_ac_number = r.attributes['dbAccessionId'].value

        if pdb_chain == chain and pdb_res_num is not None and up_res_num is not None:
            if uniprot_ac_filter is None or uniprot_ac_filter == up_ac_number:
                pdb_to_up[pdb_res_num] = up_res_num
                index_to_res_name[pdb_res_num] = up_res_name
                uniprot_ac_numbers.add(up_ac_number)

    if len(uniprot_ac_numbers) > 1:
        raise ValueError("Multiple Uniprot ACs for chain {}: {}".format(chain, uniprot_ac_numbers))

    # extract residues from 3D structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("somestructure", pdb_file)
    model = structure[model_number][chain]
    residue_list_unfiltered = [res.get_id() for res in model]

    # filter heteroatoms, but keep selenomethionines
    residue_list = sorted(
        [(het, resseq, icode) for (het, resseq, icode) in residue_list_unfiltered
         if het == " " or het == "H_MSE"]
    )

    # make sure residues have a CA atom if using C_alpha distance
    if ca_dist:
        residue_list = [res for res in residue_list if "CA" in model[res]]

    # compile atom coordinates into one NumPy matrix per residue for cdist calculation
    atom_coordinates = {}
    for res in residue_list:
        if ca_dist:
            atom_coordinates[res] = numpy.zeros((1, 3))
            atom_coordinates[res][0] = model[res]["CA"].get_coord()
        else:
            atom_coordinates[res] = numpy.zeros((len(model[res]), 3))
            for i, atom in enumerate(model[res]):
                atom_coordinates[res][i] = atom.get_coord()

    # calculate minimum atom distance for each residue pair and store
    pair_dist_list = []
    for r1 in residue_list:
        for r2 in residue_list:
            if r1 < r2:
                dist = numpy.min(cdist(atom_coordinates[r1], atom_coordinates[r2], 'euclidean'))

                # assemble string representation of residue name
                (het1, res1, icode1) = r1
                (het2, res2, icode2) = r2
                r1_str = str(res1) + icode1.strip()
                r2_str = str(res2) + icode2.strip()

                if r1_str in pdb_to_up and r2_str in pdb_to_up:
                    pair_dist_list.append(
                        (pdb_to_up[r1_str], r1_str, index_to_res_name[r1_str],
                         pdb_to_up[r2_str], r2_str, index_to_res_name[r2_str],
                         round(dist, _NUM_DIGITS))
                    )

    return pair_dist_list


def make_contact_map(
        pdb_id, chain, out_file=None, ca_dist=False, uniprot_ac_filter=None,
        pdb_file_name=None, sifts_file_name=None, model_number=0, verbose=False):
    """
    Calculates a distance map by downloading the ingredients from PDB and SIFTS, or using local files if available
    """
    if pdb_file_name is None:
        pdb_file_handle, pdb_file_name = mkstemp()
        system("curl -s http://www.rcsb.org/pdb/files/" + pdb_id + ".pdb > " + pdb_file_name)
        if verbose:
            print("PDB file:",  pdb_file_name, file=stderr)

    if sifts_file_name is None:
        sifts_file_handle, sifts_file_name = mkstemp()
        system("curl -s ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/" + pdb_id.lower() + ".xml.gz | zcat > " + sifts_file_name)
        if verbose:
            print("SIFTS file:", sifts_file_name, file=stderr)

    pair_dist_list = calculate_distance_list(pdb_file_name, chain, sifts_file_name, model_number, ca_dist, uniprot_ac_filter)
    if out_file is None:
        for pair_tuple in pair_dist_list:
            print(" ".join(map(str, pair_tuple)))
    else:
        with open(out_file, "w") as f:
            for pair_tuple in pair_dist_list:
                f.write(" ".join(map(str, pair_tuple)) + _NEWLINE)
            f.close()

    return pair_dist_list

if __name__ == "__main__":
    VERBOSE = True
    if len(argv) == 3:
        pdb_id, pdb_chain = argv[1:3]
        make_contact_map(pdb_id, pdb_chain, verbose=VERBOSE)
    elif len(argv) == 4:
        pdb_file, pdb_chain, sifts_file = argv[1:4]
        make_contact_map(None, pdb_chain, None, pdb_file, sifts_file, verbose=VERBOSE)
    else:
        print("            usage:", argv[0], "<pdb id> <chain>", file=stderr)
        print("alternative usage:", argv[0], "<pdb file> <chain> <SIFTS file>", file=stderr)
        exit(-1)
