import pandas as pd
import glob
import ruamel.yaml as yaml
from evcouplings.couplings import enrichment
from evcouplings.couplings import add_mixture_probability
import math
import sys
from evcouplings.compare import DistanceMap
import os

import pandas as pd
import glob
import ruamel.yaml as yaml
from evcouplings.couplings import enrichment
from evcouplings.couplings import add_mixture_probability
import math
import sys
from evcouplings.compare import DistanceMap
import os

def radius_distance_enrichment(ecs, distances1, distances2, num_pairs=1.0, score="cn", min_seqdist=6, distance_threshold=5,
							  ):
	"""
	Calculate EC "enrichment" as first described in
	Hopf et al., Cell, 2012.


	Parameters
	----------
	ecs : pd.DataFrame
		Dataframe containing couplings
	num_pairs : int or float, optional (default: 1.0)
		Number of ECs to use for enrichment calculation.
		- If float, will be interpreted as fraction of the
		length of the sequence (e.g. 1.0*L)
		- If int, will be interpreted as
		absolute number of pairs
	score : str, optional (default: cn)
		Pair coupling score used for calculation
	min_seqdist : int, optional (default: 6)
		Minimum sequence distance of couplings that will
		be included in the calculation

	Returns
	-------
	enrichment_table : pd.DataFrame
		Sorted table with enrichment values for each
		position in the sequence
	"""

	# check if the provided table has segments...
	if "segment_i" in ecs.columns and "segment_j" in ecs.columns:
		has_segments = True
	# ... and if not, create them
	else:
		has_segments = False
		ecs["segment_i"] = "A_1"
		ecs["segment_j"] = "A_1"

	# stack dataframe so it contains each
	# EC twice as forward and backward pairs
	# (i, j) and (j, i)
	flipped = ecs.rename(
		columns={
			"i": "j", "j": "i", "A_i": "A_j", "A_j": "A_i",
			"segment_i": "segment_j", "segment_j": "segment_i"
		}
	)

	stacked_ecs = ecs.append(flipped)

	# determine how many positions ECs are over using the combined dataframe
	num_pos = len(stacked_ecs.groupby(["i", "A_i", "segment_i"]))

	# calculate absolute number of pairs if
	# fraction of length is given
	if isinstance(num_pairs, float):
		num_pairs = int(math.ceil(num_pairs * num_pos))

	# sort the stacked ECs
	stacked_sorted_ecs = stacked_ecs.query(
		"abs(i-j) >= {}".format(min_seqdist)
	).sort_values(
		by=score, ascending=False
	)

	# take the top num *2 (because each EC represented twice)
	top_ecs = stacked_sorted_ecs[0:num_pairs * 2]

	sliding_window_data = []
	# calculate sum of EC scores for each position
	for name, _ in top_ecs.groupby(["i", "A_i", "segment_i"]):
		i, A_i, segment_i = name
		i_str = str(i)
		segment_ecs = top_ecs.query("segment_i == @segment_i")
		
		if segment_i == "A_1":
			nearby_residues = list(distances1.query("i == @i_str").j) + [i]
		else:
			nearby_residues = list(distances2.query("i == @i_str").j) + [i]

		nearby = segment_ecs.query("i in @nearby_residues")
		#print(nearby.loc[:, score])
		summed = sum(nearby.loc[:, score])
		sliding_window_data.append([i, A_i, segment_i, summed])

	ec_sums = pd.DataFrame(
		sliding_window_data,
		columns = ["i", "A_i", "segment_i", "cn"]
	)
   # print(ec_sums)
	
	# average EC strength for top ECs
	avg_degree = top_ecs.loc[:, score].sum() / len(top_ecs)

	# "enrichment" is ratio how much EC strength on
	# an individual position exceeds average strength in top
	ec_sums.loc[:, "enrichment"] = ec_sums.loc[:, score] / avg_degree

	# if the ecs had segment information, return a segment column
	if has_segments:
		e = ec_sums.reset_index().loc[:, ["i", "A_i", "segment_i", "enrichment"]]
	else:
		e = ec_sums.reset_index().loc[:, ["i", "A_i", "enrichment"]]

	return e.sort_values(by="enrichment", ascending=False)


def sliding_window_enrichment(ecs, num_pairs=1.0, score="cn", min_seqdist=0, window_size=5):
	"""
	Calculate EC "enrichment" as first described in
	Hopf et al., Cell, 2012.


	Parameters
	----------
	ecs : pd.DataFrame
		Dataframe containing couplings
	num_pairs : int or float, optional (default: 1.0)
		Number of ECs to use for enrichment calculation.
		- If float, will be interpreted as fraction of the
		length of the sequence (e.g. 1.0*L)
		- If int, will be interpreted as
		absolute number of pairs
	score : str, optional (default: cn)
		Pair coupling score used for calculation
	min_seqdist : int, optional (default: 6)
		Minimum sequence distance of couplings that will
		be included in the calculation

	Returns
	-------
	enrichment_table : pd.DataFrame
		Sorted table with enrichment values for each
		position in the sequence
	"""

	# check if the provided table has segments...
	if "segment_i" in ecs.columns and "segment_j" in ecs.columns:
		has_segments = True
	# ... and if not, create them
	else:
		has_segments = False
		ecs["segment_i"] = "A_1"
		ecs["segment_j"] = "A_1"

	# stack dataframe so it contains each
	# EC twice as forward and backward pairs
	# (i, j) and (j, i)
	flipped = ecs.rename(
		columns={
			"i": "j", "j": "i", "A_i": "A_j", "A_j": "A_i",
			"segment_i": "segment_j", "segment_j": "segment_i"
		}
	)

	stacked_ecs = ecs.append(flipped)

	# determine how many positions ECs are over using the combined dataframe
	num_pos = len(stacked_ecs.groupby(["i", "A_i", "segment_i"]))

	# calculate absolute number of pairs if
	# fraction of length is given
	if isinstance(num_pairs, float):
		num_pairs = int(math.ceil(num_pairs * num_pos))

	# sort the stacked ECs
	stacked_sorted_ecs = stacked_ecs.query(
		"abs(i-j) >= {}".format(min_seqdist)
	).sort_values(
		by=score, ascending=False
	)

	# take the top num *2 (because each EC represented twice)
	top_ecs = stacked_sorted_ecs[0:num_pairs * 2]

	sliding_window_data = []
	# calculate sum of EC scores for each position
	for name, _ in top_ecs.groupby(["i", "A_i", "segment_i"]):
		i, A_i, segment_i = name
		segment_ecs = top_ecs.query("segment_i == @segment_i")
		nearby = segment_ecs.query("abs(i - @i)<@window_size")
		#print(nearby.loc[:, score])
		summed = sum(nearby.loc[:, score])
		sliding_window_data.append([i, A_i, segment_i, summed])

	ec_sums = pd.DataFrame(
		sliding_window_data,
		columns = ["i", "A_i", "segment_i", "cn"]
	)
   # print(ec_sums)
	
	# average EC strength for top ECs
	avg_degree = top_ecs.loc[:, score].sum() / len(top_ecs)

	# "enrichment" is ratio how much EC strength on
	# an individual position exceeds average strength in top
	ec_sums.loc[:, "enrichment"] = ec_sums.loc[:, score] / avg_degree

	# if the ecs had segment information, return a segment column
	if has_segments:
		e = ec_sums.reset_index().loc[:, ["i", "A_i", "segment_i", "enrichment"]]
	else:
		e = ec_sums.reset_index().loc[:, ["i", "A_i", "enrichment"]]

	return e.sort_values(by="enrichment", ascending=False)

def combine_enrichment_scores(ecs, distancemap1, distancemap2):

	intra1_ecs = ecs.query("segment_i == segment_j == 'A_1'")
	intra2_ecs = ecs.query("segment_i == segment_j == 'B_1'")
	inter_ecs = ecs.query("segment_i != segment_j")

	# calculate intra-protein enrichment
	intra1_enrichment = enrichment(intra1_ecs, min_seqdist=6)
	intra1_enrichment["segment_i"] = "A_1"

	intra2_enrichment = enrichment(intra2_ecs, min_seqdist=6)
	intra2_enrichment["segment_i"] = "B_1"

	enrichment_table = pd.concat([intra1_enrichment, intra2_enrichment])

	# calculate inter-protein enrichment
	# for inter ECs, sequence distance is 0
	inter_enrichment = enrichment(inter_ecs, min_seqdist=0, num_pairs=50)
	inter_enrichment = inter_enrichment.rename({"enrichment": "inter_enrichment"}, axis=1)
	enrichment_table = enrichment_table.merge(inter_enrichment, on=["i", "A_i", "segment_i"])

	# calculate inter-protein sliding window enrichment
	inter_enrichment = sliding_window_enrichment(inter_ecs, min_seqdist=0, num_pairs=50, window_size=5)
	inter_enrichment = inter_enrichment.rename({"enrichment": "inter_window5_enrichment"}, axis=1)

	enrichment_table = enrichment_table.merge(inter_enrichment, on=["i", "A_i", "segment_i"])

	# calculate inter-protein sliding window enrichment
	inter_enrichment = sliding_window_enrichment(inter_ecs, min_seqdist=0, num_pairs=50, window_size=2)
	inter_enrichment = inter_enrichment.rename({"enrichment": "inter_window2_enrichment"}, axis=1)

	enrichment_table = enrichment_table.merge(inter_enrichment, on=["i", "A_i", "segment_i"])

	# calculate inter-protein radius enrichment
	inter_enrichment = radius_distance_enrichment(
		inter_ecs, distancemap1, distancemap2, min_seqdist=0, num_pairs=50
	)
	inter_enrichment = inter_enrichment.rename({"enrichment": "inter_radius_enrichment"}, axis=1)

	enrichment_table = enrichment_table.merge(inter_enrichment, on=["i", "A_i", "segment_i"])

	return enrichment_table


def create_enrichment_table(ecs, first_distance_map, second_distance_map):

	if first_distance_map:
		distances_1 = first_distance_map.contacts(max_dist=5.)
		distances_1 = pd.concat([distances_1, distances_1.rename({"i":"j","j":"i"})])
	else:
		distances_1 = pd.DataFrame({
			"i": [], 
			"j": []
		})

	if second_distance_map:
		distances_2 = second_distance_map.contacts(max_dist=5.)
		distances_2 = pd.concat([distances_2, distances_2.rename({"i":"j","j":"i"})])
	else:
		distances_2 = pd.DataFrame({
			"i": [], 
			"j": []
		})

	enrichment_table = combine_enrichment_scores(ecs, distances_1, distances_2)

	return enrichment_table

def add_enrichment(enrichment, df):
    
    def _seg_to_enrich(enrich_df, ec_df, enrichment_column):
        s_to_e = {(x,y):z for x,y,z in zip(enrich_df.i, enrich_df.segment_i, enrich_df[enrichment_column])}
        ec_df["_e_i"] =[s_to_e[(x,y)] if (x,y) in s_to_e else 0 for x,y in zip(ec_df.i, ec_df.segment_i)]
        ec_df["_e_j"] =[s_to_e[(x,y)] if (x,y) in s_to_e else 0 for x,y in zip(ec_df.j, ec_df.segment_j)]
        
        return ec_df
    
    ### intra enrichment
    df = _seg_to_enrich(enrichment, df, "enrichment")
    df["intra_enrich_max"] = df[["_e_i", "_e_j"]].max(axis=1)
    df["intra_enrich_min"] = df[["_e_i", "_e_j"]].min(axis=1)
    
    ### inter single enrichment
    df = _seg_to_enrich(enrichment, df, "inter_enrichment")
    df["inter_enrich_single_max"] = df[["_e_i", "_e_j"]].max(axis=1)
    df["inter_enrich_single_min"] = df[["_e_i", "_e_j"]].min(axis=1)
    
    ### inter window enrichment
    df = _seg_to_enrich(enrichment, df, "inter_window5_enrichment")
    df["inter_enrich5_window_max"] = df[["_e_i", "_e_j"]].max(axis=1)
    df["inter_enrich5_window_min"] = df[["_e_i", "_e_j"]].min(axis=1)
    
    df = _seg_to_enrich(enrichment, df, "inter_window2_enrichment")
    df["inter_enrich2_window_max"] = df[["_e_i", "_e_j"]].max(axis=1)
    df["inter_enrich2_window_min"] = df[["_e_i", "_e_j"]].min(axis=1)
    
    ### inter window enrichment
    df = _seg_to_enrich(enrichment, df, "inter_radius_enrichment")
    df["inter_enrich_radius_max"] = df[["_e_i", "_e_j"]].max(axis=1)
    df["inter_enrich_radius_min"] = df[["_e_i", "_e_j"]].min(axis=1)
    
    return df


