import pandas as pd
from evcouplings.utils.system import run, verify_resources
from Bio.PDB import make_dssp_dict
import ruamel.yaml as yaml
import numpy as np

AA_SURFACE_AREA = {
	"A":121,
	"R":265,
	"D":187,
	"N":187,
	"C":148,
	"E":214,
	"Q":214,
	"G":97,
	"H":216,
	"I":195,
	"L":191,
	"K":230,
	"M":103,
	"F":228,
	"P":154,
	"S":143,
	"T":163,
	"W":264,
	"Y":255,
	"V":165,
	"X":np.nan
}

# run dssp

def run_dssp(binary, infile, outfile):
	"""
	"""
	cmd = [
		binary, 
		"-i", infile,
		"-o", outfile
	]
	return_code, stdout, stderr = run(cmd)

	verify_resources(
	        "DSSP returned empty file: "
	        "stdout={} stderr={} file={}".format(
	            stdout, stderr, outfile
	        ),
	        outfile
	    )

def read_dssp_output(filename):
	
	dssp_dict, _ = make_dssp_dict(filename)
	data = []
	for key, value in dssp_dict.items():

		# keys are formatted as (chain, ("", i, ""))
		i = key[1][1]

		res = value[0]
		asa = value[2]

		data.append({
			"i": i,
			"res": res, 
			"asa": asa
		})

	return pd.DataFrame(data)

def calculate_rsa(dataframe, AA_SURFACE_AREA, output_column="rsa"):
	
	dataframe.loc[:,output_column] = [x.asa/AA_SURFACE_AREA[x.res] for _,x in dataframe.iterrows()]
	return dataframe

def asa_run(file, prefix):

	binary = "/n/groups/marks/software/dssp"

	file_prefix = file.split(".pdb")[0]

	dssp_output_file = file_prefix+".dssp"
	rsa_output_file = file_prefix+".csv"
	try:
		run_dssp(
			binary, 
			file, 
			dssp_output_file
		)
	except:
		return pd.DataFrame({
			"i": [],
			"res": [], 
			"asa": [],
			"rsa": []
	})

	d = read_dssp_output(dssp_output_file)
	d = calculate_rsa(d, AA_SURFACE_AREA)

	return d

def combine_asa(remapped_pdb_files, prefix):

	data = pd.DataFrame({
			"i": [],
			"res": [], 
			"asa": []
	})

	for file in remapped_pdb_files:
		d = asa_run(file, prefix)
		data = pd.concat([data,d])

	means = data.groupby("i").rsa.mean()
	maxes = data.groupby("i").rsa.max()
	mins = data.groupby("i").rsa.min()
	
	return pd.DataFrame({
		"i": means.index, 
		"mean": list(means),
		"max": list(maxes),
		"min": list(mins)
	})

def add_asa(ec_df, asa, asa_column):

    s_to_e = {(x,y):z for x,y,z in zip(asa.i, asa.segment_i, asa[asa_column])}
    ec_df["asa_i"] =[s_to_e[(x,y)] if (x,y) in s_to_e else 0 for x,y in zip(ec_df.i, ec_df.segment_i)]
    ec_df["asa_j"] =[s_to_e[(x,y)] if (x,y) in s_to_e else 0 for x,y in zip(ec_df.j, ec_df.segment_j)]
    
    return ec_df


