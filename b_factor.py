#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone script to extract residue temperature factor values and use them to
calculate sequence/structural averages.

Arguments:
--pdb_path - path to the PDB file
--sequence_average -- boolean. Whether sequence averages should be calculated
--window_size -- int. Size of the window for sequence averages
--sequence_average -- boolean. Whether spatial context averages should be
calculated
--distance_cutoff -- float. The distance cutoff for the spatial context
averages (only residues within distance_cutoff angstroms are considered -
except when a Gaussian kernel is used, then all are considered)
--gaussian_kernel - boolean. Whether a Gaussian kernel should be used to weigh
b-factors based on the distance
"""

__author__ = "Jose Guilherme Almeida"
__credits__ = ["Jose Guilherme Almeida"]
__email__ = "josegcpa@ebi.ac.uk"

import argparse
import os
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.stats import norm
from collections import defaultdict

class ToDirectory(argparse.Action):
    """
    Action class to use as in add_argument to automatically return the absolute
    path when a path input is provided.
    """
    def __init__(self, option_strings, dest, **kwargs):
        super(ToDirectory, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(values))

def process_atom_info(line):
    """
    Takes an ATOM line from a PDB and outputs a dictionary with the atom no.
    (atom_no), atom name (atom), residue name (res_name), chain (chain),
    residue number (res_no), coordinates (x, y, z) and temperature factor
    (temp_factor.)

    Keyword arguments:
    line -- an ATOM PDB line.

    Output:
    atom_info -- dictionary with the items described above.
    """

    atom_info = {
        'atom_no': int(line[6:11].strip()),
        'atom': line[12:16].strip(),
        'res_name': line[17:20].strip(),
        'chain': line[21].strip(),
        'res_no': int(line[22:26].strip()),
        'x': float(line[30:38].strip()),
        'y': float(line[38:46].strip()),
        'z': float(line[46:54].strip()),
        'temp_factor': float(line[60:66].strip())
        }
    return atom_info

def parse_pdb(path):
    """
    Parses a PDB file and uses process_atom_info on each ATOM line.

    Keyword arguments:
    path -- string. path to the PDB file.

    Output:
    pdb_dict -- a dictionary with PDB chains as keys and dictionaries with
    residue identifiers as keys and atom_info dictionaries as values as values
    res_dict -- a dictionary with PDB chains as keys and lists of residue
    identifiers as keys
    """

    pdb_dict = defaultdict(lambda: defaultdict(list))
    res_dict = defaultdict(list)
    with open(path) as o:
        lines = o.readlines()
    for line in lines:
        if line[:4] == 'ATOM':
            atom_info = process_atom_info(line)
            identifier = '{}{}'.format(
                atom_info['res_name'],
                atom_info['res_no']
            )
            pdb_dict[atom_info['chain']][identifier].append(atom_info)
            if identifier not in res_dict[atom_info['chain']]:
                res_dict[atom_info['chain']].append(identifier)
    return pdb_dict,res_dict

def get_b_factor(pdb_dict,
                 res_dict,
                 sequence_average,
                 window_size,
                 space_average,
                 distance_cutoff,
                 gaussian_kernel,
                 dict_key='temp_factor'):
    """
    Extracts b-factors from each atom and calculates residue averaged
    b-factors. Using these values, averages with a sliding window on the
    sequence and spatial context averages are calculated. Spatial context
    averages can be calculated using a Gaussian kernel with mean = 0 and
    std = distance_cutoff.

    Keyword arguments:
    pdb_dict -- the output from parse_pdb (a dictionary with PDB chains as keys
    and dictionaries with residue identifiers as keys and atom_info
    dictionaries as values as values)
    res_dict -- the second output from parse_pdb (a dictionary with PDB chains
    as keys and lists of residue identifiers as keys)
    sequence_average -- boolean. Whether sequence averages should be calculated
    window_size -- int. Size of the window for sequence averages
    sequence_average -- boolean. Whether spatial context averages should be
    calculated
    distance_cutoff -- float. The distance cutoff for the spatial context
    averages (only residues within distance_cutoff angstroms are considered -
    except when a Gaussian kernel is used, then all are considered)
    gaussian_kernel - boolean. Whether a Gaussian kernel should be used to
    weigh b-factors based on the distance
    dict_key -- string. Defaults to temp_factor. The values to be used from
    the pdb_dict for calculations. Allows the generalization of this function
    to other things contained in the pdb_dict.
    """

    identifiers = []
    ca_coordinates = []
    temp_factor_list = []
    sequence_average_list = []
    space_average_list = []
    for chain in pdb_dict:
        sequence_values = [0 for _ in range(len(pdb_dict[chain]))]
        sequence_counts = sequence_values.copy()
        for i,res in enumerate(res_dict[chain]):
            temp_factor = 0
            for atom in pdb_dict[chain][res]:
                temp_factor += atom[dict_key]
            temp_factor = temp_factor / len(pdb_dict[chain][res])
            temp_factor_list.append(temp_factor)
            # Sequence average
            idx_range = [x for x in range(i - window_size,
                                          i + window_size + 1)]
            idx_range = [x for x in idx_range if x < len(res_dict[chain])]
            idx_range = [x for x in idx_range if x >= 0]
            for idx in idx_range:
                sequence_values[idx] += temp_factor
                sequence_counts[idx] += 1
            # Space average
            for atom in pdb_dict[chain][res]:
                if atom['atom'] == 'CA':
                    identifiers.append(res + chain)
                    ca_coordinates.append(
                        [atom['x'],atom['y'],atom['z']]
                    )

        if sequence_average == True:
            for v,c in zip(sequence_values,sequence_counts):
                sequence_average_list.append(v/c)

    if space_average == True:
        i = 0
        ca_coordinates = np.array(ca_coordinates)
        dist_matrix = squareform(pdist(ca_coordinates))
        temp_factor_array = np.array(temp_factor_list)
        if gaussian_kernel == True:
            weight_function = norm(0,distance_cutoff)
        for chain in pdb_dict:
            for res in pdb_dict[chain]:
                if gaussian_kernel == True:
                    weights = weight_function.pdf(dist_matrix[i,:])
                    weights /= np.sum(weights)
                    temp_factor_space_average = np.sum(
                        temp_factor_array * weights)
                else:
                    valid_dist = dist_matrix[i,:] < distance_cutoff
                    temp_factor_space_average = np.mean(
                        temp_factor_array[valid_dist])
                i += 1
                space_average_list.append(temp_factor_space_average)

    output = [x for x in [
        identifiers,temp_factor_list,sequence_average_list,space_average_list
    ] if len(x) > 0]
    return output

parser = argparse.ArgumentParser(
    prog = 'b_factor.py',
    description = 'Extracts b-factor from PDBs and calculates sequence and \
    structural context averages.'
)

parser.add_argument('--pdb_path',dest = 'pdb_path',
                    action = ToDirectory,
                    default = None,
                    help = 'Path to PDB file.')

parser.add_argument('--sequence_average',dest = 'sequence_average',
                    action = 'store_true',
                    default = False,
                    help = 'Calculate sequence average.')

parser.add_argument('--window_size',dest = 'window_size',
                    action = 'store',
                    type = int,
                    default = 5,
                    help = 'Window size for sequence average.')

parser.add_argument('--space_average',dest = 'space_average',
                    action = 'store_true',
                    default = False,
                    help = 'Calculate structural context average.')

parser.add_argument('--distance_cutoff',dest = 'distance_cutoff',
                    action = 'store',
                    type = float,
                    default = 10.,
                    help = 'Cutoff for structural context_average.')

parser.add_argument('--gaussian_kernel',dest = 'gaussian_kernel',
                    action = 'store_true',
                    default = False,
                    help = "Weigh distances using a gaussian kernel with\
                    mean = 0 and standard deviation = distance_cutoff.")

args = parser.parse_args()

if __name__ == "__main__":
    pdb_dict,res_dict = parse_pdb(args.pdb_path)

    output = get_b_factor(
        pdb_dict=pdb_dict,
        res_dict=res_dict,
        sequence_average=args.sequence_average,
        window_size=args.window_size,
        space_average=args.space_average,
        distance_cutoff=args.distance_cutoff,
        gaussian_kernel=args.gaussian_kernel)

    for out in zip(*output):
        print(','.join([str(x) for x in out]))
