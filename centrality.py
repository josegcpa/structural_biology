#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone script to calculate centrality for all residues in the protein
structure, as per Antonio del Sol et al. [1].

[1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2242611/

Arguments:
--pdb_path - path to the PDB file
--distance_cutoff -- float. The distance cutoff for the spatial context
averages (only residues within distance_cutoff angstroms are considered
"""

__author__ = "Jose Guilherme Almeida"
__credits__ = ["Jose Guilherme Almeida"]
__email__ = "josegcpa@ebi.ac.uk"

import argparse
import os
from math import inf
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

def min_distance(pdb_dict,res_dict,distance_cutoff):
    """
    Uses a pdb_dict and a res_dict to get a distance matrix that gets the
    minimum distance between two atoms of two different residues.

    Keyword arguments:
    pdb_dict -- a dictionary with PDB chains as keys and dictionaries with
    residue identifiers as keys and atom_info dictionaries as values as values
    res_dict -- a dictionary with PDB chains as keys and lists of residue
    identifiers as keys
    distance_cutoff -- float. a number used to cutoff distances (any distance
    above > distance_cutoff -> inf)

    Output:
    min_dist_matrix -- a matrix containing all the minimum distances between
    all residues
    residues - list. list containing all residues.
    """

    coord_list = []
    residue_ranges = {}
    i = 0
    n_res = 0
    for chain in res_dict:
        for res in res_dict[chain]:
            residue_ranges[chain + res] = [n_res,i]
            for atom in pdb_dict[chain][res]:
                coord_list.append([atom['x'],atom['y'],atom['z']])
                i += 1
            residue_ranges[chain + res].append(i)
            n_res += 1
    dist_matrix = squareform(pdist(coord_list))
    min_dist_matrix = np.ones((n_res,n_res)) * inf
    idx = 0
    for res1 in residue_ranges:
        n_res1,idx_start1,idx_end1 = residue_ranges[res1]
        for res2 in residue_ranges:
            n_res2,idx_start2,idx_end2 = residue_ranges[res2]
            min_dist = np.min(dist_matrix[idx_start1:idx_end1,
                                          idx_start2:idx_end2])
            min_dist_matrix[n_res1,n_res2] = min_dist
            min_dist_matrix[n_res2,n_res1] = min_dist

    min_dist_matrix = np.where(
        min_dist_matrix >= distance_cutoff,
        inf,
        min_dist_matrix)
    residues = residue_ranges.keys()
    return min_dist_matrix,residues

def get_shortest_paths(dist_matrix):
    """
    This is an implementation of the Floyd-Warshall's algorithm for finding all
    shortest paths between pairs of vertices in a graph.

    Keyword arguments:
    dist_matrix -- numpy array. a NxN distance matrix.

    Output:
    shortest_path_matrix -- a NxN matrix containing all shortest paths between
    vertices.
    """

    v = dist_matrix.shape[0]
    shortest_path_matrix = dist_matrix.copy()
    for k in range(v):
        for i in range(v):
            for j in range(v):
                shortest_path_matrix[i][j] = np.minimum(
                    shortest_path_matrix[i][j],
                    shortest_path_matrix[i][k] + shortest_path_matrix[k][j])
    return shortest_path_matrix

def get_centrality(shortest_paths):
    """
    Uses a shortest path matrix calculates centrality.

    Keyword arguments:
    shortest_paths -- numpy array. a NxN matrix containing all shortest paths.

    Output:
    centrality -- a N-sized vector containing all centrality values.
    """

    n = shortest_paths.shape[0] - 1
    centrality = n / np.sum(shortest_paths,axis = 1)

    return centrality

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'b_factor.py',
        description = 'Extracts b-factor from PDBs and calculates sequence and \
        structural context averages.'
    )

    parser.add_argument('--pdb_path',dest = 'pdb_path',
                        action = ToDirectory,
                        default = None,
                        help = 'Path to PDB file.')

    parser.add_argument('--distance_cutoff',dest = 'distance_cutoff',
                        action = 'store',
                        type = float,
                        default = 10.,
                        help = 'Cutoff for structural context_average.')

    args = parser.parse_args()

    pdb_dict,res_dict = parse_pdb(args.pdb_path)
    min_dist_matrix,residues = min_distance(pdb_dict,
                                            res_dict,
                                            args.distance_cutoff)
    shortest_paths = get_shortest_paths(min_dist_matrix)
    centrality = get_centrality(shortest_paths)

    for res,c in zip(residues,centrality):
        print('{},{}'.format(res,c))
