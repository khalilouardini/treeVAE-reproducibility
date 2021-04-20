import os

import pandas as pd
import numpy as np
from ete3 import Tree
from statsmodels.stats.proportion import multinomial_proportions_confint

from itolapi import Itol
from itolapi import ItolExport


def calculate_expansion_proportion(tree):
    """ Detects clonal expansion of all children at depth 1. 
    
    :param tree: ete3.TreeNode, the root of the tree
    :return: A mapping from child to confidence interval
    """

    N = len(tree.get_leaf_names())

    props = {}
    for c in tree.children:
        props[c] = len(c.get_leaves())

    ci = multinomial_proportions_confint(list(props.values()), alpha=0.05)
    props = {c: interval for c, interval in zip(props.keys(), ci)}
    return props


def annotate_tree(tree, min_clade_size=10, min_depth=1):
    """Annotate tree with respect to expansion proportions.
    
    :param tree: ete3.TreeNode, the root of the tree.
    :param min_clade_size: minimum size of clade to be evaluated.
    :param min_depth: minimum distance from root for clade to be evaluated.
    """

    node_to_exp = {}

    # instantiate dictionary
    for n in tree.traverse():
        node_to_exp[n.name] = 0

    for n in tree.traverse():
        kn = len(n.get_leaves())
        if tree.get_distance(n) > min_depth and kn > min_clade_size:
            exp_props = calculate_expansion_proportion(n)
            for c in exp_props:
                node_to_exp[c.name] = exp_props[c][0]

    return tree, node_to_exp


def find_expanding_clones(tree, node_to_exp, thresh=0.6, _last=False):
    def ancestors_are_expanding(tree, n):

        anc_expanding = False
        while n.up:
            n = n.up
            if n.expanding:
                anc_expanding = True
        return anc_expanding

    def child_is_expanding(tree, n):

        if n.is_leaf():
            return True

        for c in n.children:
            if c.expanding:
                return True

        return False

    N = len(tree.get_leaves())

    # determine which nodes pass the desired expansion
    # threshold
    for n in tree.traverse():
        if node_to_exp[n.name] >= thresh:
            n.add_features(expanding=True)
        else:
            n.add_features(expanding=False)

    # find original expanding clone.
    expanding_nodes = {}

    if not _last:
        for n in tree.traverse():
            if n.expanding:
                if not ancestors_are_expanding(tree, n):
                    kn = len(n.get_leaves())
                    pn = len(n.up.get_leaves())
                    expanding_nodes[n.name] = (kn / pn, kn / N, node_to_exp[n.name], kn , N)

    else:
        for n in tree.traverse():
            if n.expanding:
                if not child_is_expanding(tree, n):
                    kn = len(n.get_leaves())

                    p = n.up
                    while len(p.children) == 1:
                        p = p.up
                    pn = len(p.get_leaves())
                    expanding_nodes[n.name] = (kn / pn, kn / N, node_to_exp[n.name], kn, N)
                else:
                    n.expanding = False

        # we still want to just capture the first major expansion
        for n in tree.traverse():
            if n.name in expanding_nodes.keys():
                if ancestors_are_expanding(tree, n):
                    del expanding_nodes[n.name]

    # remove temporary annotations
    for n in tree.traverse():
        if n.name not in expanding_nodes.keys():
            n.add_features(expanding=False)

    return tree, expanding_nodes


def detect_expansion(tree, thresh, _first=True, min_clade_prop=0.1, min_depth=0):

    N = len(tree.get_leaves())

    tree, node_to_prop = annotate_tree(
        tree, min_clade_size=N * min_clade_prop, min_depth=min_depth
    )

    tree, expanding_nodes = find_expanding_clones(
        tree, node_to_prop, thresh=thresh, _last=(not _first)
    )

    expansion_df = pd.DataFrame.from_dict(
        expanding_nodes, orient="index", columns=["SubProp", "TotalProp", "Lower95CI", "LeavesInExpansion", "TotalLeaves"]
    )
    return tree, expansion_df


def create_expansion_file_for_itol(tree, expansions, outfp=None):

    _leaves = tree.get_leaf_names()

    out = ""
    header = ["TREE_COLORS", "SEPARATOR SPACE", "DATA"]
    for line in header:
        out += line + "\n"
    for row in expansions.iterrows():
        out += row[0] + " clade #ff0000 normal 1\n"

    if outfp:
        with open(outfp, "w") as fOut:
            fOut.write(out)
        return outfp
    else:
        return out