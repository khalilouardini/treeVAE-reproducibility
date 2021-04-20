import os

import pandas as pd
from itolapi import Itol
from itolapi import ItolExport
from ete3 import Tree
from matplotlib.colors import hsv_to_rgb
from tqdm.auto import tqdm
import numpy as np
import pdb

#from cassiopeia.TreeSolver import utilities

def upload_to_itol(
    tree,
    apiKey,
    projectName,
    tree_name="test",
    files=[],
    outfp="test.pdf",
    fformat=None,
    rect=False,
    **kwargs,
):

    _leaves = tree.get_leaf_names()
    tree.write(outfile="tree_to_plot.tree", format=1)

    if fformat is None:
        fformat = outfp.split(".")[-1]

    itol_uploader = Itol()
    itol_uploader.add_file("tree_to_plot.tree")

    for file in files:
        itol_uploader.add_file(file)

    itol_uploader.params["treeName"] = tree_name
    itol_uploader.params["APIkey"] = apiKey
    itol_uploader.params["projectName"] = projectName

    good_upload = itol_uploader.upload()
    if not good_upload:
        print("There was an error:" + itol_uploader.comm.upload_output)
    print("iTOL output: " + str(itol_uploader.comm.upload_output))
    print("Tree Web Page URL: " + itol_uploader.get_webpage())
    print("Warnings: " + str(itol_uploader.comm.warnings))

    tree_id = itol_uploader.comm.tree_id

    itol_exporter = ItolExport()

    # set parameters:
    itol_exporter.set_export_param_value("tree", tree_id)
    itol_exporter.set_export_param_value(
        "format", outfp.split(".")[-1]
    )  # ['png', 'svg', 'eps', 'ps', 'pdf', 'nexus', 'newick']
    if rect:
        itol_exporter.set_export_param_value("display_mode", 1)  # rectangular tree
    else:
        itol_exporter.set_export_param_value("display_mode", 2)  # circular tree
        itol_exporter.set_export_param_value("arc", 359)
        itol_exporter.set_export_param_value("rotation", 270)

    itol_exporter.set_export_param_value("leaf_sorting", 1)
    itol_exporter.set_export_param_value("label_display", 1)
    itol_exporter.set_export_param_value("internal_marks", 1)
    itol_exporter.set_export_param_value("ignore_branch_length", 1)

    itol_exporter.set_export_param_value(
        "datasets_visible", ",".join([str(i) for i in range(len(files))])
    )

    itol_exporter.set_export_param_value(
        "horizontal_scale_factor", 1
    )  # doesnt actually scale the artboard

    # export!
    itol_exporter.export(outfp)

    os.remove("tree_to_plot.tree")

def create_indel_heatmap(alleletable, tree, indel_colors, dataset_name, outdir):

    _leaves = tree.get_leaf_names()

    lineage_profile = utilities.alleletable_to_lineage_profile(alleletable, write=False)
    clustered_linprof = lineage_profile.loc[_leaves[::-1]]

    # Convert colors and make colored alleleTable (rgb_heatmap)
    r, g, b = np.zeros(clustered_linprof.shape), np.zeros(clustered_linprof.shape), np.zeros(clustered_linprof.shape)
    for i in tqdm(range(clustered_linprof.shape[0])):
        for j in range(clustered_linprof.shape[1]):
            ind = str(clustered_linprof.iloc[i, j])
            if ind == 'nan':
                r[i, j], g[i, j], b[i, j] = 1, 1, 1
            elif 'None' in ind:
                r[i, j], g[i, j], b[i, j] = 192 / 255, 192/ 255, 192/255
            else:
                col = hsv_to_rgb(tuple(indel_colors.loc[ind, 'colors']))
                r[i, j], g[i, j], b[i, j] = col[0], col[1], col[2]
                
    rgb_heatmap = np.stack((r, g, b), axis=2)

    alfiles = []
    for j in range(0, rgb_heatmap.shape[1]):
        item_list = []
        for i in rgb_heatmap[:, j]:
            item = (
                "rgb("
                + str(int(round(255 * i[0])))
                + ","
                + str(int(round(255 * i[1])))
                + ","
                + str(int(round(255 * i[2])))
                + ")"
            )
            item_list.append(item)
        dfAlleleColor = pd.DataFrame()
        dfAlleleColor["cellBC"] = clustered_linprof.index.values
        dfAlleleColor["color"] = item_list

        if j == 0:
            header = [
                "DATASET_COLORSTRIP",
                "SEPARATOR TAB",
                "COLOR\t#000000",
                "MARGIN\t100",
                "DATASET_LABEL\tallele" + str(j),
                "STRIP_WIDTH\t50",
                "SHOW_INTERNAL\t0",
                "DATA",
                "",
            ]
        else:
            header = [
                "DATASET_COLORSTRIP",
                "SEPARATOR TAB",
                "COLOR\t#000000",
                "DATASET_LABEL\tallele" + str(j),
                "STRIP_WIDTH\t50",
                "SHOW_INTERNAL\t0",
                "DATA",
                "",
            ]

        if len(str(j)) == 1:
            alleleLabel_fileout = outdir + "/indelColors_0" + str(j) + ".txt"
        elif len(str(j)) == 2:
            alleleLabel_fileout = outdir + "/indelColors_" + str(j) + ".txt"
        with open(alleleLabel_fileout, "w") as ALout:
            for line in header:
                ALout.write(line + "\n")
            df_writeout = dfAlleleColor.to_csv(
                None, sep="\t", header=False, index=False
            )
            ALout.write(df_writeout)

        alfiles.append(alleleLabel_fileout)

    return alfiles, rgb_heatmap
    


def create_gradient_from_df(
    df, tree, dataset_name, color_min="#ffffff", color_max="#000000"
):

    _leaves = tree.get_leaf_names()
    #node_names = []
    #for node in tree.traverse("levelorder"):
        #node_names.append(node.name)
    #_leaves = node_names

    if type(df) == pd.Series:
        fcols = [df.name]
    else:
        fcols = df.columns

    outfps = []
    for j in range(0, len(fcols)):

        outdf = pd.DataFrame()
        outdf["cellBC"] = _leaves
        outdf["GE"] = df.loc[_leaves, fcols[j]].values

        #pdb.set_trace()
        
        header = [
            "DATASET_GRADIENT",
            "SEPARATOR TAB",
            "COLOR\t#00000",
            f"COLOR_MIN\t{color_min}",
            f"COLOR_MAX\t{color_max}",
            "MARGIN\t100",
            f"DATASET_LABEL\t{fcols[j]}",
            "STRIP_WIDTH\t50",
            "SHOW_INTERNAL\t1",
            "DATA",
            "",
        ]

        outfp = dataset_name + "." + str(fcols[j]) + ".txt"
        with open(outfp, "w") as fOut:
            for line in header:
                fOut.write(line + "\n")
            df_writeout = outdf.to_csv(None, sep="\t", header=False, index=False)
            fOut.write(df_writeout)
        outfps.append(outfp)
    return outfps


def create_colorbar(labels, tree, outfp, colormap, dataset_name="colorbar", create_legend=False):
 
    _leaves = tree.get_leaf_names()
    labelcolors_iTOL = []
    for i in labels.loc[_leaves].values:
        colors_i = colormap[i]
        color_i = (
            "rgb("
            + str(colors_i[0])
            + ","
            + str(colors_i[1])
            + ","
            + str(colors_i[2])
            + ")"
        )
        labelcolors_iTOL.append(color_i)
    dfCellColor = pd.DataFrame()
    dfCellColor["cellBC"] = _leaves
    dfCellColor["color"] = labelcolors_iTOL

    # save file with header
    header = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        "COLOR\t#FF0000",
        "MARGIN\t100",
        f"DATASET_LABEL\t{dataset_name}",
        "STRIP_WIDTH\t100",
        "SHOW_INTERNAL\t0",
        "",
    ]
    with open(outfp, "w") as SIDout:
        for line in header:
            SIDout.write(line + "\n")

        if create_legend:
            number_of_items = len(colormap)
        
            SIDout.write(f'LEGEND_TITLE\t{dataset_name} legend\n')
            SIDout.write('LEGEND_SHAPES')
            for _ in range(number_of_items):
                SIDout.write("\t1")
            
            SIDout.write("\nLEGEND_COLORS")
            for col in colormap.values():
                SIDout.write(f"\t{rgb_to_hex(col)}")
            
            SIDout.write("\nLEGEND_LABELS")
            for key in colormap.keys():
                SIDout.write(f"\t{key}")
            SIDout.write("\n")
        
        SIDout.write("\nDATA\n")
        df_writeout = dfCellColor.to_csv(None, sep="\t", header=False, index=False)
        SIDout.write(df_writeout)

    return outfp

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
                                   

def rgb_to_hex(rgb):
    
    r, g, b = rgb[0], rgb[1], rgb[2]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)