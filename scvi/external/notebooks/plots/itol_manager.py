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


def create_annotation_file_for_itol(tree, df, outfp=None):
    if type(df) == pd.Series:
        fcols = [df.name]
    else:
        fcols = df.columns
        
    _leaves = tree.get_leaf_names()

    out = ""
    header = ['DATASET_TEXT',
              "SEPARATOR SPACE", 
              "COLOR\t#00000",
              f"DATASET_LABEL\t{fcols[0]}",
              "DATA"]
    
    for line in header:
        out += line + "\n"
    for row in df.iterrows():
        #out += row[0] + " " + str(row[1].values[0]) + " 0 #0000ff bold 1.5\n"
        #out += row[0] + " " + str(row[1].values[0]) + " 0 #00b300 bold 1.5\n"
        out += row[0] + " " + str(row[1].values[0]) + " 0 #ff0000 bold 1.5\n"
        #out += row[0] + " " + str(row[1].values[0]) + " 0 #ff8000 bold 1.5\n"
    if outfp:
        with open(outfp, "w") as fOut:
            fOut.write(out)
        return outfp
    else:
        return out                            
                                   

def rgb_to_hex(rgb):
    
    r, g, b = rgb[0], rgb[1], rgb[2]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)