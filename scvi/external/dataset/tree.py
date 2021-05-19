import logging
import os
import pickle
import tarfile
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sp_io
import shutil
from scipy.sparse import csr_matrix
from ete3 import Tree
import sys

from ..dataset.dataset import (
    DownloadableDataset,
    GeneExpressionDataset,
    CellMeasurement,
)

logger = logging.getLogger(__name__)

available_specification = ["filtered", "raw"]


class TreeDataset(GeneExpressionDataset):
    """Forms a ``GeneExpressionDataset`` with a corresponding Tree structure relatating
    every cell.

    This is the dataset class that will be used to interact with the TreeVAE model. It's
    important to observe here that this function does not take in expression data from a CSV
    or sparse matrix, for example, but rather assumes that an scVI GeneExpressionDataset has
    already been created. The resulting API of the dataset remains very similar to that of a
    typical GeneExpressionDataset but with the addition of a tree (of class `ete3.Tree`) that
    will be used as a prior during model fitting.

    :param expr: ``scvi.dataset.GeneExpressionDataset`` instance.
    :param tree: file path to tree to read in from ``ete3.Tree`` instance.
    """

    def __init__(
        self, expr: GeneExpressionDataset, tree=None, filtering=True
    ):

        if tree is not None and type(tree) == str:
            self.tree = Tree(tree, 1)
            # polytomy is not a problem anymore: message passing deals with general trees
            # self.tree.resolve_polytomy(recursive=True)
        else:
            self.tree = tree

        if self.tree is None:
            logger.error(
                "Must provide a tree file path or a tree if you're using TreeDataset."
            )

        # assert we have barcode labels for cells
        if "barcodes" not in expr.cell_attribute_names:
            logger.error("Must provide cell barcode, or names, as a cell attribute.")

        super().__init__()

        # set some tree attributes
        self.populate_treedataset(expr)

        # keeping the cells in the tree and Gene expression dataset (not needed for simulations)
        # self.filter_cells_by_tree()
        if filtering:
            self.filter_cells_by_count()

    def populate_treedataset(self, expr):
        """
        Populate the TreeDataset with respect to an GeneExpressionDataset that is
        passed in.

        :param expr: A ``scvi.dataset.GeneExpressionDataset`` instance.
        """

        # set distance
        for n in self.tree.traverse():
            n.distance = self.tree.get_distance(n)

        self.populate_from_datasets([expr])

    def populate(self):

        tree = self.tree
        if tree is None and self.tree is not None:
            self.tree = Tree(tree, 1)
        else:
            logger.error(
                "Must provide a tree file path or a tree if you're using TreeDataset."
            )

        # set distance
        for n in self.tree.traverse():
            n.distance = self.tree.get_distance(n)

        self.populate_from_datasets([expr])

        self.populate_treedataset(expr=self)

        self.filter_cells_by_tree()

        self.filter_cells_by_count()

    def filter_cells_by_tree(self):
        """
        Prunes away cells that don't appear consistently between the tree object and the
        RNA expression dataset.
        """
        leaves = self.tree.get_leaf_names()
        keep_barcodes = np.intersect1d(leaves, self.barcodes)
        self.tree.prune(keep_barcodes)

        return self.filter_cells_by_attribute(keep_barcodes, on="barcodes")
