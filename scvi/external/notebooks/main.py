import os
print(os.getcwd())
import sys
sys.path.append('..')
from ete3 import Tree
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from  dataset.tree import TreeDataset
from dataset.poisson_glm import Poisson_GLM
from dataset.anndataset import AnnDatasetFromAnnData
from  models import *;
import scanpy as sc
from inference.tree_inference import TreeTrainer
from inference.inference import UnsupervisedTrainer
from inference import posterior
from models.treevae import TreeVAE
import torch



# helpers
def get_leaves(glm, tree):
    leaves_X, leaves_idx, mu = [], [], []
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            leaves_X.append(glm.X[n.index])
            leaves_idx.append(n.index)
            mu.append(glm.mu[n.index])
    leaves_X = np.array(leaves_X)
    mu = np.array(mu)
    return leaves_X, leaves_idx, mu


if __name__ == '__main__':
    # Import Tree + Simulate Data
    tree_name = "/Users/khalilouardini/Desktop/projects/scVI/scvi/data/Cassiopeia_trees/lg7_tree_hybrid_priors.alleleThresh.collapsed.txt"
    with open(tree_name, "r") as myfile:
        tree_string = myfile.readlines()
        tree = Tree(tree_string[0], 1)

    tree = Tree(tree_name, 1)
    for i, n in enumerate(tree.traverse('levelorder')):
        n.add_features(index=i)

    tree_name = "/Users/khalilouardini/Desktop/projects/scVI/scvi/data/Cassiopeia_trees/lg7_tree_hybrid_priors.alleleThresh.collapsed.txt"

    # Simulate GE data
    d = 2
    g = 20
    vis = True
    leaves_only = False

    glm = Poisson_GLM(tree_name, g, d, vis, leaves_only)
    glm.simulate_latent()
    glm.simulate_ge()

    # FIXED training set
    leaves_X, leaves_idx, mu = get_leaves(glm, tree)

    # Create Dataset
    adata = AnnData(leaves_X)
    scvi_dataset = AnnDatasetFromAnnData(adata)
    adata.obs_names = pd.read_csv('barcodes.tsv', header=None)[0]
    scvi_dataset.initialize_cell_attribute('barcodes', adata.obs_names)

    # treeDataset
    cas_dataset = TreeDataset(scvi_dataset, tree_name = tree_name)

    # Hyper parameters
    n_epochs = 1000
    lr = 1e-3
    use_batches = False
    use_cuda = False

    vae = TreeVAE(cas_dataset.nb_genes,
                  tree = cas_dataset.tree,
                  n_batch=cas_dataset.n_batches * use_batches,
                  n_latent=2,
                  n_hidden=128,
                  n_layers=1
                 )

    freq = 5
    trainer = TreeTrainer(
        model = vae,
        gene_dataset = cas_dataset,
        train_size=1.0,
        test_size=0,
        use_cuda=use_cuda,
        frequency=freq,
        )

    trainer.train(n_epochs=n_epochs,
                  lr=lr
                 )



