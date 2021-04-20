import os
import sys
import numpy as np
import random
import pandas as pd
import argparse
import copy
import torch
from sklearn.preprocessing import normalize
from ete3 import Tree
import matplotlib.pyplot as plt
sys.path.append('/Users/khalilouardini/Desktop/projects/scVI/scvi')

# Data
from anndata import AnnData
import scanpy as sc
from scvi.dataset.tree import TreeDataset, GeneExpressionDataset
from scvi.dataset.poisson_glm import Poisson_GLM
from scvi.dataset.anndataset import AnnDatasetFromAnnData

# Models
from scvi.models import *
from scvi.inference.tree_inference import TreeTrainer
from scvi.inference.inference import UnsupervisedTrainer
from scvi.inference import posterior
from scvi.models.treevae import TreeVAE

# Utils
from scvi.utils.data_util import get_leaves, get_internal
from scvi.utils.metrics import ks_pvalue, accuracy_imputation, correlations, knn_purity, knn_purity_stratified
from scvi.utils.baselines import avg_weighted_baseline, scvi_baseline, scvi_baseline_z, cascvi_baseline_z, construct_latent

class Ancestral_Imputation():

    def __init__(self, tree, branch_length, use_cuda, var, n_epochs, lr, lambda_, latent, n_genes, simulation, n_hidden):
        """
        :param tree: ete3: Cassiopeia prior tree
        :param branch_length: float: branch length of the simulation (Gaussian Random Walk)
        :param use_cuda: bool: True to train on GPUs
        :param var: float: variance parameter in the Message Passing inference
        :param n_epochs: int: number of epochs
        :param lr: float: learning rate
        :param lambda_: float: regularization parameter in the TreeVAE loss
        :param latent: int: dimension of latent space
        :param n_genes: int: number of genes
        :return:
        """
        self.tree = tree
        self.branch_length = branch_length
        self.var = None
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.lr = lr
        self.lambda_ = lambda_
        self.latent = latent
        self.n_genes = n_genes
        self.glm = None
        self.tree_dataset = None
        self.leaves_X = None
        # Models
        self.treevae = None
        self.vae = None
        # neural network parameters
        self.n_hidden = n_hidden

        for i, n in enumerate(self.tree.traverse('levelorder')):
            n.add_features(index=i)

    def simulations(self, simulation):
        """
        Generating the data with a simulation framework (default is Poisson GLM)
        :return:
        """
        if simulation == 'poisson_glm':
            self.glm = Poisson_GLM(self.tree,
                              self.n_genes,
                              self.latent,
                              False, False,
                              self.branch_length
                              )
        else:
            raise ValueError("Simulation framework not implemented")
        self.glm.simulate_latent()
        self.glm.simulate_ge()
        # Quality Control (i.e Gene Filtering)
        self.glm.gene_qc()
        #self.glm.binomial_thinning(p=0.01)

        # Query data
        # Latent vectors
        self.leaves_z, _, _ = get_leaves(self.glm.z, self.glm.mu, self.tree)
        # training set
        self.leaves_X, _, _ = get_leaves(self.glm.X, self.glm.mu, self.tree)
        # internal nodes data
        self.internal_X, _, _ = get_internal(self.glm.X, self.glm.mu, self.tree)

    def fit_cascvi(self, var):
        """
        Fitting cascVI to the gene expression data
        :return:
        """
        # anndata
        adata = AnnData(self.leaves_X)
        adata.obs_names = [n.name for n in list(self.tree.get_leaves())]
        scvi_dataset = AnnDatasetFromAnnData(adata)
        scvi_dataset.initialize_cell_attribute('barcodes', adata.obs_names)

        # treedataset
        tree_bis = copy.deepcopy(self.tree)
        self.tree_dataset = TreeDataset(scvi_dataset, tree=tree_bis)
        # full batch updates
        use_batches = False

        # treeVAE
        self.var = var
        self.treevae = TreeVAE(self.tree_dataset.nb_genes,
                      tree=self.tree_dataset.tree,
                      n_batch=self.tree_dataset.n_batches * use_batches,
                      n_latent=self.latent,
                      n_hidden=self.n_hidden,
                      n_layers=1,
                      reconstruction_loss='poisson',
                      prior_t=self.var,
                      ldvae=False
                      )
        print("Message Passing variance: ", self.var)

        freq = 10
        trainer = TreeTrainer(
            model=self.treevae,
            gene_dataset=self.tree_dataset,
            lambda_=self.lambda_,
            train_size=1.0,
            test_size=0,
            use_cuda=self.use_cuda,
            frequency=freq,
            n_epochs_kl_warmup=None
        )

        # training the VAE
        trainer.train(n_epochs=self.n_epochs,
                      lr=self.lr)

        self.cascvi_posterior = trainer.create_posterior(trainer.model, self.tree_dataset, trainer.clades,
                                indices=np.arange(len(self.tree_dataset))
                                         )

    def fit_scvi(self):
        """
        Fitting scVI to the gene expression data
        :return:
        """
        # anndata
        gene_dataset = GeneExpressionDataset()
        gene_dataset.populate_from_data(self.leaves_X)

        self.vae = VAE(gene_dataset.nb_genes,
                       n_batch=False,
                       n_hidden=self.n_hidden,
                       n_layers=1,
                       reconstruction_loss='poisson',
                       n_latent=self.latent)

        trainer_scvi = UnsupervisedTrainer(model=self.vae,
                                           gene_dataset=gene_dataset,
                                           train_size=1.0,
                                           use_cuda=self.use_cuda,
                                           frequency=10,
                                           n_epochs_kl_warmup=None)

        # train scVI
        trainer_scvi.train(n_epochs=self.n_epochs,
                             lr=self.lr)

        self.scvi_posterior = trainer_scvi.create_posterior(model=self.vae,
                                              gene_dataset=gene_dataset
                                              )

    def evaluation(self):
        """
        :return:
        """

        if not os.path.exists('results'):
            os.mkdir('results')

        # empirical library size
        empirical_l = np.mean(np.sum(self.glm.X, axis=1))

        # ========== CascVI imputations ===================
        # 0. CascVI w/ message passing
        imputed = {}
        imputed_z = {}
        imputed_gt = {}

        for n in self.tree.traverse('levelorder'):
            if not n.is_leaf():
                imputed[n.name], imputed_z[n.name] = self.cascvi_posterior.imputation_internal(n.name,
                                                                                        give_mean=False,
                                                                                        library_size=empirical_l
                                                                                        )
                imputed_gt[n.name] = self.glm.X[n.index]
        imputed_X = [x for x in imputed.values()]
        imputed_X = np.array(imputed_X).reshape(-1, self.tree_dataset.X.shape[1])

        # 1. CascVI Baseline 1 (averaging of reconstructions)
        imputed_cascvi_1 = scvi_baseline(self.tree, self.cascvi_posterior, True)

        # 2. CascVI Baseline 2 ((Reconstruction of Averaged latent space)
        imputed_cascvi_2, imputed_cascvi_2_z = scvi_baseline_z(tree=self.tree,
                                                               model=self.treevae,
                                                               posterior=self.cascvi_posterior,
                                                               n_samples_z=1,
                                                               library_size=empirical_l
                                                               )
        # 3. CascVI Baseline 3
        imputed_cascvi_3 = cascvi_baseline_z(tree=self.tree,
                                             model=self.vae,
                                             latent=imputed_z,
                                             library_size=empirical_l
                                             )

        # ========== Baselines ===================
        # 4. Baseline 0: (Un)weighted Average of gene expression
        weighted = True
        imputed_avg = avg_weighted_baseline(self.tree, True, self.glm.X)
        # get internal nodes
        avg_X = np.array([x for x in imputed_avg.values()]).reshape(-1, self.glm.X.shape[1])
        internal_avg_X, _, _ = get_internal(avg_X, self.glm.mu, self.tree)

        # 5. Baseline 1: Average of reconstructions
        imputed_scvi, imputed_scvi_z = scvi_baseline(tree=self.tree,
                                                     posterior=self.scvi_posterior,
                                                     weighted=True,
                                                     give_latent=True,
                                                     n_samples_z=1
                                                     )

        # 6. Baseline 2: Decoded averaged latent space
        imputed_scvi_2, imputed_scvi_2_z = scvi_baseline_z(tree=self.tree,
                                                           model=self.vae,
                                                           posterior=self.scvi_posterior,
                                                           weighted=True,
                                                           n_samples_z=1,
                                                           library_size=empirical_l
                                                           )

        # 7. Baseline 3: scVI w/ Message passing
        imputed_bis = {}
        imputed_z_bis = {}
        for n in tree.traverse('levelorder'):
            if not n.is_leaf():
                imputed_bis[n.name], imputed_z_bis[n.name] = self.cascvi_posterior.imputation_internal(n.name,
                                                                                                give_mean=False,
                                                                                                library_size=empirical_l,
                                                                                                other_posterior=self.scvi_posterior
                                                                                                )

        # Evaluation 1: k-nn purity
        if not os.path.exists('results/knn_purity'):
            os.mkdir('results/knn_purity')

        cascvi_latent = self.cascvi_posterior.get_latent()[0]
        scvi_latent = self.scvi_posterior.get_latent()[0]

        # Construct full latent space
        full_cascvi_latent = construct_latent(self.tree, cascvi_latent, imputed_z)
        full_cascvi_latent_2 = construct_latent(self.tree, cascvi_latent, imputed_cascvi_2_z)
        full_scvi_latent = construct_latent(self.tree, scvi_latent, imputed_scvi_2_z)
        full_scvi_latent_2 = construct_latent(self.tree, scvi_latent, imputed_z_bis)

        # leaves purity
        _ = knn_purity(max_neighbors=50,
              data=[self.leaves_z, scvi_latent, cascvi_latent],
              plot=True,
              save_fig='results/knn_purity/' + 'purity_leaves_' + str(self.branch_length) + '_' + str(self.var) + '.jpg'
              )
        plt.clf() 

        # internal nodes purity
        internal_z, _, _ = get_internal(self.glm.z, self.glm.mu, self.tree)
        internal_scvi_z, _, _ = get_internal(full_scvi_latent, self.glm.mu, self.tree)
        internal_scvi_z_2, _, _ = get_internal(full_scvi_latent_2, self.glm.mu, self.tree)
        internal_cascvi_z, _, _ = get_internal(full_cascvi_latent, self.glm.mu, self.tree)
        internal_cascvi_z_2, _, _ = get_internal(full_cascvi_latent_2, self.glm.mu, self.tree)

        _ = knn_purity(max_neighbors=50,
              data=[internal_z, internal_scvi_z, internal_scvi_z_2, internal_cascvi_z, internal_cascvi_z_2],
              plot=True,
              save_fig='results/knn_purity/' + 'purity_internal_' + str(self.branch_length) + '_' + str(self.var) + '.jpg'
              )
        plt.clf() 

        # full tree
        _ = knn_purity(max_neighbors=50,
              data=[self.glm.z, full_scvi_latent, full_scvi_latent_2, full_cascvi_latent, full_cascvi_latent_2],
              plot=True,
              save_fig='results/knn_purity/' + 'purity_full_' + str(self.branch_length) + '_' + str(self.var) + '.jpg'
              )
        plt.clf() 

        # Evalutation 2: Imputation
        if not os.path.exists('results/imputation'):
            os.mkdir('results/imputation')
        
        internal_scvi_X = np.array([x for x in imputed_scvi.values()]).reshape(-1, self.glm.X.shape[1])
        internal_scvi_X_2 = np.array([x for x in imputed_scvi_2.values()]).reshape(-1, self.glm.X.shape[1])
        internal_cascvi_X = np.array([x for x in imputed_cascvi_1.values()]).reshape(-1, self.glm.X.shape[1])
        internal_cascvi_X_2 = np.array([x for x in imputed_cascvi_2.values()]).reshape(-1, self.glm.X.shape[1])
        internal_cascvi_X_3 = np.array([x for x in imputed_cascvi_3.values()]).reshape(-1, self.glm.X.shape[1])

        norm_internal_X = sc.pp.normalize_total(AnnData(self.internal_X), target_sum=1e4, inplace=False)['X'] 
        norm_scvi_X = sc.pp.normalize_total(AnnData(internal_scvi_X), target_sum=1e4, inplace=False)['X']
        norm_scvi_X_2 = sc.pp.normalize_total(AnnData(internal_scvi_X_2), target_sum=1e4, inplace=False)['X']
        norm_avg_X = sc.pp.normalize_total(AnnData(internal_avg_X), target_sum=1e4, inplace=False)['X']
        norm_imputed_X = sc.pp.normalize_total(AnnData(imputed_X), target_sum=1e4, inplace=False)['X']
        norm_cascvi_X = sc.pp.normalize_total(AnnData(internal_cascvi_X), target_sum=1e4, inplace=False)['X']
        norm_cascvi_X_2 = sc.pp.normalize_total(AnnData(internal_cascvi_X_2), target_sum=1e4, inplace=False)['X']
        norm_cascvi_X_3 = sc.pp.normalize_total(AnnData(internal_cascvi_X_3), target_sum=1e4, inplace=False)['X']

        # Sample-Sample correlations
        data = [norm_internal_X.T, norm_imputed_X.T, norm_avg_X.T ,norm_scvi_X.T,
                        norm_scvi_X_2.T, norm_cascvi_X.T, norm_cascvi_X_2.T, norm_cascvi_X_3.T]
        _ = correlations(data=data,
                         normalization='None',
                         vis=True,
                         save_fig='results/imputation/'+'samplesample' + str(self.branch_length) + '_' + str(self.var) + '.jpg'
                         )  
        plt.clf() 

        # Gene-gene correlations
        data = [self.internal_X, imputed_X, internal_avg_X , internal_scvi_X, internal_scvi_X_2, internal_cascvi_X, internal_cascvi_X_2, internal_cascvi_X_3]
        _ = correlations(data=data,
                         normalization='rank',
                         vis=True,
                         save_fig='results/imputation/'+'genegene' + str(self.branch_length) + '_' + str(self.var) + '.jpg'
                         )  
        print(_)
        plt.clf() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_name', type=str, default='/Users/khalilouardini/Desktop/projects/scVI/scvi/data/Cassiopeia_trees/lg7_tree_hybrid_priors.alleleThresh.collapsed.txt',
                        help='Path of the Cassiopeia prior tree')
    parser.add_argument('--branch_length', type=float, default=1.0,
                        help='Branch length of the simulation (Gaussian Random Walk)')
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='Whether to use GPUs')
    parser.add_argument('--var', type=list, default=[1.0],
                        help='Branch length in the message passing inference')
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lambda_', type=float, default=1.0,
                        help='Regularization parameter in the the treeVAE')
    parser.add_argument('--latent', type=int, default=10,
                        help='dimension of latent space')
    parser.add_argument('--n_genes', type=int, default=1000,
                        help='Number of simulated genes')
    parser.add_argument('--simulation', type=str, default='poisson_glm', choices=['poisson_glm'],
                        help='Simulation framework')
    parser.add_argument('--n_hidden', type=int, default=128,
                        help='Number hidden units in the VAE')
    parser.add_argument('--seed', type=int, default=42,
                        help='random_seed')
    parser.add_argument('--n_epochs_kl_warmup', type=int, default=100,
                        help='Number of warm up epochs before introducing KL regularization in the VAE')

    #Parameters
    args = parser.parse_args()

    # Import the tree
    tree = Tree(args.tree_name, 1)
    # Args
    branch_length = args.branch_length
    var = args.var
    use_cuda = args.use_cuda
    n_epochs = args.n_epochs
    lr = args.lr
    seed = args.seed
    simulation = args.simulation
    n_hidden = args.n_hidden
    latent = args.latent
    n_genes = args.n_genes
    lambda_ = args.lambda_

    # Set random seed
    torch.manual_seed(seed), random.seed(seed), np.random.seed(seed)

    print("======== Ancestral Imputation  ======== \n")
    exp = Ancestral_Imputation(tree, branch_length, use_cuda, var, n_epochs, lr, lambda_, latent, n_genes, simulation, n_hidden)

    print("1: Simulate gene expression data \n")
    exp.simulations(simulation)

    print("2: Fitting models \n")

    print(">>>>> Fitting scVI <<<<<< \n")
    exp.fit_scvi(), print('\n')

    var = [0.01, 0.1, 1.0, 10.0, 100.0]
    for v in var:
            print(">>>>> Fitting CascVI with branch length: {} <<<<<< \n".format(v))
            exp.fit_cascvi(v), print('\n')
            print("3: Evalutation")
            exp.evaluation()

    












