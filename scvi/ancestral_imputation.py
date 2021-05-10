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

# Data
from anndata import AnnData
import scanpy as sc
from external.dataset.tree import TreeDataset, GeneExpressionDataset
from external.dataset.poisson_glm import Poisson_GLM
from external.dataset.anndataset import AnnDatasetFromAnnData

# Models
from external.models.treevae import TreeVAE
from external.inference.tree_inference import TreeTrainer, TreePosterior
from models.vae import VAE
from inference.inference import UnsupervisedTrainer
from inference import posterior

# Utils
from external.utils.data_util import get_leaves, get_internal
from external.utils.metrics import correlations, mse, update_metrics, report_results
from external.utils.baselines import avg_baseline_z, scvi_baseline_z, avg_weighted_baseline

class Ancestral_Imputation():

    def __init__(self,
                tree,
                fixed_branch_length,
                binomial_thinning,
                use_cuda,
                n_epochs,
                lr,
                lambda_,
                latent,
                n_genes,
                n_hidden,
                n_epochs_kl_warmup
        ):
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
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.lr = lr
        self.lambda_ = lambda_
        self.latent = latent
        self.n_genes = n_genes
        self.bin_thin = bin_thin

        #data
        self.glm = None
        self.tree_dataset = None
        self.leaves_X = None

        # Models
        self.treevae = None
        self.vae = None
        # neural network parameters
        self.n_hidden = n_hidden
        self.n_epochs_kl_warmup = n_epochs_kl_warmup

        # branch length
        if fixed_branch_length:
            self.branch_length = 1.0
        else:
            self.branch_length = {}
            for i, n in enumerate(self.tree.traverse('levelorder')):
                if not n.is_leaf():
                    n.name = str(i)
                n.add_features(index=i)
                if n.name == '0':
                    self.branch_length[n.name] = 0.1
                    continue
                self.branch_length[n.name] = n.dist
            self.branch_length['prior_root'] = 1.0


    def simulations(self):
        """
        Generating the data with a simulation framework (default is Poisson GLM)
        :return:
        """

        self.glm = Poisson_GLM(self.tree,
                            self.n_genes,
                            self.latent,
                            False, False,
                            self.branch_length
                            )
        self.glm.simulate_latent()
        self.glm.simulate_ge()

        # Quality Control (i.e Gene Filtering)
        #self.glm.gene_qc()

        if self.bin_thin < 1:
            self.glm.binomial_thinning(p=self.bin_thin)

        # Query data
        self.leaves_z, _, _ = get_leaves(self.glm.z, self.glm.mu, self.tree)
        # training set
        self.leaves_X, _, _ = get_leaves(self.glm.X, self.glm.mu, self.tree)
        # internal nodes data
        self.internal_X, _, _ = get_internal(self.glm.X, self.glm.mu, self.tree)

    def fit_cascvi(self):
        """
        Fitting cascVI to the gene expression data
        :return:
        """
        # anndata
        adata = AnnData(self.leaves_X)
        leaves = [n for n in tree.traverse('levelorder') if n.is_leaf()]
        adata.obs_names = [n.name for n in leaves]
        scvi_dataset = AnnDatasetFromAnnData(adata)
        scvi_dataset.initialize_cell_attribute('barcodes', adata.obs_names)

        # treedataset
        tree_bis = copy.deepcopy(self.tree)
        self.tree_dataset = TreeDataset(scvi_dataset, tree=tree_bis)

        # treeVAE
        self.treevae = TreeVAE(self.n_genes,
                      tree=self.tree_dataset.tree,
                      n_latent=self.latent,
                      n_hidden=self.n_hidden,
                      n_layers=1,
                      reconstruction_loss='poisson',
                      prior_t=self.branch_length,
                      ldvae=False,
                      use_MP=True
                      )

        freq = 100
        tree_trainer = TreeTrainer(
            model=self.treevae,
            gene_dataset=self.tree_dataset,
            lambda_=self.lambda_,
            train_size=1.0,
            test_size=0,
            use_cuda=self.use_cuda,
            frequency=freq,
            n_epochs_kl_warmup=self.n_epochs_kl_warmup
        )

        # training the VAE
        tree_trainer.train(n_epochs=self.n_epochs,
                      lr=self.lr
                      )

        self.tree_posterior = tree_trainer.create_posterior(tree_trainer.model, self.tree_dataset,
                                                     tree_trainer.clades, indices=np.arange(len(self.tree_dataset))
                                                        )

    def fit_scvi(self):
        """
        Fitting scVI to the gene expression data
        :return:
        """
        # anndata
        gene_dataset = GeneExpressionDataset()
        gene_dataset.populate_from_data(self.leaves_X)

        self.vae = VAE(self.n_genes,
                       n_batch=False,
                       n_hidden=self.n_hidden,
                       n_layers=1,
                       reconstruction_loss='poisson',
                       n_latent=self.latent,
                       ldvae=False
                       )

        trainer = UnsupervisedTrainer(model=self.vae,
                                    gene_dataset=gene_dataset,
                                    train_size=1.0,
                                    use_cuda=self.use_cuda,
                                    frequency=100,
                                    n_epochs_kl_warmup=self.n_epochs_kl_warmup
                                )

        # train scVI
        trainer.train(n_epochs=self.n_epochs,
                             lr=self.lr
                             )

        self.posterior = trainer.create_posterior(model=self.vae,
                                              gene_dataset=gene_dataset
                                              )

    def evaluation(self):
        """
        :return:
        """

        # empirical library size
        empirical_l = np.mean(np.sum(self.glm.X, axis=1))

        # 1. ========== CascVI imputations ===================
        imputed = {}
        imputed_z = {}
        for n in self.tree.traverse('levelorder'):
            if not n.is_leaf():
                imputed[n.name], imputed_z[n.name] = self.tree_posterior.imputation_internal(n.name,
                                                                                    give_mean=False,
                                                                                    library_size=empirical_l
                                                                                    )
        imputed_X = [x for x in imputed.values()]
        imputed_X = np.array(imputed_X).reshape(-1, self.n_genes)

        # 2. Baseline1: Average baseline
        imputed_avg = avg_weighted_baseline(tree=self.tree, 
                                            weighted=False, 
                                            X=self.glm.X, 
                                            rounding=True
                                            )
        avg_X = np.array([x for x in imputed_avg.values()]).reshape(-1, self.n_genes)
        internal_avg_X, _, _ = get_internal(avg_X, self.glm.mu, self.tree)

        # 3. Baseline 2: Decoded averaged latent space
        imputed_scvi, imputed_scvi_z = scvi_baseline_z(tree=self.tree,
                                                        model=self.vae,
                                                        posterior=self.posterior,
                                                        weighted=False,
                                                        n_samples_z=1,
                                                        library_size=empirical_l
                                                        )
        internal_scvi_X = np.array([x for x in imputed_scvi.values()]).reshape(-1, self.n_genes)

        #4. Baseline 3: MP Oracle
        imputed_oracle = {}
        for n in self.tree.traverse('levelorder'):
            if not n.is_leaf():
                _, z_temp = self.tree_posterior.imputation_internal(n.name,
                                                            give_mean=False,
                                                            library_size=empirical_l,
                                                            known_latent=self.leaves_z
                )

                mu_z = np.clip(a=np.exp(self.glm.W @ z_temp.cpu().numpy() + self.glm.beta),
                                a_min=0,
                                a_max=1e8
                                )
                samples = np.array([np.random.poisson(mu_z) for i in range(100)])
                imputed_oracle[n.name] = np.clip(a=np.mean(samples, axis=0),
                                                a_min=0,
                                                a_max=1e8
                                                )
        internal_oracle_X = np.array([x for x in imputed_oracle.values()]).reshape(-1, self.n_genes)

        # Gene-gene correlations
        # Normalizing data
        norm_internal_X = sc.pp.normalize_total(AnnData(self.internal_X), target_sum=1e4, inplace=False)['X'] 
        norm_scvi_X = sc.pp.normalize_total(AnnData(internal_scvi_X), target_sum=1e4, inplace=False)['X']
        norm_avg_X = sc.pp.normalize_total(AnnData(internal_avg_X), target_sum=1e4, inplace=False)['X']
        norm_imputed_X = sc.pp.normalize_total(AnnData(imputed_X), target_sum=1e4, inplace=False)['X']
        norm_oracle_X = sc.pp.normalize_total(AnnData(internal_oracle_X), target_sum=1e4, inplace=False)['X']

        data = {'groundtruth': norm_internal_X, 'cascVI': norm_imputed_X, 'scVI': norm_scvi_X, 
                'Average': norm_avg_X, 'Oracle MP': norm_oracle_X}
        
        return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_name', type=str, default='/Users/khalilouardini/Desktop/projects/scVI/scvi/data/Cassiopeia_trees/lg7_tree_hybrid_priors.alleleThresh.collapsed.txt',
                        help='Path of the Cassiopeia prior tree')
    parser.add_argument('--n_cells_tree', type=int, default=500, choices=[100, 250, 500],
                        help='number of leaves in simulations')
    parser.add_argument('--fitness', type=str, default='no_fitness', choices=['no_fitness', 'low_fitness', 'high_fitness'],
                    help='fitness regime of simulations')
    parser.add_argument('--fixed_branch_length', type=bool, default=False,
                        help='whether to use a fixed branch length in the simulations (Gaussian Random Walk)')
    parser.add_argument('--binomial_thinning', type=float, default=0.1,
                    help='proportion of binomial thinning in the Poisson simulations')
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='Whether to use GPUs')
    parser.add_argument('--n_epochs', type=int, default=800,
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
    parser.add_argument('--n_epochs_kl_warmup', type=int, default=200,
                        help='Number of warm up epochs before introducing KL regularization in the VAE')

    #Parameters
    args = parser.parse_args()

    # Import the tree
    #tree = Tree(args.tree_name, 1)

    # Args
    n_cells_tree = args.n_cells_tree
    fitness = args.fitness
    fixed_branch_length = args.fixed_branch_length
    bin_thin = args.binomial_thinning
    use_cuda = args.use_cuda
    n_epochs = args.n_epochs
    lr = args.lr
    seed = args.seed
    n_hidden = args.n_hidden
    latent = args.latent
    n_genes = args.n_genes
    lambda_ = args.lambda_
    n_epochs_kl_warmup = args.n_epochs_kl_warmup

    # Set random seed
    torch.manual_seed(seed), random.seed(seed), np.random.seed(seed)

    print("==== Loading trees ====")
    print("Collection of trees with {} leaves, and {} regime".format(n_cells_tree, fitness))
    tree_folder = '/home/eecs/khalil.ouardini/cas_scvi_topologies/newick_objects'
    tree_folder = os.path.join(tree_folder, str(n_cells_tree)+'cells')
    tree_folder = os.path.join(tree_folder, fitness)
    tree_paths = [os.path.join(tree_folder, f) for f in os.listdir(tree_folder)]

    metrics = {'correlations_ss': [], 'correlations_gg': [], 'MSE': [], 'L1': []}

    for tree_path in tree_paths:
        tree = Tree(tree_path, 1)

        print("========  Gaussian Ancestral Imputation  ======== \n")
        exp = Ancestral_Imputation(
                                tree=tree,
                                fixed_branch_length=fixed_branch_length,
                                binomial_thinning=bin_thin, 
                                use_cuda=use_cuda, 
                                n_epochs=n_epochs, 
                                lr=lr, 
                                lambda_=lambda_, 
                                latent=latent, 
                                n_genes=n_genes, 
                                n_hidden=n_hidden,
                                n_epochs_kl_warmup=n_epochs_kl_warmup
                                )

        print("I: Simulate gene expression data \n")
        print("proportion of binomial thinning in Poisson simulations: {}".format(args.binomial_thinning))
        exp.simulations()

        print("II: Fitting models \n")

        print("==== Fitting VAE ==== \n")
        exp.fit_scvi(), print('\n')

        print("==== Fitting treeVAE ==== \n")
        exp.fit_cascvi(), print('\n')

        print("III: Evalutation")
        data = exp.evaluation()
        columns2 = list(data.keys())[1:]
        update_metrics(metrics=metrics,
                        data=data,
                        normalization=None)

    results_dir = 'results/poisson'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    next_dir = os.path.join(results_dir, str(n_cells_tree))
    if not os.path.exists(next_dir):
        os.mkdir(next_dir)
    next_dir = os.path.join(next_dir, fitness)
    if not os.path.exists(next_dir):
        os.mkdir(next_dir)
    next_dir = os.path.join(next_dir, 'bin'+str(bin_thin))
    if not os.path.exists(next_dir):
        os.mkdir(next_dir)

    report_results(metrics=metrics,
                  save_path=next_dir,
                  columns2=columns2
                  )

    












