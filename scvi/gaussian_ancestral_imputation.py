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
from external.dataset.ppca import PPCA
from external.dataset.anndataset import AnnDatasetFromAnnData

# Models
from external.models.gaussian_vae import GaussianVAE
from external.models.gaussian_treevae import GaussianTreeVAE
from external.inference.gaussian_tree_inference import GaussianTreeTrainer, GaussianTreePosterior
from external.inference.gaussian_inference import GaussianTrainer, GaussianPosterior

# Utils
from external.utils.data_util import get_leaves, get_internal
from external.utils.metrics import correlations, mse, mean_variance_latent, mean_posterior_lik, update_metrics, report_results
from external.utils.baselines import avg_baseline_z, scvi_baseline_z, avg_weighted_baseline

# Wandb
import wandb


class GaussianAncestralImputation():

    def __init__(self,
                tree,
                fixed_branch_length,
                sigma,
                use_cuda,
                n_epochs,
                lr,
                lambda_,
                latent, 
                n_genes, 
                n_hidden
        ):
        """
        :param tree: ete3: Cassiopeia prior tree
        :param fixed_branch_length: bool: Whether to use a fixed branch length (=1.0) for the simulations
        :param use_cuda: bool: True to train on GPUs
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

        # data
        self.sigma = sigma
        self.ppca = None
        self.tree_dataset = None
        self.leaves_X = None

        # Models
        self.treevae = None
        self.vae = None

        # neural network parameters
        self.n_hidden = n_hidden

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
        self.ppca = PPCA(tree=self.tree,
                        dim=self.n_genes,
                        latent=self.latent,
                        vis=False,
                        only=False,
                        branch_length=self.branch_length,
                        sigma_scale=self.sigma
                                        )
        self.ppca.simulate_latent()
        self.ppca.simulate_normal()

        # Query data
        # training set
        self.leaves_X, _, _ = get_leaves(self.ppca.X, self.ppca.mu, self.tree)
        # internal nodes data
        self.internal_X, _, _ = get_internal(self.ppca.X, self.ppca.mu, self.tree)

    def fit_treevae(self):
        """
        Fitting GaussianTreeVAE to the gene expression data
        :return:
        """
        # anndata
        adata = AnnData(self.leaves_X)
        adata.obs_names = [n.name for n in self.tree.traverse('levelorder') if n.is_leaf()]
        scvi_dataset = AnnDatasetFromAnnData(adata, filtering=False)
        scvi_dataset.initialize_cell_attribute('barcodes', adata.obs_names)

        # Treedataset
        tree_bis = copy.deepcopy(self.tree)
        self.tree_dataset = TreeDataset(scvi_dataset, tree=tree_bis, filtering=False)

        # treeVAE
        self.treevae = GaussianTreeVAE(self.n_genes,
                    tree=self.tree,
                    n_latent=self.latent,
                    n_hidden=self.n_hidden,
                    n_layers=1,
                    prior_t = self.branch_length,
                    use_MP=True,
                    sigma_ldvae=None
                    )

        freq = 10
        tree_trainer = GaussianTreeTrainer(
            model=self.treevae,
            gene_dataset=self.tree_dataset,
            lambda_=self.lambda_,
            train_size=1.0,
            test_size=0,
            use_cuda=self.use_cuda,
            frequency=50,
            n_epochs_kl_warmup=None
        )

        # training the VAE
        tree_trainer.train(n_epochs=self.n_epochs,
                      lr=self.lr)

        self.tree_posterior = tree_trainer.create_posterior(tree_trainer.model,
                                                    self.tree_dataset,
                                                    tree_trainer.clades,
                                                    indices=np.arange(len(self.tree_dataset))
                                                    )

    def fit_vae(self):
        """
        Fitting Gaussian VAE to the gene expression data
        :return:
        """
        # anndata
        gene_dataset = GeneExpressionDataset()
        gene_dataset.populate_from_data(self.leaves_X)

        self.vae = GaussianVAE(self.n_genes,
                  n_latent=self.latent,
                  n_hidden=self.n_hidden,
                  n_layers=1,
                  sigma_ldvae=None
              )

        trainer = GaussianTrainer(model=self.vae,
                                    gene_dataset=gene_dataset,
                                    train_size=1.0,
                                    use_cuda=self.use_cuda,
                                    frequency=200,
                                    n_epochs_kl_warmup=None)

        # train scVI
        trainer.train(n_epochs=self.n_epochs,
                             lr=self.lr)

        self.posterior = trainer.create_posterior(model=trainer.model,
                                              gene_dataset=gene_dataset
                                              )

    def evaluation(self):
        """
        :return:
        """

        # ========== Baselines ===================

        ##### I. Unweighted average
        imputed_avg = avg_weighted_baseline(tree=self.tree, 
                                    weighted=False, 
                                    X=self.ppca.X,
                                    rounding=False
                                   )
        avg_X = np.array([x for x in imputed_avg.values()]).reshape(-1,self.n_genes)
        internal_avg_X, _, _ = get_internal(avg_X, self.ppca.mu, self.tree)


        #### II. PPCA
        self.ppca.compute_leaves_covariance()
        posterior_mean, posterior_cov = self.ppca.compute_posterior()
        predictive_mean, predictive_cov = self.ppca.compute_posterior_predictive()
        # II.1 groundtruth posterior predictive
        imputed_ppca = {}
        for n in self.tree.traverse('levelorder'):
            if not n.is_leaf():
                samples = np.array([np.random.multivariate_normal(mean=predictive_mean[n.name],
                                                                    cov=predictive_cov[n.name])
                                for i in range(20)])
                imputed_ppca[n.name] = np.mean(samples, axis=0)

        internal_ppca_X = np.array([x for x in imputed_ppca.values()]).reshape(-1, self.n_genes)
        # II.2 Approximate posterior (Oracle)
        posterior_mean_corr, posterior_cov_corr = self.ppca.compute_correlated_posterior()
        imputed_mp, _, predictive_mean_z, predictive_cov_z = self.ppca.compute_approx_posterior_predictive(iid=False, 
                                                                    use_MP=True, 
                                                                    sample_size=200
                                                                    )
        imputed_X = np.array([x for x in imputed_mp.values()]).reshape(-1, self.n_genes)

        #### III. Gaussian VAE (decoded averged latent space)
        imputed_avg_vae, imputed_avg_z, imputed_avg_cov_z = avg_baseline_z(tree=self.tree,
                                 model=self.vae,
                                 posterior=self.posterior,
                                 weighted=False,
                                 n_samples_z=1,
                                 gaussian=True,
                                 use_cuda=self.use_cuda
                                )
        internal_vae_X = np.array([x for x in imputed_avg_vae.values()]).reshape(-1, self.n_genes)

        #### IV. Gaussian Tree VAE
        # CascVI imputations
        imputed = {}
        imputed_mean_z = {}
        imputed_cov_z = {}
        for n in self.tree.traverse('levelorder'):
            if not n.is_leaf():
                imputed[n.name], _, imputed_mean_z[n.name], imputed_cov_z[n.name] = self.tree_posterior.imputation_internal(query_node=n.name,
                                                                                                                pp_averaging=200,
                                                                                                                z_averaging=None,
                                                                                                                give_mean=True                           
                                                                                                            )
        internal_treevae_X = [x for x in imputed.values()]
        internal_treevae_X = np.array(internal_treevae_X).reshape(-1, self.n_genes)

        data = {'groundtruth': imputed_X, 'average': internal_avg_X, 'gaussian VAE': internal_vae_X,
                'gaussian treeVAE': internal_treevae_X
                }
        data2 = {'groundtruth': self.internal_X, 'average': internal_avg_X, 'gaussian VAE': internal_vae_X,
                'gaussian treeVAE': internal_treevae_X
                }

        # Secondary metrics
        var_metrics = mean_variance_latent(self.tree, predictive_cov_z, imputed_avg_cov_z, imputed_cov_z)
        lik_metrics = mean_posterior_lik(self.tree, predictive_mean_z, imputed_avg_z, imputed_mean_z, predictive_cov_z, imputed_avg_cov_z, imputed_cov_z)

        return data, data2, var_metrics, lik_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_name', type=str, default='/Users/khalilouardini/Desktop/projects/scVI/scvi/data/Cassiopeia_trees/lg7_tree_hybrid_priors.alleleThresh.collapsed.txt',
                        help='Path of the Cassiopeia prior tree')
    parser.add_argument('--n_cells_tree', type=int, default=100, choices=[100, 250, 500],
                        help='number of leaves in simulations')
    parser.add_argument('--fitness', type=str, default='no_fitness', choices=['no_fitness', 'low_fitness', 'high_fitness'],
                    help='fitness regime of simulations')
    parser.add_argument('--fixed_branch_length', type=bool, default=False,
                        help='whether to use a fixed branch length in the simulations (Gaussian Random Walk)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='variance in pPCA simulations')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Whether to use GPUs')
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lambda_', type=float, default=1.0,
                        help='Regularization parameter in the treeVAE')
    parser.add_argument('--latent', type=int, default=5,
                        help='dimension of latent space')
    parser.add_argument('--n_genes', type=int, default=100,
                        help='Number of simulated genes')
    parser.add_argument('--n_hidden', type=int, default=64,
                        help='Number hidden units in the VAE')
    parser.add_argument('--seed', type=int, default=42,
                        help='random_seed')
    parser.add_argument('--n_epochs_kl_warmup', type=int, default=100,
                        help='Number of warm up epochs before introducing KL regularization in the VAE')


    #Parameters
    args = parser.parse_args()

    # Import the tree
    #tree = Tree(args.tree_name, 1)

    # Args
    n_cells_tree = args.n_cells_tree
    fitness = args.fitness
    fixed_branch_length = args.fixed_branch_length
    sigma = args.sigma
    use_cuda = args.use_cuda
    n_epochs = args.n_epochs
    lr = args.lr
    seed = args.seed
    n_hidden = args.n_hidden
    latent = args.latent
    n_genes = args.n_genes
    lambda_ = args.lambda_

    # Set random seed
    torch.manual_seed(seed), random.seed(seed), np.random.seed(seed)
    
    print("==== Loading trees ====")
    print("Collection of trees with {} leaves, and {} regime".format(n_cells_tree, fitness))
    tree_folder = '/home/eecs/khalil.ouardini/cas_scvi_topologies/newick_objects'
    tree_folder = os.path.join(tree_folder, str(n_cells_tree)+'cells')
    tree_folder = os.path.join(tree_folder, fitness)
    tree_paths = [os.path.join(tree_folder, f) for f in os.listdir(tree_folder)]

    metrics = {'correlations_ss': [], 'correlations_gg': [],
               'MSE': [], 'L1': [],
               'MSE_var': [], 'Likelihood': []
            }
    
    metrics2 = {'correlations_ss2': [], 'correlations_gg2': [],
               'MSE_2': [], 'L1_2': []
                }

    for tree_path in tree_paths:
        tree = Tree(tree_path, 1)

        print("========  Gaussian Ancestral Imputation  ======== \n")
        print("tree: {}".format(tree_path))
        print("Variance in pPCA simulations: {}".format(sigma))
        exp = GaussianAncestralImputation(
                                tree=tree,
                                fixed_branch_length=fixed_branch_length,
                                sigma=sigma, 
                                use_cuda=use_cuda, 
                                n_epochs=n_epochs, 
                                lr=lr, 
                                lambda_=lambda_, 
                                latent=latent, 
                                n_genes=n_genes, 
                                n_hidden=n_hidden
                                )

        print("I: Simulate gene expression data \n")
        exp.simulations()

        print("II: Fitting models \n")

        print("==== Fitting VAE ==== \n")
        exp.fit_vae(), print('\n')

        print("==== Fitting treeVAE ==== \n")
        exp.fit_treevae(), print('\n')

        print("III: Evalutation")
        data, data2, var_metrics, lik_metrics = exp.evaluation()
        columns2 = list(data.keys())[1:]

        # update main metrics
        update_metrics(metrics=metrics,
                        data=data,
                        normalization=None
                        )
        
        # update main metrics 2
        update_metrics(metrics=metrics2,
                        data=data2,
                        normalization=None
                        )
        
        # update secondary metrics
        metrics['MSE_var'].append(var_metrics)
        metrics['Likelihood'].append(lik_metrics)

    results_dir = 'results/gaussian'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    next_dir = os.path.join(results_dir, str(n_cells_tree))
    if not os.path.exists(next_dir):
        os.mkdir(next_dir)
    next_dir = os.path.join(next_dir, fitness)
    if not os.path.exists(next_dir):
        os.mkdir(next_dir)

    report_results(metrics=metrics,
                  save_path=next_dir,
                  columns2=columns2
                  )

    report_results(metrics=metrics2,
                save_path=next_dir,
                columns2=columns2
                )
    


    













