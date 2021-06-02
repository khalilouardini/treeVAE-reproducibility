import os
import sys
import numpy as np
import random
import pandas as pd
import argparse
import copy
import torch
from ete3 import Tree

# Data
from anndata import AnnData
import scanpy as sc
from external.dataset.tree import TreeDataset, GeneExpressionDataset
from external.dataset.ppca import PPCA
from external.dataset.anndataset import AnnDatasetFromAnnData

# Models
from external.models.gaussian_vae import GaussianVAE
from external.models.gaussian_treevae import GaussianTreeVAE
from external.inference.gaussian_tree_inference import GaussianTreeTrainer
from external.inference.gaussian_inference import GaussianTrainer

# Utils
from external.utils.data_util import get_leaves, get_internal
from external.utils.metrics import error_latent, mean_posterior_lik, update_metrics, report_results, knn_purity
from external.utils.baselines import avg_baseline_z, avg_weighted_baseline, construct_latent


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
                n.name = str(i)
                n.add_features(index=i)
                if n.is_root():
                    self.branch_length[n.name] = 0.0
                else:
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
        # latent space
        self.leaves_z, _, _ = get_leaves(self.ppca.z, self.ppca.mu, self.tree)

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
        tree_trainer.train(n_epochs=self.n_epochs+300,
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

    def evaluation(self, tree_name):
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
        cov_folder = 'data/inverse_covariances'
        cov_path = os.path.join(cov_folder, tree_name+'.npy')
        print(cov_path), print(os.path.exists(cov_path))
        if os.path.exists(cov_path):
            print("Loading saved inverse covariance: {}".format(cov_path))
            self.ppca.leaves_covariance_inv = np.load(cov_path)
        else:
            self.ppca.compute_leaves_covariance()
            np.save(cov_path, self.ppca.leaves_covariance_inv)

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
                                 use_cuda=self.use_cuda,
                                 give_cov=True
                                )
        internal_vae_X = np.array([x for x in imputed_avg_vae.values()]).reshape(-1, self.n_genes)

        #### IV. Gaussian Tree VAE
        # CascVI imputations
        imputed = {}
        imputed_mean_z = {}
        imputed_cov_z = {}
        imputed_z = {}
        for n in self.tree.traverse('levelorder'):
            if not n.is_leaf():
                imputed[n.name], z, imputed_mean_z[n.name], imputed_cov_z[n.name] = self.tree_posterior.imputation_internal(query_node=n,
                                                                                                                pp_averaging=200,
                                                                                                                z_averaging=None,
                                                                                                                give_mean=True                         
                                                                                                            )
                imputed_z[n.name] = z.cpu().numpy()
        internal_treevae_X = [x for x in imputed.values()]
        internal_treevae_X = np.array(internal_treevae_X).reshape(-1, self.n_genes)

        #### V. MCMC Mean and Variance estimates
        imputed_mcmc_cov = {}
        imputed_mcmc_mean = {}

        mcmc_mean = {}
        mcmc_cov = {}
        M = 20
        for n in self.tree.traverse('levelorder'):
            if not n.is_leaf():
                # with Approximate posterior q(z|x)
                imputed_mcmc_mean[n.name], imputed_mcmc_cov[n.name] = self.tree_posterior.mcmc_estimate(query_node=n,
                                                                                                    n_samples=M
                                                                                                    )
                # with groundtruth posterior p(z|x)
                mcmc_mean[n.name], mcmc_cov[n.name] = self.tree_posterior.mcmc_estimate(query_node=n,
                                                                    n_samples=M,
                                                                    known_latent_dist=(posterior_mean_corr, posterior_cov_corr)
                                                                    )


        #### VI. Cross Entropy ####
        vae_latent = self.posterior.get_latent()
        treevae_latent = self.tree_posterior.get_latent()
        #--> cross entropy prior | guassian VAE
        self.treevae.initialize_visit()
        self.treevae.initialize_messages(vae_latent, self.tree_dataset.barcodes, self.latent)
        self.treevae.perform_message_passing((self.treevae.tree & self.treevae.root), self.latent, False)
        ce_vae = self.treevae.aggregate_messages_into_leaves_likelihood(self.latent, add_prior=True).item()
        #--> cross entropy prior | guassianTreeVAE
        self.treevae.initialize_visit()
        self.treevae.initialize_messages(treevae_latent, self.tree_dataset.barcodes, self.latent)
        self.treevae.perform_message_passing((self.treevae.tree & self.treevae.root), self.latent, False)
        ce_treevae = self.treevae.aggregate_messages_into_leaves_likelihood(self.latent, add_prior=True).item()

        ce_metrics = [ce_vae, ce_treevae]

        ###### VII. k-NN purity #####
        full_treevae_latent = construct_latent(self.tree, treevae_latent, imputed_z)
        full_vae_latent = construct_latent(self.tree, vae_latent, imputed_avg_z)

        internal_z, _, _ = get_internal(self.ppca.z, self.ppca.mu, self.tree)
        internal_vae_z, _, _ = get_internal(full_vae_latent, self.ppca.mu, self.tree)
        internal_treevae_z, _, _ = get_internal(full_treevae_latent, self.ppca.mu, self.tree)

        max_neighbors = 30
        neighbors = list(range(2, max_neighbors))

        # Leaves
        data = {'groundtruth': self.leaves_z, 'scVI': vae_latent, 'cascVI': treevae_latent}
        scores = knn_purity(max_neighbors=max_neighbors, data=data, plot=False)      
        purity = pd.DataFrame(data={'K':neighbors, 'scVI': scores['scVI'], 'cascVI': scores['cascVI']})
        # Internal
        data = {'groundtruth': internal_z, 'scVI': internal_vae_z, 'cascVI': internal_treevae_z}
        scores = knn_purity(max_neighbors=max_neighbors, data=data, plot=False)      
        purity_internal = pd.DataFrame(data={'K':neighbors, 'scVI': scores['scVI'], 'cascVI': scores['cascVI']})
        # Full
        data = {'groundtruth': self.ppca.z, 'scVI': full_vae_latent, 'cascVI': full_treevae_latent}
        scores = knn_purity(max_neighbors=max_neighbors, data=data, plot=False)      
        purity_full = pd.DataFrame(data={'K':neighbors, 'scVI': scores['scVI'], 'cascVI': scores['cascVI']})
        
        data_purity_metrics = [purity, purity_internal, purity_full]

        data = {'groundtruth': imputed_X, 'average': internal_avg_X, 'gaussian VAE': internal_vae_X,
                'gaussian treeVAE': internal_treevae_X
                }
        data2 = {'groundtruth': self.internal_X, 'average': internal_avg_X, 'gaussian VAE': internal_vae_X,
                'gaussian treeVAE': internal_treevae_X
                }

        # Secondary metrics
        mean_metrics = error_latent(self.tree, mcmc_mean, imputed_avg_z, imputed_mcmc_mean, False)
        var_metrics = error_latent(self.tree, mcmc_cov, imputed_avg_cov_z, imputed_mcmc_cov, True)
        lik_metrics = mean_posterior_lik(self.tree, mcmc_mean, imputed_avg_z, imputed_mcmc_mean,
                                         mcmc_cov, imputed_avg_cov_z, imputed_mcmc_cov)

        return data, data2, mean_metrics, var_metrics, lik_metrics, ce_metrics, data_purity_metrics




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_name', type=str, default='data/lg7_tree_hybrid_priors.alleleThresh.collapsed.txt',
                        help='Path of the Cassiopeia prior tree')
    parser.add_argument('--n_cells_tree', type=int, default=100, choices=[100, 250, 500],
                        help='number of leaves in simulations')
    parser.add_argument('--fitness', type=str, default='low_fitness', choices=['no_fitness', 'low_fitness', 'high_fitness'],
                    help='fitness regime of simulations')
    parser.add_argument('--fixed_branch_length', type=bool, default=False,
                        help='whether to use a fixed branch length in the simulations (Gaussian Random Walk)')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='variance in pPCA simulations')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Whether to use GPUs')
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lambda_', type=float, default=2.0,
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
    tree_folder = 'data/topologies/newick_objects'
    tree_folder = os.path.join(tree_folder, str(n_cells_tree)+'cells')
    #tree_folder = os.path.join(tree_folder, fitness)
    tree_paths = [os.path.join(tree_folder, f) for f in os.listdir(tree_folder)]

    metrics = {'correlations_ss': [], 'correlations_gg': [],
               'MSE': [], 'L1': [], 'MSE_mean': [],
               'MSE_var': [], 'Likelihood': [], 'Cross_Entropy': []
            }
    
    metrics2 = {'correlations_ss2': [], 'correlations_gg2': [],
               'MSE_2': [], 'L1_2': []
                }

    purity, purity_internal, purity_full = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for tree_path in tree_paths:
        tree = Tree(tree_path, 1)

        tree_name = str(n_cells_tree) + 'cells_'+ fitness + '_' + tree_path.split('/')[-1]

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
        data, data2, mean_metrics, var_metrics, lik_metrics, ce_metrics, data_purity_metrics = exp.evaluation(tree_name)
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
        metrics['MSE_mean'].append(mean_metrics)
        metrics['MSE_var'].append(var_metrics)
        metrics['Likelihood'].append(lik_metrics)
        metrics['Cross_Entropy'].append(ce_metrics)

        # Purity metrics
        df_purity, df_purity_internal, df_purity_full = data_purity_metrics
        purity = purity.append(df_purity)
        purity_internal = purity_internal.append(df_purity_internal)
        purity_full = purity_full.append(df_purity_full)

    if not os.path.exists('results'):
        os.mkdir('results')
    results_dir = 'results/gaussian'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    next_dir = os.path.join(results_dir, 'lambda'+str(lambda_))
    if not os.path.exists(next_dir):
        os.mkdir(next_dir)
    next_dir = os.path.join(next_dir, str(n_cells_tree))
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
    
    # Save purity
    purity.to_csv(os.path.join(next_dir, 'purity_leaves'))
    purity_internal.to_csv(os.path.join(next_dir, 'purity_internal'))
    purity_full.to_csv(os.path.join(next_dir, 'purity_full'))


    













