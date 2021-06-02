import os
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
from external.dataset.poisson_glm import Poisson_GLM
from external.dataset.anndataset import AnnDatasetFromAnnData

# Models
from external.models.treevae import TreeVAE
from external.inference.tree_inference import TreeTrainer
from external.models.vae import VAE
from external.inference.inference import UnsupervisedTrainer

# Utils
from external.utils.data_util import get_leaves, get_internal
from external.utils.metrics import correlations, mse, update_metrics, report_results, knn_purity
from external.utils.baselines import avg_baseline_z, scvi_baseline_z, avg_weighted_baseline, construct_latent

class Ancestral_Imputation():

    def __init__(self,
                tree,
                fixed_branch_length,
                t_norm,
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
        self.bin_thin = binomial_thinning

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
                n.name = str(i)
                n.add_features(index=i)
                if n.is_root():
                    self.branch_length[n.name] = 0.0
                else:
                    self.branch_length[n.name] = t_norm * n.dist
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
                            self.branch_length,
                            1.0
                            )
        self.glm.simulate_latent()
        self.glm.simulate_ge(negative_binomial=False)

        # Quality Control (i.e Gene Filtering)
        self.glm.gene_qc()

        if self.bin_thin < 1:
            self.glm.binomial_thinning(p=self.bin_thin)

        # Query data
        self.leaves_z, _, _ = get_leaves(self.glm.z, self.glm.mu, self.tree)
        # training set
        self.leaves_X, _, _ = get_leaves(self.glm.X, self.glm.mu, self.tree)
        # internal nodes data
        self.internal_X, _, _ = get_internal(self.glm.X, self.glm.mu, self.tree)
        # Latent space
        self.leaves_z, _, _ = get_leaves(self.glm.z, self.glm.mu, self.tree)

    def fit_cascvi(self):
        """
        Fitting cascVI to the gene expression data
        :return:
        """
        # anndata
        adata = AnnData(self.leaves_X)
        leaves = [n for n in tree.traverse('levelorder') if n.is_leaf()]
        adata.obs_names = [n.name for n in leaves]
        scvi_dataset = AnnDatasetFromAnnData(adata, filtering=False)
        scvi_dataset.initialize_cell_attribute('barcodes', adata.obs_names)

        # treedataset
        tree_bis = copy.deepcopy(self.tree)
        self.tree_dataset = TreeDataset(scvi_dataset, tree=tree_bis, filtering=False)

        # treeVAE
        self.treevae = TreeVAE(self.tree_dataset.nb_genes,
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
        tree_trainer.train(n_epochs=self.n_epochs+100,
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

        self.vae = VAE(gene_dataset.nb_genes,
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
                imputed[n.name], z = self.tree_posterior.imputation_internal(n,
                                                                                    give_mean=False,
                                                                                    library_size=empirical_l
                                                                                    )
                imputed_z[n.name] = z
        imputed_X = [x for x in imputed.values()]
        imputed_X = np.array(imputed_X).reshape(-1, self.glm.X.shape[1])

        # 2. Baseline1: Average baseline
        imputed_avg = avg_weighted_baseline(tree=self.tree, 
                                            weighted=False, 
                                            X=self.glm.X, 
                                            rounding=True
                                            )
        avg_X = np.array([x for x in imputed_avg.values()]).reshape(-1, self.glm.X.shape[1])
        internal_avg_X, _, _ = get_internal(avg_X, self.glm.mu, self.tree)

        # 3. Baseline 2: Decoded averaged latent space
        imputed_scvi, imputed_scvi_z = scvi_baseline_z(tree=self.tree,
                                                        model=self.vae,
                                                        posterior=self.posterior,
                                                        weighted=False,
                                                        n_samples_z=1,
                                                        library_size=empirical_l,
                                                        use_cuda=self.use_cuda
                                                        )
        internal_scvi_X = np.array([x for x in imputed_scvi.values()]).reshape(-1, self.glm.X.shape[1])


        # Gene-gene correlations
        # Normalizing data
        norm_internal_X = sc.pp.normalize_total(AnnData(self.internal_X), target_sum=1e6, inplace=False)['X'] 
        norm_scvi_X = sc.pp.normalize_total(AnnData(internal_scvi_X), target_sum=1e6, inplace=False)['X']
        norm_avg_X = sc.pp.normalize_total(AnnData(internal_avg_X), target_sum=1e6, inplace=False)['X']
        norm_imputed_X = sc.pp.normalize_total(AnnData(imputed_X), target_sum=1e6, inplace=False)['X']

        # 4.. Cross Entropy 
        vae_latent = self.posterior.get_latent()[0]
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

        # 5. k-NN purity 
        full_treevae_latent = construct_latent(self.tree, treevae_latent, imputed_z)
        full_vae_latent = construct_latent(self.tree, vae_latent, imputed_scvi_z)

        internal_z, _, _ = get_internal(self.glm.z, self.glm.mu, self.tree)
        internal_vae_z, _, _ = get_internal(full_vae_latent, self.glm.mu, self.tree)
        internal_treevae_z, _, _ = get_internal(full_treevae_latent, self.glm.mu, self.tree)

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
        data = {'groundtruth': self.glm.z, 'scVI': full_vae_latent, 'cascVI': full_treevae_latent}
        scores = knn_purity(max_neighbors=max_neighbors, data=data, plot=False)      
        purity_full = pd.DataFrame(data={'K':neighbors, 'scVI': scores['scVI'], 'cascVI': scores['cascVI']})
        
        data_purity_metrics = [purity, purity_internal, purity_full]

        data = {'groundtruth': norm_internal_X, 'cascVI': norm_imputed_X, 'scVI': norm_scvi_X, 
                'Average': norm_avg_X}
        
        return data, ce_metrics, data_purity_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_name', type=str,
                        help='Path of the Cassiopeia prior tree')
    parser.add_argument('--n_cells_tree', type=int, default=500, choices=[100, 250, 500],
                        help='number of leaves in simulations')
    parser.add_argument('--fitness', type=str, default='no_fitness', choices=['no_fitness', 'low_fitness', 'high_fitness'],
                    help='fitness regime of simulations')
    parser.add_argument('--fixed_branch_length', type=bool, default=False,
                        help='whether to use a fixed branch length in the simulations (Gaussian Random Walk)')
    parser.add_argument('--t_norm', type=float, default=1,
                        help='branch length normalization parameter')
    parser.add_argument('--binomial_thinning', type=float, default=0.1,
                    help='proportion of binomial thinning in the Poisson simulations')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Whether to use GPUs')
    parser.add_argument('--n_epochs', type=int, default=600,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lambda_', type=float, default=1.0,
                        help='Regularization parameter in the the treeVAE')
    parser.add_argument('--latent', type=int, default=10,
                        help='dimension of latent space')
    parser.add_argument('--n_genes', type=int, default=1000,
                        help='Number of simulated genes')
    parser.add_argument('--simulation', type=str, default='poisson_glm', choices=['poisson_glm', 'negative_binomial'],
                        help='Simulation framework')
    parser.add_argument('--n_hidden', type=int, default=128,
                        help='Number hidden units in the VAE')
    parser.add_argument('--seed', type=int, default=0,
                        help='random_seed')
    parser.add_argument('--n_epochs_kl_warmup', type=int, default=150,
                        help='Number of warm up epochs before introducing KL regularization in the VAE')

    #Parameters
    args = parser.parse_args()

    # Import the tree
    #tree = Tree(args.tree_name, 1)

    # Args
    n_cells_tree = args.n_cells_tree
    fitness = args.fitness
    fixed_branch_length = args.fixed_branch_length
    t_norm = args.t_norm
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
    tree_folder = 'data/topologies/newick_objects'
    tree_folder = os.path.join(tree_folder, str(n_cells_tree)+'cells')
    #tree_folder = os.path.join(tree_folder, fitness)
    tree_paths = [os.path.join(tree_folder, f) for f in os.listdir(tree_folder)]

    metrics = {'correlations_ss': [], 'correlations_gg': [], 'MSE': [], 'L1': [], 'Cross_Entropy': []}

    purity, purity_internal, purity_full = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for tree_path in tree_paths:
        tree = Tree(tree_path, 1)

        print("======== Ancestral Imputation  ======== \n")
        exp = Ancestral_Imputation(
                                tree=tree,
                                fixed_branch_length=fixed_branch_length,
                                t_norm=t_norm,
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
        data, ce_metrics, data_purity_metrics = exp.evaluation()
        columns2 = list(data.keys())[1:]

        # Update metrics
        update_metrics(metrics=metrics,
                        data=data,
                        normalization=None)

        # update secondary metrics
        metrics['Cross_Entropy'].append(ce_metrics)

        # Purity metrics
        df_purity, df_purity_internal, df_purity_full = data_purity_metrics
        purity = purity.append(df_purity)
        purity_internal = purity_internal.append(df_purity_internal)
        purity_full = purity_full.append(df_purity_full)

    if not os.path.exists('results'):
        os.mkdir('results')
    results_dir = 'results/poisson'
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
    next_dir = os.path.join(next_dir, 'bin'+str(bin_thin))
    if not os.path.exists(next_dir):
        os.mkdir(next_dir)

    report_results(metrics=metrics,
                  save_path=next_dir,
                  columns2=columns2
                  )

    # Save purity
    purity.to_csv(os.path.join(next_dir, 'purity_leaves'))
    purity_internal.to_csv(os.path.join(next_dir, 'purity_internal'))
    purity_full.to_csv(os.path.join(next_dir, 'purity_full'))











