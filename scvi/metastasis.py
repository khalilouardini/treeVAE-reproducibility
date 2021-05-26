import os
import numpy as np
import random
import pandas as pd
import argparse
import torch
from ete3 import Tree

# Data
from anndata import AnnData
import scanpy as sc
from external.dataset.tree import TreeDataset, GeneExpressionDataset
from external.dataset.anndataset import AnnDatasetFromAnnData

# Models
from external.models.treevae import TreeVAE
from external.inference.tree_inference import TreeTrainer
from external.models.vae import VAE
from external.inference.inference import UnsupervisedTrainer

# Utils
from external.utils.metrics import report_results, knn_purity_tree
from external.utils.baselines import scvi_baseline_z, construct_latent

def construct_distance_matrix(tree):
    n_nodes = len([n for n in tree.traverse()])
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i, n in enumerate(tree.traverse('levelorder')):
        for j, m in enumerate(tree.traverse('levelorder')):
            if n == m:
                continue
            dist = n.get_distance(m)
            distance_matrix[i, j] = dist
    return distance_matrix

class Metastasis():

    def __init__(self,
                tree,
                data_path,
                fixed_branch_length,
                t_normalization,
                use_cuda,
                n_epochs,
                lr,
                lambda_,
                latent,
                n_hidden,
                n_epochs_kl_warmup
        ):
        """
        :param tree: ete3: Cassiopeia prior tree
        :param branch_length: float: branch length of the simulation (Gaussian Random Walk)
        :param use_cuda: bool: True to train on GPUs
        :param n_epochs: int: number of epochs
        :param lr: float: learning rate
        :param lambda_: float: regularization parameter in the TreeVAE loss
        :param latent: int: dimension of latent space
        :return:
        """
        self.tree = tree
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.lr = lr
        self.lambda_ = lambda_
        self.latent = latent

        #data
        self.leaves_X = np.load(data_path)

        # Models
        self.treevae = None
        self.vae = None
        self.vae_full = None

        # neural network parameters
        self.n_hidden = n_hidden
        self.n_epochs_kl_warmup = n_epochs_kl_warmup

        # Prune tree
        self.leaves = [n.name for n in self.tree.traverse('levelorder')]
        barcodes = pd.read_csv("/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/ALL_Samples/GRCh38/barcodes.tsv", names=['Barcodes'])
        keep_leaves = []
        keep_leaves_idx = []
        for barcode in self.leaves:
            foo = barcodes.index[barcodes['Barcodes'] == barcode].tolist()
            if foo == []:
                continue
            else:
                keep_leaves_idx.append(foo[0])
                keep_leaves.append(barcode)
        self.tree.prune(keep_leaves) 

        # branch length
        if fixed_branch_length:
            self.branch_length = 1.0
        else:
            self.branch_length = {}
            for i, n in enumerate(self.tree.traverse('levelorder')):
                if not n.is_leaf():
                    n.name = str(i)
                n.add_features(index=i)
                if n.is_root():
                    self.branch_length[n.name] = 0.0
                else:
                    self.branch_length[n.name] = n.dist / t_normalization
            self.branch_length['prior_root'] = 1.0

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
        self.tree_dataset = TreeDataset(scvi_dataset, tree=self.tree, filtering=False)

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

    def fit_scvi_full(self):
        """
        Fitting cascVI to the gene expression data
        :return:
        """

        # treeVAE
        self.vae_full = TreeVAE(self.tree_dataset.nb_genes,
                      tree=self.tree_dataset.tree,
                      n_latent=self.latent,
                      n_hidden=self.n_hidden,
                      n_layers=1,
                      reconstruction_loss='poisson',
                      prior_t=self.branch_length,
                      ldvae=False,
                      use_MP=False
                      )

        freq = 100
        trainer_full = TreeTrainer(
            model=self.vae_full,
            gene_dataset=self.tree_dataset,
            lambda_=self.lambda_,
            train_size=1.0,
            test_size=0,
            use_cuda=self.use_cuda,
            frequency=freq,
            n_epochs_kl_warmup=self.n_epochs_kl_warmup
        )

        # training the VAE
        trainer_full.train(n_epochs=self.n_epochs,
                      lr=self.lr
                      )

        self.full_posterior = trainer_full.create_posterior(trainer_full.model, self.tree_dataset,
                                                    trainer_full.clades, indices=np.arange(len(self.tree_dataset))
                                                    )


    def evaluation(self):
        """
        :return:
        """

        # empirical library size
        empirical_l = np.mean(np.sum(self.leaves_X, axis=1))

        elbo_vae, elbo_treevae, elbo_vae_full = [], [], []
        # 0. ELBO
        for i in range(50):
            with torch.no_grad():
                elbo_vae.append(self.posterior.elbo())
                elbo_treevae.append(self.tree_posterior.compute_elbo().item())
                elbo_vae_full.append(self.full_posterior.compute_elbo().item())

        elbo_metrics = [np.mean(elbo_vae), np.mean(elbo_treevae), np.mean(elbo_vae_full)]

        # 1. ========== CascVI imputations ===================
        imputed_z = {}
        for n in self.tree.traverse('levelorder'):
            if not n.is_leaf():
                _, z = self.tree_posterior.imputation_internal(n,
                                                            give_mean=False,
                                                            library_size=empirical_l
                                                            )
                
                imputed_z[n.name] = z

        # 2. Baseline 2: Decoded averaged latent space VAE
        _, imputed_scvi_z = scvi_baseline_z(tree=self.tree,
                                            model=self.vae,
                                            posterior=self.posterior,
                                            weighted=False,
                                            n_samples_z=1,
                                            library_size=empirical_l,
                                            use_cuda=self.use_cuda
                                            )

        # 4.. Cross Entropy metrics
        vae_latent = self.posterior.get_latent()[0]
        treevae_latent = self.tree_posterior.get_latent()
        full_latent = self.full_posterior.get_latent()
        #--> cross entropy | VAE
        self.treevae.initialize_visit()
        self.treevae.initialize_messages(vae_latent, self.tree_dataset.barcodes, self.latent)
        self.treevae.perform_message_passing((self.treevae.tree & '0'), self.latent, False)
        ce_vae = self.treevae.aggregate_messages_into_leaves_likelihood(self.latent, add_prior=True).item()
        #--> cross entropy | TreeVAE
        self.treevae.initialize_visit()
        self.treevae.initialize_messages(treevae_latent, self.tree_dataset.barcodes, self.latent)
        self.treevae.perform_message_passing((self.treevae.tree & '0'), self.latent, False)
        ce_treevae = self.treevae.aggregate_messages_into_leaves_likelihood(self.latent, add_prior=True).item()
        #--> cross entropy | VAE full-batch
        self.vae_full.initialize_visit()
        self.vae_full.initialize_messages(full_latent, self.tree_dataset.barcodes, self.latent)
        self.vae_full.perform_message_passing((self.vae_full.tree & '0'), self.latent, False)
        ce_full = self.vae_full.aggregate_messages_into_leaves_likelihood(self.latent, add_prior=True).item()       

        ce_metrics = [ce_vae, ce_treevae, ce_full]

        # 5. k-NN purity 
        full_treevae_latent = construct_latent(self.tree, treevae_latent, imputed_z)
        full_vae_latent = construct_latent(self.tree, vae_latent, imputed_scvi_z)

        max_neighbors = 40
        neighbors = list(range(5, max_neighbors, 5))
        
        # groundtruth k-nn graph
        distance_matrix = construct_distance_matrix(self.tree)

        # Full latent space
        data = {'scVI': full_vae_latent, 'cascVI': full_treevae_latent}
        scores = knn_purity_tree(distance_matrix, max_neighbors, data)

        purity_metrics = pd.DataFrame(data={'K':neighbors, 'scVI': scores['scVI'],
                                         'cascVI': scores['cascVI']}
                                         )
        
        return elbo_metrics, ce_metrics, purity_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=10,
                        help='# of runs')
    parser.add_argument('--tree_name', type=str, default='/home/eecs/khalil.ouardini/cas_scvi_topologies/lg7_tree_hybrid_priors.alleleThresh.processed.ultrametric.annotated.tree',
                        help='Path of the Cassiopeia prior tree')
    parser.add_argument('--data_path', type=str, default='/home/eecs/khalil.ouardini/Cassiopeia_Transcriptome/scvi/metastasis_data/Metastasis_lg7_100g.npy',
                    help='Path to the pickled gene expression metastasis data matrix')
    parser.add_argument('--fixed_branch_length', type=bool, default=False,
                        help='whether to use a fixed branch length in the simulations (Gaussian Random Walk)')
    parser.add_argument('--t_normalization', default=14, type=float,
                        help='normalizing constant for the branch length')    
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Whether to use GPUs')
    parser.add_argument('--n_epochs', type=int, default=700,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lambda_', type=float, default=1.0,
                        help='Regularization parameter in the the treeVAE')
    parser.add_argument('--latent', type=int, default=10,
                        help='dimension of latent space')
    parser.add_argument('--n_hidden', type=int, default=64,
                        help='Number hidden units in the VAE')
    parser.add_argument('--seed', type=int, default=42,
                        help='random_seed')
    parser.add_argument('--n_epochs_kl_warmup', type=int, default=150,
                        help='Number of warm up epochs before introducing KL regularization in the VAE')

    #Parameters
    args = parser.parse_args()

    # Import the tree & data
    tree = Tree(args.tree_name, 1)
    data_path = args.data_path

    # Args
    n_runs = args.n_runs
    fixed_branch_length = args.fixed_branch_length
    t_normalization = args.t_normalization
    use_cuda = args.use_cuda
    n_epochs = args.n_epochs
    lr = args.lr
    seed = args.seed
    n_hidden = args.n_hidden
    latent = args.latent
    lambda_ = args.lambda_
    n_epochs_kl_warmup = args.n_epochs_kl_warmup

    # Set random seed
    torch.manual_seed(seed), random.seed(seed), np.random.seed(seed)

    print("==== Loading tree ====")
    print("Tree in path {} \n".format(args.tree_name))
    print("Gene expression data in path {}".format(args.data_path))

    metrics = {'ELBO': [], 'Cross_Entropy': []}
    purity = pd.DataFrame()

    for i in range(n_runs):
        print("======== Metastasis run: {}  ======== \n".format(i))
        exp = Metastasis(
                        tree=tree,
                        data_path=data_path,
                        fixed_branch_length=fixed_branch_length,
                        t_normalization=t_normalization,
                        use_cuda=use_cuda, 
                        n_epochs=n_epochs, 
                        lr=lr, 
                        lambda_=lambda_, 
                        latent=latent, 
                        n_hidden=n_hidden,
                        n_epochs_kl_warmup=n_epochs_kl_warmup
                        )

        print("II: Fitting models \n")

        print("==== Fitting VAE ==== \n")
        exp.fit_scvi(), print('\n')

        print("==== Fitting treeVAE ==== \n")
        exp.fit_cascvi(), print('\n')

        print("==== Fitting scVI-full ====")
        exp.fit_scvi_full(), print('\n')

        print("III: Evalutation")
        elbo_metrics, ce_metrics, purity_metrics = exp.evaluation()

        # update secondary metrics
        metrics['Cross_Entropy'].append(ce_metrics)
        metrics['ELBO'].append(elbo_metrics)
        purity = purity.append(purity_metrics)

    results_dir = 'results/metastasis'
    data_name = data_path.split('/')[-1]
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    results_dir = os.path.join(results_dir, data_name)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    results_dir = os.path.join(results_dir, 't_norm_'+str(t_normalization))
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)    
    
    purity.to_csv(os.path.join(results_dir, 'purity_full'))

    report_results(metrics=metrics,
                save_path=results_dir,
                columns2=None,
                metastasis=True
                )












