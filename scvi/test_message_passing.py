from unittest import TestCase
from anndata import AnnData
import numpy as np
from external.dataset.tree import TreeDataset
from external.dataset.anndataset import AnnDatasetFromAnnData
from external.models.treevae import TreeVAE
from external.models.gaussian_treevae import GaussianTreeVAE
from external.utils.precision_matrix import precision_matrix, marginalize_covariance
import copy
from ete3 import Tree
import torch
from scipy.stats import multivariate_normal
import unittest
import random

class TestMessagePassing(TestCase):
    def test_mp_inference(self):
        is_random = True
        is_gaussian = False
        type_test = 'binary'

        ######## Import Tree
        if type_test == 'binary':
            tree = Tree()
            tree.populate(np.random.randint(20, 60))

        elif type_test == 'general':
            tree_name = "/home/eecs/khalil.ouardini/cas_scvi_topologies/Cassiopeia_trees/tree_test.txt"
            #tree_name = "/home/eecs/khalil.ouardini/cas_scvi_topologies/newick_objects/500cells/high_fitness/topology1.nwk"
            with open(tree_name, "r") as myfile:
                 tree_string = myfile.readlines()
            tree = Tree(tree_string[0], 1)
        
        ######### Indexing nodes
        ######### dimension of latent space
        d = 3
        N = 0
        for idx, node in enumerate(tree.traverse("levelorder")):
            N += 1
            node.name = str(idx)
            # set node index
            node.add_features(index=idx)
        
        #print("Tree:", tree)

        leaves = [node for node in tree.traverse('levelorder') if node.is_leaf()]
        internal_nodes = [node for node in tree.traverse('levelorder') if not node.is_leaf()]
        leaves_index = [n.index for n in leaves]

        ######### branch length
        branch_length = {}
        if is_random:
            for node in tree.traverse('levelorder'):
                if node.is_root():
                    branch_length['0'] = 0.0
                else:
                    branch_length[node.name] = np.random.rand()
            branch_length['prior_root'] = 1.0
        else:
            branch_length = 1.0

        ######### create toy Gene Expression dataset
        x = np.random.randint(1, 100, (len(leaves), 10))
        adata = AnnData(x)
        gene_dataset = AnnDatasetFromAnnData(adata)
        barcodes = [l.name for l in leaves]
        gene_dataset.initialize_cell_attribute('barcodes', barcodes)

        ######### create tree dataset
        tree_bis = copy.deepcopy(tree)
        cas_dataset = TreeDataset(gene_dataset, tree=tree_bis)
        
        if is_gaussian:
            vae = GaussianTreeVAE(cas_dataset.nb_genes,
                      tree=cas_dataset.tree,
                      n_latent=d,
                      prior_t=branch_length,
                      use_MP=True
                     )
        else:
            vae = TreeVAE(cas_dataset.nb_genes,
                        tree=cas_dataset.tree,
                        n_latent=d,
                        prior_t=branch_length,
                        use_MP=True
                        )

        ######### Gaussian evidence
        evidence0 = np.random.randn(len(leaves), d)

        #####################
        print("")
        print("|>>>>>>> Test 1: Message Passing Likelihood <<<<<<<<|")
        vae.initialize_visit()
        vae.initialize_messages(
            evidence0,
            barcodes,
            d
        )

        vae.perform_message_passing((vae.tree & vae.root), d, False)
        mp_lik = vae.aggregate_messages_into_leaves_likelihood(
            d,
            add_prior=True
        )
        print("Test 1: Message passing output O(nd): ", mp_lik.item())

        # likelihood via  covariance matrix Marginalization + inversion
        leaves_covariance, full_covariance = precision_matrix(tree=tree,
                                                              d=d,
                                                              branch_length=branch_length)

        leaves_mean = np.array([0] * len(leaves) * d)
        
        ######## to query evidence in levelorder
        def get_evidence_leaves_levelorder():
            num_leaves = len([n for n in vae.tree if n.is_leaf()])
            evidence_leaves_levelorder = np.zeros((num_leaves * d,))
            levelorder_idx = 0
            for node in vae.tree.traverse("levelorder"):
                if node.is_leaf():
                    evidence_leaves_levelorder[levelorder_idx * d: (levelorder_idx + 1) * d] = node.mu
                    levelorder_idx += 1
            return evidence_leaves_levelorder

        pdf_likelihood = multivariate_normal.logpdf(get_evidence_leaves_levelorder(),
                                                    leaves_mean,
                                                    leaves_covariance)

        print("Test 1: Gaussian marginalization + inversion output O(n^3d^3): ", pdf_likelihood)
        self.assertTrue((np.abs(mp_lik.item() - pdf_likelihood) < 1e-7))

        #####################
        print("")
        print("|>>>>>>> Test 2: Message Passing Posterior Predictive Density at internal nodes  <<<<<<<<| ")
        do_internal = True
        if do_internal:
            # Sample internal nodes randomly
            internal_nodes = [node for node in tree.traverse('levelorder') if not node.is_leaf()]
            if len(internal_nodes) < 20:
                internal_nodes =  int(len(internal_nodes) / 2)
            else:
                internal_nodes = random.sample(internal_nodes, 20)
            print(">>>> Sampled internal nodes:", [n.name for n in internal_nodes])
            print("")
            for n in internal_nodes:

                query_node = n.name
                print("Query node:", query_node)
                        
                # evidence
                evidence = np.random.randn(len(leaves), d)

                #Gaussian conditioning formula
                to_delete_idx_ii = [i for i in list(range(N)) if (i not in leaves_index)]
                to_delete_idx_ll = [i for i in list(range(N)) if (i in leaves_index)]

                # partition index
                I = [idx for idx in list(range(N)) if idx not in to_delete_idx_ii]
                L = [idx for idx in list(range(N)) if idx not in to_delete_idx_ll]

                # covariance marginalization
                cov_ii = marginalize_covariance(full_covariance, [to_delete_idx_ii], d)
                cov_ll = marginalize_covariance(full_covariance, [to_delete_idx_ll], d)
                cov_il = marginalize_covariance(full_covariance, [to_delete_idx_ii, to_delete_idx_ll], d)
                cov_li = np.copy(cov_il.T)

                internal_post_mean_transform = np.dot(cov_li, np.linalg.inv(cov_ii))
                internal_post_covar = cov_ll - np.dot(np.dot(cov_li, np.linalg.inv(cov_ii)), cov_il)

                # Message Passing
                vae.posterior_predictive_density(vae.tree & query_node, evidence)

                query_idx = (tree & query_node).index
                query_idx = L.index(query_idx)

                ev = get_evidence_leaves_levelorder()
                #post_mean = np.dot(internal_post_mean_transform, np.hstack(evidence_leaves))[query_idx * d: query_idx * d + 2]
                post_mean = np.dot(internal_post_mean_transform, np.hstack(ev))[
                            query_idx * d: query_idx * d + d]
                print("Test2: Gaussian conditioning formula O(n^3d^3):",
                      post_mean,
                      internal_post_covar[query_idx * d, query_idx * d])

                print("Test2: Message passing output O(nd): ",
                      (vae.tree & query_node).mu.numpy(), (vae.tree & query_node).nu
                      )

                self.assertTrue((np.abs((vae.tree & query_node).nu - internal_post_covar[query_idx * d, query_idx * d])) < 1e-7)
                for i in range(post_mean.shape[0]):
                    self.assertTrue((np.abs((vae.tree & query_node).mu.numpy()[i] - post_mean[i])) < 1e-7)

                print("")

if __name__ == '__main__':
    torch.set_printoptions(precision=10)
    unittest.main()



#tree = Tree("(A:1,(B:1,C:0.5,D:1,(E:1)):0.5, F:1);")
#branch_length1 = {'0': 0.0, '1':1/2, '2':1/3, '3':1/5, '4':1/4, 'prior_root': 1.0}
#branch_length1 = {'0': 1/3, '1':1/2, '2':1/8, '3':1/5, '4':1/4, '5':1/6, '6':1/7, '7':0.0, '8':1/9, 'prior_root': 1.0}
#branch_length2 = {'0': 0.0, '1':1/2, '2':1/3, '3':1/5, '4':1/4, '5':1/6, '6':1/7, '7':1/8, '8':1/9, 'prior_root': 1.0}
#branch_length1 = {'0': 1.0, '1':1.0, '2':1.0, '3':1.0, '4':1.0, 'prior_root': 1.0}
#branch_length2 = {'0': 0.0, '1':1.0, '2':1.0, '3':1.0, '4':1.0, 'prior_root': 1.0}
#branch_length = 1.0
