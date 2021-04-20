from unittest import TestCase
from anndata import AnnData
import numpy as np
from scvi.dataset.tree import TreeDataset
from scvi.dataset.anndataset import AnnDatasetFromAnnData
from scvi.models.treevae import TreeVAE
from scvi.utils.precision_matrix import precision_matrix, marginalize_covariance
import copy
from ete3 import Tree
import torch
from scipy.stats import multivariate_normal
import unittest
import random
import pdb

class TestMessagePassing(TestCase):
    def test_mp_inference(self):
        # Import Tree
        type_test = 'general'

        if type_test == 'binary':
            tree = Tree()
            tree.populate(np.random.randint(2, 60))

        elif type_test == 'general':
            tree_name = "../data/tree_test.txt"
            #tree_name = "../data/lg7_tree_hybrid_priors.alleleThresh.processed.txt"
            #tree_name = "../data/toy.nw"
            #tree_name = "../data/richard_tree.nw"
            with open(tree_name, "r") as myfile:
                 tree_string = myfile.readlines()
            tree = Tree(tree_string[0], 1)

        #print("Tree:", tree)

        # Indexing nodes
        # dimension of latent space
        d = 3
        N = 0
        for idx, node in enumerate(tree.traverse("levelorder")):
            N += 1
            node.name = str(idx)
            # set node index
            node.add_features(index=idx)
        leaves = [node for node in tree.traverse('levelorder') if node.is_leaf()]
        leaves_index = [n.index for n in leaves]

        # branch length
        is_random = False
        if is_random:
            eps = 1e-3
            branch_length = {}
            for node in tree.traverse('levelorder'):
                branch_length[node.name] = np.random.rand() + eps
            branch_length['prior_root'] = 1.0
        else:
            branch_length = 0.1

        #create toy Gene Expression dataset
        x = np.random.randint(1, 100, (len(leaves), 10))
        adata = AnnData(x)
        gene_dataset = AnnDatasetFromAnnData(adata)
        barcodes = [l.name for l in leaves]
        gene_dataset.initialize_cell_attribute('barcodes', barcodes)

        #create tree dataset
        tree_bis = copy.deepcopy(tree)
        cas_dataset = TreeDataset(gene_dataset, tree=tree_bis)

        use_batches = False

        vae = TreeVAE(cas_dataset.nb_genes,
                      tree=cas_dataset.tree,
                      n_batch=cas_dataset.n_batches * use_batches,
                      n_latent=d,
                      prior_t=branch_length
                     )


        # Gaussian evidence
        evidence = np.random.randn(len(leaves), d)
        #evidence = np.ones((len(leaves), d))

        #####################
        print("")
        print("|>>>>>>> Test 1: Message Passing Likelihood <<<<<<<<|")
        vae.initialize_visit()
        vae.initialize_messages(
            evidence,
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
            internal_nodes = [node for node in tree.traverse() if not node.is_leaf()]
            if len(internal_nodes) < 10:
                internal_nodes = random.sample(internal_nodes, int(len(internal_nodes) / 2))
            else:
                internal_nodes = random.sample(internal_nodes, 10)
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
                vae.initialize_visit()
                vae.initialize_messages(
                    evidence,
                    cas_dataset.barcodes,
                    d
                )

                vae.perform_message_passing((vae.tree & query_node), d, True)
                query_idx = (tree & query_node).index
                query_idx = L.index(query_idx)

                #post_mean = np.dot(internal_post_mean_transform, np.hstack(evidence_leaves))[query_idx * d: query_idx * d + 2]
                post_mean = np.dot(internal_post_mean_transform, np.hstack(get_evidence_leaves_levelorder()))[
                            query_idx * d: query_idx * d + d]
                print("Test2: Gaussian conditioning formula O(n^3d^3): ",
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




