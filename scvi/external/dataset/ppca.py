import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import ete3
import matplotlib
matplotlib.use('TkAgg')
sys.path.append('.')
from scipy.stats import multivariate_normal
import pdb
from utils.precision_matrix import precision_matrix, marginalize_covariance, marginalize_internal

class PPCA:
    def __init__(self, tree, dim, latent, vis, only, branch_length, sigma_scale):
        """
        :param tree: path to a ete3 tree .txt file | or loaded ete3 tree object
        :param dim: number of dimensions in observations space
        :param latent: desired latent space dimension
        :param vis: bool: True if visualizations are displayed
        :param only: bool: True to simulate data at the leaves only
        :param branch_length: (int or list) constant branch length or list of branch lengths of each node
        """

        if type(tree) != str and type(tree) != ete3.coretype.tree.TreeNode:
            raise ValueError("The tree param has to be a path (string) to a ete3 .txt file or a loaded ete3 tree object")

        if type(branch_length) != float and type(branch_length) != dict:
            raise ValueError("The branch length param has to be a constant float, or a dictionary")

        self.tree = tree
        # Level order indexing
        for i, n in enumerate(self.tree.traverse('levelorder')):
            n.add_features(index=i)
            if not n.is_leaf():
                n.name = str(i)

        self.latent = latent
        self.dim = dim
        self.n_nodes = len([n for n in self.tree.traverse()])
        self.n_leaves = len(self.tree.get_leaves())

        # Latent distribution
        self.branch_length = branch_length
        self.covariance_z = None
        self.leaves_covariance_z = None
        self.z = None

        # Observation model
        self.X = None
        self.covariance_x = None
        self.leaves_covariance_x = None
        self.mu = None

        # Parameters
        self.W = None
        self.sigma_scale = sigma_scale
        self.beta = None
        
        # Misc
        self.vis = vis
        self.leaves_only = only
        
        leaves_covariance, full_covariance = precision_matrix(self.tree, self.latent, self.branch_length)
        if self.leaves_only:
            self.covariance_z = leaves_covariance
        else:
            self.covariance_z = full_covariance
            self.leaves_covariance_z = leaves_covariance

    def get_evidence_leaves_levelorder(self, X, dim):
        evidence_leaves_levelorder = np.zeros((self.n_leaves * dim,))
        levelorder_idx = 0
        for node in self.tree.traverse("levelorder"):
            if node.is_leaf():
                evidence_leaves_levelorder[levelorder_idx * dim: (levelorder_idx + 1) * dim] = X[node.index]
                levelorder_idx += 1
        return evidence_leaves_levelorder

    def simulate_latent(self):
        # Define epsilon.
        epsilon = 0.00001

        # Add small perturbation for numerical stability.
        K = self.covariance_z + epsilon * np.identity(self.n_nodes * self.latent)

        #  Cholesky decomposition.
        L = np.linalg.cholesky(K)

        # sanity check
        assert (np.dot(L, np.transpose(L)).all() == K.all())

        # Number of samples.
        u = np.random.normal(loc=0, scale=1, size=self.latent * self.n_nodes).reshape(self.latent, self.n_nodes)

        #t = time.time()
        # scale samples with Cholesky factor
        self.z = L @ u.flatten()
        self.z = self.z.reshape((-1, self.latent))

        #print("Sampling with Cholesky took {} seconds".format(time.time() - t))

        if self.vis and self.latent == 2:
            sns.jointplot(x=self.z[:, 0],
                          y=self.z[:, 1],
                          kind="kde",
                          space=0,
                          color='green');
            plt.title('Latent Space')
            plt.xlabel('x axis')
            plt.ylabel('y axis')
            plt.show()

    def simulate_normal(self):
        self.W = np.random.normal(loc=0, scale=1, size=(self.dim, self.latent))

        self.mu = self.z @ self.W.T

        # Sampling 
        self.sigma = self.sigma_scale * np.identity(self.dim)
        self.X = np.zeros(shape=(self.n_nodes, self.dim))
        #for i in range(self.n_nodes):
        #    self.X[i] = np.random.multivariate_normal(mean=self.mu_z[i],
        #                                        cov=self.sigma)
        for n in self.tree.traverse('levelorder'):
            self.X[n.index] = np.random.multivariate_normal(mean=self.mu[n.index],
                                                cov=self.sigma)

    def likelihood_latent(self, leaves_only=True):
        if leaves_only:
            return multivariate_normal.logpdf(self.get_evidence_leaves_levelorder(X=self.z),
                                            np.array([0] * self.n_leaves * self.latent),
                                            self.leaves_covariance_z
                                            )
        else:
            return multivariate_normal.logpdf(self.z.flatten(),
                                            np.array([0] * self.n_nodes * self.latent),
                                            self.covariance_z
                                            )

    def query_marginalized_covariance_z(self, idx1, idx2=None):
        if idx2:
            start1, end1 = idx1 * self.latent, idx1 * self.latent + self.latent
            start2, end2 = idx2 * self.latent, idx2 * self.latent + self.latent
            return self.covariance_z[start1: end1, start2: end2]
        else:
            start = idx1 * self.latent
            end = idx1 * self.latent + self.latent
            return self.covariance_z[start: end, start: end]

    def likelihood_obs(self, leaves_only=True):
        likelihood_x = 0
        for n in self.tree.traverse('levelorder'):
            if leaves_only and not n.is_leaf():
                continue
            sigma_n = self.query_marginalized_covariance_z(idx1=n.index)
            cov = self.W @ sigma_n @ self.W.T + self.sigma
            assert np.all(np.linalg.eigvals(cov) > 0)
            evidence = self.X[n.index]
            likelihood_x += multivariate_normal.logpdf(evidence,
                                            np.array([0] * self.dim),
                                            cov
                                            )
        return likelihood_x

    def compute_leaves_covariance(self):
        eps = 1e-6
        def get_near_psd(A):
            C = (A + A.T) / 2
            eigval, eigvec = np.linalg.eigh(C)
            eigval[eigval < 0] = 0
            return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

        def is_pos_semidef(A):
            eigvals, _ = np.linalg.eigh(A)
            return np.all(eigvals >= 0)

        leaves = [n for n in self.tree.traverse('levelorder') if n.is_leaf()]
        indices = list(range(self.n_leaves))
        self.leaves_covariance_x = np.zeros(shape=(self.n_leaves * self.dim, self.n_leaves * self.dim))
        for i1, n1 in zip(indices, leaves):
            for i2, n2 in zip(indices, leaves):
                sigma_n1_n2 = self.query_marginalized_covariance_z(n1.index, n2.index)
                sigma_n2_n1 = self.query_marginalized_covariance_z(n2.index, n1.index)
                start1, end1 = i1 * self.dim, i1 * self.dim + self.dim
                start2, end2 = i2 * self.dim, i2 * self.dim + self.dim
                self.leaves_covariance_x[start1:end1, start2:end2] = self.W @ sigma_n1_n2 @ self.W.T + self.sigma
                self.leaves_covariance_x[start2:end2, start1:end1] = self.W @ sigma_n2_n1 @ self.W.T + self.sigma
        
        #self.leaves_covariance_x = get_near_psd(self.leaves_covariance_x)
        self.leaves_covariance_x += eps * np.identity(self.n_leaves * self.dim)
        if not is_pos_semidef(self.leaves_covariance_x):
            raise ValueError("The empirical covariance matrix of the leaves is not semi-definite positive")
        
        self.leaves_covariance_inv = np.linalg.inv(self.leaves_covariance_x)

    def compute_posterior(self):
        posterior_mean, posterior_cov = {}, {}
        leaves = [n for n in self.tree.traverse('levelorder') if n.is_leaf()]
        indices = list(range(self.n_leaves))
        # Evidence x_L
        evidence_leaves = self.get_evidence_leaves_levelorder(X=self.X, dim=self.dim)
        # We want to compute the posterior of n1 condtioned on all the leaves
        for i1, n1 in zip(indices, leaves):
            # Correlations between z_n and x_L
            corr_x_z = np.zeros(shape=(self.latent, self.n_leaves * self.dim))
            for i2, n2 in zip(indices, leaves):
                sigma_n1_n2 = self.query_marginalized_covariance_z(idx1=n1.index, idx2=n2.index)
                corr_x_z[:, i2 * self.dim : i2 * self.dim + self.dim] = sigma_n1_n2 @ self.W.T
            # compute posterior mean and covariance
            sigma_n1 = self.query_marginalized_covariance_z(idx1=n1.index, idx2=None)
            posterior_mean[n1.index] = corr_x_z @ self.leaves_covariance_inv @ evidence_leaves
            posterior_cov[n1.index] = sigma_n1 - corr_x_z @ self.leaves_covariance_inv @ corr_x_z.T
        return posterior_mean, posterior_cov

    def compute_correlated_posterior(self):
        leaves = [n for n in self.tree.traverse('levelorder') if n.is_leaf()]
        indices = list(range(self.n_leaves))
        # Evidence x_L
        evidence_leaves = self.get_evidence_leaves_levelorder(X=self.X, dim=self.dim)
        # We want to compute the posterior of n1 condtioned on all the leaves
        # Correlations between z_1 ... z_L and x_1, ..., x_L
        corr_x_z = np.zeros(shape=(self.n_leaves * self.latent, self.n_leaves * self.dim))
        for i1, n1 in zip(indices, leaves):
            for i2, n2 in zip(indices, leaves):
                sigma_n1_n2 = self.query_marginalized_covariance_z(idx1=n1.index, idx2=n2.index)
                corr_x_z[i1 * self.latent : i1 * self.latent + self.latent, i2 * self.dim : i2 * self.dim + self.dim] = sigma_n1_n2 @ self.W.T
        posterior_mean = corr_x_z @ self.leaves_covariance_inv @ evidence_leaves
        posterior_cov = self.leaves_covariance_z - corr_x_z @ self.leaves_covariance_inv @ corr_x_z.T


    def compute_posterior_predictive(self):
        predictive_mean, predictive_cov = {}, {}
        leaves = [n for n in self.tree.traverse('levelorder') if n.is_leaf()]
        internal_nodes = [n for n in self.tree.traverse('levelorder') if not n.is_leaf()]
        indices = list(range(self.n_leaves))
        # Evidence x_L
        evidence_leaves = self.get_evidence_leaves_levelorder(X=self.X, dim=self.dim)
        # We want to compute the posterior of n1 condtioned on all the leaves
        for n1 in internal_nodes:
            # Correlations between z_n and x_L
            corr_x_x = np.zeros(shape=(self.dim, self.n_leaves * self.dim))
            for i2, n2 in zip(indices, leaves):
                sigma_n1_n2 = self.query_marginalized_covariance_z(idx1=n1.index, idx2=n2.index)
                corr_x_x[:, i2 * self.dim : i2 * self.dim + self.dim] = self.W @ sigma_n1_n2 @ self.W.T + self.sigma
            # compute posterior mean and covariance
            sigma_n1 = self.query_marginalized_covariance_z(idx1=n1.index, idx2=None)
            predictive_mean[n1.name] = corr_x_x @ self.leaves_covariance_inv @ evidence_leaves
            predictive_cov[n1.name] = self.W @ sigma_n1 @ self.W.T - corr_x_x @ self.leaves_covariance_inv @ corr_x_x.T
        return predictive_mean, predictive_cov

        




        








        

        
        

        



