import numpy as np
from numpy.random import poisson, binomial
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import ete3
import matplotlib
matplotlib.use('WebAgg')
sys.path.append('.')
from ..utils.precision_matrix import precision_matrix


class Poisson_GLM:
    def __init__(self, tree, dim, latent, vis, only, branch_length):
        """
        :param tree: path to a ete3 tree .txt file | or loaded ete3 tree object
        :param dim: desired number of genes
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
        self.latent = latent
        self.dim = dim
        self.covariance = None
        self.leaves_covariance = None
        self.n_nodes = None
        self.z = None
        self.X = None
        self.mu = None
        self.W = None
        self.beta = None
        self.vis = vis
        self.leaves_only = only
        self.branch_length = branch_length

        leaves_covariance, full_covariance = precision_matrix(self.tree, self.latent, self.branch_length)
        if self.leaves_only:
            self.covariance = leaves_covariance
        else:
            self.covariance = full_covariance
            self.leaves_covariance = leaves_covariance
        self.n_nodes = int(self.covariance.shape[0] / self.latent)

    def simulate_latent(self):
        # Define epsilon.
        epsilon = 0.00001

        # Add small perturbation for numerical stability.
        K = self.covariance + epsilon * np.identity(self.n_nodes * self.latent)

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

    def simulate_ge(self):
        # dimension of initial space (i.e number of genes)
        self.W = np.random.normal(loc=0, scale=0.5, size=(self.dim, self.latent))
        self.beta = np.random.normal(loc=0, scale=0.5, size=self.dim)

        #self.W = np.random.normal(loc=0, scale=1.0, size=(self.latent, self.dim))
        #self.beta = np.random.normal(loc=0, scale=1.0, size=self.dim)

        self.mu = np.clip(a=np.exp(self.z @ self.W.T + self.beta),
                                 a_min=0,
                                 a_max=1e5
                          )

        self.X = np.asarray(poisson(self.mu), dtype=np.float64)

        if self.vis:
            ## Poissson distribution
            fig, axes = plt.subplots(1, 1,
                                     figsize=(14, 8),
                                     sharey=True,
                                     )

            bins = np.arange(0, 30, 5)

            cm = plt.cm.get_cmap('RdYlBu_r')

            n, binss, patches = axes.hist(self.X,
                      bins=bins,
                      edgecolor='black',
                      )
            # set color of patches
            # scale values to interval [0,1]
            bin_centers = 0.5 * (binss[:-1] + binss[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)

            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))

            axes.set_title('Histogram of simulated gene expression data')
            plt.ylabel('Counts')
            plt.xlabel('Gene Expression value')
            plt.legend(['gene_' + str(i) for i in list(range(self.dim))], loc='best')
            plt.show()

    def generate(self, n_samples, leaves_idx):
        glm_samples = []
        for i in range(n_samples):
            self.simulate_latent()
            self.simulate_ge()
            glm_samples.append(np.take(self.X, leaves_idx, 0))
        glm_samples = np.array(glm_samples)
        return glm_samples

    def generate_new_ge(self, n_samples, leaves_idx):
        glm_samples = []
        for i in range(n_samples):
            self.simulate_ge()
            glm_samples.append(np.take(self.X, leaves_idx, 0))
        #glm_samples = np.array(glm_samples)
        return glm_samples

    def generate_ge(self, n_samples, leaves_idx):
        glm_samples = []
        for i in range(n_samples):
            X = np.asarray(poisson(self.mu), dtype=np.float64)
            glm_samples.append(np.take(X, leaves_idx, 0))
        #glm_samples = np.array(glm_samples)
        return glm_samples

    def gene_qc(self, threshold=0.1):
        to_delete = []
        N_cells = self.X.shape[0]
        for g_ in range(self.dim):
            expression_rate = len([self.X[n, g_] for n in range(N_cells) if self.X[n, g_] != 0]) / N_cells
            if expression_rate < threshold:
                to_delete.append(g_)

        self.X = np.delete(arr=self.X, obj=to_delete, axis=1)
        self.mu = np.delete(arr=self.mu, obj=to_delete, axis=1)

    def normalize(self):
        for i in range(self.X.shape[0]):
            empirical_l = np.sum(self.X[i])
            self.X[i] /= empirical_l

    def binomial_thinning(self, p=0.5):
        self.X = binomial(self.X.astype(int), p)
        #self.mu *= p
        # 
 
    def is_pos_def(self):
        return np.all(np.linalg.eigvals(self.covariance) > 0), np.all(np.linalg.eigvals(self.leaves_covariance) > 0)
























