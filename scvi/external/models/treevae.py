# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from ..models.modules import Encoder, Decoder, LinearDecoder
from ..models.utils import one_hot
import copy
import numpy as np
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from ..models.distributions import NegativeBinomial

torch.backends.cudnn.benchmark = True

from ete3 import Tree

# TreeVAE Model
class TreeVAE(nn.Module):
    r"""Model class for fitting a VAE to scRNA-seq data with a tree prior.

    This is corresponding VAE class for our TreeTrainer & implements the TreeVAE model. This model
    performs training in a very specific way, in an effort to respect the tree structure. Specifically,
    we'll perform training of this model by identifying 'clades' (or groups of leaves underneath a given
    internal node) from which the cell's RNA-seq data is assumed to be iid. This is currently done crudely
    by treating every internal node at depth 3 from the root as an appropriate location to create a clade,
    though this should be improved (see TODOs).

    After creating a clustered subtree (where now the leaves correspond to the nodes where clades were induced),
    our training procedure is relativley simple. For every one of these new leaves, split the cells in this clade
    into train/test/validation and in each iteration sample a single cell from the appropriate list and assign its
    RNAseq profile to the clade's root (i.e., the leaf in the clusterd subtree).

	"""

    def __init__(
        self,
        n_input: int,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        reconstruction_loss: str = "nb",
        latent_distribution: str = "normal",
        tree: Tree = None,
        use_clades: bool = False,
        prior_t: dict or float = None,
        ldvae: bool = False,
        use_MP: bool = True
    ):

        super().__init__()
        self.use_MP = use_MP
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.reconstruction_loss = reconstruction_loss
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))

        # z encoder goes from the n_input-dimensional data to an n_latent-d latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        self.ldvae = ldvae
        if not self.ldvae:
            self.decoder = Decoder(
                n_latent,
                n_input,
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
        else:
            # linearly decoded VAE
            if self.ldvae:
                self.decoder = LinearDecoder(
                    n_input=n_latent,
                    n_output=n_input,
                    use_batch_norm=False,
                    bias=True
                )

        def cut_tree(node, distance):
            return node.distance == distance

        self.use_clades = use_clades
        leaves = [n for n in tree.traverse('levelorder') if n.is_leaf()]
        
        if self.use_clades:
            # Cluster tree into clades: After a certain depth (here = 3), all children nodes are assumed iid and grouped into
            # "clades", for the training we sample one instance of each clade.
            collapsed_tree = Tree(tree.write(is_leaf_fn=lambda x: cut_tree(x, 3)))
            for l in collapsed_tree.get_leaves():
                l.cells = tree.search_nodes(name=l.name)[0].get_leaf_names()
            self.root = collapsed_tree.name
            inf_tree = Tree("prior_root;")
            inf_tree.add_child(collapsed_tree)
        else:
            # No collapsing for simulations (and small trees)
            for l in leaves:
                l.cells = tree.search_nodes(name=l.name)[0].get_leaf_names()
            self.root = tree.name
            # add prior node
            inf_tree = Tree("prior_root;")
            inf_tree.add_child(tree)

        self.prior_root = inf_tree.name
        self.tree = inf_tree

        # leaves barcodes
        self.barcodes = [l.name for l in leaves]

        # branch length for Message Passing
        if not prior_t:
            prior_t = 1.0
        if type(prior_t) == float:
            self.prior_t = {}
            for n in self.tree.traverse('levelorder'):
                if n.is_root():
                    self.prior_t[n.name] = 0.0
                else:
                    self.prior_t[n.name] = prior_t
            self.prior_t['prior_root'] = 1.0
        else:
            self.prior_t = prior_t

        # encoder's variance
        self.encoder_variance = []

    def initialize_messages(self, evidence, barcodes, d):

        if type(evidence) == np.ndarray:
            evidence = torch.from_numpy(evidence)

        dic_nu = {}
        dic_mu = {}
        dic_log_z = {}

        for i, j in enumerate(evidence):
            dic_nu[barcodes[i]] = 0
            dic_log_z[barcodes[i]] = 0
            dic_mu[barcodes[i]] = j

        dic_nu[self.prior_root] = 0
        dic_mu[self.prior_root] = torch.zeros(d)
        dic_log_z[self.prior_root] = 0

        for n in self.tree.traverse('levelorder'):
            if n.name in dic_nu:
                n.add_features(
                    nu=dic_nu[n.name],
                    mu=dic_mu[n.name].type(torch.DoubleTensor),
                    log_z=dic_log_z[n.name],
                )
            else:
                n.add_features(
                    nu=0,
                    mu=torch.from_numpy(np.zeros(d)).type(torch.DoubleTensor),
                    log_z=0,
                )

    def initialize_visit(self):
        for node in self.tree.traverse():
            node.add_features(visited=False)

    def perform_message_passing(self, root_node, d, include_prior):
        # flag the node as visited
        prior_node = self.tree & self.prior_root
        root_node.visited = True

        incoming_messages = []
        incident_nodes = [c for c in root_node.children]
        if not root_node.is_root():
            incident_nodes += [root_node.up]

        # get list of neighbors that are not visited yet
        for node in incident_nodes:
            if node.name == 'prior_root':
                node = prior_node
            if not node.visited and (
                node != prior_node or (node == prior_node and include_prior)
            ):
                self.perform_message_passing(node, d, include_prior)
                incoming_messages.append(node)

        n = len(incoming_messages)

        # collect and return
        if n == 0:
            # nothing to do. This happens on the leaves
            return None

        elif n == 1:
            k = incoming_messages[0]
            root_node.nu = k.nu + self.prior_t[k.name]
            root_node.mu = k.mu
            root_node.log_z = 0

        elif n == 2:
            # let us give them arbitrary names k and l (could be left and right)
            k = incoming_messages[0]
            l = incoming_messages[1]

            # let us compute the updates
            k_nu_inc = k.nu + self.prior_t[k.name]
            l_nu_inc = l.nu + self.prior_t[l.name]

            root_node.nu = 1. / (1. / k_nu_inc + 1. / l_nu_inc)
            root_node.mu = k.mu / k_nu_inc + l.mu / l_nu_inc
            root_node.mu *= root_node.nu

            lambda_ = k_nu_inc + l_nu_inc
            root_node.log_z = -0.5 * torch.sum((k.mu - l.mu) ** 2) / lambda_
            root_node.log_z -= d * 0.5 * np.log(2 * np.pi * lambda_)


        elif n > 2:
            # we will keep track of mean and variances of the children nodes in 2 lists
            children_nu = [0] * n
            children_mu = [0] * n

            for i in range(n):
                k = incoming_messages[i]
                # nu
                children_nu[i] = k.nu + self.prior_t[k.name]
                if children_nu[i] != 0:
                    root_node.nu += 1. / children_nu[i]
                    # mu
                    children_mu[i] = k.mu / children_nu[i]
                else:
                    children_mu[i] = k.mu
                root_node.mu += children_mu[i]

            if root_node.nu != 0:
                root_node.nu = 1. / root_node.nu
                root_node.mu *= root_node.nu

            def product_without(L, exclude):
                """
                L: list of elements
                exclude: list of the elements indices to exlucde

                returns: product of all desired array elements
                """
                prod = 1
                for idx, x in enumerate(L):
                    if idx in exclude:
                        continue
                    else:
                        prod *= x
                return prod


            # find t
            t = 0
            for excluded_idx in range(n):
                prod = product_without(children_nu, [excluded_idx])
                t += prod

            # normalizing constants
            Z_1 = -0.5 * (n - 1) * d * np.log(2 * np.pi)
            Z_2 = -0.5 * d * np.log(t)
            Z_3 = 0

            # nested for loop --> need to optimize with numba jit
            visited = set()
            for j in range(n):
                for h in range(n):
                    if h == j:
                        continue
                    if (h, j) in visited or(j, h) in visited:
                        continue
                    else:
                        prod_2 = product_without(children_nu, [j, h])
                        visited.add((j, h))
                        k = incoming_messages[h]
                        l = incoming_messages[j]
                        Z_3 += prod_2 * torch.sum((k.mu - l.mu) ** 2)
            if t != 0:
                Z_3 *= -0.5 / t
            root_node.log_z = Z_1 + Z_2 + Z_3

    def aggregate_messages_into_leaves_likelihood(self, d, add_prior):
        res = 0
        root_node = self.tree & self.root

        # agg Z messages
        for node in self.tree.traverse():
            res += node.log_z

        if add_prior:
            # add prior
            nu_inc = 1.0 + root_node.nu
            res += -0.5 * torch.sum(root_node.mu ** 2) / nu_inc - d * 0.5 * np.log(2 * np.pi * nu_inc)
        
        # prior likelihood of the leaves p(z_1, ..., z_L) \sim Normal(0, Sigma_L)
        return res

    def posterior_predictive_density(self, query_node, evidence=None):
        """
        :param query_node: (string) barcode of a query node
               evidence: (ndarray) observation values at the leaves (used as an initialization)
               reroot_prior_t: (dict) branch lengths dictionnary of the rerooted tree
        :return: the expectation and the variance for the posterior (distribution query_node | observations)
        """

        root_node = self.tree & self.root

        # computing branch length dictionnary of rerooted tree:
        def reroot_tree(branch_length, node):
            # path from node to root
            path = []
            internal_node = copy.copy(node)
            while internal_node.up:
                path.append(internal_node)
                internal_node = internal_node.up

            # New branch length dictionnary
            new_branch_length = copy.copy(branch_length)
            new_branch_length[node.name] = 0.0

            # correct branch lengths to make 'node' the root
            for i in range(len(path) - 1):
                current_node = path[i].name
                next_node = path[i+1].name
                new_branch_length[next_node] = branch_length[current_node]

            return new_branch_length
        
        # Update branch length
        old_prior_t = copy.copy(self.prior_t)
        self.prior_t = reroot_tree(self.prior_t, query_node)

        # Message Passing
        self.initialize_visit()
        if evidence is not None:
            self.initialize_messages(evidence,
                                     self.barcodes,
                                     self.n_latent
                                     )

        self.perform_message_passing(self.tree & query_node.name, len(root_node.mu), True)

        # Update branch length 
        self.prior_t = old_prior_t

        return (self.tree & query_node.name).mu, (self.tree & query_node.name).nu

    def sample_from_posterior_z(self, x, give_mean=False, give_cov=False, n_samples=5000):
        """Samples the tensor of latent values from the posterior

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        give_mean
            is True when we want the mean of the posterior  distribution rather than sampling (Default value = False)
        n_samples
            how many MC samples to average over for transformed mean (Default value = 5000)

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``

        """
        x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x)
        if not give_mean:
            samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
            z = torch.mean(samples, dim=0)
        else:
            z = qz_m
        if give_cov:
            return z, qz_v
        else:
            return z
        
    def inference(self, x, n_samples=1):
        """Helper function used in forward pass
                """
        x_ = x
        x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x=x_)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            z = Normal(qz_m, qz_v.sqrt()).sample()

        # Library size fixed
        total_counts = x.sum(dim=1, dtype=torch.float64)
        library = torch.log(total_counts).view(-1, 1)

        if self.ldvae:
            px_scale, px_rate, raw_px_scale = self.decoder(
                self.dispersion, z, library
            )
            px_rate = torch.clamp(torch.exp(raw_px_scale), max=5000)
        else:
            px_scale, px_rate, _ = self.decoder(
                self.dispersion, z, library
            )

        if self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(self.px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            library=library,
        )

    def forward(self, x):
        r""" Returns the reconstruction loss

		:param x: tensor of values with shape (batch_size, n_input)

		:return: the reconstruction loss and the Kullback divergences
		:rtype: 2-tuple of :py:class:`torch.FloatTensor`
		"""
        # Parameters for z latent distribution
        outputs = self.inference(x)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        z = outputs["z"]
        library = outputs["library"]

        self.encoder_variance.append(np.linalg.norm(qz_v.detach().cpu().numpy(), axis=1))
        
        if self.use_MP:
            # Message passing likelihood
            self.initialize_visit()
            self.initialize_messages(z, self.barcodes, self.n_latent)
            self.perform_message_passing((self.tree & self.root), z.shape[1], False)
            mp_lik = self.aggregate_messages_into_leaves_likelihood(z.shape[1], add_prior=True)
            # Gaussian variational likelihood
            qz = Normal(qz_m, torch.sqrt(qz_v)).log_prob(z).sum(dim=-1)
        else:
            mp_lik = None
            # scVI Kl Divergence
            mean = torch.zeros_like(qz_m)
            scale = torch.ones_like(qz_v)
            qz = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
        
        # Reconstruction Loss
        if self.reconstruction_loss == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.reconstruction_loss == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)

        return reconst_loss, qz, mp_lik
