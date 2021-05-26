import logging
import random

import numpy as np
import torch
import torch.distributions as distributions

import matplotlib.pyplot as plt

from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from ..dataset.tree import TreeDataset
from ..inference.trainer import Trainer
from ..inference.gaussian_inference import GaussianPosterior
from ..models.gaussian_treevae import GaussianTreeVAE
from torch.distributions import Normal
logger = logging.getLogger(__name__)

plt.switch_backend("agg")

class SequentialCladeSampler(SubsetRandomSampler):
    """ A sampler that is used to feed observations to the VAE for model fitting.

    A `SequentiaCladeSampler` instance is instantiated with a subtree, which has had leaves
    collapsed to form 'clades', which are groups of observations (leaves) that we assume are
    drawn iid. A single iteration using the SequentailCladeSampler instance will randomly sample
    a single observation from each clade, which we will use as a batch for training our VAE.

    :param data_source: A list of 'clades', each of which corresponding to a 'leaf' of the model's tree.
    :param args: a set of arguments to be passed into ``SubsetRandomSampler``
    :param kwargs: Keyword arguments to be passed into ``SubsetRandomSampler``
    """

    def __init__(self, data_source, *args, **kwargs):
        super().__init__(data_source, *args, **kwargs)
        self.clades = data_source

    def __iter__(self):
        # randomly draw a cell from each clade (i.e. bunch of leaves)
        return iter([np.random.choice(l) for l in self.clades if len(l) > 0])



class GaussianTreePosterior(GaussianPosterior):
    """The functional data unit for treeVAE.

    A `TreePosterior` instance is instantiated with a model and
    a `gene_dataset`, and as well as additional arguments that for Pytorch's `DataLoader`.
    A subset of indices can be specified, for purposes such as splitting the data into
    train/test/validation. Each trainer instance of the `TotalTrainer` class can therefore
    have multiple `TreePosterior` instances to train a model. A `TreePosterior` instance
    also comes with many methods or utilities for its corresponding data.


    :param model: A model instance from class ``treeVAE``
    :param gene_dataset: A gene_dataset instance from class ``TreeDataset``
    :param clades: A list of clades (groups of cells, assumed to be iid) that we draw observations
    from while training.
    :param use_cuda: Default: ``True``
    :param data_loader_kwargs: Keyword arguments to passed into the `DataLoader`

    Examples:

    Let us instantiate a `trainer`, with a gene_dataset and a model

        >>> tree_dataset = TreeDataset(GeneExpressionDataset, tree)
        >>> treevae= treeVAE(tree_dataset.nb_genes, tree = tree_dataset.tree
        ... n_batch=tree_dataset.n_batches * use_batches, use_cuda=True)
        >>> trainer.train(n_epochs=400)
    """

    def __init__(
        self,
        model: GaussianTreeVAE,
        gene_dataset: TreeDataset,
        clades: list,
        use_cuda: bool = False,
        data_loader_kwargs: dict = dict(),
    ):
        super().__init__(
            model=model,
            gene_dataset=gene_dataset,
            data_loader_kwargs=data_loader_kwargs,
        )

        self.clades = clades
        self.barcodes = gene_dataset.barcodes
        self.use_cuda = use_cuda

        sampler = SequentialCladeSampler(self.clades)
        batch_size = len(self.clades)
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": batch_size})
        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)

    def elbo(self) -> float:
        elbo = self.compute_elbo(self.model)
        logger.debug("ELBO : %.4f" % elbo)
        return elbo

    def compute_elbo(self, vae, **kwargs):
        """The ELBO is the reconstruction error + the likelihood of the
        Message Passing procedure on the tree. It differs from the marginal log likelihood.
		Specifically, it is a lower bound on the marginal log likelihood
		plus a term that is constant with respect to the variational distribution.
		It still gives good insights on the modeling of the data, and is fast to compute.
		"""

        # Iterate once over the posterior and compute the elbo
        print("computing elbo")

        elbo = 0
        for i_batch, tensors in enumerate(self):
            sample_batch, _, _, _, _ = tensors[:5]
            reconst_loss, qz, mp_lik = vae.forward(sample_batch)
            elbo += torch.sum(reconst_loss)
            elbo += torch.sum(qz)
            if mp_lik:
                elbo -= mp_lik

        n_samples = len(self.indices)
        elbo /= n_samples
        return elbo

    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 100,
        batch_size: int = 128,
    ):
        """Create observation samples from the Posterior Predictive distribution

        Parameters
        ----------
        n_samples
            Number of required samples for each cell
        genes
            Indices of genes of interest
        batch_size
            Desired Batch size to generate data

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        x_old : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes)

        """
        x_old = []
        x_new = []
        for tensors in self.update({"batch_size": batch_size}):
            sample_batch, _, _, _, _ = tensors
            outputs = self.model.inference(sample_batch)

            px_m = outputs["px_m"]
            px_v = outputs["px_v"]
            dist = Normal(qz_m, qz_v.sqrt())

            gene_expressions = dist.sample() 

            x_old.append(sample_batch.cpu())
            x_new.append(gene_expressions.cpu())

        x_old = torch.cat(x_old)
        x_new = torch.cat(x_new)

        return x_new.numpy(), x_old.numpy()
    
    @torch.no_grad()
    def get_latent(self,
             give_mean=False,
             give_cov=False
     ):
        """Output posterior z mean or sample

        Parameters
        ----------
        sample
            z mean or z sample
        give_mean
             (Default value = True)

        Returns
        -------
        latent
            low-dim representation
        """
        latent = []
        for tensors in self: 
            sample_batch, _, _, _, _ = tensors
            if give_cov:
                z, qz_v = self.model.sample_from_posterior_z(
                            sample_batch, give_mean=give_mean,
                            give_cov=give_cov
                )
                latent += [z.cpu()]

                return np.array(torch.cat(latent)), qz_v
            else:
                latent += [
                    self.model.sample_from_posterior_z(
                        sample_batch, give_mean=give_mean,
                         give_cov=give_cov
                    ).cpu()
                ]
                return np.array(torch.cat(latent))

    @torch.no_grad()
    def imputation_internal(self,
                            query_node,
                            z_averaging=None,
                            pp_averaging=None,
                            known_latent=None,
                            give_mean=False
                            ):
        """
        :param self:
        :param query_node: barcode of the query node node for which we want to perform missing value imputation
        :param give_mean: bool: the mean of the NB distrbution if True
        :param z_averaging: number of samples for latent space averaging
        :param pp_averaging: number of samples for posterior predictive averaging
        :return: the imputed gene expression value at the query node
        """
        # 1. sampling from posterior z ~ q(z|x) at the leaves
        if not z_averaging:
            if known_latent is not None:
                z = known_latent
            else:
                z = self.get_latent(give_mean=False)
        else:
            latents_z = [self.get_latent(give_mean=False) for n in range(z_averaging)]
            z = np.mean(np.stack(latents_z), axis=0)

        # 2. Message passing & sampling from multivariate normal z* ~ p(z*|z)
        mu_star, nu_star = self.model.posterior_predictive_density(query_node=query_node,
                                                            evidence=z)
        if not pp_averaging:
            z_star = Normal(mu_star, torch.from_numpy(np.array([nu_star]))).sample()
            data_z = z_star
        else:
            z_star = Normal(mu_star, torch.from_numpy(np.array([nu_star]))).sample((pp_averaging,))
            z_star = torch.mean(z_star, dim=0)

        ## GPU
        if self.use_cuda:
            z_star = z_star.view(1, -1).float().to('cuda:0')
        else:
            z_star = z_star.view(1, -1).float()

        # 3. Decode latent vector x* ~ p(x*|z = z*)
        p_m, p_v = self.model.decoder.forward(z_star)
        data = Normal(p_m, p_v.sqrt()).sample((100,)).cpu().numpy()
        data = np.mean(data, axis=0)

        if give_mean:
            return data, z_star, mu_star, nu_star
            
        return data, z_star

    @torch.no_grad()
    def mcmc_estimate(self,
                        query_node,
                        n_samples=50,
                        known_latent_dist=None
                        ):
        """
        :param self:
        :param query_node: barcode of the query node node for which we want to perform missing value imputation
        :param give_mean: bool: the mean of the NB distrbution if True
        :param n_samples: number of MCMC samples
        :return: the MCMC estimate of the variance and mean parameters of internal node 'query_node'. 
        The estimate of the variance is computed with the law of total variance var(Y) = E[var(Y|X)] + var(E[Y|X])
        """
        mu_mcmc = 0
        nu_mcmc2 = []
        nu_mcmc = 0
        for i in range(n_samples):

            # 1. sampling from posterior z ~ q(z|x) at the leaves
            if known_latent_dist is not None:
                mean, cov = known_latent_dist
                z = np.random.multivariate_normal(mean=mean,
                                                cov=cov).reshape(-1, self.model.n_latent)
            else:
                z = self.get_latent(give_mean=False)

            # 2. Message passing & sampling from multivariate normal z* ~ p(z*|z)
            mu_star, nu_star = self.model.posterior_predictive_density(query_node=query_node,
                                                                evidence=z)
            
            mu_mcmc += mu_star.cpu().numpy()
            nu_mcmc2.append(mu_star.cpu().numpy())

            nu_mcmc += nu_star
        # MCMC estimate of the mean
        mu_mcmc /= n_samples

        # total variance
        nu_mcmc /= n_samples
        nu_mcmc += np.var(nu_mcmc2, axis=0)

        return mu_mcmc, nu_mcmc


    @torch.no_grad()
    def empirical_qz_v(self, n_samples, norm):
        """
        :return: empirical variance of the encoder
        """

        # Sample from posterior
        latent = []
        for n in range(n_samples):
            latent.append(self.get_latent(give_mean=False))
        latent = np.array(latent)

        qz_v = np.var(latent,
                       axis=0,
                       dtype=np.float64)

        if norm:
            norm_qz_v = [np.linalg.norm(v) for v in qz_v]
            return norm_qz_v

        return qz_v


class GaussianTreeTrainer(Trainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder
    with a latent tree structure.

	Args:
		:model: A model instance from class ``TreeVAE``
		:gene_dataset: A TreeDataset
		:train_size: The train size, either a float between 0 and 1 or an integer for the number of training samples
		 to use Default: ``0.8``.
		:test_size: The test size, either a float between 0 and 1 or an integer for the number of training samples
		 to use Default: ``None``, which is equivalent to data not in the train set. If ``train_size`` and ``test_size``
		 do not add to 1 or the length of the dataset then the remaining samples are added to a ``validation_set``.
		:n_epochs_kl_warmup: Number of epochs for linear warmup of KL(q(z|x)||p(z)) term. After `n_epochs_kl_warmup`,
			the training objective is the ELBO. This might be used to prevent inactivity of latent units, and/or to
			improve clustering of latent space, as a long warmup turns the model into something more of an autoencoder.
		:\*\*kwargs: Other keywords arguments from the general Trainer class.

	Examples:
		>>> tree_dataset = TreeDataset(GeneExpressionDataset, tree)
        >>> treevae= treeVAE(tree_dataset.nb_genes, tree = tree_dataset.tree
        ... n_batch=tree_dataset.n_batches * use_batches, use_cuda=True)
        >>> trainer = TreeTrainer(treevae, tree_dataset)
        >>> trainer.train(n_epochs=400)
	"""
    default_metrics_to_monitor = ["elbo"]

    def __init__(
        self,
        model,
        gene_dataset,
        lambda_ = 1.0,
        train_size=0.8,
        test_size=None,
        n_epochs_kl_warmup=400,
        **kwargs
    ):
        super().__init__(model, gene_dataset, **kwargs)
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.clades = []
        self.train_set = self.train_test_validation( #, self.test_set
            model, gene_dataset, train_size
        )
        self.train_set.to_monitor = ["elbo"]
        #self.test_set.to_monitor = ["elbo"]
        #self.validation_set.to_monitor = ["elbo"]

        self.barcodes = gene_dataset.barcodes

        #loss functions tracker
        self.history_train, self.history_eval = {}, {}
        self.history_train['elbo_weighted'], self.history_train['elbo'], self.history_train['Reconstruction'],\
                                            self.history_train['MP_lik'], self.history_train['Gaussian pdf'] = [], [], [], [], []
        self.history_train['ratio'] = []

        # Regularization weight
        self.lambda_ = lambda_

    @property
    def posteriors_loop(self):
        return ["train_set"]

    def loss(self, tensors):
        """ Computes the loss of the model after a specific iteration.

        Computes the mean reconstruction loss, which is derived after a forward pass
        of the model.

        :param tensors: Observations to be passed through model

        :return: Mean reconstruction loss.
        """

        sample_batch, _, _, _, _ = tensors
        reconst_loss, qz, mp_lik = self.model.forward(x=sample_batch)

        n_samples = len(self.train_set.indices)

        loss_1 = torch.mean(reconst_loss) * n_samples
        self.history_train['Reconstruction'].append(loss_1.item() / n_samples)

        loss_2 = torch.mean(qz) * n_samples
        self.history_train['Gaussian pdf'].append(loss_2.item() / n_samples)

        if mp_lik:
            loss_3 = -1 * mp_lik
            self.history_train['MP_lik'].append(loss_3.item() / n_samples)
        else:
            loss_3 = torch.from_numpy(np.array([0.0]))

        self.history_train['elbo'].append((loss_1.item() + loss_2.item() + loss_3.item()) / n_samples)
        self.history_train['ratio'].append(loss_1.item() / self.lambda_ * self.kl_weight * loss_2.item())
        self.history_train['elbo_weighted'].append(( loss_1.item() + (self.lambda_ * self.kl_weight * (loss_2.item() + loss_3.item()) ) ) / n_samples)

        #print("Encodings MP Likelihood: {}".format(self.history_train['MP_lik'][-1]))
        #print("ELBO Loss: {}".format(self.history_train['elbo'][-1]))
        #print("Varitional Likelihood: {}".format(self.history_train['Gaussian pdf'][-1]))

        return (loss_1 + (self.kl_weight * self.lambda_ * loss_2) + (self.lambda_ * self.kl_weight * loss_3)) / n_samples  

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0

    def train_test_validation(
        self,
        model: GaussianTreeVAE = None,
        gene_dataset: TreeDataset = None,
        train_size: float = 0.8,
        test_size: int = None,
        type_class=GaussianTreePosterior,
    ):
        """Creates posteriors ``train_set``, ``test_set``, ``validation_set``.
		If ``train_size + test_size < 1`` then ``validation_set`` is non-empty.

        This works a bit differently for a TreeTrainer - in order to respect the
        tree prior we need to draw our observations from within sets of cells related
        to one another (i.e in a clade).  One can think of this analagously to
        identifying clusters from the hierarchical ordering described by the tree, and splitting
        each cluster into train/test/validation.

        The procedure of actually clustering the tree into clades that contain several
        iid observations is done in the constructor function for TreeVAE (scvi.models.treevae).
        This procedure below will simply split the clades previously identified into
        train/test/validation sets according to the train_size specified.

        :param model: A ``TreeVAE` model.
        :param gene_dataset: A ``TreeDataset`` instance.
		:param train_size: float, int, or None (default is 0.1)
		:param test_size: float, int, or None (default is None)
        :param type_class: Type of Posterior object to create (here, TreePosterior)
		"""

        def get_indices_in_dataset(_subset, _subset_indices, master_list):

            _cells = np.array(_subset)[np.array(_subset_indices)]
            filt = np.array(list(map(lambda x: x in _cells, master_list)))

            return list(np.where(filt == True)[0])

        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )

        barcodes = gene_dataset.barcodes
        leaves = [n for n in model.tree.traverse('levelorder') if n.is_leaf()]

        # this is where we need to shuffle within the tree structure
        train_indices, test_indices, validate_indices = [], [], []

        # for each clade induced by an internal node at a given depth split into
        # train, test, and validation and append these indices to the master list
        # introduce an index for each leaf in the tree
        for l in leaves:
            c = l.cells
            indices = get_indices_in_dataset(c, list(range(len(c))), barcodes)
            l.indices = np.array(indices)
            self.clades.append(indices)

        # randomly split leaves into test, train, and validation sets
        for l in leaves:
            leaf_bunch = l.indices

            if len(leaf_bunch) == 1:
                #x = random.random()
                #if x < train_size:
                    #train_indices.append([leaf_bunch[0]])
                #else:
                    #test_indices.append([leaf_bunch[0]])
                train_indices.append([leaf_bunch[0]])

            else:
                n_train, n_test = _validate_shuffle_split(
                    len(leaf_bunch), test_size, train_size
                )

                random_state = np.random.RandomState(seed=self.seed)
                permutation = random_state.permutation(leaf_bunch)
                test_indices.append(list(permutation[:n_test]))
                train_indices.append(list(permutation[n_test: (n_test + n_train)]))
                # split test set in two
                validate_indices.append(list(permutation[(n_test + n_train):]))

        # some print statement to ensure test/train/validation sets created correctly
        print("train_leaves: ", train_indices)
        print("test_leaves: ", test_indices)
        print("validation leaves: ", validate_indices)
        return (
            self.create_posterior(
                model, gene_dataset, train_indices, type_class=type_class
            )
            #self.create_posterior(
                #model, gene_dataset, test_indices, type_class=type_class
            #),
            #self.create_posterior(
                #model, gene_dataset, validate_indices, type_class=type_class
            #),
        )

    def create_posterior(
        self,
        model=None,
        gene_dataset=None,
        clades=None,
        indices=None,
        type_class=GaussianTreePosterior,
    ):
        """Create a TreePosterior instance for a given set of leaves.

        This is a custom TreePoserior constructor that will take in a set of leaves (i.e. a clade)
        and return a Posterior object that can be used for training.

        :param model: A ``TreeVAE` model.
        :param gene_dataset: A ``TreeDataset`` dataset that has both gene expression data and a tree.
        :param clades: A list of clades that contain indices of sets of leaves assumed to be iid.
        :param use_cuda: Default=True.
        :param type_class: Which constructor to use (here, TreePosterior).

        :return: A ``TreePosterior`` to use for training.
        """
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )
        return type_class(
            model,
            gene_dataset,
            clades,
            use_cuda=self.use_cuda,
            data_loader_kwargs=self.data_loader_kwargs,
        )

    def __setattr__(self, name, value):
        if isinstance(value, GaussianTreePosterior):
            name = name.strip("_")
            self.register_posterior(name, value)
        else:
            object.__setattr__(self, name, value)

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        super().train(n_epochs=n_epochs, lr=lr, eps=eps, params=params)
