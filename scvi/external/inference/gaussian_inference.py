import sys
import logging
import matplotlib.pyplot as plt
from inference import Trainer, UnsupervisedTrainer
from inference.posterior import Posterior
from ..models.gaussian_vae import GaussianVAE
from ..dataset.tree import GeneExpressionDataset
import torch
import numpy as np
logger = logging.getLogger(__name__)

plt.switch_backend("agg")


class GaussianPosterior(Posterior):
    """
    :param model: A model instance from class ``GaussianVAE``
    :param gene_dataset: A gene_dataset instance from class ``GeneExpressionDataset``

    """

    def __init__(
        self,
        model: GaussianVAE,
        gene_dataset: GeneExpressionDataset,
        shuffle=False,
        indices=None,
        use_cuda=True,
        data_loader_kwargs=dict(),
    ):
        super().__init__(
            model=model,
            gene_dataset=gene_dataset,
            shuffle=shuffle,
            indices=indices,
            use_cuda=use_cuda,
            data_loader_kwargs=data_loader_kwargs
        )

    def elbo(self) -> float:
        elbo = self.compute_elbo(self.model)
        logger.debug("ELBO : %.4f" % elbo)
        return elbo

    def compute_elbo(self, vae, **kwargs):
        # Iterate once over the posterior and compute the elbo
        print("computing elbo")
        self.use_cuda = False

        elbo = 0
        for i_batch, tensors in enumerate(self):
            sample_batch, _, _, _, _ = tensors[:5]
            reconst_loss, kl, _ = vae.forward(sample_batch)
            elbo += torch.sum(reconst_loss + kl).item()
        n_samples = len(self.indices)
        print("ELBO: {}".format(elbo))
        return elbo / n_samples

    @torch.no_grad()
    def get_latent(self,
             give_mean=False
     ):
        """Output posterior z mean or sample, batch index, and label

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
        batch_indices
            batch indicies corresponding to each cell
        labels
            label corresponding to each cell

        """
        latent = []
        for tensors in self:
            sample_batch, _, _, _, _ = tensors
            latent += [
                self.model.sample_from_posterior_z(
                    sample_batch, give_mean=give_mean
                ).cpu()
            ]
        return np.array(torch.cat(latent))

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


class GaussianTrainer(Trainer):
    r"""The VariationalInference class for the unsupervised training of an autoencoder
    with a latent tree structure.

	Args:
		:model: A model instance from class ``GaussianVAE``
		:gene_dataset: A Gene Expression Dataset
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
		>>> gene_dataset = GeneExpressionDataset(X)
        >>> vae = GaussianVAE(gene_dataset.nb_genes)
        ... n_batch=tree_dataset.n_batches * use_batches, use_cuda=True)
        >>> trainer = GaussianTrainer(vae, tree_dataset)
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
        n_epochs_kl_warmup=100,
        **kwargs
    ):

        train_size = float(train_size)
        if train_size > 1.0 or train_size <= 0.0:
            raise ValueError(
                "train_size needs to be greater than 0 and less than or equal to 1"
            )
        super().__init__(model, gene_dataset, **kwargs)

        # Set up number of warmup iterations
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.normalize_loss = True

        # Total size of the dataset used for training
        self.n_samples = 1.0

        self.train_set, self.test_set, self.validation_set =  self.train_test_validation(model=model,
                                                                                         gene_dataset=gene_dataset,
                                                                                         train_size=train_size,
                                                                                         test_size=test_size,
                                                                                         type_class=GaussianPosterior
                                                                                         )
            
        self.train_set.to_monitor = ["elbo"]
        self.test_set.to_monitor = ["elbo"]
        self.validation_set.to_monitor = ["elbo"]
        self.n_samples = len(self.train_set.indices)

    @property
    def posteriors_loop(self):
        return ["train_set"]

    def loss(self, tensors):
        """ Computes the loss of the model after a specific iteration.

        Computes the mean reconstruction loss, which is derived after a forward pass
        of the model.

        :param tensors: Observations to be passed through model

        :return: Mean reconstruction loss
        """
        sample_batch, _, _, _, _ = tensors

        reconst_loss, kl, _ = self.model.forward(x=sample_batch)
        
        loss = (
            self.n_samples
            * torch.mean(reconst_loss + self.kl_weight * kl)
        )
        if self.normalize_loss:
            loss = loss / self.n_samples
        return loss

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0

    def __setattr__(self, name, value):
        if isinstance(value, GaussianPosterior):
            name = name.strip("_")
            self.register_posterior(name, value)
        else:
            object.__setattr__(self, name, value)

    def create_posterior(
        self,
        model=GaussianVAE,
        gene_dataset=None,
        indices=None,
        type_class=GaussianPosterior,
    ):
        """
        :param model: A ``GaussianVAE` model.
        :param gene_dataset: A ``TreeDataset`` dataset that has both gene expression data and a tree.
        :param use_cuda: Default=True.
        :param type_class: Which constructor to use (here, GaussianPosterior).

        :return: A ``GaussianPosterior`` to use for training.
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
            use_cuda=self.use_cuda,
            data_loader_kwargs=self.data_loader_kwargs,
        )
    

        

