import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from torch.distributions import Normal
from ..models.modules import Encoder, GaussianDecoder #, GaussianLinearDecoder

from typing import Tuple, Dict

torch.backends.cudnn.benchmark = True


# Gaussian VAE model
class GaussianVAE(nn.Module):
    """Variational auto-encoder model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks

    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        latent_distribution: str = "normal",
        sigma_ldvae: float = None,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        if not sigma_ldvae:
            self.decoder = GaussianDecoder(
                n_latent,
                n_input,
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
        else:
            self.decoder = GaussianLinearDecoder(
                n_input=n_latent,
                n_output=n_input,
                sigma=sigma_ldvae
            )

    def get_latents(self, x):
        """Returns the result of ``sample_from_posterior_z`` inside a list

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``

        Returns
        -------
        type
            one element list of tensor

        """
        return [self.sample_from_posterior_z(x)]

    def sample_from_posterior_z(self, x, give_mean=False, n_samples=5000):
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
        qz_m, qz_v, z = self.z_encoder(x)
        if not give_mean:
            samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
            z = torch.mean(samples, dim=0)
        else:
            z = qz_m
        return z

    def inference(self, x, n_samples=1):
        """Helper function used in forward pass"""
        x_ = x
        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            z = Normal(qz_m, qz_v.sqrt()).sample()

        px_m, px_v = self.decoder(z)

        return dict(
            px_m=px_m,
            px_v=px_v,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z
        )

    def forward(self, x):
        """ Returns the reconstruction loss

        :param x: tensor of values with shape (batch_size, n_input)

        :return: the reconstruction loss and the Kullback divergences
        :rtype: 2-tuple of :py:class:`torch.FloatTensor`
        """
        # Parameters for z latent distribution
        outputs = self.inference(x)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        px_m = outputs["px_m"]
        px_v = outputs["px_v"]
        z = outputs["z"]
   
        # KL divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl = kl_divergence(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)

        # Reconstruction loss
        reconst_loss = -Normal(px_m, torch.sqrt(px_v)).log_prob(x).sum(dim=-1)

        return reconst_loss, kl, 0.0






