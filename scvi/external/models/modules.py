import collections
from typing import Iterable, List

import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList

from ..models.  utils import one_hot


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def identity(x):
    return x


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.
    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_relu
        Whether to have `ReLU` layers or not
    bias
        Whether to learn bias in linear layers or not
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_relu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out, bias=bias),
                            # Below, 0.01 and 0.001 are the default values for `momentum` and `eps` from
                            # the tensorflow implementation of batch norm; we're using those settings
                            # here too so that the results match our old tensorflow code. The default
                            # setting from pytorch would probably be fine too but we haven't tested that.
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.ReLU() if use_relu else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, instance_id: int = 0):
        """Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        instance_id
            Use a specific conditional instance normalization (batchnorm)
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """

        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    x = layer(x)
        return x


# Encoder
class Encoder(nn.Module):
    """Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
        :dropout_rate: Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z

    Returns
    -------
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
    ):
        super().__init__()

        self.distribution = distribution
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor):
        """The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """

        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Decoder
class Decoder(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers

    Returns
    -------
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the NB/Poisson distribution of expression

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        z :
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple of :py:class:`torch.Tensor`
            parameters for the NB/Poisson distribution of expression
        """
        # The decoder returns values for the parameters of the NB/Poisson distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        return px_scale, px_rate, None


class LinearDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        use_batch_norm: bool = True,
        bias: bool = False,
    ):
        super().__init__()

        self.factor_regressor = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_layers=1,
            use_relu=False,
            use_batch_norm=use_batch_norm,
            bias=bias,
            dropout_rate=0,
        )

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor
    ):
        # The decoder returns values for the parameters of the NB/Poisson distribution
        raw_px_scale = self.factor_regressor(z)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_rate = torch.exp(library) * px_scale

        return px_scale, px_rate, None


class GaussianDecoder(nn.Module):
    """Decodes data from latent space to data space

    ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers

    Returns
    -------
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        x
            tensor with shape ``(n_input,)``

        Returns
        -------
        2-tuple of :py:class:`torch.Tensor`
            Mean and variance tensors of shape ``(n_output,)``

        """

        # Parameters for Likelihood
        p = self.decoder(z)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p)) + 1e-4
        return p_m, p_v

class GaussianLinearDecoder(nn.Module, sigma):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        use_batch_norm: bool = True,
        bias: bool = False,
        sigma: float = sigma
    ):
        super().__init__()

        self.factor_regressor = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_layers=1,
            use_relu=False,
            use_batch_norm=use_batch_norm,
            bias=bias,
            dropout_rate=0,
        )

    def forward(
        self, z: torch.Tensor
    ):
        # Mean of Normal
        p_m = self.factor_regressor(z)
        p_v = self.sigma * torch.ones(size=)
        px_rate = torch.exp(library) * px_scale

        return px_scale, px_rate, None