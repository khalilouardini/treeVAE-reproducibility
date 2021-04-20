from ete3 import Tree
import numpy as np
import torch
from torch.distributions import Poisson

def avg_weighted_baseline(tree, weighted, X, rounding):
    """
    :param tree: ete3 phylogenetic tree
    :param weighted: True if the average is weighted
    :param X: Gene expression data
    :return: imputed Gene Expression
    """
    imputed = {}
    for n in tree.traverse('levelorder'):
        sub_leaves = n.get_leaves()
        mean_ge = 0
        for l in sub_leaves:
            if weighted:
                dist = n.get_distance(l)
                if dist > 0:
                    mean_ge += X[l.index] / dist
                elif dist == 0:
                    mean_ge += X[l.index]
                else:
                    raise ValueError("Negative branch length value detected in the tree")
        if rounding:
            imputed[n.name] = np.round(mean_ge / len(sub_leaves))
        else:
            imputed[n.name] = mean_ge / len(sub_leaves)
    return imputed


def scvi_baseline(tree, posterior, weighted=True, give_latent=False, n_samples_z=None, rounding=True):
    """
    :param tree: ete3 phylogenetic tree
    :param posterior: scVI posterior object
    :param weighted: True if the average is weighted
    :param give_latent: True to return averaged latent representations
    :param n_samples_z: number of latent imputations
    :return: imputed Gene Expression
    """
    imputed = {}
    imputed_z = {}

    # Posterior
    reconstructed = posterior.generate(n_samples=10)[0][:, :, 0]
    if give_latent:
        latents = np.array([posterior.get_latent()[0] for i in range(n_samples_z)])
        D = latents[0].shape[1]

    # Dimensions
    G = reconstructed.shape[1]

    # Initialize
    idx = 0
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            n.add_features(ge_scvi=reconstructed[idx])
            if give_latent:
                n.add_features(latent=latents[:, idx])
            idx += 1
        #else:
            #n.add_features(ge_scvi=0)
            #if give_latent:
                #n.add_features(latent=np.zeros(shape=latent.shape[1]))

    # Averaging
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            continue
        else:
            # Init
            sub_leaves = n.get_leaves()
            mean_ge = np.zeros(shape=G)
            if give_latent:
                mean_z = np.zeros(shape=(n_samples_z, D))

            # Loop over leaves
            for l in sub_leaves:
                if weighted:
                    dist = n.get_distance(l)
                    if dist > 0:
                        mean_ge += l.ge_scvi / dist
                        if give_latent:
                            mean_z += l.latent / dist

                    elif dist == 0:
                        mean_ge += l.ge_scvi
                        if give_latent:
                            mean_z += l.latent
                    else:
                        raise ValueError("Negative branch length value detected in the tree")
                else:
                    mean_ge += l.ge_scvi
                    if give_latent:
                        mean_z += l.latent
        if rounding:
            imputed[n.name] = np.rint(mean_ge / len(sub_leaves))
        else:
            imputed[n.name] = mean_ge / len(sub_leaves)
        if give_latent:
            imputed_z[n.name] = mean_z / len(sub_leaves)

    if give_latent:
        return imputed, imputed_z

    return imputed


@torch.no_grad()
def scvi_baseline_z(tree,
                    posterior,
                    model,
                    weighted=True,
                    n_samples_z=None,
                    library_size=10000):
    """
    :param tree: ete3 phylogenetic tree
    :param posterior: scVI posterior object
    :param model: VAE or variant
    :param weighted: True if the average is weighted
    :param give_latent: True to return averaged latent representations
    :param n_samples_z: number of latent imputations
    :param library_size:
    :return: imputed Gene Expression
    """
    imputed = {}
    imputed_z = {}

    # Posterior
    latents = np.array([posterior.get_latent()[0] for i in range(n_samples_z)])
    D = latents[0].shape[1]

    # Initialize
    idx = 0
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            n.add_features(latent=latents[:, idx])
            idx += 1

    # Averaging
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            continue
        else:
            # Init
            sub_leaves = n.get_leaves()
            mean_z = np.zeros(shape=(n_samples_z, D))

            # Loop over leaves
            for l in sub_leaves:
                if weighted:
                    dist = n.get_distance(l)
                    if dist > 0:
                        mean_z += l.latent / dist
                    elif dist == 0:
                        mean_z += l.latent
                    else:
                        raise ValueError("Negative branch length value detected in the tree")
                else:
                    mean_z += l.latent
            imputed_z[n.name] = mean_z / len(sub_leaves)

            # Decoding the averaged latent vector
            with torch.no_grad():
                px_scale, px_r, px_rate, px_dropout = posterior.model.decoder.forward(model.dispersion,
                                                                                 torch.from_numpy(mean_z).float(),
                                                                                 torch.from_numpy(
                                                                                     np.array([np.log(library_size)])),
                                                                                 0)

            l_train = torch.clamp(torch.mean(px_rate, axis=0), max=1e8)

            data = Poisson(l_train).sample().cpu().numpy()

            imputed[n.name] = data

    return imputed, imputed_z

@torch.no_grad()
def cascvi_baseline_z(tree,
                    latent,
                    model,
                    library_size=10000):
    """
    :param tree: ete3 phylogenetic tree
    :param latent: dict: latent representations of internal nodes
    :param model: VAE or variant
    :param weighted: True if the average is weighted
    :param library_size:
    :return: imputed Gene Expression
    """

    imputed = {}
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            continue
        else:
            px_scale, px_r, px_rate, px_dropout = model.decoder.forward(model.dispersion,
                                                                        latent[n.name].float().view(1, -1),
                                                                        torch.from_numpy(np.array([np.log(library_size)])),
                                                                        0)

            l_train = torch.clamp(torch.mean(px_rate, axis=0), max=1e8)

            data = Poisson(l_train).sample().cpu().numpy()

            imputed[n.name] = data

    return imputed

def construct_latent(tree, leaves_z, internal_z):
    # Merge internal nodes and leaves
    full_latent = []
    idx = 0

    for i, n in enumerate(tree.traverse()):
        if n.is_leaf():
            full_latent.append(leaves_z[idx])
            idx += 1
        else:
            full_latent.append(internal_z[n.name])

    full_latent = np.vstack(full_latent)
    return full_latent

