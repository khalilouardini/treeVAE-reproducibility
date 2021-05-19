from ete3 import Tree
from matplotlib.pyplot import axis
import numpy as np
import torch
from torch.distributions import Poisson, Normal, Gamma
from ..models.distributions import NegativeBinomial

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
            else:
                mean_ge += X[l.index]
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
        D = latents.shape[2]

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
def avg_baseline_z(tree,
                    posterior,
                    model,
                    weighted=True,
                    library_size=10000,
                    n_samples_z=1,
                    gaussian=True,
                    known_latent=False,
                    latent=None,
                    use_cuda=False,
                    give_cov=False
                    ):
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
    imputed_cov_z = {}

    # Posterior
    if known_latent:
        latents = latent
        n_samples_z = latents.shape[0]
    else:
        latents = []
        for i in range(n_samples_z):
            z, qz_v = posterior.get_latent(give_mean=False, give_cov=True)
            latents.append(z)
        latents = np.array(latents)
    D = latents.shape[2]

    # Initialize
    idx = 0
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            n.add_features(latent=latents[:, idx])
            if give_cov:
                n.add_features(cov=qz_v[idx])
            idx += 1

    # Averaging
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            continue
        else:
            # Init
            sub_leaves = n.get_leaves()
            mean_z = np.zeros(shape=(n_samples_z, D))
            cov_z = 0

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
                if give_cov:
                    cov_z += l.cov
                
            mean_z /= len(sub_leaves)
            imputed_z[n.name] = mean_z
            if give_cov:
                imputed_cov_z[n.name] = cov_z / (len(sub_leaves))**2

            ## GPU
            library = torch.from_numpy(np.array([np.log(library_size)]))
            if use_cuda:
                mean_z = torch.from_numpy(mean_z).float().to('cuda:0')
                library = library.to('cuda:0')
            else:
                mean_z = torch.from_numpy(mean_z).float()

            # Decoding the averaged latent vector
            with torch.no_grad():
                if not gaussian:
                    if not model.ldvae:
                        px_scale, px_rate, px_r = posterior.model.decoder(dispersion=model.dispersion,
                                                                            z=mean_z,
                                                                            library=library
                                                                                )
                        l_train = torch.clamp(torch.mean(px_rate, axis=0), max=1e8)
                    else:
                        px_scale, px_rate, raw_px_scale = posterior.model.decoder(dispersion=model.dispersion,
                                                                                z=mean_z,
                                                                                library=library
                                                                                )
                        px_rate = torch.exp(raw_px_scale)                                                        
                        l_train = torch.clamp(torch.mean(px_rate, axis=0), max=5000)    

                    data = torch.mean(Poisson(l_train).sample((50,)), axis=0).cpu().numpy()
                    imputed[n.name] = np.clip(a=data,
                                              a_max=1e8,
                                              a_min=-1e8
                                              )
                else:
                    p_m, p_v = posterior.model.decoder(mean_z)
                    data = torch.mean(Normal(p_m, p_v.sqrt()).sample((40,)), axis=0)                                                               
                    imputed[n.name] = np.clip(a=data.cpu().numpy(),
                                             a_max=1e8,
                                             a_min=-1e8
                                             )
    if give_cov:
        return imputed, imputed_z, imputed_cov_z
    else:
        return imputed, imputed_z



@torch.no_grad()
def scvi_baseline_z(tree,
                    posterior,
                    model,
                    weighted=True,
                    n_samples_z=None,
                    library_size=10000,
                    use_cuda=False,
                    known_latent=False,
                    latent=None):
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
    if known_latent:
        latents = latent
        n_samples_z = latents.shape[0]
    else:
        latents = np.array([posterior.get_latent()[0] for i in range(n_samples_z)])
    D = latents.shape[2]

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
            mean_z /= len(sub_leaves)
            imputed_z[n.name] = mean_z

            ## GPU
            library = torch.from_numpy(np.array([np.log(library_size)]))
            if use_cuda:
                mean_z = torch.from_numpy(mean_z).float().to('cuda:0')
                library = library.to('cuda:0')
            else:
                mean_z = torch.from_numpy(mean_z).float()

            # Decoding the averaged latent vector
            with torch.no_grad():
                px_scale, px_r, px_rate, px_dropout = posterior.model.decoder.forward(model.dispersion,
                                                                                 mean_z,
                                                                                 library,
                                                                                 0)
                
                if model.reconstruction_loss=='poisson' and model.ldvae==True:
                    px_rate = px_r

                elif model.reconstruction_loss=='nb':
                    dispersion = torch.exp(model.px_r)

            if model.reconstruction_loss=='poisson':
                #l_train = torch.clamp(px_rate, max=1e5) 
                l_train = torch.clamp(torch.mean(px_rate, axis=0), max=1e8)
                data = torch.mean(Poisson(l_train).sample((50,)), axis=0).cpu().numpy()
                imputed[n.name] = np.clip(a=data,
                                            a_max=1e8,
                                            a_min=-1e8
                                            )
                #imputed[n.name] = Poisson(l_train).sample().cpu().numpy()
                
            elif model.reconstruction_loss=='nb':
                #p = px_rate / (px_rate + dispersion)
                #l_train = Gamma(dispersion, (1 - p) / p).sample()
                imputed[n.name] = NegativeBinomial(mu=px_rate, theta=dispersion).sample((200,)).cpu().numpy()
                #Poisson(l_train).sample().cpu().numpy()

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

    for i, n in enumerate(tree.traverse('levelorder')):
        if n.is_leaf():
            full_latent.append(leaves_z[idx])
            idx += 1
        else:
            full_latent.append(internal_z[n.name])

    full_latent = np.vstack(full_latent)
    return full_latent

