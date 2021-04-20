from ete3 import Tree
import numpy as np


def get_leaves(X, poisson_mu, tree):
    leaves_X, leaves_idx, mu = [], [], []
    for n in tree.traverse('levelorder'):
        if n.is_leaf():
            leaves_X.append(X[n.index])
            leaves_idx.append(n.index)
            mu.append(poisson_mu[n.index])
    leaves_X = np.array(leaves_X)
    mu = np.array(mu)
    return leaves_X, leaves_idx, mu

def get_internal(X, poisson_mu, tree):
    internal_X, internal_idx, mu = [], [], []
    for n in tree.traverse('levelorder'):
        if not n.is_leaf():
            internal_X.append(X[n.index])
            internal_idx.append(n.index)
            mu.append(poisson_mu[n.index])
    internal_X = np.array(internal_X)
    mu = np.array(mu)
    return internal_X, internal_idx, mu






