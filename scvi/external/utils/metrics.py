from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error 

def ks_pvalue(g, glm_samples, rep_samples):
    """
    :param g: number of genes
    :param glm_samples: simulated data
    :param rep_samples: cascVI-generated data
    :return -- > the pandas dataframe with p-values of K-S test for each sample
    """
    np.random.seed(42)

    data = []
    columns = ["Gene", "p-value", "H_0 accepted"]
    N = rep_samples.shape[0]

    for g_ in range(g):
        p_value = 0
        for n in range(N):
            x, y = glm_samples[:, n, g_].flatten(), rep_samples[n, g_, :].flatten()
            ks_test = stats.ks_2samp(x, y)
            p_value += ks_test[1]
        p_value /= N
        accepted = p_value >  0.05
        data.append([g_, p_value, accepted])
    df = pd.DataFrame(data,
                      columns=columns)
    return df

def accuracy_imputation(tree, groundtrtuh, imputed, gene):
    """
    :param tree: Cassiopeia ete3 tree
    :param groundtrtuh: ground truth gene expression value
    :param imputed: imputed gene expression value
    :param gene:
    :return:
    """
    accuracy = 0
    N = 0
    for n in tree.traverse('levelorder'):
        if not n.is_leaf() and groundtrtuh[n.name][gene] == imputed[n.name][gene]:
            accuracy += 1
        N += 1
    return (accuracy / N) * 100

def correlations(data, normalization=None, vis=True, save_fig=None):
    """
    :param data: dict:imputations (one of them should have the key 'ground truth')
    :param normalization: str: either "rank" or "quantile"
    :param: vis: bool: if True returns violin plots of the density of the correlation coefficients
    :return:
    """

    if 'groundtruth' not in data:
        raise ValueError('This call requires a groundtruth gene expression profile with the key "groundtruth" ')
    metrics = []
    columns = ["Method", "Spearman CC", "Pearson CC", "Kendall Tau"]

    # groundtruth and imputations
    internal_X = data['groundtruth']
    dim = internal_X.shape[1]
    for method, imputed_X in data.items():
        if method == 'groundtruth':
            continue
        for i in range(dim):
            if normalization == 'rank':
                data0 =  stats.rankdata(internal_X[:, i])
                data1 = stats.rankdata(imputed_X[:, i])
            else:
                data0 = internal_X[:, i]
                data1 = imputed_X[:, i]

            metrics.append([method, stats.spearmanr(data1, data0)[0], stats.pearsonr(data1, data0)[0],
                stats.kendalltau(data1, data0)[0]])

    df = pd.DataFrame(metrics, columns=columns)

    #plots
    if vis:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

        sns.violinplot(ax=axes[0], x="Spearman CC", y="Method",
                       data=df,
                       scale="width", palette="Set3")
        axes[0].set_title("Spearman CC")

        sns.violinplot(ax=axes[1], x="Pearson CC", y="Method",
                       data=df,
                       scale="width", palette="Set3")
        axes[1].set_title("Pearson CC")

        sns.violinplot(ax=axes[2], x="Kendall Tau", y="Method",
                       data=df,
                       scale="width", palette="Set3")
        axes[2].set_title("Kendall Tau")

        plt.suptitle("Correlations", fontsize=16)
    
    if save_fig:
        plt.savefig(save_fig)
    
    return df

def mse(data):
    if 'groundtruth' not in data:
        raise ValueError('This call requires a groundtruth gene expression profile with the key "groundtruth" ')

    # groundtruth and imputations
    internal_X = data['groundtruth']
    N = internal_X.shape[0]
    scores = []
    stds = []
    
    for method, imputed_X in data.items():
        if method == 'groundtruth':
            continue
        mse_scores = [mean_squared_error(internal_X[i], imputed_X[i]) for i in range(N)]
        scores.append(np.mean(mse_scores))
        stds.append(np.std(mse_scores))
    
    data_dict = {'MSE': scores, 'std': stds}
    results = pd.DataFrame.from_dict(data_dict, orient='index', columns=list(data.keys())[1:])
    return results

    

def knn_purity(max_neighbors, data, plot=True, do_normalize=True, save_fig=None):
    if do_normalize:
        for i in range(len(data)):
            data[i] = normalize(data[i])
    if len(data) == 3:
        query_z, query_scvi_z, query_cascvi_z = data
        n_neighbors = range(2, max_neighbors)
        score_scvi = []
        score_cascvi = []
        for k in n_neighbors:
            A = kneighbors_graph(query_scvi_z, k, mode='connectivity', include_self=True)
            B = kneighbors_graph(query_cascvi_z, k, mode='connectivity', include_self=True)
            C = kneighbors_graph(query_z, k, mode='connectivity', include_self=True)
            score_scvi.append(jaccard_score(A.toarray().flatten(), C.toarray().flatten()))
            score_cascvi.append(jaccard_score(B.toarray().flatten(), C.toarray().flatten()))

        if plot:
            plt.plot(n_neighbors, score_scvi, color='green', label='scVI', linestyle='dashed', linewidth=2, markersize=3, marker='+')
            plt.plot(n_neighbors, score_cascvi, color='blue', label='cascVI', linestyle='dashed', linewidth=2, markersize=3, marker='+')
            plt.xlabel('# neighbors'), plt.ylabel("purity"), plt.title("k-nn purity")
            plt.legend()
            plt.grid()
            plt.show()

    if len(data) == 5:
        query_z, query_scvi_z, query_scvi_z_2, query_cascvi_z, query_cascvi_z_2 = data
        n_neighbors = range(2, max_neighbors)
        score_scvi, score_scvi_2 = [], []
        score_cascvi, score_cascvi_2 = [], []
        for k in n_neighbors:
            A = kneighbors_graph(query_scvi_z, k, mode='connectivity', include_self=True)
            B = kneighbors_graph(query_scvi_z_2, k, mode='connectivity', include_self=True)
            C = kneighbors_graph(query_cascvi_z, k, mode='connectivity', include_self=True)
            D = kneighbors_graph(query_cascvi_z_2, k, mode='connectivity', include_self=True)
            E = kneighbors_graph(query_z, k, mode='connectivity', include_self=True)
            score_scvi.append(jaccard_score(A.toarray().flatten(), E.toarray().flatten()))
            score_scvi_2.append(jaccard_score(B.toarray().flatten(), E.toarray().flatten()))
            score_cascvi.append(jaccard_score(C.toarray().flatten(), E.toarray().flatten()))
            score_cascvi_2.append(jaccard_score(D.toarray().flatten(), E.toarray().flatten()))

        if plot:
            plt.plot(n_neighbors, score_scvi, color='green', label='scVI + averaging', linestyle='dashed', linewidth=2,
                     markersize=3, marker='+')
            plt.plot(n_neighbors, score_scvi_2, color='red', label='scVI + Message Passing', linestyle='dashed', linewidth=2,
                     markersize=3, marker='+')
            plt.plot(n_neighbors, score_cascvi, color='blue', label='cascVI + Message Passing', linestyle='dashed', linewidth=2,
                     markersize=3, marker='+')
            plt.plot(n_neighbors, score_cascvi_2, color='orange', label='cascVI + averaging', linestyle='dashed', linewidth=2,
                     markersize=3, marker='+')

            plt.xlabel('# neighbors'), plt.ylabel("purity"), plt.title("k-nn purity")
            plt.legend()
            plt.grid()
            plt.show()

        if save_fig:
            plt.savefig(save_fig)

    return score_scvi, score_cascvi

def knn_purity_stratified(n_neighbors, tree, data, min_depth=2, plot=True, do_normalize=True):
    if do_normalize:
        for i in range(len(data)):
            data[i] = normalize(data[i])

    full_latent, full_scvi_latent, full_scvi_latent_2, full_cascvi_latent, full_cascvi_latent_2 = data

    def get_nodes_depth(tree, d):
        nodes_index = []
        for i, n in enumerate(tree.traverse('levelorder')):
            if n.get_distance(tree) == d:
                nodes_index.append(n.index)
        return nodes_index

    def get_nodes(tree, nodes_index, X):
        query_nodes = []
        for n in tree.traverse('levelorder'):
            if n.index in nodes_index:
                query_nodes.append(X[n.index])
        return np.array(query_nodes)

    k = n_neighbors
    max_depth = int(max([n.get_distance(tree) for n in tree.traverse('levelorder')]))
    score_scvi, score_scvi_2 = [], []
    score_cascvi, score_cascvi_2 = [], []
    for d in range(min_depth, max_depth):
        nodes_index = get_nodes_depth(tree, d)
        query_scvi_z = get_nodes(tree, nodes_index, full_scvi_latent)
        query_scvi_z_2 = get_nodes(tree, nodes_index, full_scvi_latent_2)
        query_cascvi_z = get_nodes(tree, nodes_index, full_cascvi_latent)
        query_cascvi_z_2 = get_nodes(tree, nodes_index, full_cascvi_latent_2)
        query_z = get_nodes(tree, nodes_index, full_latent)
        A = kneighbors_graph(query_scvi_z, k, mode='connectivity', include_self=True)
        B = kneighbors_graph(query_scvi_z_2, k, mode='connectivity', include_self=True)
        C = kneighbors_graph(query_cascvi_z, k, mode='connectivity', include_self=True)
        D = kneighbors_graph(query_cascvi_z_2, k, mode='connectivity', include_self=True)
        E = kneighbors_graph(query_z, k, mode='connectivity', include_self=True)
        score_scvi.append(jaccard_score(A.toarray().flatten(), E.toarray().flatten()))
        score_scvi_2.append(jaccard_score(B.toarray().flatten(), E.toarray().flatten()))
        score_cascvi.append(jaccard_score(C.toarray().flatten(), E.toarray().flatten()))
        score_cascvi_2.append(jaccard_score(D.toarray().flatten(), E.toarray().flatten()))

    if plot:
        plt.plot(range(min_depth, max_depth), score_scvi, color='green', label='scVI + averaging', linestyle='dashed', linewidth=2, markersize=12, marker='o')
        plt.plot(range(min_depth, max_depth), score_scvi_2, color='red', label='scVI + Message passing', linestyle='dashed', linewidth=2,
                 markersize=12, marker='o')
        plt.plot(range(min_depth, max_depth), score_cascvi, color='orange', label='cascVI + Message Passing', linestyle='dashed', linewidth=2, markersize=12, marker='o')
        plt.plot(range(min_depth, max_depth), score_cascvi_2, color='blue', label='cascVI + Averaging', linestyle='dashed',
                 linewidth=2, markersize=12, marker='o')
        plt.xlabel('# depth'), plt.ylabel("purity"), plt.title("k-nn purity")
        plt.legend()
        plt.grid()
        plt.show()

    return score_scvi, score_scvi_2, score_cascvi, score_cascvi_2


















