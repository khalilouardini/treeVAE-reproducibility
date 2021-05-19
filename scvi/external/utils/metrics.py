from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error 
from sklearn.metrics.pairwise import manhattan_distances
from scipy.stats import multivariate_normal
import os

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


def mse(data, metric):
    if 'groundtruth' not in data:
        raise ValueError('This call requires a groundtruth gene expression profile with the key "groundtruth" ')

    # groundtruth and imputations
    internal_X = data['groundtruth']
    N = internal_X.shape[0]
    D = N = internal_X.shape[1]
    scores = []
    stds = []
    
    for method, imputed_X in data.items():
        if method == 'groundtruth':
            continue
        if metric == 'MSE':
            mse_scores = [mean_squared_error(internal_X[i], imputed_X[i]) for i in range(N)]
        elif metric == 'L1':
            mse_scores = [manhattan_distances(internal_X[i].reshape(1, -1), imputed_X[i].reshape(1, -1)) / D for i in range(N)]
        scores.append(np.mean(mse_scores))
        stds.append(np.std(mse_scores))
    
    data_dict = {'MSE': scores, 'std': stds}
    results = pd.DataFrame.from_dict(data_dict, orient='index', columns=list(data.keys())[1:])
    return results

    

def knn_purity(max_neighbors, data, plot=True, do_normalize=True, save_fig=None):
    if do_normalize:
        for method in data:
            data[method] = normalize(data[method])
    
    n_neighbors = range(2, max_neighbors)
    query_z = data['groundtruth']

    scores = {}
    for method in data:
        if method == 'groundtruth':
            continue
        query = data[method]
        scores[method] = []
        for k in n_neighbors:
            A = kneighbors_graph(query_z, k, mode='connectivity', include_self=True)
            B = kneighbors_graph(query, k, mode='connectivity', include_self=True)
            s = jaccard_score(B.toarray().flatten(), A.toarray().flatten())
            scores[method].append(s)
        
        if plot:
            plt.plot(n_neighbors, scores[method], label=method, linestyle='dashed', linewidth=2, markersize=3, marker='+')

    plt.xlabel('# neighbors'), plt.ylabel("purity"), plt.title("k-nn purity")
    plt.legend()
    plt.grid()

    if plot:
        plt.show()
        if save_fig:
            plt.savefig(save_fig)

    return scores

def knn_purity_stratified(n_neighbors, tree, data, min_depth=2, plot=True, do_normalize=True):
    if do_normalize:
        for method in data:
            data[method] = normalize(data[method])

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
    
    for method in data:
        if method == 'groundtruth':
            continue
        query = data[method]
        k = n_neighbors
        max_depth = int(max([n.get_distance(tree) for n in tree.traverse('levelorder')]))
        scores = []
        for d in range(min_depth, max_depth):

            nodes_index = get_nodes_depth(tree, d)
            query_z = get_nodes(tree, nodes_index, data['groundtruth'])
            query = get_nodes(tree, nodes_index, data[method])
            
            A = kneighbors_graph(query_z, k, mode='connectivity', include_self=True)
            B = kneighbors_graph(query, k, mode='connectivity', include_self=True)

            scores.append(jaccard_score(B.toarray().flatten(), A.toarray().flatten()))

        if plot:
            plt.plot(range(min_depth, max_depth), scores, label=method, linestyle='dashed', linewidth=2, markersize=12, marker='o')
            
    if plot:
        plt.xlabel('# depth'), plt.ylabel("purity"), plt.title("k-nn purity")
        plt.legend()
        plt.grid()
        plt.show()

    return scores


def update_metrics(metrics, data, normalization=None):
    """
    :param data: dict:imputations (one of them should have the key 'ground truth')
    :param normalization: str: either "rank" or "quantile"
    :param: metrics: list of metrics to be updated
    :return:
    """

    if 'groundtruth' not in data:
        raise ValueError('This call requires a groundtruth gene expression profile with the key "groundtruth" ')

    corr_gg, corr_ss, mse, l1 = list(metrics.keys())[:4]
    # gene-gene imputations
    internal_X = data['groundtruth']
    dim = internal_X.shape[1]
    for method, imputed_X in data.items():
        # Correlations
        if method == 'groundtruth':
            continue
        for i in range(dim):
            if normalization == 'rank':
                data0 =  stats.rankdata(internal_X[:, i])
                data1 = stats.rankdata(imputed_X[:, i])
            else:
                data0 = internal_X[:, i]
                data1 = imputed_X[:, i]

            metrics[corr_gg].append([method, stats.spearmanr(data1, data0)[0], stats.pearsonr(data1, data0)[0],
                stats.kendalltau(data1, data0)[0]])
    

    # sample-sample imputations
    internal_X = data['groundtruth'].T
    N = internal_X.shape[0]
    dim = internal_X.shape[1]
    mse_scores = []
    l1_scores = []
    for method, imputed_X in data.items():
        
        if method == 'groundtruth':
            continue

        # Correlations
        imputed_X = imputed_X.T
        for i in range(dim):
            if normalization == 'rank':
                data0 =  stats.rankdata(internal_X[:, i])
                data1 = stats.rankdata(imputed_X[:, i])
            else:
                data0 = internal_X[:, i]
                data1 = imputed_X[:, i]

            metrics[corr_ss].append([method, stats.spearmanr(data1, data0)[0], stats.pearsonr(data1, data0)[0],
                stats.kendalltau(data1, data0)[0]])
        
        # MSE error
        mse_scores.append(np.mean([mean_squared_error(internal_X[i], imputed_X[i]) for i in range(N)]))

        # L1 error
        l1_scores.append(np.mean([manhattan_distances(internal_X[i].reshape(1, -1), imputed_X[i].reshape(1, -1)) / dim for i in range(N)]))
       
    metrics[mse].append(mse_scores)
    metrics[l1].append(l1_scores)

def error_latent(tree, predictive_z, imputed_avg_z, imputed_z, do_variance=False):
    mse_treevae = 0
    mse_vae = 0
    N = 0
    for n in tree.traverse('levelorder'):
        if not n.is_leaf():
            if do_variance:
                true_cov = predictive_z[n.name]
                vae_cov = imputed_avg_z[n.name].cpu().numpy()
                treevae_cov = imputed_z[n.name]

                mse_treevae += mean_squared_error(true_cov, treevae_cov)
                mse_vae += mean_squared_error(true_cov, vae_cov)
            else:
                mse_treevae += mean_squared_error(predictive_z[n.name], imputed_z[n.name])
                mse_vae += mean_squared_error(predictive_z[n.name], imputed_avg_z[n.name][0])
            N += 1
    mse_treevae /= N
    mse_vae /= N

    return [mse_treevae, mse_vae]

def mean_posterior_lik(tree, predictive_mean_z, imputed_avg_z, imputed_mean_z, predictive_cov_z, imputed_avg_cov_z, imputed_cov_z):
    treevae_lik = 0
    vae_lik = 0
    N = 0
    for n in tree.traverse('levelorder'):
        if not n.is_leaf():
            # mean
            true_mean = predictive_mean_z[n.name]
            vae_mean = imputed_avg_z[n.name][0]
            treevae_mean = imputed_mean_z[n.name]
            # covariance
            true_cov = np.diag(predictive_cov_z[n.name])
            vae_cov = np.diag(imputed_avg_cov_z[n.name].cpu().numpy())
            treevae_cov = np.diag(imputed_cov_z[n.name])

            sample_treevae = np.random.multivariate_normal(mean=treevae_mean,
                                                            cov=treevae_cov)
            sample_vae = np.random.multivariate_normal(mean=vae_mean,
                                                        cov=vae_cov)

            treevae_lik += multivariate_normal.logpdf(sample_treevae,
                                                    true_mean,
                                                    true_cov)
            vae_lik += multivariate_normal.logpdf(sample_vae,
                                                    true_mean,
                                                    true_cov)
            
            N += 1

    vae_lik /= N
    treevae_lik /= N
    return [vae_lik, treevae_lik]


def report_results(metrics, save_path, columns2):
    columns1 = ["Method", "Spearman CC", "Pearson CC", "Kendall Tau"]
    columns2 = columns2
    columns3 = ["gaussian VAE", "gaussian treeVAE"]
    for metric, data in metrics.items():
            if metric.startswith('corr'):
                df = pd.DataFrame(data, columns=columns1)
                df.to_csv(os.path.join(save_path, metric))
            elif (metric == 'MSE_var') or (metric == 'MSE_mean') or (metric == 'Likelihood') or (metric == 'Cross_Entropy'):
                df = pd.DataFrame(data, columns=columns3)
                df.to_csv(os.path.join(save_path, metric))
            else:
                df = pd.DataFrame(data, columns=columns2)
                df.to_csv(os.path.join(save_path, metric))
        
            





















