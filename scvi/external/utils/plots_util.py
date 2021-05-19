import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
import matplotlib.cm as cmap
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pydot
import random
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import normalize

def plot_histograms(X, title):
    fig, axes = plt.subplots(1, 1,
                             figsize=(14, 8),
                             sharey=True,
                             )

    bins = np.arange(0, 50, 10)

    cm = plt.cm.get_cmap('RdYlBu_r')

    n, binss, patches = axes.hist(X,
              bins=bins,
              edgecolor='black',
              )

    # set color of patches
    # scale values to interval [0,1]
    bin_centers = 0.5 * (binss[:-1] + binss[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    axes.set_title(title)

    plt.ylabel('Counts')
    plt.xlabel('Gene Expression value')
    plt.legend(['gene_' + str(i) for i in list(range(20))], loc='best')
    plt.show()

def plot_embedding(X, color_index, sizes):
    reducer = umap.UMAP()

    # fit to data
    X_embedded_pca = PCA(n_components=2).fit_transform(X)
    X_embedded_umap = reducer.fit_transform(X)
    X_embedded_tsne = TSNE(n_components=2).fit_transform(X)

    fig, ax = plt.subplots(1, 3, figsize = (18, 5))

    # Color Maps
    group = np.copy(color_index)
    unique_group = np.unique(group)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_group)))
    c_dict = {unique_group[k]: colors[k] for k in range(len(unique_group))}

    # sequence of colors
    cmap = [c_dict[k] for k in group]

    # PCA
    scatter1 = ax[0].scatter(X_embedded_pca[:, 0],
               X_embedded_pca[:,1],
               c = cmap,
               label=group,
               s=sizes)

    legend1 = ax[0].legend(*scatter1.legend_elements(num=15),
                        loc="best", title="idx")

    ax[0].set_title('PCA', fontsize=10)

    ax[0].set_xlabel('PC 1')
    ax[0].set_ylabel('PC 2')
    ax[0].add_artist(legend1)

    # UMAP
    scatter2 = ax[1].scatter(X_embedded_umap[:, 0],
           X_embedded_umap[:,1],
           c = cmap,
           label = group,
           s=sizes)

    #legend2 = ax[1].legend(*scatter2.legend_elements(num=15),
                        #loc="best", title="idx")

    ax[1].set_title('UMAP', fontsize=10)

    ax[1].set_xlabel('UMAP 1')
    ax[1].set_ylabel('UMAP 2')
    #ax[1].add_artist(legend2)

    # tSNE
    scatter3 = ax[2].scatter(X_embedded_tsne[:, 0],
                                X_embedded_tsne[:,1],
                                c = cmap,
                                label=group,
                                s=sizes)

    #legend3 = ax[2].legend(*scatter3.legend_elements(num=15),
    #                    loc="best", title="idx")

    ax[2].set_title('tSNE', fontsize=10)

    ax[2].set_xlabel('tSNE 1')
    ax[2].set_ylabel('tSNE 2')
    #ax[2].add_artist(legend3)


    fig.suptitle('Dimensionality reduction', fontsize=10)
    patches = [mpatches.Patch(color=colors[k], label=unique_group[k]) for k in range(len(unique_group))]
    plt.legend(handles=patches)
    plt.show()


def plot_density(data):
    n = len(data)
    # Make the density plot
    evenly_spaced_interval = np.linspace(0, 1, n)
    colors = [cmap.rainbow(x) for x in evenly_spaced_interval]
    fig, ax = plt.subplots(figsize=(14,8))
    for i, c in zip(range(n), colors):
        total_counts = np.sum(data[i], axis=0)  # sum columns together
                                           # (axis=1 would sum rows)

        # Use Gaussian smoothing to estimate the density
        density = stats.kde.gaussian_kde(total_counts)

        # Make values for which to estimate the density, for plotting
        x = np.arange(min(total_counts), max(total_counts), 10000)

        ax.plot(x, density(x), color=c)
        ax.set_xlabel("Total counts per individual")
        ax.set_ylabel("Density")
        plt.show()

        print(f'Count statistics:\n  min:  {np.min(total_counts)}'
              f'\n  mean: {np.mean(total_counts)}'
              f'\n  max:  {np.max(total_counts)}')

def plot_scatter_mean(mu, imputed, g, color):
    n_rows = int(g / 5)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(45, 35),
                         sharey=True,
                         )
    g_ = 0
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col].scatter(mu[:, g_],
                          imputed[:, g_],
                          color=color,
                          marker="+",
                          s=100
                       )
            axes[row, col].loglog()
            axes[row, col].set_xlabel('Mean of Poisson', fontsize=16)
            axes[row, col].set_ylabel('Mean of NB', fontsize=16)
            axes[row, col].set_title("Gene n° {}".format(g_), fontsize=20)
            g_ += 1
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    fig.suptitle("Scatter plot of mean parameters", fontsize=36)
    plt.show()

def plot_scatter_samples(X, Y, color):
    if Y is None:
        Y = np.copy(X)

    n_rows = int(X.shape[0] / 5)
    fig, axes = plt.subplots(n_rows, n_rows,
                         figsize=(45, 35),
                         sharey=True,
                         )

    for row in range(n_rows):
        for col in range(n_rows):
            axes[row, col].scatter(X[row],
                                  Y[col],
                                  color=color,
                                  marker="+",
                                  s=400
                               )
            #axes[row, col].loglog()
            axes[row, col].set_xlabel('sample X' + str(row), fontsize=16)
            axes[row, col].set_ylabel('sample Y '+ str(col), fontsize=16)
            axes[row, col].set_title("Sample X {} & Y {}".format(row, col), fontsize=20)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    fig.suptitle("Scatter plot of samples", fontsize=36)
    plt.show()


def plot_ecdf_ks(glm_samples, rep_samples, g):
    # plot the cdf
    n_cols = 3
    n_rows = int(g / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(45, 35),
                             sharey=True,  dpi=120,
                             facecolor='w', edgecolor='k'
                             )
    g_ = 0
    for row in range(n_rows):
        for col in range(n_cols):
            # fit a cdf
            ecdf1 = ECDF(glm_samples[:, :, g_].flatten())
            ecdf2 = ECDF(rep_samples[:, g_, :].flatten())

            axes[row, col].plot(ecdf1.x, ecdf1.y, color='green', linewidth=2)
            axes[row, col].plot(ecdf2.x, ecdf2.y, color='blue', linewidth=2)
            axes[row, col].set_xlim(0, 30)
            axes[row, col].set_title("Gene {}".format(g_), fontsize=20)
            axes[row, col].set_xlabel("Gene expression value (x)", fontsize=16)
            axes[row, col].set_ylabel("|P(X < x)", fontsize=20)
            g_ += 1
    plt.suptitle("Ecdf of ground truth and generated samples", fontsize=36)
    plt.legend(['simulated ecdf', 'recplicated ecdf'], loc='best')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    plt.show()

def plot_elbo(trainer, freq):
    figure = plt.figure(figsize=(12,6))

    elbo_train_set = trainer.history["elbo_train_set"]
    epochs = np.linspace(start=0,
                         stop=trainer.n_epochs,
                         num=int(trainer.n_epochs/freq),
                        dtype=int)
    plt.plot(
             np.log(elbo_train_set),
             label="eval Elbo",
             color='green',
             linestyle=':',
             linewidth=3
             )

    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.legend()
    plt.title("log-ELBO loss evolution")
    plt.show()

def plot_losses(trainer):
    figure, (ax1, ax2) = plt.subplots(figsize=(45, 15), ncols=2)

    ax1.plot(trainer.history_train['Gaussian pdf'][1:],
             label="Gaussian pdf",
             color='orange',
             linewidth=6.0
             )

    ax1.plot(trainer.history_train['MP_lik'][1:],
             label="-1 * MP Likelihood",
             color='blue',
             linewidth=6.0,
             marker='+'
             )

    ax2.plot(np.log(trainer.history_train['elbo'][1:]),
             label="elbo",
             color='red',
             linestyle=':',
             linewidth=6.0,
             marker='o'
             )

    ax2.plot(np.log(trainer.history_train['Reconstruction'][1:]),
             label="Reconstruction error",
             color='green',
             linestyle=':',
             linewidth=6.0,
             marker='o'
             )

    ax2.plot(np.log(trainer.history_train['elbo_weighted'][1:]),
         label="Reconstruction error",
         color='green',
         linestyle=':',
         linewidth=6.0,
         marker='o'
         )

    ax2.set_title('ELBO & Reconstruction Error', fontsize=34)
    ax2.set_xlabel('Epoch', fontsize=34)
    ax2.set_ylabel('error', fontsize=34)
    ax2.legend(list(trainer.history_train.keys())[:2], loc='best', borderpad=4)

    ax1.set_xlabel('Epoch', fontsize=34)
    ax1.set_ylabel('Loss terms', fontsize=34)
    ax1.set_title("log-loss terms evolution", fontsize=44)
    ax1.legend(list(trainer.history_train.keys())[2:][::-1], loc='best', borderpad=4)

    plt.show()

def training_dashboard(trainer, encoder_variance):
    figure, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(figsize=(45, 35), nrows=3, ncols=2)

    legend = list(trainer.history_train.keys())
    ax1.plot(trainer.history_train['elbo'], color='blue', linestyle=':', linewidth=4.0, marker='+')
    ax1.set_xlabel('Epoch', fontsize=34)
    ax1.set_ylabel('Elbo', fontsize=30)
    ax1.set_title("Normalized ELBO", fontsize=40)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.grid()
    ax1.legend([legend[1]], loc='best', borderpad=4, fontsize=22)

    ax2.plot(trainer.history_train['elbo_weighted'], color='red', linestyle=':', linewidth=4.0, marker='+')
    ax2.set_xlabel('Epoch', fontsize=34)
    ax2.set_ylabel('Elbo', fontsize=30)
    ax2.set_title("Normalized weighted ELBO", fontsize=40)
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.grid()
    ax2.legend([legend[0]], loc='best', borderpad=4, fontsize=22)

    ax3.plot(trainer.history_train['Reconstruction'], color='green', linestyle=':', linewidth=4.0, marker='+')
    ax3.set_xlabel('Epoch', fontsize=34)
    ax3.set_ylabel('Reconstruction error', fontsize=30)
    ax3.set_title("Normalized Reconstruction error", fontsize=40)
    ax3.tick_params(axis='both', which='major', labelsize=25)
    ax3.grid()
    ax3.legend([legend[2]], loc='best', borderpad=4, fontsize=22)

    ax4.plot(trainer.history_train['MP_lik'], color='purple', linestyle=':', linewidth=4.0, marker='+')
    ax4.plot(trainer.history_train['Gaussian pdf'], color='orange', linestyle=':', linewidth=4.0, marker='+')
    ax4.set_xlabel('Epoch', fontsize=34)
    ax4.set_ylabel('prior  & variational likelihood', fontsize=30)
    ax4.set_title("Normalized prior  & variational likelihood", fontsize=40)
    ax4.tick_params(axis='both', which='major', labelsize=25)
    ax4.grid()
    ax4.legend(legend[3:5], loc='best', borderpad=4, fontsize=22)

    ax5.plot(trainer.history_train['ratio'], color='pink', linestyle=':', linewidth=4.0, marker='o')
    ax5.set_xlabel('Epoch', fontsize=34)
    ax5.set_ylabel('Ratio', fontsize=30)
    ax5.set_title("Reconstruction - likelihood Ratio", fontsize=40)
    ax5.tick_params(axis='both', which='major', labelsize=25)
    ax5.grid()
    ax5.legend([legend[5]], loc='best', borderpad=4, fontsize=22)

    ax6.plot(encoder_variance)
    ax6.set_xlabel('Epoch', fontsize=34)
    ax6.set_ylabel('Encoder variance', fontsize=30)
    ax6.set_title("Encoder variance", fontsize=40)
    ax6.tick_params(axis='both', which='major', labelsize=25)
    ax6.grid()
    ax6.legend([['encoder variance']], loc='best', borderpad=4, fontsize=22)

    plt.suptitle('Training dashboard', fontsize=48)
    plt.savefig('training_cascvi.png')

def plot_common_ancestor(tree, z, embedding='umap', give_labels=False):
    ##
    ancestor_color = {}
    c = 0
    for i, n in enumerate(tree.traverse('levelorder')):
        n.add_features(index=i)
        if not n.is_leaf():
            ancestor_color[n.name] = c
            c += 1
    # Nodes labelling
    ancestor = {}
    for n in tree.traverse('levelorder'):
        if not n.is_root():
            ancestor[n.name] = n.up.name

    labels = {}
    for n in tree.traverse('levelorder'):
        if n.is_root():
            labels[n.name] = ancestor_color[n.name]
        else:
            a = ancestor[n.name]
            labels[n.name] = ancestor_color[a]

    if embedding == 'tsne':
        z_embedded = TSNE(n_components=2).fit_transform(z)
    elif embedding == 'umap':
        reducer = umap.UMAP()
        z_embedded = reducer.fit_transform(z)

    # plot
    L = list(labels.values())
    fig, ax = plt.subplots(figsize = (14, 7))

    scatter = ax.scatter(z_embedded[:, 0],
               z_embedded[:, 1],
               c = L,
               cmap="Spectral")

    legend1 = ax.legend(*scatter.legend_elements(num=15),
                        loc="best", title="Common Ancestor index")

    plt.title('Simulated Latent Space')
    plt.xlabel("z_1")
    plt.ylabel("z_2")

    ax.add_artist(legend1)
    plt.grid()
    plt.show()

    if give_labels:
        return L


def tree_to_nx(tree, X, g, show_index):
    """Converts ete3 tree to a Networkx graph for visualizations

    Parameters
    ----------

    Returns
    -------

    """

    label_dict = {}
    graph = nx.DiGraph()
    for i, n in enumerate(tree.traverse()):
        if n.is_root():
            continue
        graph.add_node(n.name)
        if show_index:
            label_dict[n.name] = n.index
        else:
            label_dict[n.name] = str(int(X[i][g]))
        graph.add_edge(n.up.name, n.name)
    return graph, label_dict


def plot_one_gene(tree, X, g, node_sizes, var, size, show_index, save_fig=False, figsize=(14,8)):
    graph, label_dict = tree_to_nx(tree, X, g, show_index)
    pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")

    # variance
    node_sizes_gene = []
    if var == 'gene':
        L = [int(x) for x in node_sizes.keys()]
        L.sort()
        node_sizes_gene = [size*node_sizes[str(k)][g] for k in L]
        #node_sizes_gene = [size] + node_sizes_gene

        plt.figure(1, figsize=figsize)
        nx.draw(graph, labels=label_dict, with_labels=True, pos=pos, node_size=node_sizes_gene,
                font_size=14, font_color='white')
    else:
        plt.figure(1, figsize=figsize)
        nx.draw(graph, labels=label_dict, with_labels=True, pos=pos, node_size=node_sizes,
                font_size=14, font_color='white')

    if save_fig:
        plt.savefig("graph.pdf")
    else:
        plt.show()

def plot_contour_dist(z, c):
    z_min = np.min([np.min(z[:, 0]), np.min(z[:, 1])]) - 1.
    z_max = np.max([np.max(z[:, 0]), np.max(z[:, 1])]) + 1.

    sns.set(rc={'axes.labelsize':10,
            'figure.figsize':(20.0, 10.0),
            'xtick.labelsize':10,
            'ytick.labelsize':10})

    from itertools import chain
    df = pd.DataFrame({"x":z[:, 0], "y":z[:, 1]})
    p = sns.jointplot(data=df,
                      x='x',
                      y='y',
                      kind='kde',
                      xlim=(z_min, z_max),
                      ylim=(z_min, z_max),
                      fill=True,
                      space=0,
                      color=c,
                      stat_func=None,
                      marginal_kws={'lw':2,
                                    'bw':0.2}).set_axis_labels('X','Y')

    p.ax_marg_x.set_facecolor('#ccffccaa')
    p.ax_marg_y.set_facecolor('#ccffccaa')
    for l in chain(p.ax_marg_x.axes.lines,p.ax_marg_y.axes.lines):
        l.set_linestyle('--')
        l.set_color('black')
    plt.title("Simulated Latent Space")









