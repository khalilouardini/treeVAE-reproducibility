========================================
treeVAE-Reproducibility
========================================

Reproducing results in the "Reconstructing unobserved cellular states from  paired single-cell lineage tracing and transcriptomics data" [paper](www.biorxiv.org/content/10.1101/2021.05.28.446021v1?fbclid=IwAR17_FzhH-SyeU3LE-JyohwntDvAfnPP-s0qkHxATDE0CRfVs-MJ3zVipnE) ,  accepted at the ICML 2021 Workshop on Computational Biology. 

Contact
======

ouardini.k@gmail.com

Datasets
======

1. Gaussian process factor analysis (GPFA)

    * The simulated tree topologies used in the GPFA experiments are stored in 'scvi/data/topologies/100cells'.
    * The code for the simulations is in 'scvi/dataset/ppca.py'.

2. Gaussion process Poisson Log Normal (GPPLN)

    * The simulated tree topologies used in the GPPLN experiments are stored in 'scvi/data/topologies/500cells'.
    * The code for the simulations is in 'scvi/dataset/poisson_glm.py'.

3. Metastasis 

    * The tree toplogy fot the cancer metastasis dataset is stored in 'scvi/data/metastasis/lg7_tree_hybrid_priors.alleleThresh.processed.ultrametric.annotated.tree'
    * The gene expression data (603 cells, 100 genes) is stored in 'scvi/data/metastasis/Metastasis_lg7_100g.npy'

System requirements
======
    + Python 3
    + Pytorch

Installation guide
======
    + Clone the github repository, install the dependencies in 'requirements.txt'.

Instructions to reproduce experiments
======
    + for the GPFA experiments, run 'python3 gaussian_ancestral_imputation.py'
    + for the GPPLN experiments, run 'python3 ancestral_imputation.py'
+ for the metastasis data analysis, symply follow instructions in the notebook 'scvi/external/notebooks/Metastasis.ipynb'

At the end of each run, the raw results will be stored in *csv* format in a *results/* folder, with sub-directories automatically created in reference
to the hyperparameters used in each experiment. to reproduce the tables and figures of the paper, simply follow the instructions
in *scvi/external/notebooks/plot_results.ipynb*.

