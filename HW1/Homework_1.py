import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building an epigenetic methylation clock with scikit-learn

    Reference articles:
    - Horvath, Genome Biology 2013: https://genomebiology.biomedcentral.com/articles/10.1186/gb-2013-14-10-r115#MOESM21
    - Varshavsky et al. Cell Reports Methods 2023: https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(23)00211-4
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data loading & inspection
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    folder = 'data/'
    datasets = [ x for x in os.listdir(folder) if x.endswith('.csv') and x.startswith('GSE')]

    tables = []

    for dataset in datasets:
        name = dataset.split('.')[0]
        table = pd.read_csv(os.path.join(folder,dataset),index_col=0).transpose()
        table['Source'] = name
        print(dataset,table.shape)
        tables.append(table)

    full_table = pd.concat(tables,axis=0,join='outer')


    methylation_site_names = [x for x in full_table.columns if x not in ['age','Source']]

    if os.path.exists(os.path.join(folder, 'site_annotations.csv')):    
        methylation_site_annotations = pd.read_csv(os.path.join(folder, 'site_annotations.csv'),index_col=0)    
    else:
        methylation_site_annotations = pd.read_csv('humanmethylation450_15017482_v1-2.csv',skiprows=7,index_col='IlmnID')
        methylation_site_annotations = methylation_site_annotations.loc[methylation_site_names][['CHR','MAPINFO','Strand','SourceSeq','UCSC_RefGene_Name']]
        methylation_site_annotations.to_csv(os.path.join(folder, 'site_annotations.csv'))
    return full_table, methylation_site_annotations, methylation_site_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspecting the dataset

    Pay attention that for some individuals, not all methylation levels were measured.
    """)
    return


@app.cell
def _(full_table):
    full_table.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspecting the metadata for each methylation site.
    - IlmnID: Name.
    - CHR: Chromosome.
    - MAPINFO: start position along the reference genome.
    - STRAND: on which strand is the CpG site located.
    - Source: The corresponding DNA fragment
    """)
    return


@app.cell
def _(methylation_site_annotations):
    methylation_site_annotations.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part I: Exploratory Data Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1.	For each CpG site, calculate the Spearman correlation coefficient between its methylation level and the age of the donor. Display the result as a histogram and comment on the findings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.	For the 5 CpG sites with highest and lowest Spearman correlation coefficient values, visualize the relationship between methylation level and age using a scatter plot. Comment on the findings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3.	Select all sites located along Chromosome 10 and reorder them by their location along the genome. Calculate the matrix of Spearman correlation coefficients between all pairs of methylation levels and visualize as a heatmap. Comment on the findings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part II: Data Partition

    Here, we perform the data partition, splitting the dataset into a training and a test set. We then split the training set into five folds for cross-validation purpose.

    - The dataset was compiled from 19 separate studies. To account for potential "batch effects" (i.e., distribution shift from one study to the other), we use one of the study as test set.

    - For the cross-validation, we use a stratified K-fold approach.
    """)
    return


@app.cell
def _(full_table, methylation_site_names):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import KBinsDiscretizer
    X , Y = full_table[methylation_site_names], full_table['age']
    study = full_table['Source'] # The study chosen from.

    test_set = (study == 'GSE84727')

    X_train, Y_train = X[~test_set], Y[~test_set]
    X_test, Y_test = X[test_set], Y[test_set]


    Y_train_binned = KBinsDiscretizer(n_bins=20,encode='ordinal').fit_transform(Y_train.to_numpy()[:,None])[:,0]
    skf = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    skf.get_n_splits(X_train,Y_train_binned)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4.	Report the number of samples per study and visualize the age distribution for each of the 19 studies, using a boxplot. Comment on the findings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    5.	Read how the partition between train and test was done on the notebook.
    a.	Why did we use dataset “GSE84727” as a test set rather than by using a random split?
    b.	Why did we use a stratified split for the training set rather than a random split or a grouped split?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part III: Training and evaluating a linear model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Build a scikit-learn Pipeline consisting of:
    -	An imputer for missing values (SimpleImputer)
    -	Top-K feature selection using the F-statistics of the correlation coefficient (functions: SelectKBest and f_regression).
    -	Ridge Regression.
    For each of the following values of K: [5,10,20,30,50,100,500,1000], select the optimal L2 regularization strength over the range by cross-validation.

    6.	Report the performance of the model on the test set as a table with error bars (defined as the standard deviation of the absolute error, divided by the square root of the test set size).
    How does the performance evolve with K?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    7.	For K=50, plot the feature importance of the selected sites.
    Are some chromosomes over-represented among these sites?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    8.	For each of the five folds, the selected sites may be different. How many sites are found in all five folds? In at least 2 folds? Discuss the findings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part IV: Automated feature selection by LASSO
    Replace the previous Pipeline by a Pipeline consisting of:
    -	An imputer for missing values (SimpleImputer)
    -	LASSO Regression.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    9.	For various values of the L1 regularization strength, fit a model (on the full training set) and calculate the performance on the test set as well as the number of CpG sites with non-zero coefficients. Plot the test set performance as function of the number of sites for both models.

    Why would feature selection with LASSO outperform feature selection by Pearson correlation? Conversely, why would Ridge outperform LASSO? Suggest a “best-of-both worlds” solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    10.	Why would feature selection with LASSO outperform feature selection by Pearson correlation? Conversely, why would Ridge outperform LASSO? Suggest a “best-of-both worlds” solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part V: Experimenting with the loss function.

    Since our target metric is the MAE rather than the mean square error, we might get better results by using it as a training loss function. This is implemented in two scikit-learn classes:
    (1)	QuantileRegression (with quantile=0.5) and L1 regularization. Exact, but slow optimization by linear programming.
    (2)	SGDRegression (with loss=’huber’,epsilon=1e-2) and L1/L2 regularization. Faster, approximate optimization by stochastic gradient descent. Pay attention that the learning rate needs to be adjusted.

    11.	Repeat question 6. with the MAE loss rather than Ridge regression and compare the findings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part VI: Training a Generalized Additive Model (GAM)

    The exploratory data analysis showed a non-linear relationship between methylation level and age, suggesting that a GAM model may be more appropriate. To this end, build a Pipeline consisting of the following steps:
    -	An imputer for missing values (SimpleImputer)
    -	Top-K feature selection using the F-statistics of the correlation coefficient (functions: SelectKBest and f_regression).
    -	SplineTransformer using cubic splines, uniformly spaced knots, linear extrapolation.
    -	Ridge regression.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    12.	For each of the following values of K: [5,10,20,30,50], select the optimal regularization strength and number of knots, and report the performance over the test set. Compare the results with the ones of Part III.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    13.	Plot the feature effect functions using the Partial Dependency plot function of scikit-learn. Conclude on the benefits of using trainable non-linearities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    14.	Do you expect that performance could be improved by instead implementing the GAM with boosted trees of depth 1?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part VII: Glass-box vs Black-Box models.

    15.	Similarly train and evaluate a “black-box” Random Forest regressor on the same dataset. How does the performance compare with the one of the “glass-box” models? Conclude on the merits of each method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus
    Suggest and implement another approach to further improve the performance of the age predictor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
