import marimo

__generated_with = "0.23.3"
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
    return (
        full_table,
        methylation_site_annotations,
        methylation_site_names,
        np,
        pd,
        plt,
        sns,
        tables,
    )


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


@app.cell
def _(full_table, methylation_site_names, plt, sns):

    # Q1: Spearman correlation of each CpG site with age
    age = full_table["age"]
    methylation_sites = full_table[methylation_site_names]

    # One rho value per CpG site
    spearman_rho = methylation_sites.apply(lambda col: col.corr(age, method="spearman"))

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(spearman_rho.dropna(), bins=60)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Spearman correlation (rho) with age")
    plt.ylabel("Number of CpG sites")
    plt.title("Distribution of CpG-age Spearman correlations")
    plt.tight_layout()
    plt.show()

    # Quick numeric summary
    print(spearman_rho.describe())
    print("\nTop positive correlations:")
    print(spearman_rho.nlargest(5))
    print("\nTop negative correlations:")
    print(spearman_rho.nsmallest(5))


    # Count how many CpG sites have significant positive or negative correlation with age
    significant_positive = (spearman_rho > 0.25).sum()  #
    significant_negative = (spearman_rho < -0.25).sum()  #
    print(f"Number of CpG sites with significant positive correlation with age: {significant_positive}")
    print(f"Number of CpG sites with significant negative correlation with age: {significant_negative}")

    moderately_positive = ((0.5 > spearman_rho) & (spearman_rho > 0.25)).sum() 
    print(f"Number of CpG sites with moderate positive correlation with age: {moderately_positive}")
    return age, spearman_rho


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the dataset contains a subset of the 450K sites that showed correlation to age, we expect to see only segnificant correlation. It seems that perhaps the sites that showed correlation lower than 0.25 were removed.

    It seems that there are much more positively-correlated sites tham negatively-correlated, so we summed to get the count for each group. We got that only about a quarter of the sites are negatively-correlated, and the rest are positively correlated. We see the peak between 0.25 and 0.5 with half of the samples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.	For the 5 CpG sites with highest and lowest Spearman correlation coefficient values, visualize the relationship between methylation level and age using a scatter plot. Comment on the findings.
    """)
    return


@app.cell
def _(age, full_table, plt, sns, spearman_rho):
    # Q2: Plot top 5 positively and negatively correlated CpG sites
    top_positive_sites = spearman_rho.nlargest(5).index
    top_negative_sites = spearman_rho.nsmallest(5).index
    plt.figure(figsize=(12, 6))
    for i, site in enumerate(top_positive_sites):
        plt.subplot(2, 5, i + 1)
        sns.scatterplot(x=full_table[site], y=age)
        plt.title(f"{site} (rho={spearman_rho[site]:.2f})")
        plt.xlabel("Methylation level")
        plt.ylabel("Age")
    for i, site in enumerate(top_negative_sites):
        plt.subplot(2, 5, i + 6)
        sns.scatterplot(x=full_table[site], y=age)
        plt.title(f"{site} (rho={spearman_rho[site]:.2f})")
        plt.xlabel("Methylation level")
        plt.ylabel("Age")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As expected the 5 most-positively and negatively correlated sites exhibit clear trend when visuallising age over methylation level.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3.	Select all sites located along Chromosome 10 and reorder them by their location along the genome. Calculate the matrix of Spearman correlation coefficients between all pairs of methylation levels and visualize as a heatmap. Comment on the findings.
    """)
    return


@app.cell
def _(full_table, methylation_site_annotations, plt, sns):
    # Q3: Get all sites on chromosome 10 by order of position, and plot a heatmap of the pairwise Spearman correlation
    # for the methylation levels of these sites across samples.
    chr10_sites = methylation_site_annotations[methylation_site_annotations['CHR'] == 10].sort_values('MAPINFO').index
    chr10_methylation = full_table[chr10_sites]
    chr10_spearman = chr10_methylation.corr(method='spearman')
    plt.figure(figsize=(10, 8))
    # labels should be "site name (position)"
    sns.heatmap(chr10_spearman, cmap='coolwarm', center=0)
    plt.title("Pairwise Spearman correlation of methylation levels for CpG sites on chromosome 10")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It appears that there are multiple areas of high correlation between sites cg10530889 and cg18361940 (shows a relatively dark-red square), that hints there might be a spatial effect between the consecutive positions between them.

    The range at the end of the chromosome seems more distorted, meanning the positions there have different pattenrs of correlation, suggestiong they dont have a strong spatial dependence.

    we can also look at positions such as cg04200607 that show low correlation with other positions sampled along the entire chromosome.
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
    return X_test, X_train, Y_test, Y_train, Y_train_binned, skf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4.	Report the number of samples per study and visualize the age distribution for each of the 19 studies, using a boxplot. Comment on the findings.
    """)
    return


@app.cell
def _(full_table, plt, sns, tables):
    plt.figure(figsize=(15, 6))

    positions = []
    labels = []

    for j, t in enumerate(tables):
        source = t.iloc[0]["Source"]

        sns.boxplot(
            y=t["age"],
            positions=[j],
            width=0.6
        )

        positions.append(j)
        labels.append(source)

    # add one boxplot for all the data
    sns.boxplot(
        y=full_table["age"],
        positions=[len(tables)],
        width=0.6
    )
    positions.append(len(tables))
    labels.append("All")

    plt.xlim(-0.5, len(tables) + 1 - 0.5)
    plt.xticks(positions, labels, rotation=45)
    plt.title("Age distribution by Source")
    plt.tight_layout()
    plt.show()

    for j, t in enumerate(tables):
        source = t.iloc[0]["Source"]
        print(f"Number of {source} samples: {len(t)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    4. We can see that some of the datasets have a very different distribution than the rest. like GSE103657 that consist of samples of infants, or GSE30870 that consist only of 90-100 year old patients. GSE69270 may have the same median as the avarage median across the datasets, but it has a very small variance.
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
    5a. We added a boxplot that shows age distribution for the entire data. We see that for the selected test dataset, the median is vert simmilar to the median of the entire data. It is clearly visible that there are dataset that do not represent the distribution of the entire data either by range, median or both - for example GSE103657. Sampleing them as the test set may result in poor prediction.

    5b. It is a good practice to try and train the model based on expected real-world age distribution. using stratified k folds helps maintaining the same age distribution across all the folds.
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


@app.cell
def _(X_test, X_train, Y_test, Y_train, Y_train_binned, np, pd, skf):
    # Create an sklearn pipline that:
    #  - Imputes missing values with SimpleImputer
    #  - Selects top-k features by correlation with age (SelectKBest and f_regression) via cross-validation on the training set
    from wheel.vendored.packaging.specifiers import Specifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Ridge

    k_values = [5,10,20,30,50,100,500,1000]

    cv_splits = list(skf.split(X_train, Y_train_binned))


    best_models = []
    for k in k_values:
        pipeline = Pipeline([
            ('imputer', SimpleImputer()),
            ('feature_selection', SelectKBest(score_func=f_regression, k=k)),
            ("regressor", Ridge())
        ])

        param_grid = {'regressor__alpha': np.logspace(-4, 2, 10)}
        grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_splits,
        scoring="neg_mean_absolute_error"
    )
        # grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring="neg_mean_absolute_error")
    
        grid_search.fit(X_train, Y_train)

        best_model = grid_search.best_estimator_
        best_performance = -grid_search.best_score_
        print(f"Best MAE for k={k}: {best_performance:.4f} with alpha={grid_search.best_params_['regressor__alpha']}")
        best_models.append(best_model)

    # Report performance of the model on the test set in a table with error bars: std of abs error / sqrt(n_test_samples).
    report = []
    for k, model in zip(k_values, best_models):
        predictions = model.predict(X_test)
        mae = np.mean(np.abs(predictions - Y_test))
        std_error = np.std(np.abs(predictions - Y_test)) / np.sqrt(len(Y_test))
        report.append((k, mae, std_error))
    report_df = pd.DataFrame(report, columns=['k', 'MAE', 'Std_Error'])
    print(report_df)


    return SelectKBest, SimpleImputer, best_models, cv_splits, f_regression


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The performence slightly improves along growing K, both in MAE and error bars. Allowing small groups of features the models can't grasp the required complexity expressed in the feartures to fit well. That is probably why it improves as we incease k, howerver there is a point where the improvment in performence flattens. We can see that the L2 regulation limits from choosing more features for greater values of K, which may explain why at this point the MAE decreasement is insignificant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    7.	For K=50, plot the feature importance of the selected sites.
    Are some chromosomes over-represented among these sites?
    """)
    return


@app.cell
def _(X_train, best_models, methylation_site_annotations, np, plt, sns):
    # Plot feature importance for K=50 model, from the regression coefficients of the Ridge model,
    # sort features by genome position and color labels by chromosome
    # use methylation_site_annotations to get the chromosome and position of each feature.
    k_features_pool = best_models[4].named_steps['feature_selection'].get_support(indices=True)
    selected_features = X_train.columns[k_features_pool]
    selected_annotations = methylation_site_annotations.loc[selected_features]
    selected_annotations = selected_annotations.sort_values(['CHR', 'MAPINFO'])

    # For non gradient coloring
    selected_annotations['CHR'] = selected_annotations['CHR'].astype(int).astype(str)
    plt.figure(figsize=(10, 6))

    unique_chrs = sorted(selected_annotations['CHR'].unique())

    palette = dict(zip(
        unique_chrs,
        sns.color_palette("tab20", n_colors=len(unique_chrs))
    ))


    sns.barplot(x=selected_annotations.index, y=np.abs(best_models[4].named_steps['regressor'].coef_),
    hue=selected_annotations['CHR'], dodge=False, palette=palette)
    plt.xticks(rotation=90)
    plt.xlabel("CpG site")
    plt.ylabel("Absolute regression coefficient")
    plt.title("Feature importance for K=50 model")
    plt.legend(title="Chromosome", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some chromosomes are more represented then others. chromosomes 9 and 12 arn't even chosen, and 17,18,20 for instance have single locus present with a very low importance. Chromosome 10 has 2 sites one of wich is the most important feature (twice in size from the next-important). Chromosome 3 has the most sites but with a relatively low significance, may be interesting to check their respective correlation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    8.	For each of the five folds, the selected sites may be different. How many sites are found in all five folds? In at least 2 folds? Discuss the findings.
    """)
    return


@app.cell
def _(
    SelectKBest,
    SimpleImputer,
    X_train,
    Y_train_binned,
    cv_splits,
    f_regression,
    pd,
):
    # Q8: Count for each features in how many folds it was selected by SelectKBest, and create a df
    feature_counts = pd.DataFrame(0, index=X_train.columns, columns=range(5))
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_fold_train, Y_fold_train = X_train.iloc[train_idx], Y_train_binned[train_idx]
        # Impute missing values in the training fold
        X_fold_train = SimpleImputer(strategy='mean').fit_transform(X_fold_train)
        selector = SelectKBest(score_func=f_regression, k=50)
        selector.fit(X_fold_train, Y_fold_train)
        selected_indices = selector.get_support(indices=True)
        feature_counts.iloc[selected_indices, fold] = 1
    feature_counts['Total_Selected'] = feature_counts.sum(axis=1)

    # show features that were selected in all 5 folds
    consistently_selected = feature_counts[feature_counts['Total_Selected'] == 5]
    print("No. Features selected in all 5 folds:", len(consistently_selected))


    # show features that were selected in at least 2 folds
    selected_at_least_2 = feature_counts[feature_counts['Total_Selected'] >= 2]
    print("No. Features selected in at least 2 folds:", len(selected_at_least_2))

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
