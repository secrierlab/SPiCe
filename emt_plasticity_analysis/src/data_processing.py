import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data():
    xenium_labels = pd.read_csv('results/xenium_aligned_scores.csv', index_col=0)
    #if filtering for just DCIS
    #xenium_labels = pd.read_csv('results/xenium_aligned_scores_just_dcis.csv.csv', index_col=0)
    clone_info = pd.read_csv('results/filtered_data_with_cnv.csv', index_col=0)
    pca_data = pd.read_csv('results/X_pca_df.csv', index_col=0)
    pca_data = pca_data.drop(['array_row', 'array_col'], axis=1)
    xenium_labels = xenium_labels.drop(['EMT_hallmarks'], axis=1)
    merged_pca = pd.merge(xenium_labels, pca_data, how='left', on='cell_id')
    pca_columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
    scaler = MinMaxScaler()
    merged_pca[pca_columns] = scaler.fit_transform(merged_pca[pca_columns])
    merged_pca[pca_columns] = merged_pca[pca_columns].fillna(0)
    xenium_labels = pd.merge(merged_pca, clone_info[['subclone', 'cell_id']], how="left", on="cell_id")
    return xenium_labels, pca_columns

def assign_emt_labels(df, four_states=False):
    # Filter cells with non-NaN subclone info
    tumour_filter = df[df['subclone'].notna()]
    labels = tumour_filter['EMT_hallmarks']
    df['labels'] = np.nan
    indices_to_keep = tumour_filter.index
    if four_states:
        lower_quantile, middle_quantile, upper_quantile = labels.quantile([0.25, 0.50, 0.75])
        df.loc[indices_to_keep[labels <= lower_quantile], 'labels'] = 0  # Low
        df.loc[indices_to_keep[(labels > lower_quantile) & (labels <= middle_quantile)], 'labels'] = 1  # Low-Medium
        df.loc[indices_to_keep[(labels > middle_quantile) & (labels <= upper_quantile)], 'labels'] = 2  # Medium-High
        df.loc[indices_to_keep[labels > upper_quantile], 'labels'] = 3  # High
        mapping = {0: 'low', 1: 'low-medium', 2: 'medium-high', 3: 'high'}
    else:
        median_value = labels.median()
        df.loc[indices_to_keep[labels <= median_value], 'labels'] = 0  # Low
        df.loc[indices_to_keep[labels > median_value], 'labels'] = 1  # High
        mapping = {0: 'low', 1: 'high'}
        df.loc[df['labels'].isnull(), 'EMT_hallmarks'] = np.nan
    return df, mapping
