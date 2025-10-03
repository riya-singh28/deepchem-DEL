import numpy as np
import pandas as pd
from math import sqrt
from rdkit import Chem
from scipy.stats import chi2
from typing import Tuple, Dict, List
from utils.logging_utils import setup_logging

logger = setup_logging(__name__)


def calculate_normalized_enrichment_score(row: pd.Series, total_sum: float, row_count: int, column_name: str) -> float:
    """
    This transformation function calculates the enrichment score for each row

    Parameters
    ----------
    row: pandas.Series
        The row to be processed and calculated for enrichment score
    total_sum: float
        The total sum of the column
    row_count: int
        The number of rows
    column_name: str
        The name of the column to be processed

    Returns
    -------
    z_score: float
        The enrichment score for the row

    Examples
    --------
    >>> df = pd.DataFrame({
    ...                    'seq_target_1': [10, 20, 30],
    ...                    'seq_target_2': [40, 50, 60],
    ...                    'seq_target_3': [70, 80, 90]})
    >>> row = df.iloc[0]
    >>> total_sum = 60
    >>> row_count = 3
    >>> column_name = 'seq_target_1'
    >>> result = calculate_normalized_enrichment_score(row, total_sum, row_count, column_name)
    >>> result
    -0.35355339059327373
    """
    # get number of rows
    p0 = row[column_name] / total_sum
    p1 = 1 / row_count
    z_score = (p0 - p1) / sqrt((p1 * (1 - p1)))
    return z_score

def calculate_hit_threshold(df: pd.DataFrame, column_name: str, percentile: float) -> float:
    """
    This function calculates the hit threshold for a given column

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be processed
    column_name: str
        The name of the column to be processed
    percentile: float
        The percentile to be calculated

    Returns
    -------
    float
        The hit threshold for the given column

    Examples
    --------
    >>> df = pd.DataFrame({'seq_target_sum': [10, 20, 30]})
    >>> column_name = 'seq_target_sum'
    >>> percentile = 90
    >>> result = calculate_hit_threshold(df, column_name, percentile)
    """
    return np.percentile(df[column_name], percentile)

def poissfit(vec: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate Poisson confidence intervals.

    Parameters
    ----------
    vec: pd.Series
        The vector to be processed
    alpha: float, default 0.05
        Significance level for Poisson confidence intervals

    Examples
    --------
    >>> vec = pd.Series([10, 20, 30])
    >>> alpha = 0.05
    >>> result = poissfit(vec, alpha)
    >>> result
    (15.262106983335755, 25.743964468881106)
    """
    k_sum, n = vec.sum(), len(vec)
    lower = 0.5 * chi2.ppf(alpha / 2, 2 * k_sum) / n
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k_sum + 1)) / n
    return (lower, upper)

def get_enrichment_ratio(row: pd.Series, control_cols: List[str], target_cols: List[str]) -> float:
    """Calculate enrichment ratio.

    Parameters
    ----------
    row: pd.Series
        The row to be processed
    control_cols: List[str]
        The list of control columns
    target_cols: List[str]
        The list of target columns

    Returns
    -------
    float
        The enrichment ratio

    Examples
    --------
    >>> df = pd.DataFrame({
    ...                    'seq_matrix_1': [10, 20, 30],
    ...                    'seq_matrix_2': [40, 50, 60],
    ...                    'seq_matrix_3': [70, 80, 90],
    ...                    'seq_target_1': [10, 20, 30],
    ...                    'seq_target_2': [40, 50, 60],
    ...                    'seq_target_3': [70, 80, 90]})
    >>> row = df.iloc[0]
    >>> control_cols = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3']
    >>> target_cols = ['seq_target_1', 'seq_target_2', 'seq_target_3']
    >>> result = get_enrichment_ratio(row, control_cols, target_cols)
    >>> result
    0.6933690135129165
    """
    _, c_upper = poissfit(row[control_cols])
    t_lower, _ = poissfit(row[target_cols])
    return t_lower / c_upper

def calculate_poisson_enrichment(df: pd.DataFrame,
                                 control_cols: List[str],
                                 target_cols: List[str],
                                 alpha: float = 0.05) -> pd.DataFrame:
    """
    Calculate Poisson enrichment scores for combined datasets.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the combined data
    control_cols: List[str]
        List of column names for control (matrix) data
    target_cols: List[str]
        List of column names for target data
    alpha: float, default 0.05
        Significance level for Poisson confidence intervals

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Poisson_Enrichment' column

    Examples
    --------
    >>> df = pd.DataFrame({
    ...                    'seq_matrix_1': [10, 20, 30],
    ...                    'seq_matrix_2': [40, 50, 60],
    ...                    'seq_matrix_3': [70, 80, 90],
    ...                    'seq_target_1': [10, 20, 30],
    ...                    'seq_target_2': [40, 50, 60],
    ...                    'seq_target_3': [70, 80, 90]})
    >>> control_cols = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3']
    >>> target_cols = ['seq_target_1', 'seq_target_2', 'seq_target_3']
    >>> result = calculate_poisson_enrichment(df, control_cols, target_cols)
    >>> result['Poisson_Enrichment'].tolist()
    [0.6933690135129165, 0.72127147661966, 0.742496160144341]
    """

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Convert columns to float for calculations
    sub_df = result_df[control_cols + target_cols].astype(float)

    # Calculate enrichment scores
    result_df['Poisson_Enrichment'] = sub_df.apply(lambda row: get_enrichment_ratio(row, control_cols, target_cols),
                                                   axis=1)

    return result_df

def create_disynthon_pairs(
    df: pd.DataFrame,
    smiles_cols: List[str],
    control_cols: List[str] = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'],
    target_cols: List[str] = ['seq_target_1', 'seq_target_2', 'seq_target_3'],
    is_unified: bool = True,
) -> pd.DataFrame:
    """
    Creates disynthon pairs from the trisynthon data.

    In the non-unified setting, each replicate is treated as an independent runs 
    under identical experimental conditions. Accordingly, the three replicates of 
    the target are combined into one large experiment, while the three replicates 
    of the control are combined into a separate large experiment, after which 
    denoising is performed.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be processed
    smiles_cols: List[str]
        The list of smiles columns
    control_cols: List[str], default ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3']
        The list of control columns
    target_cols: List[str], default ['seq_target_1', 'seq_target_2', 'seq_target_3']
        The list of target columns
    is_unified: bool, default True
        Whether the data is unified

    Returns
    -------
    pd.DataFrame
        The DataFrame with the disynthon pairs

    Examples
    --------
    >>> df = pd.DataFrame({'smile_a': ['CCOc1ccc(CN)cc1', 'CCN(CC)CCOc1ccc(Cl)cc1', 'O=C(Nc1ccccc1)C2CC2'],
    ...                    'smile_b': ['CC1=CC=CC=C1C(=O)O', 'C1CCN(CC1)C(=O)C2=CC=CC=C2', 'CNC(=O)c1ccc(F)cc1'],
    ...                    'smile_c': ['CC(C)NCC(O)CO', 'Clc1ccc(cc1)C(=O)N2CCCC2', 'C1=CC=C(C=C1)C2=CN=CN2'],
    ...                    'seq_matrix_1': [10, 20, 30],
    ...                    'seq_matrix_2': [40, 50, 60],
    ...                    'seq_matrix_3': [70, 80, 90],
    ...                    'seq_target_1': [10, 20, 30],
    ...                    'seq_target_2': [40, 50, 60],
    ...                    'seq_target_3': [70, 80, 90]})
    >>> control_cols = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3']
    >>> target_cols = ['seq_target_1', 'seq_target_2', 'seq_target_3']
    >>> smiles_cols = ['smile_a', 'smile_b', 'smile_c']
    >>> disynthon_data, smiles_dict = create_disynthon_pairs(df, smiles_cols, control_cols, target_cols, is_unified=True)
    """
    smiles_set = set()
    for column_name in df.columns:
        if column_name in smiles_cols:
            smiles_set.update(df[column_name].dropna())

    smiles_list = list(smiles_set)
    smiles_dict = {smiles: f'{i}' for i, smiles in enumerate(smiles_list)}

    df[smiles_cols[0]] = df['smile_a'].apply(lambda x: smiles_dict[x] if x in smiles_dict else None)
    df[smiles_cols[1]] = df['smile_b'].apply(lambda x: smiles_dict[x] if x in smiles_dict else None)
    df[smiles_cols[2]] = df['smile_c'].apply(lambda x: smiles_dict[x] if x in smiles_dict else None)

    df = df[~df.isna().any(axis=1)]

    if is_unified:
        pair_sums = {}
        for i in range(len(smiles_cols)):
            for j in range(i, len(smiles_cols)):
                col1 = smiles_cols[i]
                col2 = smiles_cols[j]
                if col1 != col2:
                    pair_label = f"{col1}_{col2}_Pair"
                    pair_df = df.groupby([col1, col2]).agg({
                        control_cols[0]: 'sum',
                        control_cols[1]: 'sum',
                        control_cols[2]: 'sum',
                        target_cols[0]: 'sum',
                        target_cols[1]: 'sum',
                        target_cols[2]: 'sum',
                    }).reset_index()
                    pair_df.rename(columns={col1: 'Disynthon_1'}, inplace=True)
                    pair_df.rename(columns={col2: 'Disynthon_2'}, inplace=True)
                    pair_sums[pair_label] = pair_df
    else:
        df['seq_target_sum'] = df[target_cols[0]] + df[target_cols[1]] + df[target_cols[2]]
        df['seq_control_sum'] = df[control_cols[0]] + df[control_cols[1]] + df[control_cols[2]]

        pair_sums = {}
        for i in range(len(smiles_cols)):
            for j in range(i, len(smiles_cols)):
                col1 = smiles_cols[i]
                col2 = smiles_cols[j]
                if col1 != col2:
                    pair_label = f"{col1}_{col2}_Pair"
                    pair_df = df.groupby([col1, col2]).agg({
                        'seq_target_sum': 'sum',
                        'seq_control_sum': 'sum'
                    }).reset_index()
                    pair_df.rename(columns={col1: 'Disynthon_1'}, inplace=True)
                    pair_df.rename(columns={col2: 'Disynthon_2'}, inplace=True)
                    pair_sums[pair_label] = pair_df

    # Concatenate all pair DataFrames
    pair_sums_df = pd.concat(pair_sums.values(), ignore_index=True)
    return pair_sums_df, smiles_dict

def get_disynthons_from_pairs(D1_idx: str, D2_idx: str, failed_smiles: set, failed_combines: set,
                              smiles_dict_inv: dict) -> str:
    """
    Get disynthons from pairs.

    Parameters
    ----------
    D1_idx: str
        The index of the first disynthon
    D2_idx: str
        The index of the second disynthon
    failed_smiles: set
        The set of failed smiles
    failed_combines: set
        The set of failed combines
    smiles_dict_inv: dict
        The inverse of the smiles dictionary

    Returns
    -------
    str
        The disynthon

    Examples
    --------
    >>> D1_idx = '1'
    >>> D2_idx = '2'
    >>> failed_smiles = set()
    >>> failed_combines = set()
    >>> smiles_dict_inv = {'1': 'C1', '2': 'C2'}
    >>> result = get_disynthons_from_pairs(D1_idx, D2_idx, failed_smiles, failed_combines, smiles_dict_inv)
    """
    try:
        smi_1 = smiles_dict_inv[D1_idx]
        smi_2 = smiles_dict_inv[D2_idx]
        try:
            mol1 = Chem.MolFromSmiles(smi_1)
        except Exception as e:
            failed_smiles.add((smi_1, D1_idx))
            return None
        try:
            mol2 = Chem.MolFromSmiles(smi_2)
        except Exception as e:
            failed_smiles.add((smi_2, D2_idx))
            return None
        try:
            combined_molecule = Chem.CombineMols(mol1, mol2)
            return Chem.MolToSmiles(combined_molecule)
        except Exception as e:
            failed_combines.add((smi_1, smi_2, D1_idx, D2_idx))
            return None
    except Exception:
        logger.exception("Failed to combine disynthons", extra={"D1_idx": D1_idx, "D2_idx": D2_idx})
        return None
