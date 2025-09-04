from math import sqrt
import pandas as pd
import numpy as np    
from scipy.stats import poisson, chi2
from typing import Dict, List, Union, Tuple

def aggregate_columns(df, column_groups, operation='sum') -> pd.DataFrame:
    """
    Aggregate multiple columns in a DataFrame with specified operations.
    
    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to process
    column_groups: Dict[str, List[str]]
        Dictionary mapping output column names to lists of input column names
        Example: {'seq_target_sum': ['seq_target_1', 'seq_target_2', 'seq_target_3']}
    operation: str, default 'sum'
        The aggregation operation to perform. Options:
        - 'sum': Sum of columns
        - 'mean': Average of columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new aggregated columns added
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'seq_target_1': [1, 2, 3],
    ...     'seq_target_2': [4, 5, 6],
    ...     'seq_target_3': [7, 8, 9]
    ... })
    >>> result = aggregate_columns(df, {'target_sum': ['seq_target_1', 'seq_target_2', 'seq_target_3']}, 'sum')
    >>> print(result['target_sum'].tolist())
    [12, 15, 18]
    """
    
    # Validate operation
    valid_operations = ['sum', 'mean']
    if operation not in valid_operations:
        raise ValueError(f"Invalid operation '{operation}'. Must be one of: {valid_operations}")
    
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Process each column group
    for output_name, input_columns in column_groups.items():
        # Validate that all input columns exist
        missing_columns = [col for col in input_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
        
        # Perform the aggregation
        if operation == 'sum':
            result_df[output_name] = df[input_columns].sum(axis=1)
        elif operation == 'mean':
            result_df[output_name] = df[input_columns].mean(axis=1)
    
    return result_df


def calculate_enrichment_score(row, total_sum, row_count, column_name):
    """
    This transformation function calculates the enrichment score for each row

    Parameters
    ----------
    row: pandas.Series
        The row to be processed and calculated for enrichment score

    Returns
    -------
    z_score: float
        The enrichment score for the row
    """
    # get number of rows
    p0 = row[column_name] / total_sum
    p1 = 1 / row_count
    z_score = (p0 - p1) / sqrt((p1 * (1 - p1)))
    return z_score


def calculate_hit_threshold(df, column_name, percentile) -> float:
    """
    This function calculates the hit threshold for a given column
    """
    return np.percentile(df[column_name], percentile)


def poissfit(vec, alpha=0.05) -> Tuple[float, float]:
    """Calculate Poisson confidence intervals."""
    k_sum, n = vec.sum(), len(vec)
    lower = 0.5 * chi2.ppf(alpha/2, 2*k_sum) / n
    upper = 0.5 * chi2.ppf(1-alpha/2, 2*(k_sum+1)) / n
    return (lower, upper)


def get_enrichment_ratio(row, control_cols, target_cols) -> float:
    """Calculate enrichment ratio."""
    _, c_upper = poissfit(row[control_cols])
    t_lower, _ = poissfit(row[target_cols])
    return t_lower / c_upper


def calculate_poisson_enrichment(df, control_cols, target_cols, alpha=0.05) -> pd.DataFrame:
    """
    Calculate Poisson enrichment scores for combined datasets.
    
    This function replicates the Poisson enrichment calculation from tmp_disynthon.ipynb.
    
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
    >>> control_cols = ['matrix1_Sum_Counts', 'matrix2_Sum_Counts', 'matrix3_Sum_Counts']
    >>> target_cols = ['target1_Sum_Counts', 'target2_Sum_Counts', 'target3_Sum_Counts']
    >>> enriched_df = calculate_poisson_enrichment(df, control_cols, target_cols)
    """
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Convert columns to float for calculations
    sub_df = result_df[control_cols + target_cols].astype(float)
    
    # Calculate enrichment scores
    result_df['Poisson_Enrichment'] = sub_df.apply(get_enrichment_ratio, axis=1)
    
    return result_df
