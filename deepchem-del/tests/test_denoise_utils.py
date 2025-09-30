import pandas as pd
import numpy as np

from utils.denoise_utils import (
    calculate_normalized_enrichment_score,
    calculate_poisson_enrichment,
)


def test_normalized_enrichment_calculation():
    """Test normalized enrichment score calculation."""
    test_df = pd.read_csv('tests/assets/test_dataset.csv')
    row = test_df.iloc[0]
    total_sum = test_df['seq_target_1'].sum()
    row_count = test_df.shape[0]
    column_name = 'seq_target_1'
    expected = -0.1414213562373095
    result = calculate_normalized_enrichment_score(row, total_sum, row_count,
                                                   column_name)
    assert np.allclose(result, expected,
                       atol=1e-10), f"Expected {expected}, got {result}"


def test_poisson_enrichment_calculation():
    """Test Poisson enrichment calculation."""
    test_df = pd.read_csv('tests/assets/test_dataset.csv')
    control_cols = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3']
    target_cols = ['seq_target_1', 'seq_target_2', 'seq_target_3']
    result = calculate_poisson_enrichment(test_df, control_cols, target_cols)

    assert 'Poisson_Enrichment' in result.columns
    assert np.allclose(result['Poisson_Enrichment'],
                       result['target_enrichment'],
                       atol=1e-10)
