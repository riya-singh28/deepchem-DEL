import sys
from pathlib import Path
import pandas as pd
import pytest

from utils.denoise_utils import (
    aggregate_columns,
    calculate_enrichment_score,
    calculate_poisson_enrichment,
    get_enrichment_ratio,
    poissfit,
    calculate_hit_threshold,
)


@pytest.mark.parametrize("row, total_sum, row_count, column_name, expected", [
    (pd.Series({'count': 10}), 100, 5, 'count', -0.25),
    (pd.Series({'count': 50}), 100, 5, 'count', 0.75),
    (pd.Series({'count': 5}), 100, 5, 'count', -0.375),
])
def test_basic_enrichment_calculation(row, total_sum, row_count, column_name, expected):
    """Test basic enrichment score calculation with known values."""
    result = calculate_enrichment_score(row, total_sum, row_count, column_name)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    assert result > 0 if expected > 0 else result < 0, "Enrichment should be positive or negative"


def test_poisson_enrichment_calculation():
    """Test Poisson enrichment calculation."""
    test_df = pd.DataFrame({
        'Smiles': ['CCO', 'CCN', 'CCC'],
        'matrix1_Sum_Counts': [10, 20, 30],
        'matrix2_Sum_Counts': [15, 25, 35],
        'target1_Sum_Counts': [5, 10, 15],
        'target2_Sum_Counts': [8, 12, 18]
    })
    control_cols = ['matrix1_Sum_Counts', 'matrix2_Sum_Counts']
    target_cols = ['target1_Sum_Counts', 'target2_Sum_Counts']
    result = calculate_poisson_enrichment(test_df, control_cols, target_cols)
    
    # Check that enrichment column was added
    assert 'Poisson_Enrichment' in result.columns
    
    # Check that all original columns are preserved
    for col in test_df.columns:
        assert col in result.columns
    
    # Check that enrichment values are calculated (should be positive)
    assert all(result['Poisson_Enrichment'] > 0)
    assert len(result['Poisson_Enrichment']) == 3
