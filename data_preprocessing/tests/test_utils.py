import pytest
import pandas as pd
import numpy as np
from math import sqrt
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import calculate_enrichment_score, combine_deldenoised_datasets, combine_deldenoised_datasets_interactive, calculate_poisson_enrichment, create_combined_dataset


class TestCalculateEnrichmentScore:
    """Test cases for the calculate_enrichment_score function."""
    
    def test_basic_enrichment_calculation(self):
        """Test basic enrichment score calculation with known values."""
        # Create a simple test case
        row = pd.Series({'count': 10})
        total_sum = 100
        row_count = 5
        column_name = 'count'
        
        # Expected calculation:
        # p0 = 10 / 100 = 0.1
        # p1 = 1 / 5 = 0.2
        # z_score = (0.1 - 0.2) / sqrt(0.2 * (1 - 0.2)) = -0.1 / sqrt(0.16) = -0.1 / 0.4 = -0.25
        
        result = calculate_enrichment_score(row, total_sum, row_count, column_name)
        expected = -0.25
        
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    
    def test_positive_enrichment(self):
        """Test case where the value is higher than expected (positive enrichment)."""
        row = pd.Series({'count': 50})
        total_sum = 100
        row_count = 5
        column_name = 'count'
        
        # Expected calculation:
        # p0 = 50 / 100 = 0.5
        # p1 = 1 / 5 = 0.2
        # z_score = (0.5 - 0.2) / sqrt(0.2 * (1 - 0.2)) = 0.3 / sqrt(0.16) = 0.3 / 0.4 = 0.75
        
        result = calculate_enrichment_score(row, total_sum, row_count, column_name)
        expected = 0.75
        
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
        assert result > 0, "Positive enrichment should yield positive z-score"
    
    def test_negative_enrichment(self):
        """Test case where the value is lower than expected (negative enrichment)."""
        row = pd.Series({'count': 5})
        total_sum = 100
        row_count = 5
        column_name = 'count'
        
        # Expected calculation:
        # p0 = 5 / 100 = 0.05
        # p1 = 1 / 5 = 0.2
        # z_score = (0.05 - 0.2) / sqrt(0.2 * (1 - 0.2)) = -0.15 / sqrt(0.16) = -0.15 / 0.4 = -0.375
        
        result = calculate_enrichment_score(row, total_sum, row_count, column_name)
        expected = -0.375
        
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
        assert result < 0, "Negative enrichment should yield negative z-score"
    
    def test_zero_value(self):
        """Test case with zero value in the row."""
        row = pd.Series({'count': 0})
        total_sum = 100
        row_count = 5
        column_name = 'count'
        
        # Expected calculation:
        # p0 = 0 / 100 = 0.0
        # p1 = 1 / 5 = 0.2
        # z_score = (0.0 - 0.2) / sqrt(0.2 * (1 - 0.2)) = -0.2 / sqrt(0.16) = -0.2 / 0.4 = -0.5
        
        result = calculate_enrichment_score(row, total_sum, row_count, column_name)
        expected = -0.5
        
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    


class TestCombineDeldenoisedDatasets:
    """Test cases for the combine_datasets function."""
    
    def setup_method(self):
        """Set up test data by creating temporary CSV files."""
        import tempfile
        import os
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = {
            'matrix1': pd.DataFrame({
                'Smiles': ['CCO', 'CCN', 'CCC'],
                'Sum_Counts': [10, 20, 30]
            }),
            'matrix2': pd.DataFrame({
                'Smiles': ['CCO', 'CCN', 'CCC'],
                'Sum_Counts': [15, 25, 35]
            }),
            'target1': pd.DataFrame({
                'Smiles': ['CCO', 'CCN', 'CCC'],
                'Sum_Counts': [5, 10, 15]
            })
        }
        
        # Create temporary CSV files
        self.file_paths = {}
        for name, df in self.sample_data.items():
            file_path = os.path.join(self.temp_dir, f"{name}.csv")
            df.to_csv(file_path, index=False)
            self.file_paths[name] = file_path
    
    def test_basic_combination(self):
        """Test basic dataset combination."""
        result = combine_deldenoised_datasets(self.file_paths)
        
        # Check that all expected columns are present
        expected_columns = ['Smiles', 'matrix1_Sum_Counts', 'matrix2_Sum_Counts', 'target1_Sum_Counts']
        assert list(result.columns) == expected_columns
        
        # Check that data is correctly combined
        assert result['Smiles'].tolist() == ['CCO', 'CCN', 'CCC']
        assert result['matrix1_Sum_Counts'].tolist() == [10, 20, 30]
        assert result['matrix2_Sum_Counts'].tolist() == [15, 25, 35]
        assert result['target1_Sum_Counts'].tolist() == [5, 10, 15]



class TestCalculatePoissonEnrichment:
    """Test cases for the calculate_poisson_enrichment function."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_df = pd.DataFrame({
            'Smiles': ['CCO', 'CCN', 'CCC'],
            'matrix1_Sum_Counts': [10, 20, 30],
            'matrix2_Sum_Counts': [15, 25, 35],
            'target1_Sum_Counts': [5, 10, 15],
            'target2_Sum_Counts': [8, 12, 18]
        })
        
        self.control_cols = ['matrix1_Sum_Counts', 'matrix2_Sum_Counts']
        self.target_cols = ['target1_Sum_Counts', 'target2_Sum_Counts']
    
    def test_poisson_enrichment_calculation(self):
        """Test Poisson enrichment calculation."""
        result = calculate_poisson_enrichment(self.test_df, self.control_cols, self.target_cols)
        
        # Check that enrichment column was added
        assert 'Poisson_Enrichment' in result.columns
        
        # Check that all original columns are preserved
        for col in self.test_df.columns:
            assert col in result.columns
        
        # Check that enrichment values are calculated (should be positive)
        assert all(result['Poisson_Enrichment'] > 0)
        assert len(result['Poisson_Enrichment']) == 3
    
    def test_different_alpha_values(self):
        """Test with different alpha values."""
        result_alpha_01 = calculate_poisson_enrichment(self.test_df, self.control_cols, self.target_cols, alpha=0.01)
        result_alpha_05 = calculate_poisson_enrichment(self.test_df, self.control_cols, self.target_cols, alpha=0.05)
        
        # Results should be different for different alpha values
        assert not result_alpha_01['Poisson_Enrichment'].equals(result_alpha_05['Poisson_Enrichment'])
    
    def test_original_dataframe_unchanged(self):
        """Test that original DataFrame is not modified."""
        original_df = self.test_df.copy()
        result = calculate_poisson_enrichment(self.test_df, self.control_cols, self.target_cols)
        
        # Original DataFrame should be unchanged
        assert self.test_df.equals(original_df)
        # Result should have additional column
        assert len(result.columns) > len(original_df.columns)
