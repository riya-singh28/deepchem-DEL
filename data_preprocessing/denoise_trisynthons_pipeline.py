#!/usr/bin/env python3
"""
Trisynthon Denoising Pipeline

This script processes trisynthon data for DDR1 and MAPK14 targets,
calculating enrichment scores and identifying hits based on percentile thresholds.

The pipeline uses utility functions from utils.py for:
- Column aggregation
- Enrichment score calculation
- Hit threshold determination
"""

import pandas as pd
import numpy as np
import swifter
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Import utility functions
from utils import (
    aggregate_columns,
    calculate_enrichment_score,
    calculate_hit_threshold
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrisynthonDenoisingPipeline:
    """
    Pipeline for processing trisynthon data and calculating enrichment scores.
    """
    
    def __init__(self, data_dir: str = "/home/riya/Documents/Chiron/chiron/merck_trisynthon_experiments/"):
        """
        Initialize the pipeline with data directory.
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.results = {}
        
    def load_data(self, dataset_names: List[str]) -> None:
        """
        Load parquet files for specified datasets.
        
        Parameters
        ----------
        dataset_names : List[str]
            List of dataset names (e.g., ['ddr1', 'mapk14'])
        """
        logger.info(f"Loading datasets: {dataset_names}")
        
        for name in dataset_names:
            file_path = self.data_dir / f"{name}_1M.parquet"
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                continue
                
            try:
                self.datasets[name] = pd.read_parquet(file_path)
                logger.info(f"Loaded {name}: {self.datasets[name].shape[0]} rows")
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                
    def calculate_sum_columns(self, dataset_name: str) -> None:
        """
        Calculate sum columns for target and matrix sequences using aggregate_columns.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to process
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found")
            return
            
        logger.info(f"Calculating sum columns for {dataset_name}")
        
        # Define column groups for aggregation
        column_groups = {
            'seq_target_sum': ['seq_target_1', 'seq_target_2', 'seq_target_3'],
            'seq_matrix_sum': ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3']
        }
        
        try:
            # Use utility function for aggregation
            self.datasets[dataset_name] = aggregate_columns(
                self.datasets[dataset_name], 
                column_groups, 
                operation='sum'
            )
            logger.info(f"Added sum columns to {dataset_name}")
        except Exception as e:
            logger.error(f"Error calculating sum columns for {dataset_name}: {e}")
            
    def calculate_enrichment_scores(self, dataset_name: str) -> None:
        """
        Calculate enrichment scores for target and matrix data.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to process
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found")
            return
            
        logger.info(f"Calculating enrichment scores for {dataset_name}")
        
        df = self.datasets[dataset_name]
        
        try:
            # Calculate target enrichment scores
            total_target_sum = df['seq_target_sum'].sum()
            row_count = df.shape[0]
            
            df['Target_Enrichment_Score'] = df.swifter.apply(
                lambda row: calculate_enrichment_score(
                    row, total_target_sum, row_count, 'seq_target_sum'
                ), axis=1
            )
            
            # Calculate matrix enrichment scores
            total_matrix_sum = df['seq_matrix_sum'].sum()
            
            df['Matrix_Enrichment_Score'] = df.swifter.apply(
                lambda row: calculate_enrichment_score(
                    row, total_matrix_sum, row_count, 'seq_matrix_sum'
                ), axis=1
            )
            
            logger.info(f"Calculated enrichment scores for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error calculating enrichment scores for {dataset_name}: {e}")
            
    def identify_hits(self, dataset_name: str, percentile: float = 90) -> Dict[str, float]:
        """
        Identify hits based on enrichment score thresholds.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to process
        percentile : float, default 90
            Percentile threshold for hit identification
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing threshold values
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found")
            return {}
            
        logger.info(f"Identifying hits for {dataset_name} at {percentile}th percentile")
        
        df = self.datasets[dataset_name]
        thresholds = {}
        
        try:
            # Calculate thresholds using utility function
            target_threshold = calculate_hit_threshold(df, 'Target_Enrichment_Score', percentile)
            matrix_threshold = calculate_hit_threshold(df, 'Matrix_Enrichment_Score', percentile)
            
            # Create hit columns
            df['target_hits'] = (df['Target_Enrichment_Score'] >= target_threshold).astype(int)
            df['matrix_hits'] = (df['Matrix_Enrichment_Score'] >= matrix_threshold).astype(int)
            
            thresholds = {
                'target_threshold': target_threshold,
                'matrix_threshold': matrix_threshold
            }
            
            # Log hit counts
            target_hits = df['target_hits'].sum()
            matrix_hits = df['matrix_hits'].sum()
            total_rows = df.shape[0]
            
            logger.info(f"{dataset_name} - Target hits: {target_hits}/{total_rows} ({target_hits/total_rows*100:.2f}%)")
            logger.info(f"{dataset_name} - Matrix hits: {matrix_hits}/{total_rows} ({matrix_hits/total_rows*100:.2f}%)")
            logger.info(f"{dataset_name} - Target threshold: {target_threshold:.4f}")
            logger.info(f"{dataset_name} - Matrix threshold: {matrix_threshold:.4f}")
            
        except Exception as e:
            logger.error(f"Error identifying hits for {dataset_name}: {e}")
            
        return thresholds
        
    def save_results(self, dataset_name: str, output_dir: str = None) -> None:
        """
        Save processed results to CSV files.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to save
        output_dir : str, optional
            Output directory. If None, uses the data directory
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found")
            return
            
        if output_dir is None:
            output_dir = self.data_dir
        else:
            output_dir = Path(output_dir)
            
        logger.info(f"Saving results for {dataset_name}")
        
        df = self.datasets[dataset_name]
        
        try:
            # Save target enrichment results
            target_file = output_dir / f"{dataset_name}_1M_target_enrichment_trisynthon_90per.csv"
            df.to_csv(target_file, index=False)
            logger.info(f"Saved target enrichment results: {target_file}")
            
            # Save matrix enrichment results
            matrix_file = output_dir / f"{dataset_name}_1M_matrix_enrichment_trisynthon_90per.csv"
            df.to_csv(matrix_file, index=False)
            logger.info(f"Saved matrix enrichment results: {matrix_file}")
            
        except Exception as e:
            logger.error(f"Error saving results for {dataset_name}: {e}")
            
    def run_pipeline(self, dataset_names: List[str], percentile: float = 90) -> Dict[str, Dict[str, float]]:
        """
        Run the complete denoising pipeline for specified datasets.
        
        Parameters
        ----------
        dataset_names : List[str]
            List of dataset names to process
        percentile : float, default 90
            Percentile threshold for hit identification
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing threshold values for each dataset
        """
        logger.info(f"Starting trisynthon denoising pipeline for {dataset_names}")
        
        all_thresholds = {}
        
        # Load data
        self.load_data(dataset_names)
        
        # Process each dataset
        for dataset_name in dataset_names:
            if dataset_name not in self.datasets:
                continue
                
            logger.info(f"Processing {dataset_name}")
            
            # Calculate sum columns
            self.calculate_sum_columns(dataset_name)
            
            # Calculate enrichment scores
            self.calculate_enrichment_scores(dataset_name)
            
            # Identify hits
            thresholds = self.identify_hits(dataset_name, percentile)
            all_thresholds[dataset_name] = thresholds
            
            # Save results
            self.save_results(dataset_name)
            
        logger.info("Pipeline completed successfully")
        return all_thresholds


def main():
    """
    Main function to run the trisynthon denoising pipeline.
    """
    # Initialize pipeline
    pipeline = TrisynthonDenoisingPipeline()
    
    # Define datasets to process
    datasets = ['ddr1', 'mapk14']
    
    # Run pipeline
    thresholds = pipeline.run_pipeline(datasets, percentile=90)
    
    # Print summary
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    
    for dataset_name, dataset_thresholds in thresholds.items():
        print(f"\n{dataset_name.upper()}:")
        for threshold_type, value in dataset_thresholds.items():
            print(f"  {threshold_type}: {value:.4f}")
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
