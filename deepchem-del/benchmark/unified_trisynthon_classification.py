"""
Unified Trisynthon Classification Pipeline.
"""

import json
import argparse
import pandas as pd
from utils.pipeline import Pipeline
from typing import List, Dict, Any, Optional
from utils.logging_utils import setup_logging
from utils.denoise_utils import calculate_poisson_enrichment, calculate_hit_threshold

logger = setup_logging(__name__)

def denoise_data(file_path: str,
                 output_file_path: str,
                 control_cols: List[str] = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'],
                 target_cols: List[str] = ['seq_target_1', 'seq_target_2', 'seq_target_3'],
                 percentile: float = 90) -> str:
    """
    Denoise and preprocess trisynthon data for classification analysis.

    This function performs comprehensive data preprocessing including:
    - Loading and cleaning the input data
    - Calculating Poisson enrichment scores
    - Determining hit threshold based on percentile
    - Creating binary hit labels
    - Saving the processed data to output file

    Parameters
    ----------
    file_path: str
        Path to the input CSV file containing the raw trisynthon data
    output_file_path: str
        Path where the denoised data will be saved
    control_cols: List[str], optional
        List of control column names for enrichment calculation.
        Defaults to ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'].
    target_cols: List[str], optional
        List of target column names for enrichment calculation.
        Defaults to ['seq_target_1', 'seq_target_2', 'seq_target_3'].
    percentile: float, optional
        Percentile threshold for hit classification (0-100).
        Defaults to 90.

    Returns
    -------
    str
        Path to the output file containing the denoised data

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'smiles': ['CCO', 'CCN', 'CCO', 'CCC'],
    ...     'seq_matrix_1': [10, 15, 10, 20],
    ...     'seq_matrix_2': [5, 8, 5, 12],
    ...     'seq_matrix_3': [3, 4, 3, 6],
    ...     'seq_target_1': [2, 3, 2, 4],
    ...     'seq_target_2': [1, 2, 1, 2],
    ...     'seq_target_3': [0, 1, 0, 1]
    ... })
    >>> df.to_csv("example_input.csv", index=False)
    >>> control_cols = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3']
    >>> target_cols = ['seq_target_1', 'seq_target_2', 'seq_target_3']
    >>> denoised_file_path = denoise_data("example_input.csv", "example_output.csv", control_cols, target_cols, percentile=90)
    """
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['smiles'])
    data = data.drop_duplicates(subset=['smiles'])

    # calculate poisson enrichment
    data = calculate_poisson_enrichment(df=data, control_cols=control_cols, target_cols=target_cols)
    target_threshold = calculate_hit_threshold(df=data, column_name='Poisson_Enrichment', percentile=percentile)
    data['hits'] = (data['Poisson_Enrichment'] > target_threshold).astype(int)
    data.to_csv(output_file_path, index=False)
    return output_file_path

def unified_trisynthon_classification_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the unified trisynthon classification pipeline.

    This function orchestrates the complete workflow for trisynthon classification:
    1. Denoises the input data using the specified configuration
    2. Updates the pipeline configuration with the denoised file path
    3. Initializes and runs the machine learning pipeline
    4. Returns the pipeline results

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary containing:

    Returns
    -------
    Dict[str, Any]
        Pipeline execution results including model performance metrics

    Examples
    --------
    >>> import json
    >>> with open('tests/assets/config.json', 'r') as f:
    ...     config = json.load(f)
    >>> unified_trisynthon_classification_pipeline(config)
    """
    denoised_file_path = denoise_data(file_path=config['denoise_config']['file_path'],
                                      output_file_path=config['denoise_config']['denoised_file_path'],
                                      control_cols=config['denoise_config']['control_cols'],
                                      target_cols=config['denoise_config']['target_cols'],
                                      percentile=config['denoise_config']['hit_percentile'])
    config['files_to_upload']['file_path'] = denoised_file_path
    logger.info("Denoised file path set for upload", extra={"file_path": config['files_to_upload']['file_path']})
    pipe = Pipeline(config)
    result = pipe.run()
    return result

def main(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """
    Main entry point for the unified trisynthon classification pipeline.
    """
    with open(args.config, 'r') as f:
        config = json.load(f)

    result = unified_trisynthon_classification_pipeline(config)
    logger.info("Pipeline result ready", extra={"result": result})
    # save result to json
    with open('result.json', 'w') as f:
        json.dump(result, f)
    logger.info("Pipeline result ready", extra={"result_file": 'result.json'})
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
