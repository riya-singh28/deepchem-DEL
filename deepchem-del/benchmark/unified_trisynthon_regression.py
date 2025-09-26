"""
Unified Trisynthon Regression Pipeline.
"""

import json
import pandas as pd
import argparse
from utils.logging_utils import setup_logging
from utils.pipeline import Pipeline
from utils.denoise_utils import calculate_poisson_enrichment

from typing import List, Dict, Any

logger = setup_logging(__name__)

def denoise_data(file_path: str,
                 output_file_path: str,
                 control_cols: List[str] = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'],
                 target_cols: List[str] = ['seq_target_1', 'seq_target_2', 'seq_target_3']) -> str:
    """
    This function performs data preprocessing steps including:
    - Removing rows with missing SMILES
    - Removing duplicate SMILES
    - Calculating Poisson enrichment scores for trisynthon combinations

    Parameters
    ----------
    file_path : str
        Path to the input CSV file containing trisynthon data.
    output_file_path : str
        Path where the denoised data will be saved as a CSV file

    Returns
    -------
    str
        Path to the output file containing denoised data with Poisson enrichment scores

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
    >>> denoised_file_path = denoise_data("example_input.csv", "example_output.csv", control_cols, target_cols)
    >>> denoised_file_path
    'example_output.csv'
    """
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['smiles'])
    data = data.drop_duplicates(subset=['smiles'])

    # calculate poisson enrichment
    data = calculate_poisson_enrichment(data, control_cols, target_cols)
    data.to_csv(output_file_path, index=False)
    return output_file_path

def unified_trisynthon_regression_pipeline(config: dict) -> Dict[str, Any]:
    """
    Execute a complete unified trisynthon regression pipeline.

    This function orchestrates the entire machine learning pipeline for trisynthon
    regression analysis, including data preprocessing, featurization, training,
    evaluation, and inference steps using the DeepChem server infrastructure.

    The pipeline workflow:
    1. Data denoising and Poisson enrichment calculation
    2. File upload to DeepChem server
    3. Molecular featurization (e.g., ECFP fingerprints)
    4. Train/validation/test data splitting
    5. Model training (e.g., Random Forest regressor)
    6. Model evaluation with regression metrics
    7. Inference on test data

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing all pipeline parameters.

    Returns
    -------
    None
        The function executes the pipeline and logs results. Results are stored
        on the DeepChem server and can be accessed via the server API.

    Examples
    --------
    >>> import json
    >>> with open('tests/assets/config.json', 'r') as f:
    ...     config = json.load(f)
    >>> unified_trisynthon_regression_pipeline(config)
    """
    denoised_file_path = denoise_data(config['denoise_config']['file_path'],
                                      config['denoise_config']['denoised_file_path'])
    config['files_to_upload']['file_path'] = denoised_file_path
    logger.info("Denoised file path set for upload", extra={"file_path": config['files_to_upload']['file_path']})
    pipe = Pipeline(config)
    result = pipe.run()
    logger.info("Pipeline result ready", extra={"result": result})
    # save result to json
    with open('result.json', 'w') as f:
        json.dump(result, f)
    logger.info("Pipeline result ready", extra={"result_file": 'result.json'})
    return result

def main(args) -> None:
    """
    Main entry point for the unified trisynthon regression pipeline.

    This function serves as the command-line interface for running the complete
    trisynthon regression pipeline. It loads configuration from a JSON file,
    executes the pipeline, and saves the results.

    Examples
    --------
    `python unified_trisynthon_regression.py --config config.json`
    """
    with open(args.config, 'r') as f:
        config = json.load(f)
    result = unified_trisynthon_regression_pipeline(config)
    # save result to json
    with open('result.json', 'w') as f:
        json.dump(result, f)
    logger.info("Pipeline result ready", extra={"result_file": 'result.json'})

if __name__ == "__main__":
    # Command-line argument parser for the unified trisynthon regression pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to JSON configuration file containing pipeline parameters")
    args = parser.parse_args()
    main(args)
