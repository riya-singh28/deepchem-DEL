"""Unified disynthon classification benchmark utilities.
"""
import json
import yaml
import swifter
import argparse
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from utils.logging_utils import setup_logging
from utils.pipeline import Pipeline
from utils.denoise_utils import calculate_poisson_enrichment, create_disynthon_pairs, get_disynthons_from_pairs, calculate_hit_threshold

logger = setup_logging(__name__)


def denoise_data(file_path: str,
                 output_file_path: str,
                 smiles_cols: List[str],
                 aggregate_operation: str = 'sum',
                 control_cols: List[str] = [
                     'seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'
                 ],
                 target_cols: List[str] = [
                     'seq_target_1', 'seq_target_2', 'seq_target_3'
                 ],
                 percentile: float = 90) -> str:
    """
    Denoise and preprocess data for disynthon classification analysis.

    This function performs comprehensive data preprocessing including:
    - Loading and cleaning the input data
    - Creating disynthon pairs from SMILES strings
    - Aggregating duplicate disynthons
    - Calculating Poisson enrichment scores
    - Calculating hit threshold
    - Creating binary hit labels
    - Saving the processed data to output file

    Parameters
    ----------
    file_path: str
        Path to the input CSV file containing the raw data
    output_file_path: str
        Path where the denoised data will be saved
    smiles_cols: List[str]
        List of column names containing SMILES strings
    aggregate_operation: str, optional
        Operation to use for aggregating duplicates. Defaults to 'sum'.
    control_cols: List[str], optional
        List of control column names for enrichment calculation.
    target_cols: List[str], optional
        List of target column names for enrichment calculation.
    percentile: float, optional
        Percentile threshold for hit classification (0-100). Defaults to 90.

    Returns
    -------
    str
        Path to the output file containing the denoised data

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'smiles': ['CCO', 'CCN', 'CCCC', 'CCC'],
    ...     'smiles_a': ['CCO', 'CCN', 'CCO', 'CCC'],
    ...     'smiles_b': ['CCN', 'CCO', 'CCN', 'CCN'],
    ...     'smiles_c': ['CCO', 'CCN', 'CCO', 'CCC'],
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
    >>> smiles_cols = ['smiles_a', 'smiles_b', 'smiles_c']
    >>> denoised_file_path = denoise_data(file_path="example_input.csv",
    ... output_file_path="example_output.csv", smiles_cols=smiles_cols,
    ... control_cols=control_cols, target_cols=target_cols, percentile=90)
    >>> df = pd.read_csv("example_output.csv")
    >>> 'hits' in df.columns
    True
    """
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['smiles'])
    logger.info("Dropped rows (dropna)",
                extra={"rows": len(data[data['smiles'].isna()])})
    data = data.drop_duplicates(subset=['smiles'])
    logger.info("Dropped rows (drop duplicates)",
                extra={"rows": len(data[data['smiles'].duplicated()])})

    disynthon_data, smiles_dict = create_disynthon_pairs(
        data, smiles_cols, aggregate_operation, control_cols, target_cols)
    failed_smiles = set()
    failed_combines = set()
    smiles_dict_inv = {v: k for k, v in smiles_dict.items()}
    disynthon_data['disynthons'] = disynthon_data.swifter.apply(
        lambda x: get_disynthons_from_pairs(x['Disynthon_1'], x[
            'Disynthon_2'], failed_smiles, failed_combines, smiles_dict_inv),
        axis=1)
    disynthon_data = disynthon_data[~disynthon_data['disynthons'].isna()]
    # sum of duplicate disynthons
    disynthon_data = disynthon_data.groupby('disynthons').agg({
        control_cols[0]:
        aggregate_operation,
        control_cols[1]:
        aggregate_operation,
        control_cols[2]:
        aggregate_operation,
        target_cols[0]:
        aggregate_operation,
        target_cols[1]:
        aggregate_operation,
        target_cols[2]:
        aggregate_operation
    }).reset_index()
    # calculate poisson enrichment
    disynthon_data = calculate_poisson_enrichment(disynthon_data, control_cols,
                                                  target_cols)

    # calculate hit threshold
    hit_threshold = calculate_hit_threshold(df=disynthon_data,
                                            column_name='Poisson_Enrichment',
                                            percentile=percentile)
    disynthon_data['hits'] = (disynthon_data['Poisson_Enrichment']
                              > hit_threshold).astype(int)
    disynthon_data.to_csv(output_file_path, index=False)
    return output_file_path


def unified_disynthon_classification_pipeline(
        config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the unified disynthon classification pipeline end-to-end.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration dictionary for denoising and model pipeline.

    Returns
    -------
    result: Dict[str, Any]
        A dictionary containing the artifact addresses.

    Examples
    --------
    >>> import json
    >>> with open('tests/assets/unified_disynthon_classification_config.json', 'r') as f:
    ...     config = json.load(f)
    >>> unified_disynthon_classification_pipeline(config)
    """
    denoised_file_path = denoise_data(
        file_path=config['denoise_config']['file_path'],
        output_file_path=config['denoise_config']['denoised_file_path'],
        smiles_cols=config['denoise_config']['smiles_cols'],
        aggregate_operation=config['denoise_config']['aggregate_operation'],
        control_cols=config['denoise_config']['control_cols'],
        target_cols=config['denoise_config']['target_cols'],
        percentile=config['denoise_config']['hit_percentile'])
    config['files_to_upload'].append(denoised_file_path)
    logger.info("Denoised file path set for upload",
                extra={"file_path": config['files_to_upload'][-1]})
    pipe = Pipeline(config)
    result = pipe.run()
    return result


def main(args) -> None:
    """Entry point for the unified disynthon classification pipeline.
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    result = unified_disynthon_classification_pipeline(config)
    # save result to json
    result_file_name = f"{result['run_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file_name, 'w') as f:
        json.dump(result, f)
    logger.info("Pipeline result ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
