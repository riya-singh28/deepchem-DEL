"""
Non-Unified Disynthon Classification Pipeline.
"""
import yaml
import json
import swifter
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from utils.logging_utils import setup_logging
from utils.pipeline import Pipeline
from utils.denoise_utils import calculate_normalized_enrichment_score, create_disynthon_pairs, get_disynthons_from_pairs

logger = setup_logging(__name__)


def denoise_data(
    file_path: str,
    output_target_file_path: str,
    output_control_file_path: str,
    smiles_cols: List[str],
    control_cols: List[str] = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'],
    target_cols: List[str] = ['seq_target_1', 'seq_target_2', 'seq_target_3'],
    hit_percentile: float = 90,
) -> pd.DataFrame:
    """Create disynthons from trisynthons and generate hit labels.

    In the non-unified setting, each replicate is treated as an independent runs
    under identical experimental conditions. Accordingly, the three replicates of
    the target are combined into one large experiment, while the three replicates
    of the control are combined into a separate large experiment, after which
    denoising is performed.

    Steps
    -----
    - Drop duplicates/NaNs from smiles column.
    - Create disynthon pairs and sum counts.
    - Calculate enrichment scores for target and control.
    - Determine target and control hit threshold based on percentile.
    - Create binary hit labels for target and control.
    - Save denoised disynthon CSV with labels.

    Parameters
    ----------
    file_path
        Path to the raw trisynthon CSV.
    output_target_file_path
        Path to write the denoised disynthon CSV with target hits.
    output_control_file_path
        Path to write the denoised disynthon CSV with control hits.
    smiles_cols
        Names of the three trisynthon `smiles` columns to derive disynthons.
    control_cols
        Column names for control selection counts.
    target_cols
        Column names for target selection counts.
    hit_percentile
        Percentile threshold (0-100) to call hits.

    Returns
    -------
    Tuple[str, str]
        Path to the denoised files with target hits and control hits.

    Example
    -------
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
    >>> denoised_target_file_path, denoised_control_file_path = denoise_data("example_input.csv", "example_control_output.csv", "example_target_output.csv", control_cols, target_cols, percentile=90)
    >>> denoised_target_data = pd.read_csv(denoised_target_file_path)
    >>> 'target_hits' in denoised_target_data.columns
    True
    >>> denoised_control_data = pd.read_csv(denoised_control_file_path)
    >>> 'control_hits' in denoised_control_data.columns
    True
    """
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['smiles'])
    logger.info("Dropped rows (dropna)",
                extra={"rows": len(data[data['smiles'].isna()])})
    data = data.drop_duplicates(subset=['smiles'])
    logger.info("Dropped rows (drop duplicates)",
                extra={"rows": len(data[data['smiles'].duplicated()])})

    # create disynthon pairs from trisynthon data
    disynthon_data, smiles_dict = create_disynthon_pairs(
        df=data,
        smiles_cols=smiles_cols,
        control_cols=control_cols,
        target_cols=target_cols,
        is_unified=False)
    failed_smiles = set()
    failed_combines = set()
    smiles_dict_inv = {v: k for k, v in smiles_dict.items()}

    # get disynthons from disynthon pairs
    disynthon_data['disynthons'] = disynthon_data.swifter.apply(
        lambda x: get_disynthons_from_pairs(x['Disynthon_1'], x[
            'Disynthon_2'], failed_smiles, failed_combines, smiles_dict_inv),
        axis=1)
    disynthon_data = disynthon_data[~disynthon_data['disynthons'].isna()]

    # sum of duplicate disynthons
    print(disynthon_data.head())
    disynthon_data = disynthon_data.groupby('disynthons').agg({
        'seq_target_sum':
        'sum',
        'seq_control_sum':
        'sum'
    }).reset_index()

    # calculate enrichment scores
    disynthon_data['Target_Enrichment_Score'] = disynthon_data.swifter.apply(
        lambda row: calculate_normalized_enrichment_score(
            row, disynthon_data['seq_target_sum'].sum(), len(disynthon_data),
            'seq_target_sum'),
        axis=1)
    disynthon_data['Control_Enrichment_Score'] = disynthon_data.swifter.apply(
        lambda row: calculate_normalized_enrichment_score(
            row, disynthon_data['seq_control_sum'].sum(), len(disynthon_data),
            'seq_control_sum'),
        axis=1)

    target_hit_threshold = np.percentile(
        disynthon_data['Target_Enrichment_Score'], hit_percentile)
    disynthon_data['target_hits'] = (disynthon_data['Target_Enrichment_Score']
                                     > target_hit_threshold).astype(int)
    disynthon_data.to_csv(output_target_file_path, index=False)

    control_hit_threshold = np.percentile(
        disynthon_data['Control_Enrichment_Score'], hit_percentile)
    disynthon_data['control_hits'] = (
        disynthon_data['Control_Enrichment_Score']
        > control_hit_threshold).astype(int)
    disynthon_data.to_csv(output_control_file_path, index=False)
    return output_target_file_path, output_control_file_path


def non_unified_disynthon_classification_pipeline(
        config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the non-unified disynthon classification pipeline.

    Parameters
    ----------
    config
        Configuration dictionary.

    Returns
    -------
    Dict[str, Any]
        Results for target and control pipelines.
    """
    denoised_target_file_path, denoised_control_file_path = denoise_data(
        file_path=config['denoise_config']['input_file_path'],
        output_target_file_path=config['denoise_config']
        ['target_output_file_path'],
        output_control_file_path=config['denoise_config']
        ['control_output_file_path'],
        smiles_cols=config['denoise_config']['smiles_cols'],
        control_cols=config['denoise_config']['control_cols'],
        target_cols=config['denoise_config']['target_cols'],
        hit_percentile=config['denoise_config']['hit_percentile'])

    # target pipeline
    target_config = config['target_config']
    target_config['files_to_upload'].append(denoised_target_file_path)
    logger.info("Denoised Target file path set for upload",
                extra={"file_path": denoised_target_file_path})
    target_pipe = Pipeline(target_config)
    result_target = target_pipe.run()

    # control pipeline
    control_config = config['control_config']
    control_config['files_to_upload'].append(denoised_control_file_path)
    logger.info("Denoised Control file path set for upload",
                extra={"file_path": denoised_control_file_path})
    control_pipe = Pipeline(control_config)
    result_control = control_pipe.run()
    return {"result_target": result_target, "result_control": result_control}


def main(args) -> None:
    """Main entry point for the non-unified disynthon classification pipeline.
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    result = non_unified_disynthon_classification_pipeline(config)
    result_target = result['result_target']
    result_control = result['result_control']

    result_file_name_target = f"{result_target['run_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file_name_target, 'w') as f:
        json.dump(result_target, f)
    logger.info("Target Pipeline result ready",
                extra={"result_file": result_file_name_target})
    result_file_name_control = f"{result_control['run_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file_name_control, 'w') as f:
        json.dump(result_control, f)
    logger.info("Control Pipeline result ready",
                extra={"result_file": result_file_name_control})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
