"""
Non-Unified Trisynthon Classification Pipeline.
"""
import json
import yaml
import swifter
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.pipeline import Pipeline
from utils.logging_utils import setup_logging
from utils.denoise_utils import calculate_normalized_enrichment_score, calculate_hit_threshold

logger = setup_logging(__name__)


def denoise_data(file_path: str,
                 target_output_file_path: str,
                 control_output_file_path: str,
                 control_cols: List[str] = [
                     'seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'
                 ],
                 target_cols: List[str] = [
                     'seq_target_1', 'seq_target_2', 'seq_target_3'
                 ],
                 percentile: float = 90) -> str:
    """Denoise and preprocess trisynthon data for classification analysis.

    This function performs comprehensive data preprocessing including:
    - Loading and cleaning the input data
    - Calculating normalized enrichment scores for target and control
    - Determining hit threshold based on percentile
    - Creating binary hit labels for target and control
    - Saving the processed data to output file

    Parameters
    ----------
    file_path
        Path to the raw trisynthon CSV file.
    target_output_file_path
        Path where the denoised CSV with target hit labels will be written.
    control_output_file_path
        Path where the denoised CSV with control hit labels will be written.
    control_cols
        Column names representing control selection counts.
    target_cols
        Column names representing target selection counts.
    percentile
        Percentile threshold (0-100) used to call hits from enrichment scores.

    Returns
    -------
    str
        The path to the written denoised CSV file.

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
    >>> denoised_file_path = denoise_data("example_input.csv", "example_output.csv", control_cols, target_cols, percentile=90)
    >>> denoised_data = pd.read_csv(denoised_file_path)
    >>> 'target_hits' in denoised_data.columns
    True
    >>> 'control_hits' in denoised_data.columns
    True
    """

    data = pd.read_csv(file_path)
    data = data.dropna(subset=['smiles'])
    logger.info("Dropped rows (dropna)",
                extra={"rows": len(data[data['smiles'].isna()])})
    data = data.drop_duplicates(subset=['smiles'])
    logger.info("Dropped rows (drop duplicates)",
                extra={"rows": len(data[data['smiles'].duplicated()])})

    data['seq_target_sum'] = data[target_cols].agg('sum', axis=1)
    data['seq_control_sum'] = data[control_cols].agg('sum', axis=1)

    # calculate enrichment scores
    data['Target_Enrichment_Score'] = data.swifter.apply(
        lambda row: calculate_normalized_enrichment_score(
            row, data['seq_target_sum'].sum(), data.shape[0], 'seq_target_sum'
        ),
        axis=1)
    data['Control_Enrichment_Score'] = data.swifter.apply(
        lambda row: calculate_normalized_enrichment_score(
            row, data['seq_control_sum'].sum(), data.shape[0],
            'seq_control_sum'),
        axis=1)

    print(data.head())

    target_threshold = calculate_hit_threshold(
        df=data, column_name='Target_Enrichment_Score', percentile=percentile)
    control_threshold = calculate_hit_threshold(
        df=data, column_name='Control_Enrichment_Score', percentile=percentile)
    data['target_hits'] = (data['Target_Enrichment_Score']
                           > target_threshold).astype(int)
    data.to_csv(target_output_file_path, index=False)
    data['control_hits'] = (data['Control_Enrichment_Score']
                            > control_threshold).astype(int)
    data.to_csv(control_output_file_path, index=False)
    return target_output_file_path, control_output_file_path


def non_unified_trisynthon_classification_pipeline(
        config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full non-unified trisynthon classification pipeline.

    This function denoises the input according to `config['denoise_config']`,
    uploads the resulting file path into `config['files_to_upload']`, and then
    executes two classification runs using `Pipeline`: one for target hits and
    one for control hits.

    Parameters
    ----------
    config
        Dictionary loaded from YAML providing `denoise_config`,
        `featurizer_config`, and `files_to_upload`.

    Returns
    -------
    Dict[str, Any]
        Mapping with keys `result_target` and `result_control` containing the
        outputs from the two pipeline runs.
    """

    target_denoised_file_path, control_denoised_file_path = denoise_data(
        file_path=config['denoise_config']['input_file_path'],
        target_output_file_path=config['denoise_config']
        ['target_output_file_path'],
        control_output_file_path=config['denoise_config']
        ['control_output_file_path'],
        control_cols=config['denoise_config']['control_cols'],
        target_cols=config['denoise_config']['target_cols'],
        percentile=config['denoise_config']['hit_percentile'],
    )

    # target pipeline
    target_config = config['target_config']
    target_config['files_to_upload'].append(target_denoised_file_path)
    logger.info("Denoised file path set for upload",
                extra={"file_path": target_denoised_file_path})
    target_pipe = Pipeline(target_config)
    result_target = target_pipe.run()

    # control pipeline
    control_config = config['control_config']
    control_config['files_to_upload'].append(control_denoised_file_path)
    logger.info("Denoised file path set for upload",
                extra={"file_path": control_denoised_file_path})
    control_pipe = Pipeline(control_config)
    result_control = control_pipe.run()
    return {"result_target": result_target, "result_control": result_control}


def main(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Main entry point for the non-unified trisynthon classification pipeline.
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    result = non_unified_trisynthon_classification_pipeline(config)
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
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file containing pipeline parameters")
    args = parser.parse_args()
    main(args)
