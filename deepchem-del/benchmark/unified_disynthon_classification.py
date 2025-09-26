import json
import swifter
import pandas as pd
import argparse
# from utils.logging_utils import setup_logging
# from utils.pipeline import Pipeline
# from utils.denoise_utils import calculate_poisson_enrichment, create_disynthon_pairs, get_disynthons_from_pairs

from logging_utils import setup_logging
from denoise_utils import calculate_poisson_enrichment, create_disynthon_pairs, get_disynthons_from_pairs
from pipeline import Pipeline
from typing import List, Dict, Any
import numpy as np

logger = setup_logging(__name__)

def denoise_data(
    file_path: str,
    output_file_path: str,
    smiles_cols: List[str],
    aggregate_operation: str = 'sum',
    control_cols: List[str] = ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'],
    target_cols: List[str] = ['seq_target_1', 'seq_target_2', 'seq_target_3'],
    hit_percentile: float = 90,
) -> pd.DataFrame:
    """
    Denoise the data.
    """
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['smiles'])
    data = data.drop_duplicates(subset=['smiles'])

    disynthon_data, smiles_dict = create_disynthon_pairs(df=data,
                                                         smiles_cols=smiles_cols,
                                                         aggregate_operation=aggregate_operation,
                                                         control_cols=control_cols,
                                                         target_cols=target_cols,
                                                         is_unified=True)
    failed_smiles = set()
    failed_combines = set()
    smiles_dict_inv = {v: k for k, v in smiles_dict.items()}
    disynthon_data['disynthons'] = disynthon_data.swifter.apply(lambda x: get_disynthons_from_pairs(
        x['Disynthon_1'], x['Disynthon_2'], failed_smiles, failed_combines, smiles_dict_inv),
                                                                axis=1)
    disynthon_data = disynthon_data[~disynthon_data['disynthons'].isna()]
    # sum of duplicate disynthons
    disynthon_data = disynthon_data.groupby('disynthons').agg({
        control_cols[0]: aggregate_operation,
        control_cols[1]: aggregate_operation,
        control_cols[2]: aggregate_operation,
        target_cols[0]: aggregate_operation,
        target_cols[1]: aggregate_operation,
        target_cols[2]: aggregate_operation
    }).reset_index()

    # calculate poisson enrichment
    disynthon_data = calculate_poisson_enrichment(df=disynthon_data, control_cols=control_cols, target_cols=target_cols)
    hit_threshold = np.percentile(disynthon_data['Poisson_Enrichment'], hit_percentile)
    disynthon_data['hits'] = (disynthon_data['Poisson_Enrichment'] > hit_threshold).astype(int)
    logger.info(f"Hit threshold: {hit_threshold}")
    logger.info(f"Number of hits: {disynthon_data['hits'].sum()}")
    logger.info(f"Number of rows: {disynthon_data.shape[0]}")
    logger.info(f"Hit percentage: {disynthon_data['hits'].sum() / disynthon_data.shape[0] * 100:.2f}%")
    disynthon_data.to_csv(output_file_path, index=False)

    return output_file_path

def unified_disynthon_classification_pipeline(config: dict) -> Dict[str, Any]:
    """
    Unified disynthon classification pipeline.
    """
    denoised_file_path = denoise_data(file_path=config['denoise_config']['file_path'],
                                      output_file_path=config['denoise_config']['denoised_file_path'],
                                      smiles_cols=config['denoise_config']['smiles_cols'],
                                      aggregate_operation=config['denoise_config']['aggregate_operation'],
                                      control_cols=config['denoise_config']['control_cols'],
                                      target_cols=config['denoise_config']['target_cols'],
                                      hit_percentile=config['denoise_config']['hit_percentile'])
    config['files_to_upload']['file_path'] = denoised_file_path
    logger.info("Denoised file path set for upload", extra={"file_path": config['files_to_upload']['file_path']})
    pipe = Pipeline(config)
    result = pipe.run()
    return result

def main(args) -> None:
    """
    Main function to run the unified disynthon classification pipeline.
    """
    with open(args.config, 'r') as f:
        config = json.load(f)
    result = unified_disynthon_classification_pipeline(config)
    # save result to json
    with open('result.json', 'w') as f:
        json.dump(result, f)
    logger.info("Pipeline result ready", extra={"result_file": 'result.json'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
