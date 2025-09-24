import json
import pandas as pd
import argparse
import logging
from utils.pipeline import Pipeline
from utils.denoise_utils import calculate_poisson_enrichment


def denoise_data(file_path: str, output_file_path: str) -> pd.DataFrame:
    """
    Denoise the data.
    """
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['smiles'])
    data = data.drop_duplicates(subset=['smiles'])

    # calculate poisson enrichment
    data = calculate_poisson_enrichment(data, ['seq_matrix_1', 'seq_matrix_2', 'seq_matrix_3'], ['seq_target_1', 'seq_target_2', 'seq_target_3'])
    data.to_csv(output_file_path, index=False)
    return output_file_path

def unified_trisynthon_regression_pipeline(config: dict) -> None:
    """
    Unified trisynthon regression pipeline.
    """
    denoised_file_path = denoise_data(config['denoise_config']['file_path'], config['denoise_config']['denoised_file_path'])
    config['files_to_upload']['file_path'] = denoised_file_path
    print(config['files_to_upload']['file_path'])
    pipe = Pipeline(config)
    pipe.run()

def main(args) -> None:
    """
    Main function to run the unified trisynthon regression pipeline.
    """
    # extract denoising key from config
    with open(args.config, 'r') as f:
        config = json.load(f)

    result = unified_trisynthon_regression_pipeline(config)
    print(result)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
