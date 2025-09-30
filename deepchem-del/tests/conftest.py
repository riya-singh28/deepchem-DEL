import pytest


@pytest.fixture
def pipeline_config():
    """Provide a reusable pipeline configuration for tests."""
    return {
        "settings_file": "tests/assets/test_settings.json",
        "files_to_upload": ["tests/assets/test_dataset.csv"],
        "featurizer_config": {
            "featurizer": "ecfp",
            "output": "feat_out_4",
            "dataset_column": "smiles",
            "label_column": "target_enrichment",
        },
        "split_config": {
            "splitter_type": "random",
            "frac_train": 0.8,
            "frac_valid": 0.1,
            "frac_test": 0.1
        },
        "train_config": {
            "model_type": "random_forest_regressor",
            "model_name": "test_rf_reg_try4",
            "init_kwargs": {
                "n_jobs": -1
            },
            "train_kwargs": {}
        },
        "evaluate_config": {
            "metrics": ["rms_score"],
            "output_key": "test_evaluation_result_4",
            "is_metric_plots": False
        },
        "infer_config": {
            "output": "test_inference_result_4",
            "dataset_column": "smiles",
            "label_column": "target_enrichment"
        }
    }
