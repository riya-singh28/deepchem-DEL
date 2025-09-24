from utils.pipeline import Pipeline


def test_pipeline_end_to_end() -> None:
    """End-to-end pipeline test using a temporary JSON config.
    """
    config_file = {
        "files_to_upload": {
            "file_path": "tests/assets/test_dataset.csv",
            "filename": "name",
            "description": "desc",
        },
        "featurizer_config": {
            "featurizer": "circular_fps",
            "output": "feat_out",
            "dataset_column": "smiles",
            "label_column": "y",
        },
        "split_config": {
            "splitter_type": "random",
            "frac_train": 0.8,
            "frac_valid": 0.1,
            "frac_test": 0.1
        },
        "train_config": {
            "model_type": "random_forest_regressor",
            "model_name": "mapk14_1M_rf_reg_try1",
            "init_kwargs": {"n_jobs": -1},
            "train_kwargs": {}
        },
        "evaluate_config": {
            "metrics": ["rms_score"],
            "output_key": "mapk14_1M_evaluation_result",
            "is_metric_plots": False
        },
        "infer_config": {
            "output": "mapk14_1M_inference_result",
            "dataset_column": "smiles",
            "shard_size": 8192,
            "threshold": 0.5
        }
    }
    p = Pipeline(config_file)

    result = p.run()
    
    assert set(result.keys()) == {
        "uploaded_files",
        "featurized_dataset_address",
        "train_split_address",
        "val_split_address",
        "test_split_address",
        "trained_model_address",
        "evaluation_result_address",
        "inference_result_address",
    }
