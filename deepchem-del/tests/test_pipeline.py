import json
from utils.pipeline import Pipeline


def test_pipeline_end_to_end() -> None:
    """End-to-end pipeline test using a temporary JSON config.

    This test writes a minimal configuration file, initializes the
    ``Pipeline`` with it, runs the full pipeline, and asserts that
    the returned mapping contains all expected artifact keys.

    Returns
    -------
    None
        Pytest test function; assertions validate behavior.
    """

    # make a config file
    config_file: Dict[str, Any] = {
        "files_to_upload": {
            "file_path": "data/dataset.csv",
            "filename": "name",
            "description": "desc",
        },
        "featurizer_config": {
            "featurizer": "circular_fps",
            "output": "feat_out",
            "dataset_column": "smiles",
            "feat_kwargs": {"radius": 2, "size": 1024},
            "label_column": "y",
        },
        "split_config": {
            "split_type": "random",
            "split_sizes": [0.8, 0.1, 0.1],
            "output": "split_out",
        },
        "train_config": {
            "model_type": "sklearn",
            "model_name": "rf",
            "init_kwargs": {"n_estimators": 10},
            "train_kwargs": {},
        },
        "evaluate_config": {
            "metrics": ["roc_auc_score"],
            "output_key": "eval_out",
            "is_metric_plots": False,
        },
        "infer_config": {
            "output": "preds",
            "dataset_column": "smiles",
            "shard_size": 4096,
            "threshold": 0.5,
        },
    }
    # make a config.json file with the config file
    with open("config.json", "w") as f:
        json.dump(config_file, f)
    p: Pipeline = Pipeline()
    p.init("config.json")

    result: Dict[str, Any] = p.run()
    
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
