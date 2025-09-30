from utils.pipeline import Pipeline
from utils.logging_utils import setup_logging

logger = setup_logging(__name__)


def test_pipeline_end_to_end(pipeline_config):
    """End-to-end pipeline test using a temporary JSON config.
    """
    p = Pipeline(pipeline_config)
    try:
        result = p.run()
    except Exception as e:
        raise Exception(f"Pipeline run failed: {e}")

    tvt_split_address = result["train_valid_test_split_address"]
    assert len(
        tvt_split_address['train_valid_test_split_results_address']) == 3

    assert set(result.keys()) == {
        "uploaded_files",
        "featurized_dataset_address",
        "train_valid_test_split_address",
        "trained_model_address",
        "evaluation_result_address",
        "inference_result_address",
    }
