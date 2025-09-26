from uuid import uuid4
from typing import Any, Dict
from time import perf_counter
from logging_utils import setup_logging
from pyds import Evaluate, Featurize, Infer, Train, Settings, Data, TVTSplit

logger = setup_logging(__name__)


class Pipeline:
    """Run featurize → split → train → evaluate → infer using configured Deepchem server clients.
    """

    def __init__(self, config: dict) -> None:
        """Initialize client instances and load a configuration JSON file.

        Parameters
        ----------
        config: dict
            Dictionary containing the configuration for the pipeline.
        """

        logger.info("Initializing pipeline settings and clients")

        # Initialize the settings
        test_settings = Settings(
            settings_file='settings.json',
            profile="test_profile",
            project="test_project",
            base_url="http://localhost:8000",
        )

        # Initialize the clients
        self.featurize_client = Featurize(settings=test_settings)
        self.tvt_split_client = TVTSplit(settings=test_settings)
        self.train_client = Train(settings=test_settings)
        self.evaluate_client = Evaluate(settings=test_settings)
        self.infer_client = Infer(settings=test_settings)
        self.data_client = Data(settings=test_settings)

        # Load the configuration
        if config is None:
            logger.error("Initialization failed: config was None")
            raise ValueError("No config provided in config.")
        self.config = config
        # Attach a unique run id for correlating logs across the pipeline execution
        self.run_id = uuid4().hex
        logger.info("Pipeline initialization complete")

    def upload_data(self) -> Dict[str, Any]:
        """Upload data using the configured ``Data`` client.

        Returns
        -------
        Dict[str, Any]
            Result mapping returned by the data client, including an address
            for the uploaded dataset.

        Raises
        ------
        ValueError
            If required configuration or client is missing.
        """

        logger.info("Uploading data via Data client", extra={"run_id": self.run_id})
        config = self.config
        files_to_upload = config.get("files_to_upload")
        if files_to_upload is None:
            logger.error("Missing 'files_to_upload' in configuration")
            raise ValueError("No files_to_upload provided in config_file.")
        file_path = files_to_upload.get("file_path")
        if file_path is None:
            logger.error("Missing 'file_path' under 'files_to_upload' in configuration")
            raise ValueError("No dataset_path provided in config_file.")
        if self.data_client is None:
            logger.error("Data client not initialized")
            raise ValueError("No data_client provided in config_file.")
        result = self.data_client.upload_data(
            file_path=file_path,
            filename=files_to_upload.get("filename"),
            description=files_to_upload.get("description"),
        )
        logger.info("Data upload complete", extra={"run_id": self.run_id})
        logger.debug("Uploaded data response", extra={"run_id": self.run_id, "payload": result})
        return result

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline using parameters from the loaded config.

        Returns
        -------
        Dict[str, Any]
            Mapping of artifact names to addresses/identifiers produced by each
            stage of the pipeline.
        """
        logger.info("Starting pipeline run", extra={"run_id": self.run_id})
        try:
            # Upload the data
            t_upload_start = perf_counter()
            denoised_file_path = self.upload_data()
            logger.info(
                "Stage complete: upload",
                extra={
                    "run_id": self.run_id,
                    "elapsed_s": round(perf_counter() - t_upload_start, 4),
                },
            )

            # Get the configuration
            featurizer_config = self.config.get("featurizer_config")
            split_config = self.config.get("split_config")
            train_config = self.config.get("train_config")
            evaluate_config = self.config.get("evaluate_config")
            infer_config = self.config.get("infer_config")

            # Featurize the data
            logger.info("Featurization started", extra={"run_id": self.run_id})
            t_feat_start = perf_counter()
            featurized_dataset_address = self.featurize_client.run(
                dataset_address=denoised_file_path['dataset_address'],
                featurizer=featurizer_config.get("featurizer"),
                output=featurizer_config.get("output"),
                dataset_column=featurizer_config.get("dataset_column"),
                feat_kwargs=featurizer_config.get("feat_kwargs", None),
                label_column=featurizer_config.get("label_column", None),
            )
            logger.info(
                "Featurization complete",
                extra={
                    "run_id": self.run_id,
                    "elapsed_s": round(perf_counter() - t_feat_start, 4),
                },
            )
            logger.debug(
                "Featurized dataset response",
                extra={
                    "run_id": self.run_id,
                    "payload": featurized_dataset_address
                },
            )

            # Split the data
            logger.info("Train/Valid/Test split started", extra={"run_id": self.run_id})
            t_split_start = perf_counter()
            train_valid_test_split_address = self.tvt_split_client.run(
                dataset_address=featurized_dataset_address["featurized_file_address"],
                splitter_type=split_config.get("splitter_type"),
                frac_train=split_config.get("frac_train"),
                frac_valid=split_config.get("frac_valid"),
                frac_test=split_config.get("frac_test"),
            )
            logger.info(
                "Split complete",
                extra={
                    "run_id": self.run_id,
                    "elapsed_s": round(perf_counter() - t_split_start, 4),
                },
            )
            logger.debug(
                "TVT split response",
                extra={
                    "run_id": self.run_id,
                    "payload": train_valid_test_split_address
                },
            )

            # Train the model
            logger.info("Training started", extra={"run_id": self.run_id})
            t_train_start = perf_counter()
            trained_model_address = self.train_client.run(
                dataset_address=train_valid_test_split_address["train_valid_test_split_results_address"][0],
                model_type=train_config.get("model_type"),
                model_name=train_config.get("model_name"),
                init_kwargs=train_config.get("init_kwargs", None),
                train_kwargs=train_config.get("train_kwargs", None),
            )
            logger.info(
                "Training complete",
                extra={
                    "run_id": self.run_id,
                    "elapsed_s": round(perf_counter() - t_train_start, 4),
                },
            )
            logger.debug(
                "Trained model response",
                extra={
                    "run_id": self.run_id,
                    "payload": trained_model_address
                },
            )

            # Evaluate the model
            logger.info("Evaluation started", extra={"run_id": self.run_id})
            t_eval_start = perf_counter()
            evaluation_result_address = self.evaluate_client.run(
                dataset_addresses=(train_valid_test_split_address["train_valid_test_split_results_address"][0],
                                   train_valid_test_split_address["train_valid_test_split_results_address"][1],
                                   train_valid_test_split_address["train_valid_test_split_results_address"][2]),
                model_address=trained_model_address["trained_model_address"],
                metrics=evaluate_config.get("metrics"),
                output_key=evaluate_config.get("output_key"),
                is_metric_plots=evaluate_config.get("is_metric_plots"),
            )
            logger.info(
                "Evaluation complete",
                extra={
                    "run_id": self.run_id,
                    "elapsed_s": round(perf_counter() - t_eval_start, 4),
                },
            )
            logger.debug(
                "Evaluation response",
                extra={
                    "run_id": self.run_id,
                    "payload": evaluation_result_address
                },
            )

            # Inference the model
            logger.info("Inference started", extra={"run_id": self.run_id})
            t_infer_start = perf_counter()
            inference_result_address = self.infer_client.run(
                model_address=trained_model_address["trained_model_address"],
                data_address=train_valid_test_split_address["train_valid_test_split_results_address"][2],
                output=infer_config.get("output"),
                dataset_column=infer_config.get("dataset_column"),
                shard_size=infer_config.get("shard_size"),
                threshold=infer_config.get("threshold"),
            )
            logger.info(
                "Inference complete",
                extra={
                    "run_id": self.run_id,
                    "elapsed_s": round(perf_counter() - t_infer_start, 4),
                },
            )
            logger.debug(
                "Inference response",
                extra={
                    "run_id": self.run_id,
                    "payload": inference_result_address
                },
            )

            # Return the results
            result: Dict[str, Any] = {
                "uploaded_files": denoised_file_path,
                "featurized_dataset_address": featurized_dataset_address,
                "train_valid_test_split_results_address": train_valid_test_split_address,
                "trained_model_address": trained_model_address,
                "evaluation_result_address": evaluation_result_address,
                "inference_result_address": inference_result_address
            }
            logger.info("Pipeline run complete", extra={"run_id": self.run_id})
            logger.debug("Pipeline result payload", extra={"run_id": self.run_id, "payload": result})
            return result
        except Exception:
            logger.exception("Pipeline run failed", extra={"run_id": self.run_id})
            raise
