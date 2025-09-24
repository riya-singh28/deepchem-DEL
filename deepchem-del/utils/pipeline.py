from typing import Any, Dict, Mapping, Optional, Tuple
from pyds import Evaluate, Featurize, Infer, Train, Settings, Data, Split, TvtSplit
import json


class Pipeline:
    """Run featurize → split → train → evaluate → infer using configured Deepchem server clients.
    """

    def init(self, config_file: str) -> None:
        """Initialize client instances and load a configuration JSON file.

        Parameters
        ----------
        config_file: str
            Path to a JSON file containing the configuration for the pipeline.
        """

        # Initialize the settings
        test_settings = Settings(
        settings_file='settings.json',
        profile="test_profile",
        project="test_project",
        base_url="http://localhost:8000",
        )

        # Initialize the clients
        self.featurize_client = Featurize(settings=test_settings)
        self.tvt_split_client = TvtSplit(settings=test_settings)
        self.train_client = Train(settings=test_settings)
        self.evaluate_client = Evaluate(settings=test_settings)
        self.infer_client = Infer(settings=test_settings)
        self.data_client = Data(settings=test_settings)

        # Load the configuration
        if config_file is None:
            raise ValueError("No config_file provided in config_file.")
        with open(config_file, "r") as f:
            self.config = json.load(f)

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

        config = self.config
        files_to_upload = config.get("files_to_upload")
        if files_to_upload is None:
            raise ValueError("No files_to_upload provided in config_file.")
        file_path = files_to_upload.get("file_path")
        if file_path is None:
            raise ValueError("No dataset_path provided in config_file.")
        if self.data_client is None:
            raise ValueError("No data_client provided in config_file.")
        return self.data_client.upload_data(
            file_path=file_path,
            filename=files_to_upload.get("filename"),
            description=files_to_upload.get("description"),
        )

    def denoise_data(self) -> Dict[str, Any]:
        raise NotImplementedError("Need to implement denoise_data method.")

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline using parameters from the loaded config.

        Returns
        -------
        Dict[str, Any]
            Mapping of artifact names to addresses/identifiers produced by each
            stage of the pipeline.
        """
        # Upload the data
        file_path = self.upload_data()

        # Get the configuration
        featurizer_config = self.config.get("featurizer_config")
        split_config = self.config.get("split_config")
        train_config = self.config.get("train_config")
        evaluate_config = self.config.get("evaluate_config")
        infer_config = self.config.get("infer_config")

        # Featurize the data
        featurized_dataset_address = self.featurize_client.run(
            dataset_address=file_path,
            featurizer=featurizer_config.get("featurizer"),
            output=featurizer_config.get("output"),
            dataset_column=featurizer_config.get("dataset_column"),
            feat_kwargs=featurizer_config.get("feat_kwargs"),
            label_column=featurizer_config.get("label_column"),
        )

        # Split the data
        train_split_address, val_split_address, test_split_address = self.tvt_split_client.run(
            dataset_address=featurized_dataset_address["featurized_file_address"],
            split_type=split_config.get("split_type"),
            split_sizes=split_config.get("split_sizes"),
            output=split_config.get("output"),
        )

        # Train the model
        trained_model_address = self.train_client.run(
            dataset_address=train_split_address["train_split_address"],
            model_type=train_config.get("model_type"),
            model_name=train_config.get("model_name"),
            init_kwargs=train_config.get("init_kwargs"),
            train_kwargs=train_config.get("train_kwargs"),
        )

        # Evaluate the model
        evaluation_result_address = self.evaluate_client.run(
            dataset_addresses=(train_split_address, val_split_address, test_split_address),
            model_address=trained_model_address["trained_model_address"],
            metrics=evaluate_config.get("metrics"),
            output_key=evaluate_config.get("output_key"),
            is_metric_plots=evaluate_config.get("is_metric_plots"),
        )

        # Inference the model
        inference_result_address = self.infer_client.run(
            model_address=trained_model_address["trained_model_address"],
            data_address=test_split_address,
            output=infer_config.get("output"),
            dataset_column=infer_config.get("dataset_column"),
            shard_size=infer_config.get("shard_size"),
            threshold=infer_config.get("threshold"),
        )

        # Return the results
        return {
            "uploaded_files": file_path,
            "featurized_dataset_address": featurized_dataset_address,
            "train_split_address": train_split_address,
            "val_split_address": val_split_address,
            "test_split_address": test_split_address,
            "trained_model_address": trained_model_address,
            "evaluation_result_address": evaluation_result_address,
            "inference_result_address": inference_result_address
        }
