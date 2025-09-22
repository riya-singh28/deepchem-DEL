from pyds import Evaluate, Featurize, Infer, Train, Settings, Data, Split
import os
import tempfile


class Pipeline:
    def __init__(self, pipeline_config):
        self.pipeline_config = pipeline_config
        self.featurize_client = pipeline_config.get("featurize_client")
        self.split_client = pipeline_config.get("split_client")
        self.train_client = pipeline_config.get("train_client")
        self.evaluate_client = pipeline_config.get("evaluate_client")
        self.infer_client = pipeline_config.get("infer_client")
        self.upload_client = pipeline_config.get("upload_client")

    def upload_data(self, file_path, output=None, **kwargs):
        """
        Handles data upload using the upload_client.
        """
        if self.upload_client is None:
            raise ValueError("No upload_client provided in pipeline_config or notebook.")
        return self.upload_client.upload_data(
            file_path=file_path,
            output=output or "uploaded_dataset",
            **kwargs
        )

    def run(
        self,
        dataset_address=None,
        featurizer=None,
        featurizer_kwargs=None,
        split_type=None,
        split_sizes=None,
        model=None,
        model_init_kwargs=None,
        model_train_kwargs=None,
        metrics=None,
        output_prefix="test"
    ):
        # If config has any files to upload, upload them at the start
        uploaded_files = []
        config_files = self.pipeline_config.get("files_to_upload", [])
        if config_files and self.upload_client is None:
            raise ValueError("No upload_client available to upload files specified in config.")

        for file_info in config_files:
            # file_info can be a string (file path) or dict with more info
            if isinstance(file_info, str):
                upload_result = self.upload_data(file_info)
            elif isinstance(file_info, dict):
                file_path = file_info.get("file_path")
                upload_kwargs = {k: v for k, v in file_info.items() if k != "file_path"}
                upload_result = self.upload_data(file_path, **upload_kwargs)
            else:
                raise ValueError("Invalid file info in files_to_upload: {}".format(file_info))
            uploaded_files.append(upload_result)
        # If dataset_address is not provided, use the first uploaded file's dataset_address
        if dataset_address is None:
            if uploaded_files:
                dataset_address = uploaded_files[0].get("dataset_address")
            else:
                raise ValueError("No dataset_address provided and no files uploaded.")

        # Use config defaults if not provided
        featurizer = featurizer or self.pipeline_config.get("featurizer")
        featurizer_kwargs = featurizer_kwargs or self.pipeline_config.get("feat_kwargs", {})
        split_type = split_type or self.pipeline_config.get("split_type", "random")
        split_sizes = split_sizes or self.pipeline_config.get("split_sizes", [0.8, 0.1, 0.1])
        model = model or self.pipeline_config.get("model")
        model_init_kwargs = model_init_kwargs or self.pipeline_config.get("init_kwargs", {})
        model_train_kwargs = model_train_kwargs or self.pipeline_config.get("train_kwargs", {})
        metrics = metrics or self.pipeline_config.get("metrics")

        # Featurize
        featurized_dataset_address = self.featurize_client.run(
            dataset_address=dataset_address,
            featurizer=featurizer,
            featurizer_kwargs=featurizer_kwargs,
            output=f"{output_prefix}_featurized_dataset"
        )

        # Split
        train_split_address, val_split_address, test_split_address = self.split_client.run(
            dataset_address=featurized_dataset_address,
            split_type=split_type,
            split_sizes=split_sizes,
            output=f"{output_prefix}_split_output"
        )

        # Train
        trained_model_address = self.train_client.run(
            featurized_dataset_address=train_split_address,
            model=model,
            init_kwargs=model_init_kwargs,
            train_kwargs=model_train_kwargs,
            output=f"{output_prefix}_trained_model"
        )

        # Evaluate
        evaluation_result_address = self.evaluate_client.run(
            trained_model_address=trained_model_address,
            featurized_dataset_address=(train_split_address, val_split_address, test_split_address),
            metrics=metrics,
            output=f"{output_prefix}_evaluation_result"
        )

        # Inference
        inference_result_address = self.infer_client.run(
            trained_model_address=trained_model_address,
            featurized_dataset_address=test_split_address,
            output=f"{output_prefix}_inference_result"
        )

        return {
            "uploaded_files": uploaded_files,
            "featurized_dataset_address": featurized_dataset_address,
            "train_split_address": train_split_address,
            "val_split_address": val_split_address,
            "test_split_address": test_split_address,
            "trained_model_address": trained_model_address,
            "evaluation_result_address": evaluation_result_address,
            "inference_result_address": inference_result_address
        }
