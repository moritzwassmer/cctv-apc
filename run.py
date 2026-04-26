import os
os.environ["HYDRA_FULL_ERROR"] = "1" 
#os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'
os.environ["TORCH_CUDA_ARCH_LIST"]="8.0;8.6;9.0"

from dataclasses import dataclass

from napc.model import *
from napc.helpers import setup_mlflow

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import yaml

#from pytorch_lightning.loggers.mlflow import MLFlowLogger
#from napc.logger import BatchedMLFlowLogger as MLFlowLogger

from pytorch_lightning import Trainer

from typing import Dict, List, Tuple, Any

import mlflow
import time
import random
import shutil


class Run:
    """Orchestrates training/testing pipelines with Hydra configuration and MLflow logging."""
    def __init__(self, cfg: DictConfig) -> None:
        """Initialize Run with configuration and setup components.

        Args:
            cfg: DictConfig with full configuration tree
        """
        self.cfg: DictConfig = cfg
        self.run_cfg: DictConfig = cfg.run
        self.mlflow_creds: str = self.run_cfg.mlflow_creds

        # Initialize MLflow with credentials
        config: Dict[str, Any] = setup_mlflow(self.mlflow_creds)
        self.run_cfg.trainer.logger.tracking_uri = config["uri"]

        self._prepare_output_directory_and_logger()

        self.mode: str = self.run_cfg.mode
        self.ckpt_path: str = self.run_cfg.ckpt_path
        self.skip_meta: bool = self.run_cfg.skip_meta
        self.do_compile: bool = self.run_cfg.do_compile
        self.delete_cache_finally: bool = True

        self._init_components()


    def _prepare_output_directory_and_logger(self) -> None:
        """Initialize MLflow logger and create output directory for run artifacts."""
        # Instantiate MLflow logger early
        self.logger = instantiate(self.run_cfg.trainer.logger)

        # Create output directory based on run_id
        self.task_id: str = self.logger.run_id
        self.output_dir: str = os.path.join(os.getcwd(), "out", self.task_id)
        os.makedirs(self.output_dir, exist_ok=True)
        os.environ["RUN_OUT"] = self.output_dir

    def _init_components(self) -> None:
        """Initialize DataModule, Model, and Trainer components from configuration.

        Sets model input/output dimensions based on DataModule shapes and optionally compiles model.
        """
        self.dm = instantiate(self.run_cfg.dm)
        image_shape = self.dm.image_shape
        label_shape = self.dm.label_shape

        # Configure model dimensions
        self.run_cfg.model.input_dimensions = image_shape
        self.run_cfg.model.output_dimensions = label_shape // 2

        self.model = instantiate(self.run_cfg.model)

        if self.do_compile:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(self.output_dir, "compilation_cache")
            self.model.custom_compile()

        # Instantiate trainer with logger
        self.trainer: Trainer = instantiate(self.run_cfg.trainer, logger=self.logger)

    def _log_metadata(self) -> None:
        """Log configuration parameters and artifacts to MLflow.

        Flattens configuration tree, saves resolved/unresolved configs, and logs additional artifacts.
        """
        if self.cfg is None:
            return

        # Serialize configuration to dictionaries
        conf: Dict[str, Any] = OmegaConf.to_container(self.cfg.run, resolve=True, enum_to_str=True)
        conf_unresolved: Dict[str, Any] = OmegaConf.to_container(self.cfg.run, resolve=False, enum_to_str=True)
        flattened_conf: List[Tuple[str, Any]] = self._flatten_dict(conf)

        # Log parameters to MLflow
        params_dict: Dict[str, str] = {str(key): str(value) for key, value in flattened_conf}
        mlflow.log_params(params_dict, run_id=self.logger.run_id)

        # Save configuration files
        self._save_dict_config(conf, "config.yaml")
        self._save_dict_config(conf_unresolved, "config_unresolved.yaml")

        # Log additional artifacts if they exist
        for path in ["backup.7z", "environment.yaml"]:
            if os.path.exists(path):
                self.logger.experiment.log_artifact(self.logger.run_id, path)

    def execute(self) -> None:
        """Execute the run based on mode (train, test, train_test, or load).

        Logs metadata, runs training/testing as configured, handles cleanup and artifact logging.
        """
        try:
            # Log metadata for training/testing modes
            if self.mode in ("train", "test", "train_test"):
                if not self.skip_meta:
                    self._log_metadata()

            # Execute based on mode
            if self.mode == "train_test":
                self._train()
                self._test()
            elif self.mode == "train":
                self._train()
            elif self.mode == "test":
                self._test()
            elif self.mode == "load":
                self.delete_cache_finally = False
                print("Loaded config only, no action taken.")
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
        finally:
            # Clean up compilation cache
            if self.delete_cache_finally:
                cache_dir: str = os.path.join(self.output_dir, "compilation_cache")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)

            # Log remaining artifacts to MLflow
            self._log_directory()

    def _train(self) -> None:
        """Train the model, optionally resuming from a checkpoint."""
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            print(f"Continue training using checkpoint: {self.ckpt_path}")
            self.trainer.fit(self.model, self.dm, ckpt_path=self.ckpt_path)
        else:
            print("Training from scratch")
            self.trainer.fit(self.model, self.dm)

    def _test(self) -> None:
        """Test the model on test dataset, optionally using a checkpoint."""
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            print(f"Test Dataset using checkpoint: {self.ckpt_path}")
            self.trainer.test(self.model, datamodule=self.dm, ckpt_path=self.ckpt_path)
        else:
            print("Test Dataset using no checkpoint")
            self.trainer.test(self.model, datamodule=self.dm)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> List[Tuple[str, Any]]:
        """Flatten nested dictionary into list of (key_path, value) tuples.

        Args:
            d: Nested dictionary to flatten
            parent_key: Prefix for keys (used in recursion)
            sep: Separator for nested keys (default: '.')

        Returns:
            List of (flattened_key_path, value) tuples
        """
        stack: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = [((), d)]
        flat_items: List[Tuple[str, Any]] = []

        while stack:
            path, current = stack.pop()
            for k, v in current.items():
                new_path: Tuple[str, ...] = path + (k,)
                if isinstance(v, dict):
                    stack.append((new_path, v))
                else:
                    flat_items.append((sep.join(new_path), v))

        return flat_items

    def _save_dict_config(self, cfg: Dict[str, Any], out_name: str = "config.yaml") -> str:
        """Save configuration dictionary to YAML file.

        Args:
            cfg: Configuration dictionary to save
            out_name: Output filename (default: 'config.yaml')

        Returns:
            Path to saved configuration file
        """
        out_path: str = os.path.join(self.output_dir, out_name)
        with open(out_path, "w") as file:
            yaml.safe_dump(cfg, file, sort_keys=False)
        return out_path

    def _log_directory(self, exclude: List[str] = None) -> None:
        """Log output directory contents to MLflow as artifacts.

        Args:
            exclude: List of directory/file names to skip (default: ['compilation_cache'])
        """
        if exclude is None:
            exclude = ["compilation_cache"]

        for item in os.listdir(self.output_dir):
            if item in exclude:
                continue
            item_path: str = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                self.logger.experiment.log_artifacts(self.logger.run_id, item_path, artifact_path=item)
            else:
                self.logger.experiment.log_artifact(self.logger.run_id, item_path)



# Hydra entrypoint
@hydra.main(version_base=None, config_path="./napc/configs")
def main(cfg: DictConfig) -> None:
    """Main entry point orchestrating Hydra initialization and run execution.

    Adds random delay to stagger parallel job submissions, then creates and executes a Run.

    Args:
        cfg: Hydra configuration object
    """
    time.sleep(random.randint(0, 30))
    runner: Run = Run(cfg)
    runner.execute()


if __name__ == "__main__":
    main()
