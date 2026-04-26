# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/loggers/mlflow.html#MLFlowLogger
# modified by Moritz Wassmer 2025 & 2026

import os
import re
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional

from typing_extensions import override

from lightning_fabric.utilities.logger import _add_prefix
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.loggers.mlflow import MLFlowLogger

class BatchedMLFlowLogger(MLFlowLogger):
    """MLFlow logger that batches metric logging to reduce overhead.

    Caches metrics and logs them in batches to reduce the number of calls to MLFlow,
    improving performance during training.
    """

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: Optional[str] = None,
        run_id: Optional[str] = None,
        synchronous: Optional[bool] = None,
        log_metrics_every: int = 50,
    ) -> None:
        """Initialize BatchedMLFlowLogger.

        Args:
            experiment_name: Name of the MLFlow experiment.
            run_name: Name of the MLFlow run.
            tracking_uri: URI for MLFlow tracking server.
            tags: Tags to assign to the run.
            save_dir: Local directory to save MLFlow runs.
            log_model: Whether to log model checkpoints.
            prefix: Prefix to add to all metric names.
            artifact_location: Location to store artifacts.
            run_id: ID of an existing run to continue logging to.
            synchronous: Whether to log synchronously.
            log_metrics_every: Number of log_metrics calls before flushing cache.
        """
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            artifact_location=artifact_location,
            run_id=run_id,
            synchronous=synchronous,
        )
        # Number of log_metrics calls before flushing cache to MLFlow
        self.log_metrics_every = log_metrics_every
        # Counter for tracking number of log_metrics calls
        self._log_metrics_counter = 0
        # Buffer for cached metrics to be logged in batch
        self._cached_metrics: List[Dict[str, Any]] = []

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLFlow, batching them for efficiency.

        Caches metrics and only flushes them to MLFlow after log_metrics_every calls.
        Sanitizes metric names to comply with MLFlow requirements.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional training step number to associate with metrics.

        Returns:
            None
        """
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        from mlflow.entities import Metric

        # Increment counter and prepare metrics
        self._log_metrics_counter += 1
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        timestamp_ms = int(time() * 1000)

        # Process and cache each metric
        for key, value in metrics.items():
            if isinstance(value, str):
                rank_zero_warn(f"Discarding metric with string value {key}={value}.")
                continue

            # Sanitize metric name to comply with MLFlow naming rules
            sanitized_key = re.sub(r"[^a-zA-Z0-9_/. -]+", "", key)
            if key != sanitized_key:
                rank_zero_warn(
                    f"MLFlow only allows '_', '/', '.' and ' ' in metric names. "
                    f"Renaming {key} to {sanitized_key}.",
                    category=RuntimeWarning,
                )
                key = sanitized_key

            self._cached_metrics.append(
                Metric(key=key, value=value, timestamp=timestamp_ms, step=step or 0)
            )

        # Flush cached metrics when batch size is reached
        if self._log_metrics_counter >= self.log_metrics_every:
            self._cached_metrics.sort(key=lambda m: (m.step, m.timestamp))
            self.experiment.log_batch(
                run_id=self.run_id, metrics=self._cached_metrics, **self._log_batch_kwargs
            )
            self._cached_metrics.clear()
            self._log_metrics_counter = 0

    @override
    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """Finalize the run: flush remaining metrics, log checkpoints, and terminate run.

        Args:
            status: Final status of the run ("success", "failed", or "finished").

        Returns:
            None
        """
        if not self._initialized:
            return

        # Normalize status to MLFlow format
        status_map = {"success": "FINISHED", "failed": "FAILED", "finished": "FINISHED"}
        normalized_status = status_map.get(status, status.upper())

        # Flush any remaining cached metrics before finalizing
        if self._cached_metrics:
            self._cached_metrics.sort(key=lambda m: (m.step, m.timestamp))
            self.experiment.log_batch(
                run_id=self.run_id, metrics=self._cached_metrics, **self._log_batch_kwargs
            )
            self._cached_metrics.clear()
            self._log_metrics_counter = 0

        # Log checkpoints as artifacts if available
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

        # Mark run as terminated with final status
        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, normalized_status)