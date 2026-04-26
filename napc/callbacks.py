import time
from typing import Any, Dict, Optional, Tuple

import GPUtil
import mlflow
import pandas as pd
import psutil
import torch
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import MetricCollection

from .metrics import GRB, MAE, SMAPE, Accuracy, AverageMetric, GRBAwareAccuracy

class CSV_log_pred_cctv(Callback):
    """Logs test predictions to CSV by matching stream IDs to metadata.

    Note:
        Assumes no concatenated streams; uses the first stream in each batch.
        Not tested with multiple dataset workers.
    """

    def __init__(
        self,
        metadata_path: str = "/net/vericon/napc_data/cctv_data/cctv_passenger_bb_sel_350_100_25_2.csv",
        output_path: str = "predictions.csv",
    ) -> None:
        super().__init__()
        self.metadata_path = metadata_path
        self.output_path = output_path
        self.metadata: Optional[pd.DataFrame] = None
        self.stream_index_map: Dict[str, int] = {}
    
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Load metadata CSV and initialize prediction columns."""
        self.metadata = pd.read_csv(
            self.metadata_path,
            header=None,
            names=[
                "fname",
                "n_boarding",
                "n_alighting",
                "n_frame",
                "event_boarding",
                "bb_boarding",
                "event_alighting",
                "bb_alighting",
                "n_boarding_N",
                "n_alighting_N",
                "frame_N",
            ],
        )
        self.metadata["b_pred"] = None
        self.metadata["a_pred"] = None

        # Create a mapping for stream indexing
        self.stream_index_map = {stream: i for i, stream in enumerate(self.metadata["fname"])}

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Tuple[torch.Tensor, ...],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Record predictions for each stream at the final frame (hit_mask=1)."""
        x, y_true, seq_lengths, hit_mask, stream_ids = batch

        y_pred = pl_module.predict_step(batch, batch_idx)
        y_pred = y_pred.detach().cpu().numpy()

        for i, stream in enumerate(stream_ids):
            stream_name = stream[0]  # Take first stream in list
            index = self.stream_index_map.get(stream_name)
            if index is not None:
                hit_index = (hit_mask[i].squeeze(-1) == 1).nonzero(as_tuple=True)[0].item()
                hit_index2 = seq_lengths[i] - 1
                assert (
                    hit_index == hit_index2
                ), f"Hit index from mask {hit_index} does not match seq length -1 {hit_index2}"
                self.metadata.at[index, "b_pred"] = y_pred[i, hit_index, 0]  # Boarding prediction
                self.metadata.at[index, "a_pred"] = y_pred[i, hit_index, 1]  # Alighting prediction

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save predictions to CSV, dropping rows without predictions."""
        self.metadata = self.metadata.dropna(subset=["a_pred", "b_pred"])
        self.metadata.to_csv(self.output_path, index=False)

class SystemMetricsLogger(Callback):
    """Logs CPU, RAM, and GPU metrics during training; samples every 50 batches and logs maximums."""

    def __init__(self) -> None:
        super().__init__()
        self._batch_counter: int = 0
        self._total_gpu_memory_logged: bool = False

        # Max values collected every 50 batches
        self._max_cpu_usage: float = 0.0
        self._max_ram_usage: float = 0.0
        self._max_gpu_mem_used: float = 0.0  # MB
        self._max_gpu_mem_util: float = 0.0  # Percent
        self._max_gpu_load: float = 0.0  # Percent

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log total GPU memory once at start."""
        if not self._total_gpu_memory_logged and torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                total_mem = gpus[0].memoryTotal  # MB
                pl_module.log("zsys_total_gpu_memory_mb_gputil", total_mem, on_step=False, on_epoch=True)
                self._total_gpu_memory_logged = True

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset peak memory stats and tracking counters."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)

        self._batch_counter = 0
        self._max_cpu_usage = 0.0
        self._max_ram_usage = 0.0
        self._max_gpu_mem_used = 0.0
        self._max_gpu_mem_util = 0.0
        self._max_gpu_load = 0.0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Sample CPU, RAM, and GPU metrics every 50 batches."""
        self._batch_counter += 1
        if self._batch_counter % 50 == 0:
            # CPU & RAM (psutil)
            self._max_cpu_usage = max(self._max_cpu_usage, psutil.cpu_percent())
            self._max_ram_usage = max(self._max_ram_usage, psutil.virtual_memory().percent)

            # GPU (GPUtil)
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self._max_gpu_mem_used = max(self._max_gpu_mem_used, gpu.memoryUsed)
                self._max_gpu_mem_util = max(self._max_gpu_mem_util, gpu.memoryUtil * 100)
                self._max_gpu_load = max(self._max_gpu_load, gpu.load * 100)

    def log_metrics(self, pl_module: LightningModule) -> None:
        """Log all accumulated max metrics to MLflow/logger."""
        # PyTorch CUDA metrics
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
            peak_reserved = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
        else:
            peak_allocated = 0.0
            peak_reserved = 0.0

        pl_module.log("zsys_train_peak_gpu_memory_allocated_mb_torch", peak_allocated, on_step=False, on_epoch=True)
        pl_module.log("zsys_train_peak_gpu_memory_reserved_mb_torch", peak_reserved, on_step=False, on_epoch=True)

        # Log max CPU/RAM metrics (psutil)
        if self._max_cpu_usage > 0:
            pl_module.log("zsys_train_max_cpu_usage_percent_psutil", self._max_cpu_usage, on_step=False, on_epoch=True)
        if self._max_ram_usage > 0:
            pl_module.log("zsys_train_max_ram_usage_percent_psutil", self._max_ram_usage, on_step=False, on_epoch=True)

        # Log max GPU metrics (gputil)
        if self._max_gpu_mem_used > 0:
            pl_module.log(
                "zsys_train_max_gpu_memory_reserved_mb_gputil", self._max_gpu_mem_used, on_step=False, on_epoch=True
            )
        if self._max_gpu_mem_util > 0:
            pl_module.log(
                "zsys_train_max_gpu_memory_util_percent_gputil", self._max_gpu_mem_util, on_step=False, on_epoch=True
            )
        if self._max_gpu_load > 0:
            pl_module.log("zsys_train_max_gpu_load_percent_gputil", self._max_gpu_load, on_step=False, on_epoch=True)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log metrics at epoch end."""
        self.log_metrics(pl_module)

class SystemMetricsLoggerBenchmark(Callback):
    """Lightweight version of SystemMetricsLogger focused on GPU memory for benchmarking."""

    def __init__(self, reset_first_n_epochs: int = 1) -> None:
        super().__init__()
        self._total_gpu_memory_logged: bool = False
        self.reset_first_n_epochs = reset_first_n_epochs

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log total GPU memory once at start."""
        if not self._total_gpu_memory_logged and torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                total_mem = gpus[0].memoryTotal  # MB
                pl_module.log("zsys_total_gpu_memory_mb_gputil", total_mem, on_step=False, on_epoch=True)
                self._total_gpu_memory_logged = True

    def log_metrics(self, pl_module: LightningModule) -> None:
        """Log peak GPU memory usage."""
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
            peak_reserved = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
        else:
            peak_allocated = 0.0
            peak_reserved = 0.0

        pl_module.log("zsys_train_peak_gpu_memory_allocated_mb_torch", peak_allocated, on_step=False, on_epoch=True)
        pl_module.log("zsys_train_peak_gpu_memory_reserved_mb_torch", peak_reserved, on_step=False, on_epoch=True)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log metrics at epoch end."""
        self.log_metrics(pl_module)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset peak memory stats after the first N epochs."""
        if torch.cuda.is_available() and self.reset_first_n_epochs <= trainer.current_epoch:
            torch.cuda.reset_peak_memory_stats(0)

class ShuffleCallback(Callback):
    """Shuffles training dataset indices at the start of each epoch."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Shuffle the training dataset if it has a shuffle_dataset method."""
        datamodule = trainer.datamodule
        if hasattr(datamodule, "train") and hasattr(datamodule.train, "shuffle_dataset"):
            datamodule.train.shuffle_dataset()

class LogBatchSizeCallback(Callback):
    """Logs the batch size to MLflow after the first training epoch."""

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log batch size to MLflow after the first epoch."""
        if trainer.current_epoch == 0:
            batch_size = trainer.datamodule.batch_size
            mlflow.log_params({"batch_size": str(batch_size)}, run_id=trainer.logger.run_id)

class TrainingTimeTrackerCallback(Callback):
    """Tracks elapsed training time in minutes and logs it each epoch."""

    def __init__(self, reset_first_n_epochs: int = 0) -> None:
        super().__init__()
        self.start_time: float = time.time()  # Default to avoid errors if on_train_epoch_start not called
        self.reset_first_n_epochs = reset_first_n_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset timer after the first N epochs."""
        if trainer.current_epoch <= self.reset_first_n_epochs:
            self.start_time = time.time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log elapsed time in minutes."""
        elapsed_time = (time.time() - self.start_time) / 60
        elapsed_time = round(elapsed_time, 2)
        pl_module.log("train_minutes", elapsed_time, on_step=False, on_epoch=True)

class NumParamsCallback(Callback):
    """Logs total and trainable parameter counts to MLflow at training start."""

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Count and log model parameters."""
        model = pl_module
        all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({"nParameters": str(all_params)}, run_id=trainer.logger.run_id)
        mlflow.log_params({"nParameters_trainable": str(trainable_params)}, run_id=trainer.logger.run_id)

class DebugLRScheduler(Callback):
    """Prints the current learning rate at each training batch for debugging."""

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Print learning rate at the start of each batch."""
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        print(f"Step {trainer.global_step}: lr={lr:.6f}")

class MetricLoggerCallback(Callback):
    """Logs custom metrics (Accuracy, MAE, SMAPE, GRB, GRB-aware accuracy) from model outputs."""

    def __init__(self) -> None:
        super().__init__()

        # Helper to construct the complete metric collection
        def make_metric_collection():
            return MetricCollection({
                # --- Basic metrics ---
                "Acc_a": Accuracy(class_idx=1),
                "Acc_b": Accuracy(class_idx=0),
                "avg_Acc": AverageMetric([
                    Accuracy(class_idx=1),
                    Accuracy(class_idx=0),
                ]),

                "MAE_a": MAE(class_idx=1),
                "MAE_b": MAE(class_idx=0),
                "avg_MAE": AverageMetric([
                    MAE(class_idx=1),
                    MAE(class_idx=0),
                ]),

                "SMAPE_a": SMAPE(class_idx=1),
                "SMAPE_b": SMAPE(class_idx=0),
                "avg_SMAPE": AverageMetric([
                    SMAPE(class_idx=1),
                    SMAPE(class_idx=0),
                ]),

                # --- GRB and variants ---
                "GRB_a": GRB(class_idx=1),
                "GRB_b": GRB(class_idx=0),
                "avg_GRB": AverageMetric([
                    GRB(class_idx=1),
                    GRB(class_idx=0),
                ]),

                "abs_grb_a": GRB(class_idx=1, absolute=True),
                "abs_grb_b": GRB(class_idx=0, absolute=True),
                "avg_abs_grb": AverageMetric([
                    GRB(class_idx=1, absolute=True),
                    GRB(class_idx=0, absolute=True),
                ]),

                # --- GRB-aware accuracy ---
                "grbacc_a": GRBAwareAccuracy(class_idx=1),
                "grbacc_b": GRBAwareAccuracy(class_idx=0),
                "avg_grbacc": AverageMetric([
                    GRBAwareAccuracy(class_idx=1),
                    GRBAwareAccuracy(class_idx=0),
                ]),
            })

        # Separate metric sets for train/val/test
        self.train_metrics = make_metric_collection()
        self.val_metrics = make_metric_collection()
        self.test_metrics = make_metric_collection()

    @torch.no_grad()
    def _update_from_outputs(
        self, pl_module: LightningModule, outputs: Optional[Dict[str, Any]], batch: Tuple[torch.Tensor, ...], stage: str
    ) -> None:
        """Update metrics using y_true and y_pred from step outputs.

        Args:
            pl_module: Lightning module
            outputs: Dict with 'y_true' and 'y_pred' keys (B, F, 2)
            batch: Tuple containing hit_mask (B, F, 1)
            stage: 'train', 'val', or 'test'
        """
        if outputs is None or "y_true" not in outputs or "y_pred" not in outputs:
            return

        _, _, _, hit_mask, _ = batch
        device = pl_module.device
        hit_mask = hit_mask.to(device)

        y_true = outputs["y_true"].to(device)
        y_pred = outputs["y_pred"].to(device)

        metric_collection = getattr(self, f"{stage}_metrics").to(device)
        metric_collection.update(y_true, y_pred, hit_mask)

    def _compute_and_log(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Compute and log all metrics at the end of an epoch."""
        metric_collection = getattr(self, f"{stage}_metrics")
        results = metric_collection.compute()

        for name, value in results.items():
            val = value.item() if torch.is_tensor(value) else float(value)
            pl_module.log(f"{stage}_{name}", val, on_step=False, on_epoch=True)

        metric_collection.reset()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Update train metrics after each batch."""
        self._update_from_outputs(pl_module, outputs, batch, "train")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log train metrics at epoch end."""
        self._compute_and_log(trainer, pl_module, "train")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update validation metrics after each batch."""
        self._update_from_outputs(pl_module, outputs, batch, "val")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log validation metrics at epoch end."""
        self._compute_and_log(trainer, pl_module, "val")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update test metrics after each batch."""
        self._update_from_outputs(pl_module, outputs, batch, "test")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log test metrics at epoch end."""
        self._compute_and_log(trainer, pl_module, "test")
