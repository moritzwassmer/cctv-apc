import torch
from abc import ABC, abstractmethod
from torch import Tensor
from torchmetrics import Metric
from torchmetrics import MeanAbsoluteError


def prepare_y(
    y_true: Tensor, y_pred: Tensor, hit_mask: Tensor, class_idx: int
) -> tuple[Tensor, Tensor]:
    """Prepare tensors by extracting a class and applying a mask.

    Extracts a specific class dimension and applies a mask to keep only valid frames (no padding).

    Args:
        y_true: True values shaped (B, F, 2) where the last dim is (boarding, alighting).
        y_pred: Predicted values shaped (B, F, 2).
        hit_mask: Mask shaped (B, F, 1) where values > 0 indicate valid frames.
        class_idx: Index of the class/channel to extract.

    Returns:
        Tuple of (masked_y_true, masked_y_pred) flattened to 1D.
    """
    # Extract class dimension
    y_true = y_true[:, :, class_idx].unsqueeze(-1)
    y_pred = y_pred[:, :, class_idx].unsqueeze(-1)

    # Create and apply mask for valid positions
    mask = (hit_mask > 0)
    mask = torch.cat([mask], -1).bool()

    masked_y_true = y_true[mask]
    masked_y_pred = y_pred[mask]
    return masked_y_true, masked_y_pred


class BaseMetric(Metric, ABC):
    """Abstract base class for all metrics

    Handles class filtering and masking of labels and predictions before
    delegating to subclass-specific metric computation.
    """

    def __init__(self, class_idx: int | None) -> None:
        """Initialize BaseMetric.

        Args:
            class_idx: Index of the class to compute metric for (boarding=0, alighting=1), or None
        """
        self.class_idx = class_idx
        super(BaseMetric, self).__init__()

    def update(self, y_true: Tensor, y_pred: Tensor, hit_mask: Tensor | None = None) -> None:
        """Update metric with new batch of predictions.

        Args:
            y_true: True values shaped (B, F, 2) where dim2=(boarding, alighting).
            y_pred: Predicted values shaped (B, F, 2).
            hit_mask: Mask shaped (B, F, 1) for valid frames (no padding, only last frame).

        Returns:
            None
        """
        if self.class_idx is not None and hit_mask is None:
            raise ValueError("hit_mask must be provided if class_idx is not None")

        if self.class_idx is not None:
            y_true, y_pred = prepare_y(y_true, y_pred, hit_mask, self.class_idx)

        self.custom_update(y_true, y_pred)

    @abstractmethod
    def custom_update(self, y_true: Tensor, y_pred: Tensor) -> None:
        """Update metric with preprocessed tensors.

        Note: If class_idx is set, tensors are flattened after masking and class extraction.

        Args:
            y_true: Processed true values with shape (flat_shape,).
            y_pred: Processed predicted values with shape (flat_shape,).

        Returns:
            None
        """
        pass


class MAE(BaseMetric):
    """Mean Absolute Error metric"""

    def __init__(self, class_idx: int | None = None) -> None:
        """Initialize MAE metric.

        Args:
            class_idx: Index of the class to compute metric for.
        """
        super().__init__(class_idx)
        self.mae = MeanAbsoluteError()

    def custom_update(self, y_true: Tensor, y_pred: Tensor) -> None:
        """Update MAE with batch of predictions.

        Args:
            y_true: True values with shape (flat_shape,).
            y_pred: Predicted values with shape (flat_shape,).

        Returns:
            None
        """
        self.mae(y_pred, y_true)

    def compute(self) -> Tensor:
        """Compute mean absolute error.

        Returns:
            Scalar MAE value.
        """
        return self.mae.compute()

    def reset(self) -> None:
        """Reset metric state.

        Returns:
            None
        """
        self.mae.reset()


class Accuracy(BaseMetric):
    """Classification accuracy metric"""

    def __init__(self, class_idx: int | None = None) -> None:
        """Initialize Accuracy metric.

        Args:
            class_idx: Index of the class to compute metric for.
        """
        super().__init__(class_idx)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def custom_update(self, y_true: Tensor, y_pred: Tensor) -> None:
        """Update accuracy with batch of predictions.

        Args:
            y_true: True values with shape (flat_shape,).
            y_pred: Predicted values with shape (flat_shape,).

        Returns:
            None
        """
        # Clamp and round predictions to nearest integer for classification
        y_pred = torch.clamp(y_pred, min=1e-8)
        y_pred = torch.round(y_pred).to(torch.int64)
        y_true = torch.round(y_true).to(torch.int64)

        self.correct += torch.sum(y_true == y_pred)
        self.total += torch.numel(y_true)

    def compute(self) -> Tensor:
        """Compute accuracy.

        Returns:
            Scalar accuracy value in [0, 1].
        """
        return torch.nan_to_num(self.correct / self.total)
    

class AverageMetric(Metric):
    """Aggregates results from multiple metrics by averaging.

    Useful for computing metrics across multiple classes or dimensions.
    """

    def __init__(self, metrics: list[BaseMetric]) -> None:
        """Initialize AverageMetric.

        Args:
            metrics: List of metric instances to aggregate.
        """
        super().__init__()
        self.metrics = metrics

    def update(self, y_true: Tensor, y_pred: Tensor, hit_mask: Tensor) -> None:
        """Update all metrics with new batch.

        Args:
            y_true: True values shaped (B, F, 2) where dim2=(boarding, alighting).
            y_pred: Predicted values shaped (B, F, 2).
            hit_mask: Mask shaped (B, F, 1) for valid frames.

        Returns:
            None
        """
        # Move metrics to device and update each one
        for metric in self.metrics:
            metric = metric.to(y_pred.device)

        for metric in self.metrics:
            metric.update(y_true, y_pred, hit_mask)

    def compute(self) -> Tensor:
        """Compute average of all metrics.

        Returns:
            Scalar average value, or 0.0 if no valid results.
        """
        # Collect results from each metric, filtering out NaNs
        results = []
        for metric in self.metrics:
            result = metric.compute()
            if not torch.isnan(result):
                results.append(result)

        # Average valid results
        if results:
            results_tensor = torch.stack(results)
            return torch.mean(results_tensor.float())
        else:
            return torch.tensor(0, device=self.device)

    def reset(self) -> None:
        """Reset all metrics.

        Returns:
            None
        """
        for metric in self.metrics:
            metric.reset()


class SMAPE(BaseMetric):
    """Symmetric Mean Absolute Percentage Error metric.

    Measures the average percentage difference between predictions and targets.
    """

    def __init__(self, class_idx: int | None = None) -> None:
        """Initialize SMAPE metric.

        Args:
            class_idx: Index of the class to compute metric for.
        """
        super().__init__(class_idx)
        self.add_state("sum_abs_per_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def custom_update(self, y_true: Tensor, y_pred: Tensor) -> None:
        """Update SMAPE with batch of predictions.

        Args:
            y_true: True values with shape (flat_shape,).
            y_pred: Predicted values with shape (flat_shape,).

        Returns:
            None
        """
        # Symmetric percentage error: |pred - true| / ((true + pred)/2)
        abs_per_error = torch.abs(y_pred - y_true) / ((y_true + y_pred) / 2 + 1e-8)
        self.sum_abs_per_error += torch.sum(abs_per_error)
        self.total += len(y_pred)

    def compute(self) -> Tensor:
        """Compute SMAPE.

        Returns:
            Scalar SMAPE value.
        """
        return self.sum_abs_per_error / self.total


class DebugMetric(BaseMetric):
    """Debug metric that stores predictions for inspection without computing a score."""

    def __init__(self, class_idx: int | None = None) -> None:
        """Initialize DebugMetric.

        Args:
            class_idx: Index of the class to compute metric for.
        """
        self.add_state("y_true_list", default=[], dist_reduce_fx=None)
        self.add_state("y_pred_list", default=[], dist_reduce_fx=None)
        super().__init__(class_idx)

    def custom_update(self, y_true: Tensor, y_pred: Tensor) -> None:
        """Store predictions for debugging.

        Args:
            y_true: True values with shape (flat_shape,).
            y_pred: Predicted values with shape (flat_shape,).

        Returns:
            None
        """
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)

    def compute(self) -> Tensor:
        """Return dummy value (use y_true_list and y_pred_list for inspection).

        Returns:
            Zero tensor.
        """
        return torch.tensor(0, device=self.device)


class GRB(BaseMetric):
    """Global Relative Bias metric (mean prediction bias / mean true value).

    Measures systematic over- or under-counting.
    """

    def __init__(self, class_idx: int | None = None, absolute: bool = False) -> None:
        """Initialize GRB metric.

        Args:
            class_idx: Index of the class to compute metric for.
            absolute: If True, return absolute value of GRB.
        """
        super().__init__(class_idx)
        self.add_state("sum_y_true", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("diffs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.absolute = absolute

    def custom_update(self, y_true: Tensor, y_pred: Tensor) -> None:
        """Update GRB with batch of predictions.

        Args:
            y_true: True values with shape (flat_shape,).
            y_pred: Predicted values with shape (flat_shape,).

        Returns:
            None
        """
        self.sum_y_true += y_true.sum()
        self.diffs += (y_pred - y_true).sum()

    def compute(self) -> Tensor:
        """Compute GRB.

        Returns:
            Scalar GRB value. Positive means over-counting, negative means under-counting.
        """
        if self.sum_y_true == 0:
            return torch.tensor(0, device=self.device)

        grb = self.diffs / self.sum_y_true

        return torch.abs(grb) if self.absolute else grb


class GRBAwareAccuracy(BaseMetric):
    """Accuracy metric penalized by systematic bias (GRB).

    Combines classification accuracy with penalty for large GRB deviations.
    Final score = max(0, min(1, accuracy - grb_weight * |GRB|)).
    """

    def __init__(self, class_idx: int | None = None, grb_weight: float = 100.0) -> None:
        """Initialize GRBAwareAccuracy metric.

        Args:
            class_idx: Index of the class to compute metric for.
            grb_weight: Penalty weight for GRB deviations (0.1 GRB change ≈ 0.01 accuracy change).
        """
        super().__init__(class_idx)
        self.grb_weight = grb_weight
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y_true", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("diffs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def custom_update(self, y_true: Tensor, y_pred: Tensor) -> None:
        """Update GRB-aware accuracy with batch of predictions.

        Args:
            y_true: True values with shape (flat_shape,).
            y_pred: Predicted values with shape (flat_shape,).

        Returns:
            None
        """
        # Clamp and round predictions for classification
        y_pred = torch.clamp(y_pred, min=1e-8)
        y_pred = torch.round(y_pred).to(torch.int64)
        y_true = torch.round(y_true).to(torch.int64)

        # Accumulate accuracy components
        self.correct += torch.sum(y_true == y_pred).float()
        self.total += torch.numel(y_true)

        # Accumulate GRB components
        self.sum_y_true += y_true.sum().float()
        self.diffs += (y_pred - y_true).sum().float()

    def compute(self) -> Tensor:
        """Compute GRB-aware accuracy.

        Returns:
            Scalar score in [0, 1]. Higher is better.
        """
        if self.total == 0 or self.sum_y_true == 0:
            return torch.tensor(0.0, device=self.device)

        # Compute components
        accuracy = self.correct / self.total
        grb = self.diffs / self.sum_y_true

        # Apply penalty for bias
        score = accuracy - self.grb_weight * torch.abs(grb)

        # Clamp to valid range
        score = torch.clamp(score, min=0.0, max=1.0)
        return torch.nan_to_num(score, nan=0.0)
