import torch
from abc import ABC, abstractmethod
from torch import Tensor
from .metrics import prepare_y

def bound_loss(
    y_pred: Tensor,
    upper_bound: Tensor,
    lower_bound: Tensor,
    mask: Tensor,
    zero: Tensor,
    over_w: float = 1,
    under_w: float = 1,
) -> Tensor:
    """
    Compute bounded loss penalizing predictions outside the bounds.

    Calculates the absolute loss for predictions that exceed upper or lower bounds,
    with separate weighting for overcounting and undercounting violations.

    Args:
        y_pred: Model predictions with shape (batch, seq_len, output_dimensions)
        upper_bound: Upper bound constraints with shape (batch, seq_len, output_dimensions)
        lower_bound: Lower bound constraints with shape (batch, seq_len, output_dimensions)
        mask: Binary mask indicating valid (non-padded) positions with shape (batch, seq_len, output_dimensions)
        zero: Zero tensor on the same device as y_pred for numerical stability
        over_w: Weight for overcounting violations (default: 1)
        under_w: Weight for undercounting violations (default: 1)

    Returns:
        Absolute bounded loss with shape (batch, seq_len, output_dimensions)
    """
    # Compute overcounting: penalize predictions exceeding upper bound
    overcounting = over_w * torch.maximum(zero, y_pred - upper_bound)
    # Compute undercounting: penalize predictions below lower bound
    undercounting = under_w * torch.minimum(zero, y_pred - lower_bound)

    # Combine both violations
    error = overcounting - undercounting
    # Apply mask to ignore padded positions
    error = mask * error

    return torch.abs(error)

class BaseLoss(ABC):
    """Abstract base class for all loss functions."""

    def __init__(self) -> None:
        """Initialize the base loss class."""
        super(BaseLoss, self).__init__()

    @abstractmethod
    def __call__(
        self, y_true: Tensor, y_pred: Tensor, hit_mask: Tensor
    ) -> Tensor:
        """
        Compute the loss between true and predicted values.

        Args:
            y_true: True values with shape (batch, seq_len, output_dimensions)
            y_pred: Predicted values with shape (batch, seq_len, output_dimensions)
            hit_mask: Mask tensor with shape (batch, seq_len, output_dimensions)

        Returns:
            Scalar loss tensor
        """
        pass

class BoundOnlyLoss(BaseLoss):
    """Computes loss based on bound violations without considering temporal changes."""

    def __init__(
        self,
        balanced: bool = False,
        squared: bool = False,
        squared_scale: float = 1.0,
    ) -> None:
        """
        Initialize BoundOnlyLoss.

        Args:
            balanced: If True, normalize loss by the number of valid (non-padded) positions per batch.
            squared: If True, apply squared scaling to the loss.
            squared_scale: Scaling factor applied when squared=True.
        """
        super(BoundOnlyLoss, self).__init__()
        self.balanced = balanced
        self.squared = squared
        self.squared_scale = squared_scale

    def __call__(
        self, y_true: Tensor, y_pred: Tensor, hit_mask: Tensor
    ) -> Tensor:
        """
        Compute bounded loss for predictions.

        Args:
            y_true: True bounds with shape (batch, seq_len, 4), where first 2 channels are upper
                bounds and next 2 channels are lower bounds.
            y_pred: Predicted values with shape (batch, seq_len, 2).
            hit_mask: Not used in this loss function.

        Returns:
            Scalar loss value.
        """
        output_dimensions = 2
        zero = torch.tensor(0.0, device=y_pred.device)

        upper_bound = y_true[:, :, :output_dimensions]
        lower_bound = y_true[:, :, output_dimensions : 2 * output_dimensions]

        mask = (upper_bound >= zero).float()
        loss_values = bound_loss(y_pred, upper_bound, lower_bound, mask, zero)

        if self.squared:
            loss_values = self.squared_scale * loss_values**2

        if self.balanced:
            loss_values = torch.sum(loss_values, dim=1) / torch.sum(mask[:, :, 0], dim=1)

        return torch.mean(loss_values)

### NOT USED IN THESIS

class IncrementBoundsOnlyLoss(BaseLoss):
    """Computes loss based on bound violations and penalizes temporal changes in loss magnitude."""

    def __init__(
        self,
        squared: bool = False,
        squared_scale: float = 1.0,
        punish_corrections: bool = False,
        scale: float = 100,
        over_w: float = 1.0,
        under_w: float = 1.0,
    ) -> None:
        """
        Initialize IncrementBoundsOnlyLoss.

        Args:
            squared: If True, apply squared scaling to the loss.
            squared_scale: Scaling factor applied when squared=True.
            punish_corrections: If False, only penalizes increasing loss; if True, penalizes
                both increases and decreases.
            scale: Divisor for scaling the final loss value.
            over_w: Weight for overcounting violations in bound_loss.
            under_w: Weight for undercounting violations in bound_loss.
        """
        super(IncrementBoundsOnlyLoss, self).__init__()
        self.squared = squared
        self.squared_scale = squared_scale
        self.punish_corrections = punish_corrections
        self.scale = scale
        self.over_w = over_w
        self.under_w = under_w

    def __call__(
        self, y_true: Tensor, y_pred: Tensor, hit_mask: Tensor
    ) -> Tensor:
        """
        Compute increment-aware bounded loss for predictions.

        Args:
            y_true: True bounds with shape (batch, seq_len, 4), where first 2 channels are upper
                bounds and next 2 channels are lower bounds.
            y_pred: Predicted values with shape (batch, seq_len, 2).
            hit_mask: Not used in this loss function.

        Returns:
            Scalar loss value.
        """
        output_dimensions = 2
        zero = torch.tensor(0.0, device=y_pred.device)

        upper_bound = y_true[:, :, :output_dimensions]
        lower_bound = y_true[:, :, output_dimensions : 2 * output_dimensions]

        mask = (upper_bound >= zero).float()
        loss_values = bound_loss(
            y_pred,
            upper_bound,
            lower_bound,
            mask,
            zero,
            over_w=self.over_w,
            under_w=self.under_w,
        )

        if self.squared:
            loss_values = self.squared_scale * loss_values**2

        # Calculate temporal increments: loss change from t-1 to t
        zero_pad = torch.zeros_like(loss_values[:, :1, :], device=y_pred.device)
        loss_previous = torch.cat([zero_pad, loss_values[:, :-1, :]], dim=1)
        difference = loss_values - loss_previous

        if not self.punish_corrections:
            # Only penalize increasing loss (clamp to non-negative)
            loss = torch.abs(torch.clamp(difference, min=0))
        else:
            # Penalize both increases and decreases
            loss = torch.abs(difference)

        return torch.sum(loss) / self.scale

class MaskedLoss(BaseLoss):
    """Compute MAE on masked frames (only last frames of a sequence)."""

    def __init__(self) -> None:
        """Initialize MaskedLoss."""
        super(MaskedLoss, self).__init__()

    def __call__(
        self, y_true: Tensor, y_pred: Tensor, hit_mask: Tensor
    ) -> Tensor:
        """
        Compute masked loss by splitting predictions into separate channels.

        Args:
            y_true: True values with shape (batch, seq_len, 2).
            y_pred: Predicted values with shape (batch, seq_len, 2).
            hit_mask: Mask tensor with shape (batch, seq_len, output_dimensions).

        Returns:
            Scalar loss value (mean of losses for both channels).
        """
        y_true_a, y_pred_a = prepare_y(y_true, y_pred, hit_mask, 1)
        y_true_b, y_pred_b = prepare_y(y_true, y_pred, hit_mask, 0)

        loss_a = torch.mean(torch.abs(y_true_a - y_pred_a))
        loss_b = torch.mean(torch.abs(y_true_b - y_pred_b))

        return (loss_a + loss_b) / 2