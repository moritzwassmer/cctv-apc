import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from .lr_scheduler import LinearWarmupCosineAnnealing
from .losses import *

from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)

class DenseInput(nn.Module):
    """Flattens spatial dimensions and projects frames to an embedding; (B, F, H, W) -> (B, F, E)."""

    def __init__(
        self,
        input_dimensions: List[int] = [30, 40],
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dimensions = input_dimensions
        self.embedding_dim = embedding_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=int(np.prod(self.input_dimensions)), out_features=self.embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten each frame and embed to `embedding_dim`. Expects x: (B, F, H, W)."""
        batch_size, seq_len, height, width = x.shape
        x = x.view(batch_size, seq_len, -1)
        x = self.layers(x)
        return x

class LSTMModule(nn.Module):
    """Stacked single-layer LSTMs that can progressively shrink the embedding size."""

    def __init__(self, embedding_dims: List[int] = [130, 70, 50], dropout: float = 0.2) -> None:
        super().__init__()

        self.embedding_dims = embedding_dims
        self.num_layers = len(embedding_dims)
        self.dropout = dropout

        self.lstm_layers = nn.ModuleList()

        in_dim = embedding_dims[0]
        for out_dim in embedding_dims:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=out_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout,
                )
            )
            in_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the sequence through each LSTM layer; input/outputs shaped (B, F, E)."""
        output = x
        for lstm in self.lstm_layers:
            output, _ = lstm(output)
        return output

class LSTMModuleFast(nn.Module):
    """Single multi-layer LSTM for faster execution than a Python-level stack."""

    def __init__(
        self,
        input_dim: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.layers = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.embedding_dim,
            batch_first=True,
            dropout=self.dropout_rate,
            num_layers=self.num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence (B, F, E) with a multi-layer LSTM and return all steps."""
        x, _ = self.layers(x)
        return x

class DenseOutput(nn.Module):
    """Projection head for per-frame outputs; (B, F, E) -> (B, F, output_dim)."""

    def __init__(
        self,
        embedding_dim: int = 128,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.embedding_dim, self.output_dim),
            nn.LeakyReLU(),  # TODO: really leaky relu as output activation?
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings to outputs; keeps shape (B, F, output_dim)."""
        x = self.layers(x)
        return x
    
class ConvInput(nn.Module):
    """3D CNN encoder over (frames, H, W) producing per-frame embeddings."""
    def __init__(
        self,
        input_dimensions: Tuple[int, int] = (30, 40),
        embedding_dim: int = 128,
        conv_pool_layer: int = 3,
        conv_filters: List[int] = [8, 16, 32],
        conv_kernel: List[Tuple[int, int, int]] = [(1, 2, 2)] * 3,
        conv_pad: List[Tuple[int, int, int]] = [(0, 1, 1)] * 3,
        conv_stride: List[Tuple[int, int, int]] = [(1, 1, 1)] * 3,
        pool_pad: List[Tuple[int, int, int]] = [(0, 0, 0)] * 3,
        pool_kernel: List[Tuple[int, int, int]] = [(1, 2, 2)] * 3,
        pool_stride: List[Tuple[int, int, int]] = [(1, 2, 2)] * 3,
        use_batchnorm: bool = True,
        use_groupnorm: bool = False,
        num_groups: int = 4,
    ) -> None:
        super().__init__()
        self.input_dimensions = input_dimensions
        self.embedding_dim = embedding_dim
        self.conv_pool_layer = conv_pool_layer
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.conv_pad = conv_pad
        self.conv_stride = conv_stride
        self.pool_pad = pool_pad
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.use_batchnorm = use_batchnorm
        self.use_groupnorm = use_groupnorm
        self.num_groups = num_groups

        if self.use_batchnorm and self.use_groupnorm:
            raise ValueError("Cannot use both batchnorm and groupnorm at the same time.")

        self.conv_pool_layers = nn.ModuleList()
        current_shape = (1, 1270, *self.input_dimensions)  # (C, F, H, W)

        for idx in range(self.conv_pool_layer):
            conv = nn.Conv3d(
                in_channels=current_shape[0],
                out_channels=self.conv_filters[idx],
                kernel_size=self.conv_kernel[idx],
                padding=self.conv_pad[idx],
                stride=self.conv_stride[idx]
            )
            self.conv_pool_layers.append(conv)
            current_shape = self.calculate_output_shape(
                current_shape,
                self.conv_kernel[idx],
                self.conv_pad[idx],
                self.conv_stride[idx],
                self.conv_filters[idx]
            )

            if self.use_batchnorm:
                self.conv_pool_layers.append(nn.BatchNorm3d(self.conv_filters[idx]))
            
            if self.use_groupnorm:
                # GroupNorm will be applied in forward after reshaping (B*F, C, H, W)
                gn = nn.GroupNorm(num_groups=min(self.num_groups, self.conv_filters[idx]),
                                  num_channels=self.conv_filters[idx])
                self.conv_pool_layers.append(gn)

            self.conv_pool_layers.append(nn.LeakyReLU())
            self.conv_pool_layers.append(nn.MaxPool3d(
                kernel_size=self.pool_kernel[idx],
                padding=self.pool_pad[idx],
                stride=self.pool_stride[idx]
            ))

            current_shape = self.calculate_output_shape(
                current_shape,
                self.pool_kernel[idx],
                self.pool_pad[idx],
                self.pool_stride[idx],
                self.conv_filters[idx],
            )

            #print("Layer:", idx)
            #print("Shape:", current_shape)

        # Flatten shape for linear layer
        lin_size = current_shape[0] * current_shape[2] * current_shape[3]  # ignore depth F
        self.intermediate = nn.Sequential(
            nn.Linear(lin_size, self.embedding_dim),
            nn.LeakyReLU()
        )


    def calculate_output_shape(
        self,
        input_shape: Tuple[int, int, int, int],
        kernel_size: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        channels: int,
    ) -> Tuple[int, int, int, int]:
        in_channels, depth, height, width = input_shape
        depth_out = (depth + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        height_out = (height + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        width_out = (width + 2 * padding[2] - kernel_size[2]) // stride[2] + 1
        return (channels, depth_out, height_out, width_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frames via 3D convs; accepts (B, F, H, W) or (B, 1, F, H, W), returns (B, F, E)."""
        if len(x.shape) == 4:  # (B, F, H, W)
            x = x.unsqueeze(1)  # -> (B, 1, F, H, W)

        for layer in self.conv_pool_layers:
            if self.use_groupnorm and isinstance(layer, nn.GroupNorm):
                # Merge batch and frame dimensions for GroupNorm
                B, C, F, H, W = x.shape
                x = x.permute(0, 2, 1, 3, 4).reshape(B*F, C, H, W)
                x = layer(x)
                # Reshape back
                x = x.view(B, F, C, H, W).permute(0, 2, 1, 3, 4)
            else:
                x = layer(x)
        x = x.transpose(1, 2)  # (B, F, C, H, W)
        batch_size, seq_len, channels, height, width = x.shape
        x = x.reshape(batch_size * seq_len, channels * height * width)  # (B*F, C*H*W)
        x = self.intermediate(x)  # Linear(C*H*W -> embedding_dim)
        x = x.view(batch_size, seq_len, -1)  # (B, F, embedding_dim)
        # -> (B, F', E)
        return x

class BaseModel(LightningModule, ABC):
    """Shared LightningModule scaffold for NAPC models."""

    def __init__(
        self,
        output_dimensions: int = 2,
        input_dimensions: List[int] = [30, 40],
        loss: BaseLoss = BoundOnlyLoss(),
        lr: float = 0.001,
        min_lr: float = 0.0001,
        optimizer: str = "Adam",
        lr_scheduler: Optional[str] = None,
        weight_decay: float = 0.1,
        warmup_steps: int = 20000,
        decay_until_step: int = 200000,
    ) -> None:
        super().__init__()
        self.output_dimensions = output_dimensions
        self.input_dimensions = input_dimensions
        self.loss = loss
        self.lr = lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.decay_until_step = decay_until_step

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

    def step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int, prefix: str) -> Dict[str, torch.Tensor]:
        """Run a shared step for train/val/test: forward, loss, metric prep.

        Args:
            batch: Tuple(x, y_true, seq_lengths, hit_mask, stream_ids) with shapes
                x: (B, F, H, W) or already embedded
                y_true: (B, F, 2)
                seq_lengths: (B,) lengths tensor
                hit_mask: (B, F) mask
                stream_ids: unused here
            batch_idx: batch index.
            prefix: Logging prefix (e.g. "train_").
        """
        x, y_true, seq_lengths, hit_mask, _ = batch
        y_pred = self(x, seq_lengths)

        loss = self.loss(y_true, y_pred, hit_mask)

        self.log(prefix + "loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        y_true, y_pred = self.prep_y_true(y_true), self.prep_y_pred(y_pred)
        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass returning predictions shaped like y_true."""

    @abstractmethod
    def custom_compile(self) -> None:
        """Optionally wrap submodules in torch.compile for speed."""

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, "train_")

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, "val_")

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, "test_")

    def predict_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int = 0) -> torch.Tensor:
        x, _, seq_lengths, _, _ = batch
        y_pred = self(x, seq_lengths)
        return self.prep_y_pred(y_pred)

    def prep_y_true(self, y_true: torch.Tensor) -> torch.Tensor:
        """Round labels and keep the first two dims for metric computation; expects (B, F, >=2)."""
        y_true = torch.round(y_true[:, :, :2]).to(torch.int64)
        return y_true

    def prep_y_pred(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Clamp predictions to positive, round to ints"""
        y_pred = torch.clamp(y_pred, min=1e-08)
        y_pred = torch.round(y_pred).to(torch.int64)
        return y_pred

    def configure_optimizers(self):
        if self._optimizer is None:
            if self.optimizer == "Adam":
                self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            elif self.optimizer == "beck24":
                self._optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.lr,
                    betas=(0.9, 0.95),
                    eps=1e-5,
                    weight_decay=self.weight_decay,
                )
            elif self.optimizer == "SGD":
                self._optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

        optimizer = self._optimizer

        if self.lr_scheduler == "beck24":
            if self._scheduler is None:
                self._scheduler = LinearWarmupCosineAnnealing(
                    optimizer=optimizer,
                    warmup_steps=self.warmup_steps,
                    decay_until_step=self.decay_until_step,
                    max_lr=self.lr,
                    min_lr=self.min_lr,
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self._scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "beck24_cosine",
                },
            }

        return optimizer

class NAPC(BaseModel):
    """NAPC class. compose input encoder, LSTM stack, and dense head."""

    def __init__(
        self,
        lr: float = 0.001,
        min_lr: float = 0.0001,
        optimizer: str = "Adam",
        lr_scheduler: Optional[str] = None,
        weight_decay: float = 0.1,
        output_dimensions: int = 2,
        warmup_steps: int = 20000,
        decay_until_step: int = 200000,
        input_dimensions: List[int] = [30, 40],
        loss: BaseLoss = BoundOnlyLoss(),
        input: nn.Module = DenseInput(),
        intermediate: nn.Module = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                    )
                ),
                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="vanilla",
                        num_heads=4,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),
                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                ),
                context_length=1024,
                num_blocks=7,
                embedding_dim=128,
                slstm_at=[1],
                dropout=0.2,
            )
        ),
        output: nn.Module = DenseOutput(),
    ) -> None:
        super().__init__(
            lr=lr,
            min_lr=min_lr,
            output_dimensions=output_dimensions,
            input_dimensions=input_dimensions,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            decay_until_step=decay_until_step,
        )

        self.pipeline = nn.Sequential(input, intermediate, output)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply input encoder, xLSTM stack, and dense head; expects x: (B, F, H, W)."""
        return self.pipeline(x)

    def custom_compile(self) -> None:
        """Compile the full pipeline for potential speedups."""
        self.pipeline = torch.compile(self.pipeline, dynamic=True)

    def load_compiled_state_dict(self, checkpoint_path: str) -> None:
        """Load a checkpoint saved from a compiled model, stripping `_orig_mod` prefixes."""

        ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        state_dict = ckpt["state_dict"]

        new_state_dict: Dict[str, torch.Tensor] = {}
        prefix_to_remove = "pipeline._orig_mod."
        prefix_replacement = "pipeline."

        for k, v in state_dict.items():
            if k.startswith(prefix_to_remove):
                new_k = k.replace(prefix_to_remove, prefix_replacement, 1)
            else:
                new_k = k

            new_state_dict[new_k] = v
        self.load_state_dict(new_state_dict, strict=True)

class DeconvOutput(nn.Module):
    """1D deconvolutional decoder that upsamples along time."""

    def __init__(
        self,
        embedding_dim: int = 128,
        output_dim: int = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        layers: List[nn.Module] = []
        in_channels = embedding_dim

        for _ in range(num_layers - 1):
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = in_channels // 2
            layers.append(nn.LeakyReLU())

        layers.append(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=output_dim,
                kernel_size=3,
                stride=2,
            )
        )

        self.layers = nn.Sequential(*layers)
        self.final_proj = nn.Sequential(nn.LeakyReLU())

    def forward(self, x: torch.Tensor, lengths: int) -> torch.Tensor:
        """Upsample embeddings (B, F, E) to target length and project to outputs."""
        x = x.transpose(1, 2)
        x = self.layers(x)

        if x.shape[-1] != lengths:
            x = nn.functional.interpolate(x, size=lengths, mode="linear", align_corners=False)

        x = x.transpose(1, 2)
        x = self.final_proj(x)
        return x

class InterpolateOutput(nn.Module):
    """Linear projection head with optional temporal interpolation."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()

        self.final_proj = nn.Sequential(
            nn.Linear(embedding_dim, 2),
        )

    def forward(self, x: torch.Tensor, lengths: int) -> torch.Tensor:
        """Project (B, F, E) to (B, F, 2) and interpolate to `lengths` if needed."""
        x = self.final_proj(x)

        x = x.transpose(1, 2)
        if x.shape[-1] != lengths:
            x = nn.functional.interpolate(x, size=lengths, mode="linear")
        x = x.transpose(1, 2)

        return x

class DecoderOutput(nn.Module):
    """Upsampling decoder that doubles the temporal length per block before projection."""

    def __init__(
        self,
        embedding_dim: int = 128,
        num_blocks: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.initial_proj = nn.Linear(embedding_dim, hidden_dim)

        blocks: List[nn.Module] = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, lengths: int) -> torch.Tensor:
        """Upsample (B, F, E) by ~2^num_blocks in time then project to 2 channels."""
        x = self.initial_proj(x)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = nn.functional.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
            x = block(x)

        if x.shape[-1] != lengths:
            x = nn.functional.interpolate(x, size=lengths, mode="linear", align_corners=False)

        x = x.transpose(1, 2)
        x = self.final_proj(x)

        return x

### NOT USED IN THESIS

class mLSTMModule(nn.Module):
    """Wrapper around the xLSTM large stack with chunk-wise padding for sequence alignment."""

    """
    ChunkwiseKernelType = Literal[
        "chunkwise--native_autograd",
        "chunkwise--native_custbw",
        "chunkwise--triton_limit_chunk",
        "chunkwise--triton_xl_chunk",
        "chunkwise--triton_xl_chunk_siging",
        "parallel--native_autograd",
        "parallel--native_custbw",
        "parallel--native_stablef_autograd",
        "parallel--native_stablef_custbw",
        "parallel--triton_limit_headdim",
        "parallel--native_siging_autograd",
        "parallel--native_siging_custbw",
    ]
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        chunkwise_kernel: str = "chunkwise--native_autograd",
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        from xlstm.xlstm_large.model import xLSTMLargeBlockStack, xLSTMLargeConfig

        cfg = xLSTMLargeConfig(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            vocab_size=1,  # will be ignored
            chunkwise_kernel=chunkwise_kernel,
            return_last_states=True,
            mode="train",
            chunk_size=chunk_size,
        )
        self.mLSTM = xLSTMLargeBlockStack(cfg)
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad to chunk multiple, run mLSTM, then trim back; expects (B, F, E)."""
        _, seq_len, _ = x.shape

        next_multiple = ((seq_len + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        pad_amount = next_multiple - seq_len

        if pad_amount > 0:
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, pad_amount, 0, 0))
        else:
            x_padded = x

        x_padded, _ = self.mLSTM(x_padded)

        x = x_padded[:, :seq_len, :]
        return x

class BN_NAPC(BaseModel):
    """
    BN_NAPC = Bottleneck NAPC. In contrast to NAPC it allows to downsample along the temporal dimension. 
    The decoder is passed the original sequence length and does upsample back to the original resolution.
    """

    def __init__(
        self,
        lr: float = 0.001,
        min_lr: float = 0.0001,
        optimizer: str = "Adam",
        lr_scheduler: Optional[str] = None,
        weight_decay: float = 0.1,
        output_dimensions: int = 2,
        input_dimensions: List[int] = [30, 40],
        loss: BaseLoss = BoundOnlyLoss(),
        warmup_steps: int = 20000,
        decay_until_step: int = 200000,
        input: nn.Module = ConvInput(),
        intermediate: nn.Module = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                    )
                ),
                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="vanilla",
                        num_heads=4,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),
                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                ),
                context_length=1024,
                num_blocks=7,
                embedding_dim=128,
                slstm_at=[1],
                dropout=0.2,
            )
        ),
        output: nn.Module = DenseOutput(),
    ) -> None:
        super().__init__(
            lr=lr,
            min_lr=min_lr,
            output_dimensions=output_dimensions,
            input_dimensions=input_dimensions,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            decay_until_step=decay_until_step,
        )

        self.input = input
        self.intermediate = intermediate
        self.output = output

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode video frames with ConvInput, xLSTM, then decode to outputs."""
        lengths = x.shape[1]

        x = self.input(x)
        x = self.intermediate(x)
        x = self.output(x, lengths=lengths)
        return x

    def custom_compile(self) -> None:
        """Compile encoder and xLSTM; output stays eager due to interpolation."""
        self.input = torch.compile(self.input, dynamic=True)
        self.intermediate = torch.compile(self.intermediate, dynamic=True)

class R2Plus1DBlock(nn.Module):
    """Factorized 3D conv: spatial (H,W) then temporal (T) sub-convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        spatial_kernel: int = 3,
        temporal_kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = int(
                (spatial_kernel * spatial_kernel * in_channels * out_channels * temporal_kernel)
                / (spatial_kernel * spatial_kernel * in_channels + temporal_kernel * out_channels)
            )

        self.spatial_conv = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm3d(mid_channels)

        self.temporal_conv = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial then temporal convs; input/output: (B, C, F, H, W)."""
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.relu(x)

        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.relu(x)

        return x

class R2Plus1DResidualBlock(nn.Module):
    """Two R(2+1)D blocks with an optional projection skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.block1 = R2Plus1DBlock(in_channels, out_channels, stride=stride)
        self.block2 = R2Plus1DBlock(out_channels, out_channels)

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(stride, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual path; input/output: (B, C, F, H, W)."""
        identity = x

        out = self.block1(x)
        out = self.block2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity.clone()
        out = self.relu(out)

        return out

class R2Plus1DNet(nn.Module):
    """Residual R(2+1)D encoder producing per-frame embeddings."""

    def __init__(
        self,
        num_blocks: int = 8,
        base_channels: int = 8,
        embedding_dim: int = 128,
        input_shape: Tuple[int, int, int, int] = (1, 1215, 40, 40),
        stride_every: int = 2,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose

        self.stem = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        layers: List[nn.Module] = []
        in_c = base_channels
        for i in range(num_blocks):
            out_c = base_channels * (2 ** (i // 2))
            stride = 2 if i % stride_every == 0 and i != 0 else 1
            layers.append(R2Plus1DResidualBlock(in_c, out_c, stride=stride))
            in_c = out_c
        self.blocks = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros((1, *input_shape))
            dummy_out = self.stem(dummy)
            dummy_out = self.blocks(dummy_out)
            _, C, F, H, W = dummy_out.shape
            fc_in_features = C * H * W

        self.fc = nn.Linear(fc_in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video input (B, 1, F, H, W) -> (B, F, embedding_dim)."""
        if x.dim() == 4:
            x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.blocks(x)

        batch_size, channels, frames, height, width = x.shape
        x = x.reshape(batch_size, frames, channels * height * width)
        if self.verbose:
            print(x.shape)
        x = self.fc(x)
        return x
    
class IdentityModule(torch.nn.Module):
    """No-op module used as a configurable placeholder."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
