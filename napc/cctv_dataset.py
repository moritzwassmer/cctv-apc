# originally from https://git.tu-berlin.de/deepAPC/napc-cctv (Dr. Sambu Seo) # translated to pytorch and modified by Moritz Wassmer 

import ast
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from .standardize import calcGlobalMeanStd, calcMeanStdPerPixel

@dataclass
class DatasetConfig:
    """Configuration for CCTV passenger counting dataset with augmentation and preprocessing options."""

    hdf5_path: str = "/net/vericon/napc_data/cctv/data_vd2_cut2.hdf5"
    context_length: int = 1024  # Max sequence length; sequences are padded/truncated to this value
    concat: int = 1  # Number of consecutive sequences to concatenate
    padding_value: int = -1
    prob_time_reverse: float = 0  # Probability of reversing time (augmentation)
    prob_mirror: float = 0  # Probability of horizontal flip (augmentation)
    bounding_box_extend: float = 0
    do_pad_to_max: bool = False
    do_trunc_max: bool = True
    mean: float = None # if both mean and std are None, calculate global mean and std from the dataset.
    std: float = None
    do_cut: bool = False  # If True, use n_boarding_N and n_alighting_N instead of n_boarding and n_alighting

    collate_fn: str = "collate_pad"
    pin_memory: bool = True
    num_workers: int = 4
    shuffle: bool = False  # KEEP FALSE; shuffling is done manually in the dataset class
    ext_up_bb: int = 0  # Bounding box extension upper
    ext_low_bb: int = 0  # Bounding box extension lower
        
        
class Dataset(TorchDataset):
    """Custom Dataset for loading and processing grayscale video streams with passenger counting labels.

    Expected napci_ DataFrame columns:
        ['fname', 'n_boarding', 'n_alighting', 'n_frame', 'event_boarding', 'bb_boarding',
         'event_alighting', 'bb_alighting', 'n_boarding_N', 'n_alighting_N', 'frame_N']
    """

    def __init__(
        self,
        napci_: pd.DataFrame,
        hdf5_path: str,
        context_length: int,
        concat: int,
        padding_value: int,
        prob_time_reverse: float,
        prob_mirror: float,
        bounding_box_extend: float,
        do_pad_to_max: bool,
        do_trunc_max: bool,
        #standardize: str,
        mean: float,
        std: float,
        do_cut: bool,
        collate_fn: str,
        pin_memory: bool,
        num_workers: int,
        shuffle: bool,
        ext_up_bb: int,
        ext_low_bb: int,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.context_length = context_length
        self.concat = concat
        self.padding_value = padding_value
        self.prob_time_reverse = prob_time_reverse
        self.prob_mirror = prob_mirror
        self.bounding_box_extend = bounding_box_extend
        self.do_pad_to_max = do_pad_to_max
        self.do_trunc_max = do_trunc_max
        #self.standardize = standardize
        self.mean = mean
        self.std = std
        self.do_cut = do_cut
        self.collate_fn: Callable = collate_fn
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.ext_up_bb = ext_up_bb
        self.ext_low_bb = ext_low_bb

        self.napci_ = napci_.copy()
        self.hdf5 = h5py.File(self.hdf5_path, "r")
        self.indices: np.ndarray = np.arange(len(self.napci_))

        if self.do_cut:
            self.napci_.n_boarding = self.napci_.n_boarding_N
            self.napci_.n_alighting = self.napci_.n_alighting_N
            self.napci_.n_frame = self.napci_.frame_N

        # Calculate mean and std for standardization
        if self.mean is not None and self.std is not None:
            self.mean = self.mean
            self.std = self.std
        elif self.mean is None and self.std is None:
            self.mean, self.std = calcGlobalMeanStd(self.hdf5_path)
        else:
            raise ValueError(f"invalid mean/std combination")
        print(f"Dataset: mean: {self.mean}, std: {self.std}")

        # Infer image and label shape from a sample
        sample_index = 0
        sample_stream = self.next_cstream([sample_index], apply_time_reverse=False, apply_left_right_flip=False)
        sample_label = self.next_label([sample_index], apply_time_reverse=False)
        self.image_shape: Tuple[int, int] = sample_stream.shape[1:]  # (height, width)
        self.label_shape: int = sample_label.shape[1]  # Number of label features

        print(f"Dataset: image_shape: {self.image_shape}, label_shape: {self.label_shape}")

        if self.collate_fn == "collate_pad":
            self.collate_fn = self.collate_pad
        else:
            raise ValueError(f"collate_fn {self.collate_fn} not supported. Use 'collate_pad'")
        
    def collate_pad(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, List[str]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[List[str], ...]]:
        """Pad sequences in a batch to the longest sequence length.

        Args:
            batch: List of tuples (sequence, label, length, accuracy, fnames)
                sequence: (F, H, W) video frames
                label: (F, 4) boarding/alighting bounds
                length: int, original sequence length
                accuracy: (F, 1) hit mask
                fnames: List[str] filenames

        Returns:
            Tuple of (padded_sequences, padded_labels, lengths_tensor, padded_accuracy, fnames)
                padded_sequences: (B, F_max, H, W)
                padded_labels: (B, F_max, 4)
                lengths_tensor: (B,)
                padded_accuracy: (B, F_max, 1)
        """
        sequences, labels, lengths, accuracy, fnames = zip(*batch)

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.padding_value)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        padded_accuracy = pad_sequence(accuracy, batch_first=True, padding_value=-1)

        # Convert lengths into a tensor
        lengths_tensor = torch.tensor(lengths)

        return padded_sequences, padded_labels, lengths_tensor, padded_accuracy, fnames

    def __len__(self) -> int:
        """Return the number of batches after concatenation."""
        return math.ceil(len(self.indices) / self.concat)

    def shuffle_dataset(self) -> None:
        """Randomly permute the dataset indices for shuffling."""
        self.indices = np.random.permutation(self.indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, List[str]]:
        """Get a batch item: video frames, labels, length, accuracy mask, and filenames.

        Returns:
            Tuple of (cstream, label, length, accuracy, fnames)
                cstream: (F, H, W) video frames tensor
                label: (F, 4) bounds tensor
                length: int, sequence length
                accuracy: (F, 1) hit mask tensor
                fnames: List[str] filenames
        """
        next_idx_ = self.get_next_indices(idx)

        apply_time_reverse = np.random.rand() > 1 - self.prob_time_reverse
        apply_left_right_flip = np.random.rand() > 1 - self.prob_mirror

        next_cstream = self.next_cstream(next_idx_, apply_time_reverse, apply_left_right_flip)
        next_label = self.next_label(next_idx_, apply_time_reverse)
        next_accuracy = self.next_accuracy(next_idx_, apply_time_reverse)

        assert next_cstream.shape[0] == next_label.shape[0]

        if self.do_pad_to_max:
            next_cstream = self.pad_to_max(next_cstream, self.padding_value)
            next_label = self.pad_to_max(next_label, -1)
            next_accuracy = self.pad_to_max(next_accuracy, -1)

        if self.do_trunc_max:
            next_cstream = self.trunc_max(next_cstream)
            next_label = self.trunc_max(next_label)
            next_accuracy = self.trunc_max(next_accuracy)

        length = next_cstream.shape[0]  # if padding and truncating always the same

        # Convert sequences and labels to tensors
        next_cstream = torch.as_tensor(next_cstream).float()
        next_label = torch.as_tensor(next_label).float()
        next_accuracy = torch.as_tensor(next_accuracy).float()

        fnames = (
            [self.napci_.fname.iloc[i] for i in next_idx_]
            #if not self.augment else None
        )

        return next_cstream, next_label, length, next_accuracy, fnames

    def get_next_indices(self, idx: int) -> np.ndarray:
        """Get the indices of sequences to concatenate for a given batch index."""
        selection = []
        for i in range(self.concat):
            absolute_index = idx * self.concat + i
            if absolute_index < len(self.indices):
                selection.append(self.indices[absolute_index])
        return np.array(selection)

    def get_dataloader(self, batch_size: int) -> DataLoader:
        """Create a DataLoader for the dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=self.shuffle,  # False; shuffling is done manually
        )

    def next_cstream(
        self, indices: np.ndarray, apply_time_reverse: bool, apply_left_right_flip: bool
    ) -> np.ndarray:
        """Load and preprocess video frames from HDF5 for given indices.

        Args:
            indices: Array of dataset indices to load
            apply_time_reverse: Whether to reverse the temporal order
            apply_left_right_flip: Whether to flip horizontally

        Returns:
            Concatenated array of shape (F_total, H, W) where F_total is the sum of all frame counts
        """
        one_iter = []
        for idx in indices:
            sequence = np.array(self.hdf5[self.napci_.fname.iloc[idx]])
            image_shape = sequence.shape[1:]  # (height, width)
            sequence = np.reshape(sequence, (-1, *image_shape))

            sequence = (sequence - self.mean) / self.std  # Standardize

            # Fix for n_frame sequence length mismatch
            cut_length = min(sequence.shape[0], self.napci_.n_frame.iloc[idx])
            self.napci_.loc[idx, "n_frame"] = cut_length
            sequence = sequence[:cut_length, :, :]

            if apply_time_reverse:
                sequence = sequence[::-1, :, :]
            if apply_left_right_flip:
                sequence = sequence[:, :, ::-1]

            one_iter += [sequence]
        return np.concatenate(one_iter)
    
    def next_label(self, indices: np.ndarray, apply_time_reverse: bool) -> np.ndarray:
        """Generate cumulative boarding/alighting bound labels for given indices.

        Args:
            indices: Array of dataset indices
            apply_time_reverse: Whether to reverse temporal order

        Returns:
            Array of shape (F_total, 4) with columns [boarding_upper, alighting_upper, boarding_lower, alighting_lower]
        """
        boarding_mask_low_ = []
        alighting_mask_low_ = []
        boarding_mask_upper_ = []
        alighting_mask_upper_ = []

        for idx in indices:
            la = self.napci_.iloc[idx]
            fNum = int(la.n_frame)
            boarding_mask_lo = np.zeros(fNum)
            alighting_mask_lo = np.zeros(fNum)
            boarding_mask_up = np.zeros(fNum)
            alighting_mask_up = np.zeros(fNum)
           
            if la.n_boarding > 0:
                b_bb = ast.literal_eval(la.bb_boarding)[: la.n_boarding]
                if apply_time_reverse:
                    for i, eb in enumerate(ast.literal_eval(la.event_boarding)[: la.n_boarding]):
                        ixx1 = fNum - eb + b_bb[i] + self.ext_up_bb
                        ixx2 = fNum - eb - b_bb[i] - self.ext_low_bb
                        ixx1 = min(ixx1, fNum - 1)
                        ixx2 = max(0, ixx2)
                        alighting_mask_lo[ixx1] += 1
                        alighting_mask_up[ixx2] += 1
                else:
                    for i, eb in enumerate(ast.literal_eval(la.event_boarding)[: la.n_boarding]):
                        ixx1 = eb - 1 + b_bb[i] + self.ext_low_bb
                        ixx2 = eb - 1 - b_bb[i] - self.ext_up_bb
                        ixx1 = min(ixx1, fNum - 1)
                        ixx2 = max(0, ixx2)
                        boarding_mask_lo[ixx1] += 1
                        boarding_mask_up[ixx2] += 1
                                                                                                 
            if la.n_alighting > 0:
                a_bb = ast.literal_eval(la.bb_alighting)[: la.n_alighting]
                if apply_time_reverse:
                    for i, ea in enumerate(ast.literal_eval(la.event_alighting)[: la.n_alighting]):
                        ixx1 = fNum - ea + a_bb[i] + self.ext_up_bb
                        ixx2 = fNum - ea - a_bb[i] - self.ext_low_bb
                        ixx1 = min(ixx1, fNum - 1)
                        ixx2 = min(max(0, ixx2), fNum - 1)
                        boarding_mask_lo[ixx1] += 1
                        boarding_mask_up[ixx2] += 1
                else:
                    for i, ea in enumerate(ast.literal_eval(la.event_alighting)[: la.n_alighting]):
                        ixx1 = ea - 1 + a_bb[i] + self.ext_low_bb
                        ixx2 = ea - 1 - a_bb[i] - self.ext_up_bb
                        ixx1 = min(ixx1, fNum - 1)
                        ixx2 = min(max(0, ixx2), fNum - 1)
                        alighting_mask_lo[ixx1] += 1
                        alighting_mask_up[ixx2] += 1            

            boarding_mask_low_.append(boarding_mask_lo)
            boarding_mask_upper_.append(boarding_mask_up)
            alighting_mask_low_.append(alighting_mask_lo)
            alighting_mask_upper_.append(alighting_mask_up)

        boarding_mask_low_ = np.concatenate(boarding_mask_low_)
        boarding_mask_upper_ = np.concatenate(boarding_mask_upper_)
        boarding_mask_lower = np.cumsum(boarding_mask_low_)
        boarding_mask_upper = np.cumsum(boarding_mask_upper_) + self.bounding_box_extend
        alighting_mask_low_ = np.concatenate(alighting_mask_low_)
        alighting_mask_upper_ = np.concatenate(alighting_mask_upper_)
        alighting_mask_lower = np.cumsum(alighting_mask_low_)
        alighting_mask_upper = np.cumsum(alighting_mask_upper_) + self.bounding_box_extend

        upper_bound = np.column_stack((boarding_mask_upper, alighting_mask_upper))
        lower_bound = np.column_stack((boarding_mask_lower, alighting_mask_lower))
        return np.concatenate([upper_bound, lower_bound], axis=1)

    def next_accuracy(self, indices: np.ndarray, apply_time_reverse: bool) -> np.ndarray:
        """Generate hit mask marking the last frame of each sequence.

        Args:
            indices: Array of dataset indices
            apply_time_reverse: Whether to reverse temporal order

        Returns:
            Array of shape (F_total, 1) with 1 at last frame of each sequence
        """
        hit_mask = []
        for idx in indices:
            mask = np.zeros((self.napci_.n_frame.iloc[idx], 1))
            mask[-1] = 1  # Mark the last frame
            if apply_time_reverse:
                mask = mask[::-1]
            hit_mask += [mask]
        return np.concatenate(hit_mask)

    def trunc_max(self, array: np.ndarray) -> np.ndarray:
        """Truncate sequence to context_length along the first dimension."""
        slices = (slice(self.context_length),) + (slice(None),) * (array.ndim - 1)
        return array[slices]

    def pad_to_max(self, array: np.ndarray, pad_value: float) -> np.ndarray:
        """Pad sequence to context_length along the first dimension."""
        if array.shape[0] < self.context_length:
            padding = [(0, self.context_length - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
            array = np.pad(array, padding, mode="constant", constant_values=pad_value)
        return array
    
class DataModule(LightningDataModule):
    """PyTorch Lightning DataModule for CCTV passenger counting with automatic train/val/test splits."""

    def partition_data(self) -> None:
        """Load NAPCI metadata and partition into train/val/test splits by video-day groups."""
        self.napci_  = pd.read_csv(
            self.napci_path,
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
        if self.max_framenum is not None:
            self.napci_ = self.napci_.query(f"n_frame <= {self.max_framenum}").reset_index(drop=True)

        # when seperate test set is provided, no splitting. Use development dataset as train and validation set, use test set as test set
        if self.napci_path_test is not None:
            self.napci_test = pd.read_csv(
                self.napci_path_test,
                header=None,
                names=[
                    "fname",
                    "n_boarding",
                    "n_alighting",
                    "n_frame",
                ],
            )
            # convert to development dataset format
            df = self.napci_test
            df["event_boarding"] = df["n_boarding"].apply(lambda x: str(list(np.ones(x, dtype=int))))
            df["bb_boarding"] = df["n_boarding"].apply(lambda x: str(list(np.ones(x, dtype=int)*1)))
            df["event_alighting"] = df["n_alighting"].apply(lambda x: str(list(np.ones(x, dtype=int))))
            df["bb_alighting"] = df["n_alighting"].apply(lambda x: str(list(np.ones(x, dtype=int)*1)))
            df["n_boarding_N"] = df["n_boarding"]
            df["n_alighting_N"] = df["n_alighting"]
            df["frame_N"] = df["n_frame"]
            self.napci_test = df

            self.napci_test = self.napci_test.query(f"n_frame <= {self.max_framenum}").reset_index(drop=True)
            self.test_ = self.napci_test.reset_index(drop=True)
            if self.ttdv_ratio is None:
                print("DataPartitioner: No ttdv_ratio provided, train and validate on training dataset")
                self.train_ = self.napci_.copy()
                self.validate_ = self.napci_.copy()
                return
        
        # Extract unique date-door combinations
        vday_ = (
            (self.napci_.fname.str.slice(0, 2) + self.napci_.fname.str.slice(13, 21))
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        # Split list of combinations according to ttdv_ratio (test:train:disuse:validate)
        test, train, disuse, validate = map(float, self.ttdv_ratio.split(":"))
        n = len(vday_)

        # Set random seed for reproducible splits
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        idx_ = np.random.permutation(n)
        split1 = int(test * n)
        split2 = int((test + train) * n)
        split3 = int((test + train + disuse) * n)

        # Reset seed after partitioning
        if self.random_seed is not None:
            np.random.seed(int(time.time()))

        if not self.napci_path_test:
            test_vday_ = vday_[idx_[:split1]]
        train_vday_ = vday_[idx_[split1:split2]]
        disuse_vday_ = vday_[idx_[split2:split3]]
        validate_vday_ = vday_[idx_[split3:]]

        # Split napci_ according to partitioned date-door combinations
        vday_col = self.napci_.fname.str.slice(0, 2) + self.napci_.fname.str.slice(13, 21)
        if not self.napci_path_test:
            self.test_ = self.napci_[vday_col.isin(test_vday_)].reset_index(drop=True)
        self.train_ = self.napci_[vday_col.isin(train_vday_)].reset_index(drop=True)
        self.disuse_ = self.napci_[vday_col.isin(disuse_vday_)].reset_index(drop=True)
        self.validate_ = self.napci_[vday_col.isin(validate_vday_)].reset_index(drop=True)

        # Optionally use only a fraction of the training data
        if self.train_fraction < 1.0:
            n_train = len(self.train_)
            n_subset = int(self.train_fraction * n_train)
            self.train_ = self.train_.sample(n=n_subset, random_state=self.random_seed).reset_index(drop=True)
            print(f"Using only {n_subset}/{n_train} samples ({100 * self.train_fraction:.1f}%) of training set.")

        # Print partition summary
        print(f"DataPartitioner: Found {n} date-door combinations with {len(self.napci_)} streams.")
        if not self.napci_path_test:
            print(
                f"DataPartitioner: {len(test_vday_)} date-door combinations with {len(self.test_)} "
                f"({len(self.test_) / len(self.napci_):.2%}) streams for testing: {list(test_vday_)}"
            )
        print(
            f"DataPartitioner: {len(train_vday_)} date-door combinations with {len(self.train_)} "
            f"({len(self.train_) / len(self.napci_):.2%}) streams for training: {list(train_vday_)}"
        )
        print(
            f"DataPartitioner: {len(disuse_vday_)} date-door combinations with {len(self.disuse_)} "
            f"({len(self.disuse_) / len(self.napci_):.2%}) streams for disuse: {list(disuse_vday_)}"
        )
        print(
            f"DataPartitioner: {len(validate_vday_)} date-door combinations with {len(self.validate_)} "
            f"({len(self.validate_) / len(self.napci_):.2%}) streams for validation: {list(validate_vday_)}"
        )

    def __init__(
        self,
        train_ds_conf: DatasetConfig = DatasetConfig(
            prob_time_reverse=0.5,
            prob_mirror=0.5,
            concat=2,
        ),
        val_ds_conf: DatasetConfig = DatasetConfig(),
        test_ds_conf: DatasetConfig = DatasetConfig(),
        napci_path: str = "/net/vericon/napc_data/cctv/cctv_passenger_bb_sel_350_100_25_2.csv",
        napci_path_test: str = None,
        ttdv_ratio: str = "0.2:0.6:0.0:0.2", # if None then no splitting, train and validate on training dataset
        max_framenum: int = 1024,
        batch_size: int = 32,
        random_seed: int = 0,
        train_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        self.train_ds_conf = train_ds_conf
        self.val_ds_conf = val_ds_conf
        self.test_ds_conf = test_ds_conf
        self.napci_path = napci_path
        self.napci_path_test = napci_path_test
        self.ttdv_ratio = ttdv_ratio
        self.max_framenum = max_framenum
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.train_fraction = train_fraction

        # Ensure all configs use the same HDF5 file
        """if train_ds_conf.hdf5_path != val_ds_conf.hdf5_path or train_ds_conf.hdf5_path != test_ds_conf.hdf5_path:
            raise ValueError("hdf5_path must be the same for train, val and test dataset")"""

        self.partition_data()

        # Infer image and label shapes from a small sample
        self.sample = Dataset(self.train_.head(5), **asdict(self.train_ds_conf))
        self.image_shape: Tuple[int, int] = self.sample.image_shape
        self.label_shape: int = self.sample.label_shape
            
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize datasets for training, validation, or testing.

        Args:
            stage: 'fit' for train/val, 'test' for test, or None for all
        """
        if stage == "fit" or stage is None:
            self.train = Dataset(self.train_.copy(), **asdict(self.train_ds_conf))
            self.validate = Dataset(self.validate_.copy(), **asdict(self.val_ds_conf))

        if stage == "test" or stage is None:
            self.test = Dataset(self.test_.copy(), **asdict(self.test_ds_conf))

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self.train.get_dataloader(batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self.validate.get_dataloader(batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return self.test.get_dataloader(batch_size=self.batch_size)
