"""
Dataset for plane tracking data with GT homographies.

Each sample is a pair of frames from the same video clip with:
  - A known homography H_0to1 between the two frames
  - A polygon mask of the tracked plane region (for each view)

Directory structure expected:
  <data_dir>/
    <clip_name>/
      original.mp4
      ae_data/
        homographies.npy   (N, 3, 3) float64  -- maps reference→frame_i
        corners.csv        (N, 9)              -- tl/tr/bl/br pixel corners
        track_masks.npy    (N, M) bool         -- valid tracks per frame
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..utils.tools import fork_rng
from .augmentations import IdentityAugmentation, augmentations
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def _corners_to_mask(tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y, img_h, img_w):
    """Rasterize the quad defined by corner coordinates into a (H, W) bool mask."""
    poly = np.array(
        [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, np.round(poly).astype(np.int32), 1)
    return mask.astype(bool)


def _load_frame(video_path, frame_idx):
    """Load a single frame from a video file by index."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class TrackingPlanesDataset(BaseDataset):
    default_conf = {
        "data_dir": "tracking_whisper_clean_dataset",
        "train_size": 10000,   # virtual epoch size (random pairs)
        "val_size": 1000,
        "val_fraction": 0.1,   # fraction of clips held out for val
        "shuffle_seed": 0,
        "min_frame_gap": 1,
        "max_frame_gap": 30,
        "grayscale": False,
        "photometric": {
            "name": "lg",
            "p": 0.75,
        },
        "seed": 0,
        "num_threads": 1,
        # Resize images before returning: [W, H] or null to keep original
        "resize": None,
    }

    def _init(self, conf):
        data_dir = Path(conf.data_dir)
        if not data_dir.is_absolute():
            data_dir = DATA_PATH / conf.data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        clips = sorted([
            d for d in data_dir.iterdir()
            if d.is_dir()
            and (d / "original.mp4").exists()
            and (d / "ae_data" / "homographies.npy").exists()
        ])
        if not clips:
            raise ValueError(f"No valid clips found in {data_dir}")

        logger.info("Found %d clips in %s", len(clips), data_dir)

        rng = np.random.RandomState(conf.shuffle_seed)
        order = rng.permutation(len(clips))
        clips = [clips[i] for i in order]

        n_val = max(1, int(len(clips) * conf.val_fraction))
        self.clips = {
            "train": clips[n_val:],
            "val": clips[:n_val],
        }
        logger.info(
            "Split: %d train clips, %d val clips",
            len(self.clips["train"]),
            len(self.clips["val"]),
        )

    def get_dataset(self, split):
        assert split in ("train", "val"), split
        size = self.conf.train_size if split == "train" else self.conf.val_size
        return _Dataset(self.conf, self.clips[split], split, size)


class _ClipData:
    """Lightweight container holding pre-loaded ae_data for one clip."""

    def __init__(self, clip_dir):
        self.clip_dir = clip_dir
        ae = clip_dir / "ae_data"
        self.homographies = np.load(ae / "homographies.npy")   # (N, 3, 3)
        self.corners = pd.read_csv(ae / "corners.csv")         # (N, 9)
        self.track_masks = np.load(ae / "track_masks.npy")     # (N, M)
        self.n_frames = len(self.homographies)
        self.video_path = clip_dir / "original.mp4"

    def frame_h_to_ref(self, i):
        """H that maps reference (frame-0 coords) → frame i."""
        return self.homographies[i]  # (3, 3)

    def h_i_to_j(self, i, j):
        """Homography mapping pixel coords in frame i → frame j."""
        Hi = self.homographies[i]
        Hj = self.homographies[j]
        return Hj @ np.linalg.inv(Hi)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, clips, split, size):
        self.conf = conf
        self.split = split
        self.size = size
        self.clip_data = [_ClipData(c) for c in clips]

        aug_conf = conf.photometric
        aug_name = aug_conf.name
        assert aug_name in augmentations, (
            f"Unknown augmentation '{aug_name}'. "
            f"Available: {list(augmentations.keys())}"
        )
        self.photo_aug = augmentations[aug_name](aug_conf)
        self.identity_aug = IdentityAugmentation()

        self.resize = list(conf.resize) if conf.resize is not None else None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_pair(self, rng):
        """Sample a random (clip, frame_i, frame_j) triple."""
        clip = self.clip_data[rng.randint(len(self.clip_data))]
        n = clip.n_frames
        gap = rng.randint(self.conf.min_frame_gap, self.conf.max_frame_gap + 1)
        i = rng.randint(0, n - gap)
        j = i + gap
        return clip, i, j

    def _load_and_prepare(self, clip, frame_idx, aug):
        """Load a frame, optionally resize, apply augmentation, build mask."""
        img = _load_frame(clip.video_path, frame_idx)   # (H, W, 3) uint8 RGB
        orig_h, orig_w = img.shape[:2]

        if self.resize is not None:
            target_w, target_h = self.resize
            img = cv2.resize(img, (target_w, target_h))
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
        else:
            scale_x = scale_y = 1.0
            target_w, target_h = orig_w, orig_h

        # Build polygon mask in output-image coordinates
        row = clip.corners.iloc[frame_idx]
        mask = _corners_to_mask(
            tl_x=row["tl_x"] * scale_x,
            tl_y=row["tl_y"] * scale_y,
            tr_x=row["tr_x"] * scale_x,
            tr_y=row["tr_y"] * scale_y,
            br_x=row["br_x"] * scale_x,
            br_y=row["br_y"] * scale_y,
            bl_x=row["bl_x"] * scale_x,
            bl_y=row["bl_y"] * scale_y,
            img_h=target_h,
            img_w=target_w,
        )

        img_f = img.astype(np.float32) / 255.0
        img_t = aug(img_f, return_tensor=True)   # (3, H, W) float32 [0,1]

        if self.conf.grayscale:
            gs = img_t.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            img_t = (img_t * gs).sum(0, keepdim=True)

        return {
            "image": img_t,
            "image_size": torch.tensor([target_w, target_h], dtype=torch.float32),
            "mask": torch.from_numpy(mask),   # (H, W) bool
        }, scale_x, scale_y

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.conf.get("reseed", False):
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        return self.getitem(idx)

    def getitem(self, idx):
        rng = np.random.RandomState(
            (self.conf.seed + idx) if self.split == "val" else None
        )
        clip, i, j = self._sample_pair(rng)

        # For the reference view use identity aug (or left aug); for target use photo aug
        view0, sx0, sy0 = self._load_and_prepare(clip, i, self.identity_aug)
        view1, sx1, sy1 = self._load_and_prepare(clip, j, self.photo_aug)

        H_ij = clip.h_i_to_j(i, j).astype(np.float32)

        # Adjust H for any resize scaling
        # H operates on pixel coords; rescaling changes pixel coords
        if self.resize is not None:
            S0 = np.diag([sx0, sy0, 1.0]).astype(np.float32)
            S1 = np.diag([sx1, sy1, 1.0]).astype(np.float32)
            H_ij = S1 @ H_ij @ np.linalg.inv(S0)

        return {
            "view0": view0,
            "view1": view1,
            "H_0to1": H_ij,
            "name": f"{clip.clip_dir.name}/{i}_{j}",
            "idx": idx,
        }


