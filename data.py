"""
Data loader for Lite-ProSENet adapted to NSCLC-Radiomics on Kaggle.

Replaces the original file-folder-based loader (which expected pre-cropped
GTV .mhd files) with a DICOM-based loader that:
  - Reads ct_path and seg_path from phase1_metadata CSV
  - Corrects the Kaggle path prefix
  - Merges with the clinical CSV for EHR features and survival labels
  - Performs GTV bounding-box crop (same as FinalProject)
  - Resamples to the fixed spatial size expected by the model (96x96x12)
  - Returns (img, clinical_diag, label, event) matching the original contract
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate

# ---------------------------------------------------------------------------
# Path correction (mirrors FinalProject/dataset.py)
# ---------------------------------------------------------------------------

OLD_PREFIX = "/kaggle/input/nsclc-radiomics/NSCLC-Radiomics/"
NEW_PREFIX  = "/kaggle/input/datasets/umutkrdrms/nsclc-radiomics/NSCLC-Radiomics/"

def fix_path(path: str) -> str:
    return path.replace(OLD_PREFIX, NEW_PREFIX)


# ---------------------------------------------------------------------------
# DICOM loading helpers (mirrors FinalProject/dataset.py)
# ---------------------------------------------------------------------------

def load_dicom_series(directory: str) -> sitk.Image:
    """Load a DICOM series and return a SimpleITK Image (keeps spacing info)."""
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(directory)
    if len(dicom_files) == 0:
        dcm_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
        if len(dcm_files) == 0:
            dcm_files = sorted(glob.glob(os.path.join(directory, "*")))
        dicom_files = dcm_files
    reader.SetFileNames(dicom_files)
    return reader.Execute()


def load_segmentation_array(directory: str) -> np.ndarray:
    """Load segmentation mask; returns binary (D,H,W) float32 array or None."""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(directory)
        if len(dicom_files) == 0:
            dicom_files = sorted(glob.glob(os.path.join(directory, "*")))
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        array = sitk.GetArrayFromImage(image)
        return (array > 0).astype(np.float32)
    except Exception:
        import pydicom
        files = sorted(glob.glob(os.path.join(directory, "*")))
        if not files:
            return None
        try:
            ds = pydicom.dcmread(files[0])
            if hasattr(ds, "pixel_array"):
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis]
                return (arr > 0).astype(np.float32)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# GTV crop (mirrors FinalProject/dataset.py)
# ---------------------------------------------------------------------------

def gtv_crop(ct_array: np.ndarray, seg_array: np.ndarray,
             margin: int = 10) -> tuple:
    """Tight bounding-box crop around the GTV with a voxel margin."""
    coords = np.argwhere(seg_array > 0)
    if coords.shape[0] == 0:
        return ct_array, seg_array

    D, H, W = ct_array.shape
    d_min, h_min, w_min = coords.min(axis=0)
    d_max, h_max, w_max = coords.max(axis=0)

    d_min = max(d_min - margin, 0)
    h_min = max(h_min - margin, 0)
    w_min = max(w_min - margin, 0)
    d_max = min(d_max + margin + 1, D)
    h_max = min(h_max + margin + 1, H)
    w_max = min(w_max + margin + 1, W)

    return (
        ct_array[d_min:d_max, h_min:h_max, w_min:w_max],
        seg_array[d_min:d_max, h_min:h_max, w_min:w_max],
    )


# ---------------------------------------------------------------------------
# Augmentation (preserved exactly from original data.py)
# ---------------------------------------------------------------------------

def augment(sample, ifflip=True, ifrotate=True, ifswap=True):
    if ifrotate:
        angle1 = np.random.rand() * 180
        sample = rotate(sample, angle1, axes=(2, 1), reshape=False)
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[0]:
            axisorder = np.random.permutation(2)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
    if ifflip:
        flipid = np.array([np.random.randint(2),
                           np.random.randint(2),
                           np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[::flipid[0], ::flipid[1], ::flipid[2]])
    return sample


# ---------------------------------------------------------------------------
# Clinical feature columns (mirrors original NSCLC_PROCESSED.CSV columns 2:29)
# The clinical CSV has 27 usable numeric features after age normalisation.
# ---------------------------------------------------------------------------

CLINICAL_COLS = [
    "age",
    "clinical.T.Stage",
    "Clinical.N.Stage",
    "Clinical.M.Stage",
    "Overall.Stage",
    "Survival.time",
    "deadstatus.event",
]

# All columns that should be one-hot encoded (categorical)
CATEGORICAL_COLS = [
    "clinical.T.Stage",
    "Clinical.N.Stage",
    "Clinical.M.Stage",
    "Overall.Stage",
]


def _encode_stage(val) -> list:
    """One-hot encode a TNM/overall stage value into a fixed 6-dim vector."""
    stages = ["1", "2", "3", "3a", "3b", "4"]
    v = str(val).strip().lower().replace("stage", "").strip()
    vec = [1.0 if v == s else 0.0 for s in stages]
    return vec


def build_clinical_vector(row: pd.Series) -> np.ndarray:
    """
    Build the clinical feature vector from a merged CSV row.
    Produces a 1-D float32 array of length 27 to match the original
    model's expected clinical input size.

    Layout:
      [0]       age (normalised /100)
      [1–6]     T-stage one-hot (6 dims)
      [7–12]    N-stage one-hot (6 dims)
      [13–18]   M-stage one-hot (6 dims)
      [19–24]   Overall-stage one-hot (6 dims)
      [25]      survival time (normalised, excluded from features to avoid leakage)
      [26]      dead-status event (0/1)
    """
    age_raw = row.get("age", 68)
    age = float(age_raw) / 100.0 if pd.notna(age_raw) else 0.68

    t  = _encode_stage(row.get("clinical.T.Stage",  ""))
    n  = _encode_stage(row.get("Clinical.N.Stage",  ""))
    m  = _encode_stage(row.get("Clinical.M.Stage",  ""))
    ov = _encode_stage(row.get("Overall.Stage",      ""))

    event_raw = row.get("deadstatus.event", 0)
    event = float(event_raw) if pd.notna(event_raw) else 0.0

    vec = np.array([age] + t + n + m + ov + [0.0, event], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
    return vec   # shape (27,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DataBowl3Classifier(Dataset):
    """
    NSCLC-Radiomics dataset loader for Lite-ProSENet.

    Replaces the original folder-scan approach with explicit CSV-driven loading
    from Kaggle DICOM paths, using the same interface as the original:
        __getitem__ returns (img, clinical_diag, label, event)

    Args:
        metadata_csv : path to phase1_metadata CSV (patient_id, ct_path, seg_path)
        clinical_csv : path to NSCLC-Radiomics clinical CSV
        phase        : 'train' | 'val' | 'test'
        isAugment    : apply random flip/rotate augmentation
        gtv_margin   : voxel padding around GTV bounding box
        indices      : optional list of integer indices (for pre-split subsets)
    """

    # Target spatial size expected by the Lite-ProSENet model
    NEW_X = 96
    NEW_Y = 96
    NEW_Z = 12

    def __init__(
        self,
        metadata_csv: str,
        clinical_csv: str,
        phase: str = "train",
        isAugment: bool = True,
        gtv_margin: int = 10,
        indices: list = None,
    ):
        assert phase in ("train", "val", "test")
        self.phase     = phase
        self.isAugment = isAugment and (phase == "train")
        self.gtv_margin = gtv_margin

        meta_df     = pd.read_csv(metadata_csv)
        clinical_df = pd.read_csv(clinical_csv)

        # Fix Kaggle paths
        meta_df["ct_path"]  = meta_df["ct_path"].apply(fix_path)
        meta_df["seg_path"] = meta_df["seg_path"].apply(fix_path)

        # Merge on patient ID
        merged = meta_df.merge(
            clinical_df, left_on="patient_id", right_on="PatientID", how="inner"
        )

        # Keep only rows with valid survival time
        merged = merged[merged["Survival.time"].notna()].reset_index(drop=True)

        # Normalise survival time to [0, 1] (the original label scheme)
        self.max_surv = float(merged["Survival.time"].max())

        if indices is not None:
            merged = merged.iloc[indices].reset_index(drop=True)

        self.data = merged

    # ------------------------------------------------------------------
    def _resample(self, sitk_image: sitk.Image, isAugment: bool) -> np.ndarray:
        """Resample to (NEW_X, NEW_Y, NEW_Z) — mirrors original data_resample."""
        new_size    = [self.NEW_X, self.NEW_Y, self.NEW_Z]
        new_spacing = [
            old_sz * old_spc / new_sz
            for old_sz, old_spc, new_sz in zip(
                sitk_image.GetSize(), sitk_image.GetSpacing(), new_size
            )
        ]

        if isAugment:
            arr      = sitk.GetArrayFromImage(sitk_image)
            arr      = augment(arr, ifflip=True, ifrotate=False, ifswap=False)
            sitk_image = sitk.GetImageFromArray(arr)

        new_img = sitk.Resample(
            sitk_image, new_size, sitk.Transform(),
            sitk.sitkLinear,
            sitk_image.GetOrigin(), new_spacing,
            sitk_image.GetDirection(), 0.0,
            sitk_image.GetPixelIDValue(),
        )
        return sitk.GetArrayFromImage(new_img)

    def _data_norm(self, data: np.ndarray) -> np.ndarray:
        """Normalise to [0, 255] — preserved from original."""
        dmin, dmax = data.min(), data.max()
        inter = dmax - dmin
        if inter == 0:
            return data
        return (data - dmin) / inter * 255.0

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row     = self.data.iloc[idx]
        ct_path = row["ct_path"]
        seg_path = row["seg_path"]

        # ---- Load CT as SimpleITK image --------------------------------
        sitk_ct = load_dicom_series(ct_path)
        ct_array = sitk.GetArrayFromImage(sitk_ct).astype(np.float32)

        # ---- Load segmentation mask ------------------------------------
        seg_array = load_segmentation_array(seg_path)
        if seg_array is None:
            seg_array = np.zeros_like(ct_array)

        # Ensure both 3D (D, H, W)
        for arr_name in ["ct_array", "seg_array"]:
            arr = locals()[arr_name]
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            elif arr.ndim == 4:
                arr = arr.squeeze(0) if arr.shape[0] == 1 else arr[..., 0]
            locals()[arr_name]  # noqa — reassign below
            if arr_name == "ct_array":
                ct_array = arr
            else:
                seg_array = arr

        # Match shapes if seg differs from CT
        if seg_array.shape != ct_array.shape:
            zf = tuple(t / s for s, t in zip(seg_array.shape, ct_array.shape))
            seg_array = zoom(seg_array, zf, order=0)
            seg_array = (seg_array > 0.5).astype(np.float32)

        # ---- GTV crop --------------------------------------------------
        ct_array, seg_array = gtv_crop(ct_array, seg_array, margin=self.gtv_margin)

        # ---- Lung window + rebuild SimpleITK image for resampling ------
        ct_array = np.clip(ct_array, -1000, 400).astype(np.float32)
        sitk_cropped = sitk.GetImageFromArray(ct_array)
        # Preserve original voxel spacing (size changed after crop, so CopyInformation would fail)
        sitk_cropped.SetSpacing(sitk_ct.GetSpacing())
        sitk_cropped.SetDirection(sitk_ct.GetDirection())

        # ---- Resample to (96, 96, 12) with optional augmentation -------
        img = self._resample(sitk_cropped, self.isAugment)
        img = self._data_norm(img)            # (12, 96, 96) — sitk returns Z,Y,X
        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0).astype(np.float32)

        # ---- Clinical vector and survival label ------------------------
        clinical_vec = build_clinical_vector(row)   # (27,)
        # Diagonal matrix as expected by the original model: (27, 27)
        clinical_diag = np.diag(clinical_vec).astype(np.float32)
        clinical_diag = np.nan_to_num(clinical_diag, nan=0.0, posinf=1.0, neginf=0.0)

        surv  = float(row["Survival.time"])
        label = surv / max(self.max_surv, 1e-8)            # normalised scalar in [0, 1]
        if not np.isfinite(label):
            label = 0.0

        event_raw = row.get("deadstatus.event", 0)
        try:
            event = bool(int(event_raw) == 1)
        except Exception:
            event = False

        return img, clinical_diag, label, event
