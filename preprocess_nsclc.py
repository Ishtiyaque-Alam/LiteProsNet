# preprocess_nsclc.py
# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline: NSCLC-Radiomics DICOM → Lite-ProSENet inputs
#
# OUTPUT STRUCTURE (matches data.py exactly):
#   data/
#   ├── train/  LUNG1-XXXGTV.mha   (60%)
#   ├── val/    LUNG1-XXXGTV.mha   (20%)
#   ├── test/   LUNG1-XXXGTV.mha   (20%)
#   └── NSCLC_PROCESSED.CSV
#       col[0]  = index
#       col[1]  = PatientID          ← data.py reads name = filename[pos-9:pos]
#       col[2:29] = 27 clinical feats (one-hot expanded from 7 raw columns)
#       col[29] = Survival.time
#       col[30] = deadstatus.event
#
# CONFIGURE THE TWO PATHS BELOW BEFORE RUNNING ON KAGGLE
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT    = "/kaggle/input/nsclc-radiomics/NSCLC-Radiomics"
CLINICAL_CSV = "/kaggle/input/nsclc-radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
OUT_DIR      = "/kaggle/working/data"

# ── target volume size expected by data.py ────────────────────────────────────
TARGET_X = 96
TARGET_Y = 96
TARGET_Z = 12

# ── train / val / test split (paper: 6:2:2) ──────────────────────────────────
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
# TEST_RATIO  = 0.2  (remainder)
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
import os, warnings, traceback, random, time
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def print_section(title):
    print("\n" + "═"*65)
    print(f"  {title}")
    print("═"*65)

def show_grid(images, titles, suptitle, rows=2, cols=3, cmap="gray",
              vmin=None, vmax=None, figsize=(14, 8)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, (img, ttl) in enumerate(zip(images, titles)):
        if i >= len(axes): break
        axes[i].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(ttl, fontsize=8)
        axes[i].axis("off")
    for j in range(len(images), len(axes)):
        axes[j].axis("off")
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 0 ── DISCOVER CT + SEG PATHS FOR ALL PATIENTS
# ═════════════════════════════════════════════════════════════════════════════

def discover_paths(data_root):
    print_section("STAGE 0 — Path Discovery")
    root = Path(data_root)
    records = []
    missing_ct, missing_seg = [], []

    for patient_dir in sorted(root.glob("LUNG1-*")):
        patient_id = patient_dir.name

        # collect all study sub-dirs
        study_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]

        ct_folder = None
        seg_file  = None

        for study_dir in study_dirs:
            subdirs = [d for d in study_dir.iterdir() if d.is_dir()]

            # SEG: folder whose name starts with "300."
            for d in subdirs:
                if d.name.startswith("300."):
                    candidates = list(d.rglob("*.dcm"))
                    if candidates:
                        seg_file = str(candidates[0])
                        break

            # CT: non-300.* folder with the most .dcm files
            best, best_count = None, 0
            for d in subdirs:
                if d.name.startswith("300."):
                    continue
                n = len(list(d.rglob("*.dcm")))
                if n > best_count:
                    best, best_count = d, n
            if best is not None and best_count >= 10:
                ct_folder = str(best)

        if ct_folder and seg_file:
            records.append({"patient_id": patient_id,
                             "ct_folder":  ct_folder,
                             "seg_file":   seg_file})
        else:
            if not ct_folder: missing_ct.append(patient_id)
            if not seg_file:  missing_seg.append(patient_id)

    df = pd.DataFrame(records)
    print(f"  Patients with both CT + SEG : {len(df)}")
    print(f"  Missing CT                  : {len(missing_ct)}  {missing_ct[:5]}")
    print(f"  Missing SEG                 : {len(missing_seg)}  {missing_seg[:5]}")
    print(f"\n  First 5 records:")
    print(df.head().to_string(index=False))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 ── BUILD NSCLC_PROCESSED.CSV
# ═════════════════════════════════════════════════════════════════════════════

def build_clinical_csv(clinical_csv_path, path_df, out_dir):
    print_section("STAGE 1 — Clinical CSV Processing")

    # na_values ensures literal "NA" strings are read as NaN
    raw = pd.read_csv(clinical_csv_path, na_values=["NA", "N/A", "", " "])
    print(f"  Raw CSV shape : {raw.shape}")
    print(f"  Columns       : {list(raw.columns)}")

    # ── missing value summary ─────────────────────────────────────────────
    miss = raw.isnull().sum()
    print(f"\n  Missing values per column:")
    for col, n in miss.items():
        if n > 0:
            print(f"    {col:35s}: {n}")

    # ── impute age with column mean (22 NA rows) ──────────────────────────
    age_mean      = raw["age"].dropna().mean()
    n_age_imputed = raw["age"].isnull().sum()
    raw["age"]    = raw["age"].fillna(age_mean)
    raw["age_norm"] = raw["age"] / 100.0
    print(f"\n  Age: {raw['age'].min():.1f} – {raw['age'].max():.1f}  "
          f"(mean {age_mean:.1f}, imputed {n_age_imputed} rows)")

    # ── encode T Stage (1-5) → one-hot 5 cols ────────────────────────────
    raw["clinical.T.Stage"] = pd.to_numeric(raw["clinical.T.Stage"],
                                            errors="coerce").fillna(2).astype(int)
    t_dummies = pd.get_dummies(raw["clinical.T.Stage"],
                               prefix="T_stage").reindex(
        columns=[f"T_stage_{i}" for i in range(1, 6)], fill_value=0)

    # ── encode N Stage (0-4) → one-hot 5 cols ────────────────────────────
    raw["Clinical.N.Stage"] = pd.to_numeric(raw["Clinical.N.Stage"],
                                            errors="coerce").fillna(0).astype(int)
    n_dummies = pd.get_dummies(raw["Clinical.N.Stage"],
                               prefix="N_stage").reindex(
        columns=[f"N_stage_{i}" for i in range(0, 5)], fill_value=0)

    # ── encode M Stage (0/1/3) → one-hot 3 cols ──────────────────────────
    raw["Clinical.M.Stage"] = pd.to_numeric(raw["Clinical.M.Stage"],
                                            errors="coerce").fillna(0).astype(int)
    m_dummies = pd.get_dummies(raw["Clinical.M.Stage"],
                               prefix="M_stage").reindex(
        columns=[f"M_stage_{i}" for i in [0, 1, 3]], fill_value=0)

    # ── encode Histology (5 categories) → one-hot 5 cols ─────────────────
    # Unique values: adenocarcinoma, large cell, nos,
    #                squamous cell carcinoma, NaN (originally "NA" → 42 rows)
    hist_cats = ["adenocarcinoma", "large cell", "nos",
                 "squamous cell carcinoma", "unknown"]
    raw["hist_clean"] = raw["Histology"].fillna("unknown")
    h_dummies = pd.get_dummies(raw["hist_clean"],
                               prefix="hist").reindex(
        columns=[f"hist_{c}" for c in hist_cats], fill_value=0)

    # ── encode Overall Stage → one-hot 5 cols ────────────────────────────
    stage_cats = ["I", "II", "IIIa", "IIIb", "IV"]
    raw["stage_clean"] = raw["Overall.Stage"].fillna("IIIb")  # 1 NA → mode
    s_dummies = pd.get_dummies(raw["stage_clean"],
                               prefix="stage").reindex(
        columns=[f"stage_{s}" for s in stage_cats], fill_value=0)

    # ── encode gender → 1 col ────────────────────────────────────────────
    raw["gender_male"] = (raw["gender"].str.lower() == "male").astype(int)

    # ── continuous numeric features ───────────────────────────────────────
    raw["T_num_norm"] = raw["clinical.T.Stage"] / 5.0
    raw["N_num_norm"] = raw["Clinical.N.Stage"]  / 4.0
    numeric_cols = raw[["age_norm", "T_num_norm", "N_num_norm"]].copy()

    # ── assemble exactly 27 clinical features ─────────────────────────────
    # 3 numeric   : age_norm, T_num_norm, N_num_norm
    # 5 T one-hot : T_stage_1 … T_stage_5
    # 5 N one-hot : N_stage_0 … N_stage_4
    # 3 M one-hot : M_stage_0, M_stage_1, M_stage_3
    # 5 stage OH  : stage_I … stage_IV
    # 5 hist  OH  : hist_adenocarcinoma … hist_unknown
    # 1 gender    : gender_male
    # TOTAL       : 3+5+5+3+5+5+1 = 27  ✓
    clinical_df = pd.concat(
        [numeric_cols, t_dummies, n_dummies, m_dummies,
         s_dummies, h_dummies, raw[["gender_male"]]],
        axis=1
    )

    print(f"\n  Clinical feature breakdown:")
    print(f"    numeric      : {numeric_cols.shape[1]}  (age_norm, T_num_norm, N_num_norm)")
    print(f"    T one-hot    : {t_dummies.shape[1]}")
    print(f"    N one-hot    : {n_dummies.shape[1]}")
    print(f"    M one-hot    : {m_dummies.shape[1]}")
    print(f"    Stage one-hot: {s_dummies.shape[1]}")
    print(f"    Hist one-hot : {h_dummies.shape[1]}")
    print(f"    gender       : 1")
    print(f"    TOTAL        : {clinical_df.shape[1]}  (need 27)")
    assert clinical_df.shape[1] == 27, \
        f"Expected 27 clinical cols, got {clinical_df.shape[1]}"

    # ── survival time + event ──────────────────────────────────────────────
    surv = raw["Survival.time"].astype(float)
    evt  = raw["deadstatus.event"].astype(int)

    # ── final CSV: index | PatientID | 27 clinical | surv_time | event ───
    out = pd.concat(
        [raw[["PatientID"]].reset_index(drop=True),
         clinical_df.reset_index(drop=True),
         surv.reset_index(drop=True),
         evt.reset_index(drop=True)],
        axis=1
    )
    # DataFrame has 30 cols: PatientID(1) + clinical(27) + surv(1) + event(1)
    # When saved with index=True and read back by data.py via pd.read_csv,
    # the index becomes col[0], so col[1]=PatientID, col[2:29]=clinical,
    # col[29]=Survival.time, col[30]=deadstatus.event  ✓
    assert out.shape[1] == 30, \
        f"Expected 30 cols (PatientID + 27 clinical + surv + event), got {out.shape[1]}"

    # ── keep only patients that have CT+SEG ───────────────────────────────
    valid_ids = set(path_df["patient_id"].tolist())
    out_filt = out[out["PatientID"].isin(valid_ids)].reset_index(drop=True)
    print(f"\n  Patients in CSV              : {len(out)}")
    print(f"  Patients with CT+SEG match   : {len(out_filt)}")
    print(f"  Final CSV shape              : {out_filt.shape}  "
          f"(30 cols; after index=True save → col[0]=idx, col[1]=PatientID, "
          f"col[2:29]=clinical, col[29]=surv, col[30]=event)")
    print(f"  Column[0]  = {out_filt.columns[0]}   (PatientID)")
    print(f"  Column[28] = {out_filt.columns[28]}  (Survival.time)")
    print(f"  Column[29] = {out_filt.columns[29]}  (deadstatus.event)")
    print(f"\n  Clinical feature columns [2:29]:")
    for i, c in enumerate(out_filt.columns[2:29], start=2):
        print(f"    [{i:2d}] {c}")

    # ── save ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "NSCLC_PROCESSED.CSV")
    out_filt.to_csv(csv_out, index=True)
    print(f"\n  Saved → {csv_out}")

    # ── visualizations ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Survival time distribution
    axes[0].hist(surv.dropna(), bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Survival Time Distribution (days)")
    axes[0].set_xlabel("Days"); axes[0].set_ylabel("Count")

    # 2. Event pie
    ec = evt.value_counts()
    axes[1].pie(ec, labels=["Dead (1)", "Censored (0)"] if 1 in ec.index else ec.index,
                autopct="%1.1f%%", colors=["#DD8452", "#4C72B0"])
    axes[1].set_title("Event Status")

    # 3. Age distribution
    axes[2].hist(raw["age"].dropna(), bins=30, color="#55A868", edgecolor="white")
    axes[2].set_title("Age Distribution")
    axes[2].set_xlabel("Age (years)"); axes[2].set_ylabel("Count")

    plt.suptitle("STAGE 1 — Clinical Data Overview", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # 4. Clinical feature heatmap (first 20 patients)
    sample = out_filt[out_filt.columns[2:29]].head(20).astype(float)
    fig2, ax2 = plt.subplots(figsize=(16, 4))
    im = ax2.imshow(sample.T.values, aspect="auto", cmap="viridis")
    ax2.set_yticks(range(27))
    ax2.set_yticklabels(out_filt.columns[2:29].tolist(), fontsize=7)
    ax2.set_xlabel("Patient index (first 20)")
    ax2.set_title("Clinical Feature Matrix (first 20 patients)", fontweight="bold")
    plt.colorbar(im, ax=ax2, fraction=0.03)
    plt.tight_layout()
    plt.show()

    return out_filt, csv_out


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 ── SINGLE-PATIENT DEMO (full visual pipeline)
# ═════════════════════════════════════════════════════════════════════════════

def load_ct_series(ct_folder):
    reader = sitk.ImageSeriesReader()
    fnames = reader.GetGDCMSeriesFileNames(ct_folder)
    if not fnames:
        # fallback: collect dcm files sorted by z
        dcm_files = sorted(Path(ct_folder).rglob("*.dcm"))
        slices = []
        for f in dcm_files:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            z  = float(getattr(ds, "ImagePositionPatient", [0,0,0])[2])
            slices.append((z, str(f)))
        slices.sort(key=lambda t: t[0])
        fnames = [s[1] for s in slices]
    reader.SetFileNames(fnames)
    ct_img = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    return ct_img


def extract_gtv_mask(seg_file, ct_img):
    """Extract GTV segment from SEG DICOM as a binary SimpleITK mask
       aligned to ct_img grid."""
    seg_ds = pydicom.dcmread(str(seg_file), force=True)

    # ── find tumor segment number ──────────────────────────────────────────
    GTV_KEYWORDS = ["neoplasm", "gtv", "tumor", "primary"]
    tumor_num = None
    if hasattr(seg_ds, "SegmentSequence"):
        for seg in seg_ds.SegmentSequence:
            lbl = getattr(seg, "SegmentLabel", "").lower()
            if any(kw in lbl for kw in GTV_KEYWORDS):
                tumor_num = int(getattr(seg, "SegmentNumber", 0))
                break
        if tumor_num is None:                           # fallback: first segment
            tumor_num = int(getattr(seg_ds.SegmentSequence[0],
                                    "SegmentNumber", 1))
    else:
        tumor_num = 1

    # ── get CT z-positions ──────────────────────────────────────────────────
    ct_arr = sitk.GetArrayFromImage(ct_img)             # [z, y, x]
    Z, H, W = ct_arr.shape
    ct_origin  = np.array(ct_img.GetOrigin())
    ct_spacing = np.array(ct_img.GetSpacing())
    ct_dir     = np.array(ct_img.GetDirection()).reshape(3, 3)
    ct_zs = np.array([ct_origin[2] + i * ct_spacing[2]
                      for i in range(Z)])

    # ── build mask volume ──────────────────────────────────────────────────
    mask_vol = np.zeros((Z, H, W), dtype=np.uint8)
    n_frames = int(getattr(seg_ds, "NumberOfFrames", 0))

    if n_frames > 0 and hasattr(seg_ds, "PerFrameFunctionalGroupsSequence"):
        px = seg_ds.pixel_array                         # (n_frames, rows, cols)
        px = (px > 0).astype(np.uint8)
        pfg = seg_ds.PerFrameFunctionalGroupsSequence

        for i in range(n_frames):
            try:
                seg_num = int(pfg[i].SegmentIdentificationSequence[0]
                              .ReferencedSegmentNumber)
                if seg_num != tumor_num:
                    continue
                z_mm = float(pfg[i].PlanePositionSequence[0]
                             .ImagePositionPatient[2])
                k = int(np.argmin(np.abs(ct_zs - z_mm)))
                if np.abs(ct_zs[k] - z_mm) < 2 * ct_spacing[2]:
                    mask_vol[k] = np.maximum(mask_vol[k], px[i])
            except Exception:
                continue
    else:
        # simple single-label SEG
        seg_sitk = sitk.ReadImage(str(seg_file))
        seg_sitk = sitk.Cast(seg_sitk > 0, sitk.sitkUInt8)
        f = sitk.ResampleImageFilter()
        f.SetReferenceImage(ct_img)
        f.SetInterpolator(sitk.sitkNearestNeighbor)
        f.SetDefaultPixelValue(0)
        seg_on_ct = f.Execute(seg_sitk)
        mask_vol  = sitk.GetArrayFromImage(seg_on_ct).astype(np.uint8)

    mask_sitk = sitk.GetImageFromArray(mask_vol)
    mask_sitk.CopyInformation(ct_img)
    return mask_sitk, tumor_num


def crop_gtv_bbox(ct_img, mask_sitk, pad_vox=4):
    """Crop a bounding-box around the GTV in all 3 dims, with padding."""
    ct_arr   = sitk.GetArrayFromImage(ct_img)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)

    nz = np.argwhere(mask_arr > 0)
    if nz.size == 0:
        return None, None, None    # no GTV found

    zmin, ymin, xmin = nz.min(axis=0)
    zmax, ymax, xmax = nz.max(axis=0)

    Z, Y, X = ct_arr.shape
    zmin = max(0, zmin - pad_vox); zmax = min(Z-1, zmax + pad_vox)
    ymin = max(0, ymin - pad_vox); ymax = min(Y-1, ymax + pad_vox)
    xmin = max(0, xmin - pad_vox); xmax = min(X-1, xmax + pad_vox)

    ct_crop   = ct_arr  [zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    mask_crop = mask_arr[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    bbox = (zmin, zmax, ymin, ymax, xmin, xmax)
    return ct_crop, mask_crop, bbox


def resample_to_target(np_arr, target=(TARGET_Z, TARGET_Y, TARGET_X),
                       is_label=False):
    sitk_img = sitk.GetImageFromArray(np_arr.astype(
        np.uint8 if is_label else np.float32))
    orig_size    = sitk_img.GetSize()          # (x, y, z) in sitk
    orig_spacing = sitk_img.GetSpacing()
    new_size = [target[2], target[1], target[0]]   # sitk: (x,y,z)
    new_spacing = [old_sz * old_spc / new_sz
                   for old_sz, old_spc, new_sz
                   in zip(orig_size, orig_spacing, new_size)]
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    resampled = sitk.Resample(sitk_img, new_size, sitk.Transform(),
                              interp, sitk_img.GetOrigin(),
                              new_spacing, sitk_img.GetDirection(),
                              0.0, sitk_img.GetPixelIDValue())
    return sitk.GetArrayFromImage(resampled)


def normalize_0_255(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn) * 255).astype(np.float32)


def demo_single_patient(row):
    print_section("STAGE 2 — Single Patient Demo")
    pid = row["patient_id"]
    print(f"  Patient : {pid}")

    # ── 1. Load CT ──────────────────────────────────────────────────────────
    ct_img = load_ct_series(row["ct_folder"])
    ct_arr = sitk.GetArrayFromImage(ct_img)
    print(f"  CT shape    : {ct_arr.shape}   spacing: {ct_img.GetSpacing()}")
    print(f"  CT HU range : [{ct_arr.min():.0f}, {ct_arr.max():.0f}]")

    # ── 2. Extract GTV mask ─────────────────────────────────────────────────
    mask_sitk, tumor_num = extract_gtv_mask(row["seg_file"], ct_img)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    nz_slices = int((mask_arr > 0).any(axis=(1,2)).sum())
    print(f"  Tumor segment #     : {tumor_num}")
    print(f"  Mask non-zero slices: {nz_slices} / {ct_arr.shape[0]}")
    print(f"  Mask voxel count    : {int(mask_arr.sum())}")

    if nz_slices == 0:
        print("  ⚠️  Empty mask — skipping demo.")
        return

    # ── 3. Crop bounding box ────────────────────────────────────────────────
    ct_crop, mask_crop, bbox = crop_gtv_bbox(ct_img, mask_sitk)
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    sp = ct_img.GetSpacing()
    bbox_mm = ((zmax-zmin)*sp[2], (ymax-ymin)*sp[1], (xmax-xmin)*sp[0])
    print(f"  Bbox (vox)  : z[{zmin}:{zmax}] y[{ymin}:{ymax}] x[{xmin}:{xmax}]")
    print(f"  Bbox (mm)   : z={bbox_mm[0]:.1f} y={bbox_mm[1]:.1f} x={bbox_mm[2]:.1f}")
    print(f"  Crop shape  : {ct_crop.shape}")

    # ── 4. Resample to 96×96×12 ─────────────────────────────────────────────
    ct_resampled   = resample_to_target(ct_crop,   is_label=False)
    mask_resampled = resample_to_target(mask_crop, is_label=True)
    print(f"  After resample : {ct_resampled.shape}")

    # ── 5. Normalize ────────────────────────────────────────────────────────
    ct_norm = normalize_0_255(ct_resampled)
    print(f"  Normalized range : [{ct_norm.min():.0f}, {ct_norm.max():.0f}]")

    # ── VISUALIZATION ────────────────────────────────────────────────────────

    # Panel A: mid-slice full CT + GTV overlay
    mid = ct_arr.shape[0] // 2
    gtv_slices = np.where((mask_arr > 0).any(axis=(1,2)))[0]
    gtv_mid    = int(gtv_slices[len(gtv_slices)//2])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(ct_arr[mid], cmap="gray", vmin=-1000, vmax=400)
    axes[0].set_title(f"Full CT — mid slice ({mid})")

    axes[1].imshow(ct_arr[gtv_mid], cmap="gray", vmin=-1000, vmax=400)
    axes[1].imshow(np.ma.masked_where(mask_arr[gtv_mid]==0, mask_arr[gtv_mid]),
                   cmap="Reds", alpha=0.45)
    axes[1].set_title(f"CT + GTV overlay (slice {gtv_mid})")

    axes[2].imshow(ct_crop[ct_crop.shape[0]//2], cmap="gray")
    axes[2].set_title(f"Bounding-box crop\n{ct_crop.shape}")

    axes[3].imshow(ct_norm[ct_norm.shape[0]//2], cmap="gray", vmin=0, vmax=255)
    axes[3].set_title(f"After resample+norm\n{ct_norm.shape}")

    for ax in axes: ax.axis("off")
    fig.suptitle(f"STAGE 2 — {pid}", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()

    # Panel B: all GTV slices grid
    n_show = min(12, len(gtv_slices))
    idxs   = gtv_slices[np.linspace(0, len(gtv_slices)-1, n_show).astype(int)]
    imgs   = [ct_arr[k] for k in idxs]
    ttls   = [f"slice {k}" for k in idxs]
    show_grid(imgs, ttls, f"{pid} — GTV-containing slices",
              rows=2, cols=6, vmin=-1000, vmax=400, figsize=(20, 7))

    # Panel C: resampled slices (all 12 z)
    imgs2  = [ct_norm[k] for k in range(ct_norm.shape[0])]
    ttls2  = [f"z={k}" for k in range(ct_norm.shape[0])]
    show_grid(imgs2, ttls2, f"{pid} — Resampled 96×96×12 (all z slices)",
              rows=2, cols=6, vmin=0, vmax=255, figsize=(20, 7))

    # Panel D: intensity histogram before / after
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.hist(ct_crop.flatten(),   bins=60, color="#4C72B0", edgecolor="white")
    a1.set_title("HU histogram — cropped bbox")
    a1.set_xlabel("HU value")
    a2.hist(ct_norm.flatten(),   bins=60, color="#55A868", edgecolor="white")
    a2.set_title("Intensity histogram — after norm [0-255]")
    a2.set_xlabel("Intensity")
    plt.suptitle(f"STAGE 2 — {pid} — Histogram", fontweight="bold")
    plt.tight_layout(); plt.show()

    return ct_norm, mask_resampled


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 ── BATCH PREPROCESSING: ALL 422 PATIENTS
# ═════════════════════════════════════════════════════════════════════════════

def split_patients(patient_ids, train_r=TRAIN_RATIO, val_r=VAL_RATIO,
                   seed=RANDOM_SEED):
    ids = sorted(patient_ids)
    train_ids, temp = train_test_split(ids, test_size=1-train_r,
                                       random_state=seed)
    val_ids, test_ids = train_test_split(temp,
                                         test_size=0.5,
                                         random_state=seed)
    return train_ids, val_ids, test_ids


def process_one_patient(row, out_split_dir):
    """
    Full pipeline for one patient.
    Returns: ("ok"|"no_gtv"|"error", patient_id, message)
    """
    pid = row["patient_id"]
    out_path = os.path.join(out_split_dir, f"{pid}GTV.mha")

    try:
        ct_img = load_ct_series(row["ct_folder"])
        mask_sitk, _ = extract_gtv_mask(row["seg_file"], ct_img)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        if mask_arr.sum() == 0:
            return ("no_gtv", pid, "Empty mask after extraction")

        ct_crop, mask_crop, bbox = crop_gtv_bbox(ct_img, mask_sitk)
        if ct_crop is None:
            return ("no_gtv", pid, "crop_gtv_bbox returned None")

        ct_resampled = resample_to_target(ct_crop, is_label=False)
        ct_norm      = normalize_0_255(ct_resampled)

        # Save as SimpleITK mha (what data.py reads with sitk.ReadImage)
        out_sitk = sitk.GetImageFromArray(ct_norm)
        sitk.WriteImage(out_sitk, out_path)
        return ("ok", pid, out_path)

    except Exception as e:
        return ("error", pid, f"{type(e).__name__}: {e}")


def batch_preprocess(path_df, clinical_df, out_dir):
    print_section("STAGE 3 — Batch Preprocessing (all patients)")

    # ── 1. split ────────────────────────────────────────────────────────────
    all_ids = path_df["patient_id"].tolist()
    # keep only patients present in clinical CSV
    valid_ids = set(clinical_df["PatientID"].tolist())
    all_ids = [p for p in all_ids if p in valid_ids]

    train_ids, val_ids, test_ids = split_patients(all_ids)
    split_map = {pid: "train" for pid in train_ids}
    split_map.update({pid: "val"   for pid in val_ids})
    split_map.update({pid: "test"  for pid in test_ids})

    print(f"  Total usable patients : {len(all_ids)}")
    print(f"  Train : {len(train_ids)}  |  Val : {len(val_ids)}  |  Test : {len(test_ids)}")

    # ── 2. create output dirs ───────────────────────────────────────────────
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    # ── 3. process each patient ──────────────────────────────────────────────
    results = {"ok": [], "no_gtv": [], "error": []}
    id_to_row = {r["patient_id"]: r for _, r in path_df.iterrows()}
    t0 = time.time()

    for i, pid in enumerate(all_ids):
        split     = split_map[pid]
        out_split = os.path.join(out_dir, split)
        row       = id_to_row[pid]

        status, pid_out, msg = process_one_patient(row, out_split)
        results[status].append((pid_out, msg))

        # progress every 10 patients
        elapsed = time.time() - t0
        eta     = elapsed / (i+1) * (len(all_ids) - i - 1)
        tag = "✅" if status=="ok" else ("⚠️ " if status=="no_gtv" else "❌")
        if (i+1) % 10 == 0 or status != "ok":
            print(f"  [{i+1:3d}/{len(all_ids)}] {tag} {pid_out:12s}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  — {msg if status!='ok' else 'saved'}")

    # ── 4. summary ──────────────────────────────────────────────────────────
    print_section("STAGE 3 — Summary")
    print(f"  ✅ Saved    : {len(results['ok'])}")
    print(f"  ⚠️  No GTV  : {len(results['no_gtv'])}")
    print(f"  ❌ Errors   : {len(results['error'])}")
    if results["error"]:
        print(f"\n  Error details:")
        for pid, msg in results["error"][:10]:
            print(f"    {pid}: {msg}")

    # ── 5. visual: random GTV thumbnails grid ──────────────────────────────
    saved_files = []
    for split in ["train", "val", "test"]:
        sdir = os.path.join(out_dir, split)
        saved_files += [os.path.join(sdir, f)
                        for f in os.listdir(sdir) if f.endswith(".mha")]

    if saved_files:
        sample_files = random.sample(saved_files, min(9, len(saved_files)))
        thumbs, ttls = [], []
        for fp in sample_files:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(fp))
            mid = arr.shape[0] // 2
            thumbs.append(arr[mid])
            ttls.append(Path(fp).stem[:12])
        show_grid(thumbs, ttls,
                  "STAGE 3 — Random saved GTV crops (mid-slice, normalized 0-255)",
                  rows=3, cols=3, vmin=0, vmax=255, figsize=(12, 12))

    # ── 6. split count bar chart ────────────────────────────────────────────
    counts = {s: len([f for f in saved_files
                       if f"/{s}/" in f.replace("\\", "/")])
              for s in ["train", "val", "test"]}
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(counts.keys(), counts.values(),
                  color=["#4C72B0", "#55A868", "#DD8452"])
    for b, v in zip(bars, counts.values()):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                str(v), ha="center", va="bottom", fontweight="bold")
    ax.set_title("STAGE 3 — Files saved per split", fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout(); plt.show()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 ── SANITY CHECK: run data.py DataBowl3Classifier on output
# ═════════════════════════════════════════════════════════════════════════════

def sanity_check(out_dir):
    print_section("STAGE 4 — Sanity Check (DataBowl3Classifier)")

    # inline the data.py loader to avoid import path issues on Kaggle
    import sys
    sys.path.insert(0, "/kaggle/input/lite-prosenet/Lite-ProSENet")   # adjust if needed

    try:
        from data import DataBowl3Classifier
        from torch.utils.data import DataLoader

        ds = DataBowl3Classifier(os.path.join(out_dir, "train"),
                                 phase="train", isAugment=False)
        print(f"  Dataset length : {len(ds)}")

        img, clinical, label, event = ds[0]
        print(f"  img shape      : {img.shape}")
        print(f"  clinical shape : {clinical.shape}")
        print(f"  label          : {label:.4f}")
        print(f"  event          : {event}")

        # visualize one sample
        mid = img.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img[mid], cmap="gray")
        axes[0].set_title(f"img[{mid}]  shape={img.shape}")
        axes[1].imshow(img[img.shape[0]//4], cmap="gray")
        axes[1].set_title(f"img[{img.shape[0]//4}]")
        axes[2].imshow(img[img.shape[0]*3//4], cmap="gray")
        axes[2].set_title(f"img[{img.shape[0]*3//4}]")
        for ax in axes: ax.axis("off")
        fig.suptitle(f"STAGE 4 — DataBowl3Classifier sample  "
                     f"(label={label:.3f}, event={event})",
                     fontweight="bold")
        plt.tight_layout(); plt.show()

        print("\n  ✅ data.py loads correctly — pipeline verified!")

    except ImportError:
        print("  ℹ️  data.py not found on this path — skipping import check.")
        print("     Verify shapes manually using the saved .mha files.")

        # manual check on saved mha
        train_dir = os.path.join(out_dir, "train")
        mha_files = list(Path(train_dir).glob("*.mha"))
        if mha_files:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mha_files[0])))
            print(f"\n  Loaded {mha_files[0].name}")
            print(f"  Array shape : {arr.shape}    (expected 12×96×96)")
            print(f"  Value range : [{arr.min():.0f}, {arr.max():.0f}]   (expected 0–255)")

            mid = arr.shape[0] // 2
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for j, k in enumerate([0, mid, arr.shape[0]-1]):
                axes[j].imshow(arr[k], cmap="gray", vmin=0, vmax=255)
                axes[j].set_title(f"z={k}")
                axes[j].axis("off")
            fig.suptitle(f"STAGE 4 — {mha_files[0].name}  shape={arr.shape}",
                         fontweight="bold")
            plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start = time.time()

    # ── STAGE 0: discover paths ──────────────────────────────────────────────
    path_df = discover_paths(DATA_ROOT)

    # ── STAGE 1: build clinical CSV ──────────────────────────────────────────
    clinical_df, csv_path = build_clinical_csv(CLINICAL_CSV, path_df, OUT_DIR)

    # ── STAGE 2: single patient demo ────────────────────────────────────────
    demo_row = path_df.iloc[0]
    demo_single_patient(demo_row)

    # ── STAGE 3: batch preprocess all patients ───────────────────────────────
    results = batch_preprocess(path_df, clinical_df, OUT_DIR)

    # ── STAGE 4: sanity check ────────────────────────────────────────────────
    sanity_check(OUT_DIR)

    print_section("DONE")
    print(f"  Total time : {(time.time()-start)/60:.1f} minutes")
    print(f"  Output dir : {OUT_DIR}")
    print(f"  Files:")
    for split in ["train", "val", "test"]:
        n = len(list(Path(os.path.join(OUT_DIR, split)).glob("*.mha")))
        print(f"    {split:5s}/  {n} .mha files")
    print(f"    NSCLC_PROCESSED.CSV")
