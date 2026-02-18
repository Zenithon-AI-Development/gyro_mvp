"""Data loading: HDF5 (CGYRO targets) and NPZ (TGLF conditioning inputs).

Species conventions:
  CGYRO: species 0 = IONS, species 1 = ELECTRONS
  TGLF:  species 0 / spec_1 = ELECTRONS, species 1 / spec_2 = IONS

Target ordering: [Q_electron, Q_ion]
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import build_feature_vector, build_structured_vector


def parse_experiment_from_run_name(run_name: str) -> str:
    """Derive experiment group from run_name.

    Examples:
        "2019_03-isotope_d_an2_q1_filtered" -> "isotope_d"
        "r90_dlntdr_dlt1_filtered" -> "r90_scans"
        "STD_AC_a1_filtered" -> "STD_AC"
        "2020_09-rmin_r025_filtered" -> "rmin"
    """
    name = run_name.replace('_filtered', '')

    if 'isotope_d' in name:
        return 'isotope_d'
    elif 'isotope_h' in name:
        return 'isotope_h'
    elif 'isotope_t' in name:
        return 'isotope_t'
    elif 'r90' in name:
        return 'r90_scans'
    elif 'r70_ge' in name:
        return 'r70_ge'
    elif 'Belli' in name:
        return 'Belli_Lmode_iso'
    elif 'rmin' in name:
        return 'rmin'
    elif 'theta' in name:
        return 'theta_diagnostic'
    elif 'nuei_lor' in name:
        return 'nuei_lor'
    elif 'garyshift' in name:
        return 'garyshift'
    elif 'exb_2res' in name:
        return 'exb_2res'
    elif 'exb_kappa' in name:
        return 'exb_kappa'
    elif 'exb_shift' in name:
        return 'exb_shift'
    elif 'STD_AC' in name:
        return 'STD_AC'
    elif 'STD_A' in name:
        return 'STD_A'
    elif 'STD_gamma' in name:
        return 'STD_gamma'
    else:
        return 'other'


def load_run_list(csv_path: str) -> pd.DataFrame:
    """Load validated run list from CSV.

    Handles both old format (folder_name column) and new format (run_name column).
    If 'experiment' column is missing, derives it from run_name.

    Returns:
        DataFrame with at least 'folder_name' and 'experiment' columns.
    """
    df = pd.read_csv(csv_path)

    # Handle column naming: old CSV has 'folder_name', new CSV has 'run_name'
    if 'folder_name' not in df.columns and 'run_name' in df.columns:
        df['folder_name'] = df['run_name']

    # Derive experiment column if missing
    if 'experiment' not in df.columns:
        df['experiment'] = df['folder_name'].apply(parse_experiment_from_run_name)

    return df


def load_cgyro_targets(h5_path: str) -> Tuple[float, float]:
    """Extract Q_ion and Q_electron from CGYRO HDF5 file.

    Takes mean of last 30% of timesteps for energy flux (index 1).

    Args:
        h5_path: Path to the .h5 file.

    Returns:
        (Q_electron, Q_ion) — note ordering matches target convention.
    """
    with h5py.File(h5_path, "r") as f:
        # total_flux_species0 = ions, shape (1, N_steps, 3)
        # total_flux_species1 = electrons, shape (1, N_steps, 3)
        flux_ion = f["fluxes/total_flux_species0"][0, :, 1]   # (N_steps,)
        flux_elec = f["fluxes/total_flux_species1"][0, :, 1]  # (N_steps,)

    n_steps = len(flux_ion)
    cutoff = int(0.7 * n_steps)
    q_ion = float(np.mean(flux_ion[cutoff:]))
    q_elec = float(np.mean(flux_elec[cutoff:]))

    return q_elec, q_ion


def load_tglf_conditioning(npz_path: str) -> Dict[str, np.ndarray]:
    """Load TGLF conditioning data from NPZ file.

    Returns:
        Dict with keys: 'QL_weights', 'gamma', 'freq', 'flux_spectrum'.
        flux_spectrum is a dict with keys 'spec_1_field_1', etc.
    """
    data = np.load(npz_path, allow_pickle=True)
    return {
        "QL_weights": data["QL_weights"],
        "gamma": data["gamma"],
        "freq": data["freq"],
        "flux_spectrum": data["flux_spectrum"].item(),
    }


def load_all_data(
    data_dir: str,
    run_list: pd.DataFrame,
    global_param_columns: List[str] = None,
    feature_builder: str = "flat",
    use_csv_targets: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load features and targets for runs.

    Args:
        data_dir: Path to root data directory containing run folders.
        run_list: DataFrame with 'folder_name' column.
        global_param_columns: List of column names to extract as global params.
                              If None or empty, returns TGLF-only features (378 dims).
                              Otherwise, appends these params to get (378 + len) dims.
        feature_builder: "flat" uses build_feature_vector (378-dim aggregated),
                         "structured" uses build_structured_vector().ravel() (378-dim ky-major).
        use_csv_targets: If True, read Q_electron/Q_ion from CSV columns instead of .h5 files.

    Returns:
        features: np.ndarray of shape (N, 378) or (N, 378+len(global_param_columns))
        targets: np.ndarray of shape (N, 2) = [Q_electron, Q_ion]
        metadata: DataFrame with folder_name, experiment, and targets
    """
    features_list = []
    targets_list = []
    loaded_folders = []
    skipped = []

    # Check if CSV has target columns (for use_csv_targets)
    has_csv_targets = use_csv_targets and "Q_electron" in run_list.columns and "Q_ion" in run_list.columns

    for _, row in tqdm(run_list.iterrows(), total=len(run_list), desc="Loading data"):
        folder_name = row["folder_name"]
        folder_path = os.path.join(data_dir, folder_name)

        if not os.path.isdir(folder_path):
            skipped.append(folder_name)
            continue

        # Find the .h5 file (named after the folder)
        h5_path = os.path.join(folder_path, f"{folder_name}.h5")
        npz_path = os.path.join(folder_path, "conditioning_data.npz")

        # NPZ always required (for TGLF features); h5 only needed if not using CSV targets
        if not os.path.exists(npz_path):
            skipped.append(folder_name)
            continue
        if not has_csv_targets and not os.path.exists(h5_path):
            skipped.append(folder_name)
            continue

        # Load targets
        if has_csv_targets:
            q_elec = float(row["Q_electron"])
            q_ion = float(row["Q_ion"])
        else:
            q_elec, q_ion = load_cgyro_targets(h5_path)
        targets_list.append([q_elec, q_ion])

        # Load features from TGLF
        raw_tglf = load_tglf_conditioning(npz_path)
        if feature_builder == "structured":
            feat_tglf = build_structured_vector(raw_tglf).ravel()  # (378,) ky-major
        else:
            feat_tglf = build_feature_vector(raw_tglf)  # (378,) channel-major

        # Optionally append global parameters from CSV
        if global_param_columns:
            global_params = np.array([float(row[col]) for col in global_param_columns], dtype=np.float64)
            feat_full = np.concatenate([feat_tglf, global_params])
        else:
            feat_full = feat_tglf  # TGLF-only, 378 dims

        features_list.append(feat_full)
        loaded_folders.append(folder_name)

    if skipped:
        print(f"WARNING: Skipped {len(skipped)} folders (not found): {skipped[:5]}...")

    features = np.array(features_list, dtype=np.float64)
    targets = np.array(targets_list, dtype=np.float64)

    # Build metadata for loaded runs only
    metadata = run_list[run_list["folder_name"].isin(loaded_folders)].reset_index(drop=True)
    metadata["Q_electron"] = targets[:, 0]
    metadata["Q_ion"] = targets[:, 1]

    print(f"Loaded {len(features)} runs: features {features.shape}, targets {targets.shape}")
    return features, targets, metadata
