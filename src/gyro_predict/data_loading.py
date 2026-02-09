"""Data loading: HDF5 (CGYRO targets) and NPZ (TGLF conditioning inputs).

Species conventions:
  CGYRO: species 0 = IONS, species 1 = ELECTRONS
  TGLF:  species 0 / spec_1 = ELECTRONS, species 1 / spec_2 = IONS

Target ordering: [Q_electron, Q_ion]
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import build_feature_vector


def load_run_list(csv_path: str) -> pd.DataFrame:
    """Load validated run list from CSV.

    Returns:
        DataFrame with at least 'folder_name' and 'experiment' columns.
    """
    df = pd.read_csv(csv_path)
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
    data_dir: str, run_list: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load features and targets for all 105 validated runs.

    Args:
        data_dir: Path to root data directory containing run folders.
        run_list: DataFrame with 'folder_name' column.

    Returns:
        features: np.ndarray of shape (N, 378)
        targets: np.ndarray of shape (N, 2) = [Q_electron, Q_ion]
        metadata: DataFrame with folder_name, experiment, and targets
    """
    features_list = []
    targets_list = []
    loaded_folders = []
    skipped = []

    for _, row in tqdm(run_list.iterrows(), total=len(run_list), desc="Loading data"):
        folder_name = row["folder_name"]
        folder_path = os.path.join(data_dir, folder_name)

        if not os.path.isdir(folder_path):
            skipped.append(folder_name)
            continue

        # Find the .h5 file (named after the folder)
        h5_path = os.path.join(folder_path, f"{folder_name}.h5")
        npz_path = os.path.join(folder_path, "conditioning_data.npz")

        if not os.path.exists(h5_path) or not os.path.exists(npz_path):
            skipped.append(folder_name)
            continue

        # Load targets from CGYRO
        q_elec, q_ion = load_cgyro_targets(h5_path)
        targets_list.append([q_elec, q_ion])

        # Load features from TGLF
        raw_tglf = load_tglf_conditioning(npz_path)
        feat = build_feature_vector(raw_tglf)
        features_list.append(feat)

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
