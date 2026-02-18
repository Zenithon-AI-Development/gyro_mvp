"""Feature engineering: TGLF conditioning data -> 378-dim input vector.

Species conventions (critical):
  TGLF: species index 0 / spec_1 = ELECTRONS, species index 1 / spec_2 = IONS
  CGYRO: species 0 = IONS, species 1 = ELECTRONS

Feature vector layout (378 total):
  [0:84]   gamma aggregated (max, sum, mean, std) x 21 ky
  [84:168] freq aggregated (max, sum, mean, std) x 21 ky
  [168:252] QL_elec aggregated (max, sum, mean, std) x 21 ky
  [252:336] QL_ion aggregated (max, sum, mean, std) x 21 ky
  [336:357] flux_spectrum electrons (21 ky)
  [357:378] flux_spectrum ions (21 ky)
"""

import numpy as np
from typing import Dict


def _aggregate_over_modes(arr: np.ndarray, mode_axis: int) -> np.ndarray:
    """Aggregate array over mode dimension using max, sum, mean, std.

    Args:
        arr: Array with a mode dimension.
        mode_axis: Which axis is the mode dimension.

    Returns:
        Stacked array of shape (4, ...) where 4 = [max, sum, mean, std].
        The mode_axis is replaced by the aggregation axis.
    """
    return np.stack([
        np.max(arr, axis=mode_axis),
        np.sum(arr, axis=mode_axis),
        np.mean(arr, axis=mode_axis),
        np.std(arr, axis=mode_axis),
    ], axis=0)


def build_feature_vector(raw: Dict[str, np.ndarray]) -> np.ndarray:
    """Build 378-dim feature vector from raw TGLF conditioning data.

    Args:
        raw: Dict with keys 'QL_weights', 'gamma', 'freq', 'flux_spectrum'.
             flux_spectrum is a dict with keys 'spec_1_field_1', etc.

    Returns:
        1D numpy array of shape (378,).
    """
    # --- QL_weights processing ---
    # Shape: (2, 2, 4, 21, 5) = (species, fields, modes, ky, flux_type)
    ql = raw["QL_weights"]
    # Slice energy flux only (index 1 in flux_type dim)
    ql_energy = ql[:, :, :, :, 1]  # (2, 2, 4, 21)
    # Sum over fields dimension (axis=1)
    ql_summed = ql_energy.sum(axis=1)  # (2, 4, 21) = (species, modes, ky)
    # Species 0 = electrons, species 1 = ions (TGLF convention)
    ql_elec = ql_summed[0]  # (4, 21) = (modes, ky)
    ql_ion = ql_summed[1]   # (4, 21) = (modes, ky)

    # --- gamma and freq ---
    # Shape: (21, 4) = (ky, modes)
    gamma = raw["gamma"]  # (21, 4)
    freq = raw["freq"]    # (21, 4)

    # Aggregate over modes
    # gamma: aggregate axis=1 (modes) -> (4_aggs, 21_ky)
    gamma_agg = _aggregate_over_modes(gamma, mode_axis=1)  # (4, 21)
    freq_agg = _aggregate_over_modes(freq, mode_axis=1)    # (4, 21)

    # QL weights: aggregate axis=0 (modes) -> (4_aggs, 21_ky)
    ql_elec_agg = _aggregate_over_modes(ql_elec, mode_axis=0)  # (4, 21)
    ql_ion_agg = _aggregate_over_modes(ql_ion, mode_axis=0)    # (4, 21)

    # --- flux_spectrum processing ---
    # Dict with keys: spec_1_field_1, spec_1_field_2, spec_2_field_1, spec_2_field_2
    # Each value shape: (21, 5) = (ky, flux_type)
    # TGLF: spec_1 = electrons, spec_2 = ions
    fs = raw["flux_spectrum"]
    # Slice energy flux (index 1), sum over fields per species
    fs_elec = fs["spec_1_field_1"][:, 1] + fs["spec_1_field_2"][:, 1]  # (21,)
    fs_ion = fs["spec_2_field_1"][:, 1] + fs["spec_2_field_2"][:, 1]   # (21,)

    # --- Concatenate ---
    feature_vector = np.concatenate([
        gamma_agg.ravel(),     # 84
        freq_agg.ravel(),      # 84
        ql_elec_agg.ravel(),   # 84
        ql_ion_agg.ravel(),    # 84
        fs_elec,               # 21
        fs_ion,                # 21
    ])  # total: 378

    assert feature_vector.shape == (378,), f"Expected 378 features, got {feature_vector.shape[0]}"
    return feature_vector


def build_structured_vector(raw: Dict[str, np.ndarray]) -> np.ndarray:
    """Build (21, 18) structured feature matrix from raw TGLF conditioning data.

    Rows = 21 ky values, columns = 18 channels:
      gamma(4) | freq(4) | ql_elec(4) | ql_ion(4) | fs_elec(1) | fs_ion(1)

    Args:
        raw: Dict with keys 'QL_weights', 'gamma', 'freq', 'flux_spectrum'.
             flux_spectrum is a dict with keys 'spec_1_field_1', etc.

    Returns:
        2D numpy array of shape (21, 18).
    """
    # --- QL_weights processing ---
    # Shape: (2, 2, 4, 21, 5) = (species, fields, modes, ky, flux_type)
    ql = raw["QL_weights"]
    # Slice energy flux only (index 1 in flux_type dim)
    ql_energy = ql[:, :, :, :, 1]  # (2, 2, 4, 21)
    # Sum over fields dimension (axis=1)
    ql_summed = ql_energy.sum(axis=1)  # (2, 4, 21) = (species, modes, ky)
    # Species 0 = electrons, species 1 = ions (TGLF convention)
    ql_elec = ql_summed[0]  # (4, 21) = (modes, ky)
    ql_ion = ql_summed[1]   # (4, 21) = (modes, ky)

    # --- gamma and freq ---
    # Shape: (21, 4) = (ky, modes)
    gamma = raw["gamma"]  # (21, 4)
    freq = raw["freq"]    # (21, 4)

    # --- flux_spectrum processing ---
    # Dict with keys: spec_1_field_1, spec_1_field_2, spec_2_field_1, spec_2_field_2
    # Each value shape: (21, 5) = (ky, flux_type)
    # TGLF: spec_1 = electrons, spec_2 = ions
    fs = raw["flux_spectrum"]
    # Slice energy flux (index 1), sum over fields per species
    fs_elec = fs["spec_1_field_1"][:, 1] + fs["spec_1_field_2"][:, 1]  # (21,)
    fs_ion = fs["spec_2_field_1"][:, 1] + fs["spec_2_field_2"][:, 1]   # (21,)

    # --- Build structured (21, 18) matrix ---
    # Columns: gamma(4) | freq(4) | ql_elec(4) | ql_ion(4) | fs_elec(1) | fs_ion(1)
    structured_vector = np.column_stack([
        gamma,              # (21, 4)
        freq,               # (21, 4)
        ql_elec.T,          # (4, 21).T -> (21, 4)
        ql_ion.T,           # (4, 21).T -> (21, 4)
        fs_elec[:, None],   # (21,) -> (21, 1)
        fs_ion[:, None],    # (21,) -> (21, 1)
    ])

    assert structured_vector.shape == (21, 18), f"Expected 21x18 dim, got {structured_vector.shape}"
    return structured_vector