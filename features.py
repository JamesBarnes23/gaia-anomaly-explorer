"""
Derives astrophysically meaningful features from raw Gaia columns, then
scores each source for how "anomalous" it is relative to the rest of the
sample.

This stays purely statistical/deterministic - the LLM never sees raw numbers
without this stage having already decided what counts as unusual.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)

# Constant: parsec conversion for parallax (mas) -> distance (pc)
MAS_TO_DISTANCE_PC = 1000.0

# Constant for converting proper motion + parallax into tangential velocity
# v_tan [km/s] = 4.74 * pm [arcsec/yr] * distance [pc]
PM_TO_VTAN_CONSTANT = 4.74


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - distance_pc: naive distance from parallax (only valid for parallax > 0)
      - abs_g_mag: absolute G magnitude (position on color-magnitude diagram)
      - pm_total_mas_yr: total proper motion magnitude
      - v_tan_km_s: tangential velocity (a classic "unusual star" signal -
        very high v_tan suggests a halo star, runaway star, or unresolved binary)
    """
    df = df.copy()

    # Distance only meaningful for positive parallax; others get NaN and are
    # dropped before scoring rather than silently mishandled.
    df["distance_pc"] = np.where(
        df["parallax"] > 0, MAS_TO_DISTANCE_PC / df["parallax"], np.nan
    )

    # Absolute magnitude: m - 5*log10(d) + 5
    df["abs_g_mag"] = df["phot_g_mean_mag"] - 5 * np.log10(df["distance_pc"]) + 5

    df["pm_total_mas_yr"] = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)

    # Convert proper motion from mas/yr to arcsec/yr for the v_tan formula
    df["v_tan_km_s"] = (
        PM_TO_VTAN_CONSTANT * (df["pm_total_mas_yr"] / 1000.0) * df["distance_pc"]
    )

    return df


def score_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores each source using the configured method. Higher anomaly_score
    means more unusual. Adds an `anomaly_score` and `anomaly_rank` column.

    Features used (deliberately simple + interpretable, so the LLM stage can
    reason about *why* something scored high):
      - bp_rp (color)
      - abs_g_mag (luminosity)
      - v_tan_km_s (kinematics)
      - parallax_error / parallax (relative parallax precision, flags noisy
        astrometry that could produce a spurious 'anomaly')
    """
    df = df.copy()
    df["parallax_rel_error"] = df["parallax_error"] / df["parallax"].abs()
    df["parallax_significance"] = df["parallax"].abs() / df["parallax_error"]

    feature_cols = ["bp_rp", "abs_g_mag", "v_tan_km_s", "parallax_rel_error"]

    # Drop rows missing any feature - can't score what we can't compute.
    clean = df.dropna(subset=feature_cols).copy()
    dropped = len(df) - len(clean)
    if dropped:
        logger.info("Dropping %d rows with missing derived features", dropped)

    # Critical filter: reject stars where the parallax isn't statistically
    # distinguishable from zero. Without this, abs_g_mag and v_tan_km_s become
    # nonsense (sometimes megaparsec-scale "distances") for stars that are
    # simply too faint/distant for Gaia to measure a real parallax - these
    # numerical artifacts otherwise dominate the anomaly ranking.
    n_before = len(clean)
    clean = clean[clean["parallax_significance"] >= config.MIN_PARALLAX_SIGNIFICANCE].copy()
    n_rejected = n_before - len(clean)
    if n_rejected:
        logger.info(
            "Rejecting %d/%d rows with parallax significance < %.1f sigma "
            "(distance-derived features would be unreliable)",
            n_rejected, n_before, config.MIN_PARALLAX_SIGNIFICANCE,
        )

    X = clean[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    if config.ANOMALY_METHOD == "isolation_forest":
        model = IsolationForest(
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            random_state=42,
        )
        model.fit(X_scaled)
        # decision_function: lower (more negative) = more anomalous.
        # We flip sign so higher anomaly_score = more unusual, which is
        # more intuitive when sorting/reporting.
        clean["anomaly_score"] = -model.decision_function(X_scaled)
    elif config.ANOMALY_METHOD == "zscore":
        z = np.abs(X_scaled)
        clean["anomaly_score"] = z.max(axis=1)
    else:
        raise ValueError(f"Unknown ANOMALY_METHOD: {config.ANOMALY_METHOD}")

    clean = clean.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    clean["anomaly_rank"] = clean.index + 1

    return clean


if __name__ == "__main__":
    import gaia_query

    logging.basicConfig(level=logging.INFO)
    raw = gaia_query.fetch_gaia_sample()
    enriched = add_derived_features(raw)
    scored = score_anomalies(enriched)
    print(scored[["source_id", "anomaly_score", "bp_rp", "abs_g_mag", "v_tan_km_s"]].head(10))