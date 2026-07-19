"""
Pulls a sample of Gaia DR3 sources for a sky region via the ESA Gaia
archive's TAP service (ADQL query), using astroquery.

This is the deterministic data-acquisition stage. No LLM involved here -
we want reliable, reproducible numbers before any narrative generation.
"""

import os
import logging

import pandas as pd
from astroquery.gaia import Gaia

import config

logger = logging.getLogger(__name__)


def build_adql_query() -> str:
    """
    Construct the ADQL query string for a cone search against gaiadr3.gaia_source,
    pulling the columns we need for anomaly detection:
      - astrometry: parallax, proper motion, ruwe (data-quality flag)
      - photometry: G, BP, RP magnitudes (for color-magnitude diagram features)
      - identifiers: source_id, ra, dec (for cross-matching later)
    """
    ruwe_filter = f"AND ruwe < {config.RUWE_MAX}" if config.REQUIRE_GOOD_ASTROMETRY else ""
    parallax_filter = (
        f"AND parallax > {config.MIN_PARALLAX_MAS}"
        if config.MIN_PARALLAX_MAS > 0
        else ""
    )

    query = f"""
    SELECT TOP {config.MAX_SOURCES}
        source_id, ra, dec,
        parallax, parallax_error,
        pmra, pmdec,
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        bp_rp, ruwe,
        radial_velocity, radial_velocity_error
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {config.SEARCH_RA_DEG}, {config.SEARCH_DEC_DEG}, {config.SEARCH_RADIUS_DEG})
    )
    AND parallax IS NOT NULL
    AND phot_g_mean_mag IS NOT NULL
    AND phot_bp_mean_mag IS NOT NULL
    AND phot_rp_mean_mag IS NOT NULL
    {parallax_filter}
    {ruwe_filter}
    """
    return query.strip()


def fetch_gaia_sample(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch Gaia sources for the configured region, or load from cache if present.
    """
    if use_cache and os.path.exists(config.RAW_GAIA_CACHE):
        logger.info("Loading cached Gaia sample from %s", config.RAW_GAIA_CACHE)
        return pd.read_csv(config.RAW_GAIA_CACHE)

    query = build_adql_query()
    logger.info("Submitting ADQL query to Gaia archive...\n%s", query)

    job = Gaia.launch_job_async(query)
    table = job.get_results()
    df = table.to_pandas()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    df.to_csv(config.RAW_GAIA_CACHE, index=False)
    logger.info("Fetched %d sources; cached to %s", len(df), config.RAW_GAIA_CACHE)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_gaia_sample(use_cache=False)
    print(df.head())
    print(f"\nTotal sources fetched: {len(df)}")
