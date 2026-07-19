"""
Cross-matches top anomaly candidates against SIMBAD to see whether each
source is already a known/classified object (e.g. a catalogued white dwarf,
binary, or variable star) or appears to be unclassified.

This matters a lot for the LLM stage: "unusual AND unclassified" is a much
more interesting candidate than "unusual but already a known eclipsing
binary in SIMBAD".
"""

import logging

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad

import config

logger = logging.getLogger(__name__)

custom_simbad = Simbad()
custom_simbad.add_votable_fields("otype", "ids")


def crossmatch_source(ra_deg: float, dec_deg: float) -> dict:
    """
    Queries SIMBAD for the nearest known object within
    config.CROSSMATCH_RADIUS_ARCSEC of the given coordinates.

    Returns a dict with keys: simbad_match (bool), main_id, otype (object
    type classification), separation_arcsec.
    """
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    try:
        result = custom_simbad.query_region(
            coord, radius=config.CROSSMATCH_RADIUS_ARCSEC * u.arcsec
        )
    except Exception as exc:
        logger.warning("SIMBAD query failed for RA=%s DEC=%s: %s", ra_deg, dec_deg, exc)
        return {"simbad_match": False, "main_id": None, "otype": None, "separation_arcsec": None}

    if result is None or len(result) == 0:
        return {"simbad_match": False, "main_id": None, "otype": None, "separation_arcsec": None}

    # Take the closest match (first row; SIMBAD sorts by separation by default
    # for query_region in recent astroquery versions).
    row = result[0]
    return {
        "simbad_match": True,
        "main_id": str(row["main_id"]) if "main_id" in row.colnames else str(row["MAIN_ID"]),
        "otype": str(row["otype"]) if "otype" in row.colnames else None,
        "separation_arcsec": None,  # populate if needed via coord math on returned RA/DEC
    }


def crossmatch_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs crossmatch_source for the top N anomaly candidates (by anomaly_rank)
    and appends the results as new columns. Deliberately limited to top N -
    SIMBAD's public interface is not meant for bulk queries of thousands of
    rows, and we only need this for the shortlist anyway.
    """
    df = df.copy()
    top = df[df["anomaly_rank"] <= config.TOP_N_CANDIDATES].copy()

    matches = []
    for _, row in top.iterrows():
        logger.info("Cross-matching source_id=%s (rank %d)...", row["source_id"], row["anomaly_rank"])
        matches.append(crossmatch_source(row["ra"], row["dec"]))

    match_df = pd.DataFrame(matches, index=top.index)
    top = pd.concat([top, match_df], axis=1)

    return top


if __name__ == "__main__":
    import gaia_query
    import features

    logging.basicConfig(level=logging.INFO)
    raw = gaia_query.fetch_gaia_sample()
    enriched = features.add_derived_features(raw)
    scored = features.score_anomalies(enriched)
    matched = crossmatch_candidates(scored)
    print(matched[["source_id", "anomaly_rank", "simbad_match", "main_id", "otype"]])
