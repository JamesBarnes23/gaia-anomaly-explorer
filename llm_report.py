"""
Turns the statistical output (features + anomaly score + SIMBAD match info)
for each top candidate into a readable vetting report, using a locally
hosted LLM via Ollama.

The LLM is never asked to compute anything - only to reason over and
explain numbers that were already calculated deterministically upstream.
This keeps hallucination risk contained to "interpretation" rather than
"arithmetic."
"""

import json
import logging

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an assistant helping an amateur astronomer triage candidate stars \
flagged as statistical outliers in ESA Gaia DR3 data. You will be given a \
JSON object describing one star's measured and derived properties, plus \
whether it has a known classification in SIMBAD.

Write a short vetting report (150-250 words) that:
1. States in plain language what makes this star statistically unusual, \
   using the specific numbers provided (color, luminosity, tangential \
   velocity, parallax precision).
2. Proposes 1-3 plausible astrophysical explanations (e.g. white dwarf, \
   subdwarf, high-velocity/runaway star, unresolved binary, or a spurious \
   result from noisy astrometry) with brief reasoning for each.
3. Notes explicitly if the parallax has poor relative precision, since that \
   alone can produce a fake-looking anomaly.
4. States whether it is already a known object per the SIMBAD match, and if \
   so, tempers the "discovery" framing accordingly - already-classified \
   objects are not novel.
5. Ends with a one-line recommendation: worth closer follow-up, or likely \
   noise/already explained.

Do not invent facts not present in the JSON. Be precise, measured, and \
avoid dramatic language - this is a technical triage report, not a press \
release.
"""


def _build_user_prompt(candidate: dict) -> str:
    return f"Candidate data:\n{json.dumps(candidate, indent=2, default=str)}"


def _candidate_to_dict(row: pd.Series) -> dict:
    """Selects and labels the fields relevant to the LLM prompt."""
    return {
        "gaia_source_id": row.get("source_id"),
        "ra_deg": round(row.get("ra"), 5) if pd.notna(row.get("ra")) else None,
        "dec_deg": round(row.get("dec"), 5) if pd.notna(row.get("dec")) else None,
        "bp_rp_color": round(row.get("bp_rp"), 3) if pd.notna(row.get("bp_rp")) else None,
        "absolute_g_magnitude": round(row.get("abs_g_mag"), 2) if pd.notna(row.get("abs_g_mag")) else None,
        "tangential_velocity_km_s": round(row.get("v_tan_km_s"), 1) if pd.notna(row.get("v_tan_km_s")) else None,
        "distance_pc": round(row.get("distance_pc"), 1) if pd.notna(row.get("distance_pc")) else None,
        "parallax_mas": round(row.get("parallax"), 3) if pd.notna(row.get("parallax")) else None,
        "parallax_relative_error": round(row.get("parallax_rel_error"), 3) if pd.notna(row.get("parallax_rel_error")) else None,
        "ruwe": round(row.get("ruwe"), 2) if pd.notna(row.get("ruwe")) else None,
        "anomaly_score": round(row.get("anomaly_score"), 3) if pd.notna(row.get("anomaly_score")) else None,
        "anomaly_rank": int(row.get("anomaly_rank")) if pd.notna(row.get("anomaly_rank")) else None,
        "known_in_simbad": bool(row.get("simbad_match", False)),
        "simbad_main_id": row.get("main_id"),
        "simbad_object_type": row.get("otype"),
    }


def generate_report(candidate: dict) -> str:
    """
    Calls the local Ollama API to generate one vetting report.
    Raises requests.RequestException on connection failure (e.g. Ollama not running).
    """
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(candidate)},
        ],
        "stream": False,
    }

    response = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


def generate_all_reports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a report for every row in df (expected to be the cross-matched
    top-N candidates) and returns df with a new `llm_report` column.
    """
    df = df.copy()
    reports = []

    for _, row in df.iterrows():
        candidate = _candidate_to_dict(row)
        try:
            report = generate_report(candidate)
        except requests.RequestException as exc:
            logger.error(
                "Could not reach local Ollama server at %s (%s). "
                "Is Ollama running? Try: `ollama serve` and `ollama pull %s`",
                config.OLLAMA_HOST, exc, config.OLLAMA_MODEL,
            )
            report = "[LLM report unavailable - could not reach local Ollama server]"
        reports.append(report)

    df["llm_report"] = reports
    return df


def write_markdown_report(df: pd.DataFrame, path: str = None) -> str:
    """
    Writes all candidate reports to a single Markdown file, sorted by
    anomaly rank, for easy human review.
    """
    path = path or config.REPORTS_OUTPUT
    lines = ["# Gaia Anomaly Explorer - Candidate Vetting Reports\n"]

    for _, row in df.sort_values("anomaly_rank").iterrows():
        lines.append(f"## Candidate #{int(row['anomaly_rank'])} - Gaia DR3 {row['source_id']}\n")
        lines.append(f"- **Coordinates (ICRS):** RA {row['ra']:.5f}, Dec {row['dec']:.5f}")
        lines.append(f"- **Anomaly score:** {row['anomaly_score']:.3f}")
        known = "Yes" if row.get("simbad_match") else "No"
        lines.append(f"- **Known in SIMBAD:** {known}" + (f" (as {row.get('main_id')}, type: {row.get('otype')})" if row.get("simbad_match") else ""))
        lines.append("")
        lines.append(row["llm_report"])
        lines.append("\n---\n")

    content = "\n".join(lines)

    import os
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

    logger.info("Wrote %d reports to %s", len(df), path)
    return path
