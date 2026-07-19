"""
Gaia Anomaly Explorer - main pipeline.

Runs the full flow:
  1. Query Gaia DR3 for a sky region (gaia_query.py)
  2. Derive astrophysical features + score anomalies (features.py)
  3. Cross-match top candidates against SIMBAD (crossmatch.py)
  4. Generate LLM vetting reports via local Ollama (llm_report.py)
  5. Write a Markdown report for human review

Usage:
    python pipeline.py                 # run full pipeline, using caches if present
    python pipeline.py --no-cache      # force re-fetch/re-score from scratch
    python pipeline.py --skip-llm      # stop after cross-match, skip LLM step
                                        # (useful if Ollama isn't running yet)
"""

import argparse
import logging

import config
import gaia_query
import features
import crossmatch
import llm_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def run(use_cache: bool = True, skip_llm: bool = False):
    logger.info("=== Step 1/4: Fetching Gaia sample ===")
    raw = gaia_query.fetch_gaia_sample(use_cache=use_cache)
    logger.info("Sample size: %d sources", len(raw))

    logger.info("=== Step 2/4: Feature engineering + anomaly scoring ===")
    enriched = features.add_derived_features(raw)
    scored = features.score_anomalies(enriched)
    scored.to_csv(config.SCORED_CACHE, index=False)
    logger.info(
        "Scored %d sources; top anomaly_score = %.3f",
        len(scored), scored["anomaly_score"].max(),
    )

    logger.info("=== Step 3/4: Cross-matching top %d candidates against SIMBAD ===", config.TOP_N_CANDIDATES)
    matched = crossmatch.crossmatch_candidates(scored)
    matched.to_csv(config.CROSSMATCHED_CACHE, index=False)
    n_known = matched["simbad_match"].sum()
    logger.info(
        "%d/%d top candidates already have a SIMBAD classification",
        n_known, len(matched),
    )

    if skip_llm:
        logger.info("Skipping LLM report generation (--skip-llm set).")
        print(matched[["source_id", "anomaly_rank", "anomaly_score", "simbad_match", "otype"]])
        return matched

    logger.info("=== Step 4/4: Generating LLM vetting reports (local Ollama, model=%s) ===", config.OLLAMA_MODEL)
    reported = llm_report.generate_all_reports(matched)
    out_path = llm_report.write_markdown_report(reported)

    logger.info("Done. Full report written to: %s", out_path)
    return reported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaia Anomaly Explorer pipeline")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch/re-score, ignoring cached CSVs")
    parser.add_argument("--skip-llm", action="store_true", help="Stop after cross-match, skip LLM report generation")
    args = parser.parse_args()

    run(use_cache=not args.no_cache, skip_llm=args.skip_llm)
