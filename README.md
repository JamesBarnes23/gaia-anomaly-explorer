# Gaia Anomaly Explorer

Pulls a sample of stars from **ESA Gaia DR3** (the public astrometric/photometric
catalogue of ~1.8 billion stars), scores them for statistical unusualness, cross-checks
which ones are already known/classified in **SIMBAD**, and uses a **locally hosted LLM**
(via [Ollama](https://ollama.com)) to turn the surviving candidates into readable
vetting reports.

## Pipeline

```
Gaia TAP archive (ADQL query)
        │
        ▼
gaia_query.py        → raw astrometry + photometry for a sky region
        │
        ▼
features.py           → derived features (color, absolute magnitude,
                         tangential velocity) + anomaly scoring
                         (Isolation Forest, purely statistical)
        │
        ▼
crossmatch.py          → SIMBAD lookup: is this already a known/classified object?
        │
        ▼
llm_report.py          → local LLM (Ollama) writes a plain-language vetting
                         report per candidate, grounded only in the numbers
                         computed above
        │
        ▼
output/anomaly_reports.md
```

The LLM never computes anything — it only explains numbers that were already
derived deterministically. This keeps the "creative" part of the pipeline
contained to interpretation, not arithmetic, which matters a lot for a
science-adjacent tool.

## Setup

```bash
pip install -r requirements.txt

# Install and start Ollama (https://ollama.com), then pull a model:
ollama pull llama3.1
ollama serve   # if not already running as a background service
```

## Usage

```bash
# Full run: query Gaia, score, cross-match, generate LLM reports
python pipeline.py

# Re-run from scratch (ignore cached CSVs in cache/)
python pipeline.py --no-cache

# Stop before the LLM step (useful while Ollama isn't set up yet,
# or if you just want the ranked candidate table)
python pipeline.py --skip-llm
```

Output:
- `cache/gaia_raw.csv` — raw Gaia query result (cached so you don't re-hit the archive every run)
- `cache/gaia_scored.csv` — full sample with derived features + anomaly scores
- `cache/gaia_crossmatched.csv` — top candidates with SIMBAD match info
- `output/anomaly_reports.md` — final human-readable report, one section per candidate

## Configuration

All tunable parameters live in `config.py`:
- **Search region**: `SEARCH_RA_DEG` / `SEARCH_DEC_DEG` / `SEARCH_RADIUS_DEG` — pick any
  patch of sky. Smaller/sparser regions run faster; the default is a modest test patch.
- **Sample size**: `MAX_SOURCES`
- **Data-quality cuts**: `REQUIRE_GOOD_ASTROMETRY` / `RUWE_MAX` — RUWE is Gaia's own
  "is this astrometric solution trustworthy" flag; filtering on it removes a lot of
  spurious-looking anomalies caused by bad fits rather than real physics.
- **Anomaly method**: `ANOMALY_METHOD` (`"isolation_forest"` or `"zscore"`), `TOP_N_CANDIDATES`
- **LLM**: `OLLAMA_MODEL`, `OLLAMA_HOST`

## What counts as "anomalous" here

Each source gets scored on four features together (via Isolation Forest, an
unsupervised outlier-detection algorithm that doesn't need labeled examples
of "normal" vs "weird" — it just finds points that are hard to isolate from
the bulk of the distribution):

1. **`bp_rp`** — color index (temperature proxy)
2. **`abs_g_mag`** — absolute magnitude (luminosity, computed from parallax distance)
3. **`v_tan_km_s`** — tangential velocity (fast movers can indicate halo stars,
   runaway stars, or nearby unresolved binaries)
4. **`parallax_rel_error`** — relative parallax precision (included deliberately:
   a star with garbage astrometry will *look* anomalous on a color-magnitude
   diagram for boring reasons, and the report explicitly flags this so you
   don't chase noise)

## Realistic expectations

Gaia DR3 has already been mined heavily by professional astronomers, so:

- You are very unlikely to find a genuinely new class of object.
- You *can* plausibly turn up individual stars that are unusual and
  **not yet in SIMBAD** — Gaia's footprint is so large that individual
  interesting objects do fall through the cracks, especially faint ones
  or ones outside well-studied fields. Projects like *Backyard Worlds* have
  had real (if rare) citizen discoveries this way.
- The much more likely — and still genuinely useful — outcome is a working
  **triage tool**: something that quickly separates "boring/noisy" from
  "worth a human second look," which is exactly the kind of grunt work
  professional astronomers are happy to offload.

If a candidate looks genuinely interesting and unclassified, the next step
outside this tool would be to check it against additional catalogs (2MASS,
WISE for infrared colors; ZTF/ASAS-SN for variability) before considering
any kind of report to a service like the AAVSO or a professional contact —
this pipeline is a triage/vetting aid, not a confirmation.

## GUI

A Streamlit interface (`app.py`) sits on top of the same pipeline modules -
no duplicate logic, it just calls `gaia_query`, `features`, `crossmatch`, and
`llm_report` directly.

```bash
streamlit run app.py
```

What it gives you:
- **Sidebar** — edit search region (RA/Dec/radius) and trigger the fetch+score
  and cross-match steps without touching the command line
- **Candidate table** — sortable/filterable view of all top candidates
- **Candidate detail** — full feature breakdown, SIMBAD status, an embedded
  Aladin Lite finder chart, an on-demand "Generate report" button that calls
  your local Ollama model just for the selected candidate, and a notes box
  that saves your own follow-up findings to `output/candidate_notes.json`
- **Color-magnitude diagram** — the full sample plotted with top candidates
  highlighted, so you can see at a glance how a candidate's color/luminosity
  compares to the bulk of "normal" stars in your search region

The GUI reads from the same `cache/gaia_scored.csv` and
`cache/gaia_crossmatched.csv` files the CLI pipeline produces, so you can
freely mix `python pipeline.py` runs and GUI-triggered runs - whichever ran
most recently is what the GUI displays.

## Notes on scaling up

- SIMBAD's public interface isn't meant for bulk querying — that's why
  cross-matching is deliberately limited to `TOP_N_CANDIDATES`, not the full sample.
- For serious large-area searches, consider batching the Gaia ADQL query
  by HEALPix pixel or magnitude range, and running the anomaly scoring
  incrementally rather than loading everything into memory at once.
- The feature set here is intentionally small and interpretable. You could
  extend `features.py` with more Gaia columns (e.g. `phot_variable_flag`,
  astrometric excess noise) for richer scoring — just make sure whatever
  you add still has an intuitive physical meaning, since the LLM report
  stage explains features by name.