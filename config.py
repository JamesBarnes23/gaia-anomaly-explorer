"""
Central configuration for the Gaia Anomaly Explorer.

Edit these values to change the search region, sample size, and
anomaly-detection thresholds without touching the pipeline code.
"""

# --- Sky region to search --------------------------------------------------
# Default: a patch near the Galactic anticenter with modest crowding,
# good for a first test run. Change RA/DEC/RADIUS to explore elsewhere.
SEARCH_RA_DEG = 180.0        # Right ascension, degrees
SEARCH_DEC_DEG = 43.0       # Declination, degrees
SEARCH_RADIUS_DEG = 0.5     # Cone search radius, degrees

# --- Query limits ------------------------------------------------------------
MAX_SOURCES = 20000          # Cap on rows pulled from Gaia (keeps queries fast)
MIN_PARALLAX_MAS = 0.0       # Set > 0 to exclude negative/zero parallax outright
REQUIRE_GOOD_ASTROMETRY = True  # Filter on ruwe < RUWE_MAX if True
RUWE_MAX = 1.4               # Renormalised Unit Weight Error threshold (Gaia data-quality cut)

# Minimum parallax significance (parallax / parallax_error) required to trust
# any distance-derived feature (abs_g_mag, v_tan_km_s). Standard Gaia practice
# uses 5-sigma as a "the parallax is real" cutoff. Without this, stars with
# noisy near-zero parallax produce nonsense distances (sometimes MPc-scale)
# that look like extreme anomalies but are pure numerical artifacts.
MIN_PARALLAX_SIGNIFICANCE = 5.0

# --- Anomaly scoring ----------------------------------------------------------
ANOMALY_METHOD = "isolation_forest"   # "isolation_forest" or "zscore"
ISOLATION_FOREST_CONTAMINATION = 0.02  # Expected fraction of outliers
TOP_N_CANDIDATES = 5                  # How many top anomalies to carry to cross-match + LLM stage

# --- SIMBAD cross-match --------------------------------------------------------
CROSSMATCH_RADIUS_ARCSEC = 5.0

# --- Local LLM (Ollama) --------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"     # Any model you've pulled via `ollama pull <model>`
LLM_TIMEOUT_SECONDS = 120

# --- File paths -----------------------------------------------------------------
CACHE_DIR = "cache"
OUTPUT_DIR = "output"
RAW_GAIA_CACHE = f"{CACHE_DIR}/gaia_raw.csv"
SCORED_CACHE = f"{CACHE_DIR}/gaia_scored.csv"
CROSSMATCHED_CACHE = f"{CACHE_DIR}/gaia_crossmatched.csv"
REPORTS_OUTPUT = f"{OUTPUT_DIR}/anomaly_reports.md"