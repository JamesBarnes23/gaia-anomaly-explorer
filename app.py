"""
Streamlit GUI for the Gaia Anomaly Explorer.

Run with:
    streamlit run app.py

Lets you:
  - Adjust search region / re-run the pipeline from the sidebar
  - Browse ranked anomaly candidates in a sortable table
  - Drill into one candidate: full feature breakdown, SIMBAD status,
    an Aladin Lite finder chart, and on-demand LLM report generation
  - Log your own follow-up notes per candidate (saved to disk)
  - View a color-magnitude diagram with the selected candidate highlighted
    against the full sample
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

import config
import crossmatch
import features
import gaia_query
import llm_report

st.set_page_config(page_title="Gaia Anomaly Explorer", layout="wide")

NOTES_PATH = os.path.join(config.OUTPUT_DIR, "candidate_notes.json")


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------

def load_notes() -> dict:
    if os.path.exists(NOTES_PATH):
        with open(NOTES_PATH) as f:
            return json.load(f)
    return {}


def save_note(source_id: int, text: str):
    notes = load_notes()
    notes[str(source_id)] = text
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(NOTES_PATH, "w") as f:
        json.dump(notes, f, indent=2)


def load_scored() -> pd.DataFrame | None:
    if os.path.exists(config.SCORED_CACHE):
        return pd.read_csv(config.SCORED_CACHE)
    return None


def load_crossmatched() -> pd.DataFrame | None:
    if os.path.exists(config.CROSSMATCHED_CACHE):
        return pd.read_csv(config.CROSSMATCHED_CACHE)
    return None


def aladin_iframe_html(ra: float, dec: float, fov: float = 0.02) -> str:
    """Embeds Aladin Lite centered on the given coordinates."""
    return f"""
    <div id="aladin-lite-div" style="width:100%;height:400px;"></div>
    <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
    <script>
        let aladin;
        A.init.then(() => {{
            aladin = A.aladin('#aladin-lite-div', {{
                target: '{ra} {dec}',
                fov: {fov},
                survey: 'P/DSS2/color'
            }});
        }});
    </script>
    """


# --------------------------------------------------------------------------
# Sidebar: pipeline controls
# --------------------------------------------------------------------------

st.sidebar.title("Pipeline controls")

st.sidebar.subheader("Search region")
ra = st.sidebar.slider(
    "RA (deg)", min_value=0.0, max_value=359.9, value=float(config.SEARCH_RA_DEG), step=0.1,
    help="Right ascension - the sky's full 0-360 degree coordinate.",
)
dec = st.sidebar.slider(
    "Dec (deg)", min_value=-90.0, max_value=90.0, value=float(config.SEARCH_DEC_DEG), step=0.1,
    help="Declination - measured from the celestial equator, so it only ranges +/-90 degrees.",
)
radius = st.sidebar.slider(
    "Radius (deg)", min_value=0.01, max_value=10.0, value=float(config.SEARCH_RADIUS_DEG), step=0.01,
    help="Cone search radius. Capped at 10 degrees to keep queries a reasonable size - "
         "MAX_SOURCES in config.py limits how many rows come back regardless.",
)

st.sidebar.subheader("Actions")
run_query = st.sidebar.button("1. Fetch Gaia sample + score anomalies", use_container_width=True)
run_crossmatch = st.sidebar.button("2. Cross-match top candidates (SIMBAD)", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Steps run in order. Step 1 must complete (and produce "
    f"`{config.SCORED_CACHE}`) before step 2 will find anything to match."
)

if run_query:
    config.SEARCH_RA_DEG = ra
    config.SEARCH_DEC_DEG = dec
    config.SEARCH_RADIUS_DEG = radius
    try:
        gaia_query.validate_search_region(ra, dec, radius)
    except ValueError as exc:
        st.sidebar.error(str(exc))
    else:
        with st.spinner("Querying Gaia archive and scoring anomalies..."):
            raw = gaia_query.fetch_gaia_sample(use_cache=False)
            enriched = features.add_derived_features(raw)
            scored = features.score_anomalies(enriched)
            scored.to_csv(config.SCORED_CACHE, index=False)
        st.sidebar.success(f"Scored {len(scored)} sources.")

if run_crossmatch:
    scored = load_scored()
    if scored is None:
        st.sidebar.error("Run step 1 first - no scored data found.")
    else:
        with st.spinner(f"Cross-matching top {config.TOP_N_CANDIDATES} candidates against SIMBAD..."):
            matched = crossmatch.crossmatch_candidates(scored)
            matched.to_csv(config.CROSSMATCHED_CACHE, index=False)
        st.sidebar.success(f"Cross-matched {len(matched)} candidates.")


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------

st.title("🔭 Gaia Anomaly Explorer")

matched_df = load_crossmatched()

if matched_df is None:
    st.info(
        "No candidate data found yet. Use the sidebar to run step 1 "
        "(fetch + score), then step 2 (cross-match), or run `python pipeline.py` "
        "from the command line first."
    )
    st.stop()

tab_table, tab_detail, tab_cmd = st.tabs(["Candidate table", "Candidate detail", "Color-magnitude diagram"])

# --- Tab 1: table ---
with tab_table:
    st.subheader(f"Top {len(matched_df)} anomaly candidates")
    display_cols = [
        "anomaly_rank", "source_id", "anomaly_score", "bp_rp", "abs_g_mag",
        "v_tan_km_s", "parallax_rel_error", "ruwe", "simbad_match", "otype",
    ]
    display_cols = [c for c in display_cols if c in matched_df.columns]
    display_df = matched_df[display_cols].sort_values("anomaly_rank").copy()
    # source_id is a 19-digit identifier that can exceed JavaScript's safe
    # integer range (2^53) - render as a string so Streamlit's table widget
    # doesn't risk silently rounding the last couple of digits.
    display_df["source_id"] = display_df["source_id"].astype(str)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

# --- Tab 2: detail view ---
with tab_detail:
    source_ids = matched_df.sort_values("anomaly_rank")["source_id"].tolist()
    labels = [
        f"#{int(row.anomaly_rank)} - {row.source_id} (score {row.anomaly_score:.3f})"
        for _, row in matched_df.sort_values("anomaly_rank").iterrows()
    ]
    selected_label = st.selectbox("Select a candidate", labels)
    selected_source_id = source_ids[labels.index(selected_label)]
    row = matched_df[matched_df["source_id"] == selected_source_id].iloc[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Measured / derived features")
        feature_table = pd.DataFrame({
            "Field": [
                "RA / Dec", "Anomaly score", "bp_rp (color)", "Absolute G mag",
                "Tangential velocity (km/s)", "Distance (pc)", "Parallax (mas)",
                "Parallax relative error", "RUWE",
            ],
            "Value": [
                f"{row['ra']:.5f}, {row['dec']:.5f}",
                f"{row['anomaly_score']:.3f}",
                f"{row['bp_rp']:.3f}",
                f"{row['abs_g_mag']:.2f}",
                f"{row['v_tan_km_s']:.1f}",
                f"{row['distance_pc']:.1f}",
                f"{row['parallax']:.4f}",
                f"{row['parallax_rel_error']:.3f}",
                f"{row['ruwe']:.3f}",
            ],
        })
        st.table(feature_table.set_index("Field"))

        known = row.get("simbad_match", False)
        if known:
            st.success(f"Known in SIMBAD: **{row.get('main_id')}** (type: {row.get('otype')})")
        else:
            st.warning("Not found in SIMBAD - unclassified as of last cross-match.")

    with col2:
        st.subheader("Finder chart (Aladin Lite / DSS2)")
        components.html(aladin_iframe_html(row["ra"], row["dec"]), height=420)

    st.divider()
    st.subheader("LLM vetting report")

    llm_col1, llm_col2 = st.columns([1, 3])
    with llm_col1:
        generate = st.button("Generate report", key=f"gen_{selected_source_id}")
    if generate:
        with st.spinner(f"Querying local Ollama model ({config.OLLAMA_MODEL})..."):
            try:
                candidate = llm_report._candidate_to_dict(row)
                report_text = llm_report.generate_report(candidate)
                st.session_state[f"report_{selected_source_id}"] = report_text
            except Exception as exc:
                st.error(
                    f"Could not reach Ollama at {config.OLLAMA_HOST}. "
                    f"Is it running? (`ollama serve`) Error: {exc}"
                )

    report_key = f"report_{selected_source_id}"
    if report_key in st.session_state:
        st.markdown(st.session_state[report_key])

    st.divider()
    st.subheader("Your notes")
    notes = load_notes()
    existing_note = notes.get(str(selected_source_id), "")
    note_text = st.text_area(
        "Follow-up notes for this candidate (saved to output/candidate_notes.json)",
        value=existing_note,
        height=120,
        key=f"note_{selected_source_id}",
    )
    if st.button("Save note", key=f"save_{selected_source_id}"):
        save_note(selected_source_id, note_text or "")
        st.success("Note saved.")

# --- Tab 3: color-magnitude diagram ---
with tab_cmd:
    st.subheader("Color-magnitude diagram")
    scored_df = load_scored()
    if scored_df is None:
        st.info("Run step 1 to populate the full sample for this plot.")
    else:
        scored_df = scored_df.copy()
        scored_df["is_candidate"] = scored_df["source_id"].isin(matched_df["source_id"])
        scored_df["label"] = scored_df["is_candidate"].map(
            {True: "Top anomaly candidate", False: "Sample star"}
        )
        # Cast for hover display - same JS integer precision issue as the table.
        scored_df["source_id_str"] = scored_df["source_id"].astype(str)
        fig = px.scatter(
            scored_df,
            x="bp_rp",
            y="abs_g_mag",
            color="label",
            color_discrete_map={
                "Sample star": "#888888",
                "Top anomaly candidate": "#e74c3c",
            },
            hover_data=["source_id_str", "anomaly_score"],
            opacity=0.6,
        )
        # Absolute magnitude: brighter (more negative-ish/smaller) stars plot higher,
        # so flip the y-axis to match standard astronomical convention.
        fig.update_yaxes(autorange="reversed", title="Absolute G magnitude")
        fig.update_xaxes(title="BP - RP color")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Standard astronomical convention: brighter stars (lower/more negative "
            "absolute magnitude) appear higher on this plot. Candidates sitting "
            "noticeably above the main bulk of same-color stars are overluminous "
            "for their color - one of the signals this pipeline flags."
        )