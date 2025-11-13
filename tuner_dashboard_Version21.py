"""
tuner_dashboard.py

Vehicle Log Tuner - Updated to include timing readout and suggestions.

Features:
- Robust VCM Scanner / HP Tuners CSV parsing with header-candidate selection UI
- Persist uploaded/local file bytes so reruns/preset clicks don't lose the file
- Presets: HP Tuners (Gen3/Gen4), EFILive, SCT, MegaSquirt
- Column mapping selectboxes list real column names (not numeric channel tokens)
- HP‑Tuners export (multiple header variants), SCT/MegaSquirt generic exports
- MAF export fixed and preview
- VE binning, heatmap, simulator
- Timing readout and suggestions based on AFR/STFT/LTFT
- Multi-file comparison for VE and timing changes
- CSV repository: Save/load previous logs for comparison
- File switching: Clear old data when new file uploaded
- Large file handling: Warn for >50MB, adjust pandas options
- UI: light-grey preset/file uploader buttons for readability
- Future-ready for power/boost enrichment, forced induction, timing adjustments
"""
from io import StringIO, BytesIO
import os
import csv
import json
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import pickle  # For saving CSV repository
import hashlib  # For file hash checking

st.set_page_config(page_title="Vehicle Log Tuner", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# UI Styling (buttons + file uploader Browse button)
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&display=swap');

    .stApp { background: #ffffff; color: #0b2b45; }
    header, footer { visibility: hidden; }
    .hpt-title { font-family: 'Montserrat', 'Helvetica', sans-serif; font-size: 34px; font-weight:700; color: #004A99; margin-bottom: 0px; }
    .hpt-sub { color:#2b2b2b; margin: 6px 0 18px 0; font-size: 13px; }
    .card { background:#ffffff; border:1px solid #e6eef8; border-radius:8px; padding:12px 18px; margin-bottom:12px; }
    .stButton>button { background-color: #f0f0f0 !important; color: #0b2b45 !important; border: 1px solid #cfcfcf !important; box-shadow: none !important; }
    .stButton>button:hover { background-color: #e6e6e6 !important; }
    /* File uploader browse button */
    .stFileUploader button, .stFileUploader [role="button"], input[type="file"]::file-selector-button {
        background-color: #f0f0f0 !important; color: #0b2b45 !important; border:1px solid #cfcfcf !important; padding:0.4rem 0.75rem !important; border-radius:4px !important;
    }
    .stFileUploader button:hover, input[type="file"]::file-selector-button:hover { background-color: #e6e6e6 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hpt-title">📈 Vehicle Log Tuner</div><div class="hpt-sub">VE & MAF correction suggestions — HP‑Tuners / SCT / MegaSquirt export — Multi-file comparison & CSV repository — Timing adjustments</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Settings & Presets")
sep_option = st.sidebar.radio("Delimiter (auto-detect if unknown)", ["auto-detect","comma (,)", "semicolon (;)", "tab (\\t)"], index=0)

st.sidebar.subheader("Steady-state filter")
rpm_thresh = st.sidebar.number_input("RPM delta (abs)", value=150, step=10)
load_thresh = st.sidebar.number_input("Load delta (abs)", value=5, step=1)
thr_thresh = st.sidebar.number_input("Throttle delta (abs)", value=3, step=1)

st.sidebar.subheader("Binning & samples")
rpm_bin = st.sidebar.number_input("RPM bin size", value=250, step=50)
map_bin = st.sidebar.number_input("MAP bin size (same units as log)", value=10, step=1)
maf_bin = st.sidebar.number_input("MAF freq bin size (Hz)", value=50, step=10)
min_samples = st.sidebar.number_input("Min samples per bin", value=5, step=1)

st.sidebar.subheader("MAP units handling")
map_unit_option = st.sidebar.selectbox("Interpret MAP units as:", ["Auto-detect from file", "kPa (no conversion)", "psi (convert to kPa)"], index=0)

st.sidebar.subheader("Timing suggestion settings")
target_afr = st.sidebar.number_input("Target AFR", value=14.7, step=0.1)
timing_adj_factor = st.sidebar.number_input("Timing adj factor (deg per AFR unit error)", value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("HP‑Tuners export options")
hpt_variant = st.sidebar.selectbox("Header variant", ["Variant A (blank TL, RPM headers)", "Variant B (blank TL + units)", "Variant C (Manifold label row then RPM)", "Variant D (two header rows)"])
hpt_delimiter = st.sidebar.selectbox("Delimiter for export", ["comma (,)", "tab (\\t)", "semicolon (;)"], index=0)
hpt_line_end = st.sidebar.selectbox("Line endings", ["CRLF (Windows)", "LF (Unix)"], index=0)
hpt_utf8_bom = st.sidebar.checkbox("Write UTF‑8 BOM (legacy/VCM compatibility)", value=True)
hpt_export_type = st.sidebar.selectbox("Cell values", ["Percent adjustments (recommended)", "Absolute VE values (as-is)"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Other exporters")
other_export = st.sidebar.multiselect("Also export", ["SCT (generic pivot CSV)", "MegaSquirt (generic pivot CSV)", "Timing adjustments (generic CSV)"], default=[])

# -----------------------------
# CSV Repository (save/load previous logs)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("CSV Repository")
if 'csv_repo' not in st.session_state:
    st.session_state['csv_repo'] = {}

repo_names = list(st.session_state['csv_repo'].keys())
if repo_names:
    selected_repo = st.sidebar.selectbox("Load saved log", ["None"] + repo_names, index=0)
    if selected_repo != "None":
        if st.sidebar.button("Load from Repository"):
            st.session_state['uploaded_bytes'] = st.session_state['csv_repo'][selected_repo]
            st.success(f"Loaded {selected_repo} from repository.")
            st.experimental_rerun()
else:
    st.sidebar.info("No saved logs yet.")

save_name = st.sidebar.text_input("Save current log as (name)")
if st.sidebar.button("Save to Repository") and save_name and 'uploaded_bytes' in st.session_state:
    st.session_state['csv_repo'][save_name] = st.session_state['uploaded_bytes']
    st.sidebar.success(f"Saved as {save_name}.")

# -----------------------------
# Utility helpers
# -----------------------------
def detect_sep_from_text(raw_text: str) -> str:
    counts = {";": raw_text.count(";"), "\t": raw_text.count("\t"), ",": raw_text.count(",")}
    return max(counts, key=counts.get)

def is_units_like(tokens):
    if not tokens:
        return False
    short_count = sum(1 for t in tokens if 0 < len(t.strip()) <= 8)
    symbol_count = sum(1 for t in tokens if any(sym in t for sym in ["%", "°", "psi", "kPa", "kpa", "V", "lb", "g/s", "λ", "rpm"]))
    return short_count >= max(3, len(tokens)//6) or symbol_count >= 2

def detect_header_candidates(raw: str, max_lines=30):
    """
    Return list of (line_index, line_text) candidates for header rows.
    Exclude pure numeric channel-id lines if possible.
    """
    lines = raw.splitlines()
    candidates = []
    for i in range(min(len(lines), max_lines)):
        ln = lines[i].rstrip("\n")
        s = ln.strip()
        if not s:
            continue
        # heuristics:
        # - prefer lines with alphabetic characters (column names)
        # - or lines starting with "Offset" or containing "Engine RPM"
        has_alpha = any(ch.isalpha() for ch in s)
        if has_alpha or s.lower().startswith("offset") or "engine rpm" in s.lower():
            candidates.append((i, ln))
    # fallback: include lines with many delimiters (may be header)
    if not candidates:
        for i in range(min(len(lines), max_lines)):
            ln = lines[i].rstrip("\n")
            if ln.count(",") >= 3 or ln.count(";") >= 3 or ln.count("\t") >= 3:
                candidates.append((i, ln))
    # always include first non-empty line as last resort
    if not candidates:
        for i in range(min(len(lines), max_lines)):
            if lines[i].strip():
                candidates.append((i, lines[i]))
                break
    return candidates

def extract_dataframe_from_raw(raw_bytes: bytes, header_line_index: int | None, sep_guess: str | None):
    """
    Build a pandas DataFrame using the chosen header_line_index (if provided),
    otherwise attempt auto-detection using the [Channel Data] trick.
    """
    try:
        raw = raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        raw = raw_bytes.decode("latin1", errors="ignore")

    lines = raw.splitlines()
    # If header_line_index provided, find Channel Data block (if present) and use header + channel data
    if header_line_index is not None:
        header_line = lines[header_line_index]
        # find [Channel Data] after header
        idx_channel_data = None
        for i in range(header_line_index+1, len(lines)):
            if lines[i].strip().lower().startswith("[channel data]"):
                idx_channel_data = i + 1
                break
        # If we found channel data section, use header + data; otherwise use header + remaining lines
        if idx_channel_data is not None:
            data_lines = [ln for ln in lines[idx_channel_data:] if ln.strip() != ""]
        else:
            data_lines = [ln for ln in lines[header_line_index+1:] if ln.strip() != ""]
        csv_text = header_line + "\n" + "\n".join(data_lines)
        delim = sep_guess if sep_guess else detect_sep_from_text(header_line)
        try:
            df = pd.read_csv(StringIO(csv_text), sep=delim, engine="c")  # Use c engine for speed, fallback if fails
        except Exception:
            df = pd.read_csv(StringIO(csv_text), sep=delim, engine="python")
        # attempt to parse units row if directly after header and matches units heuristics
        units_map = {}
        if header_line_index + 1 < len(lines):
            cand = lines[header_line_index + 1].strip()
            tokens = [t.strip() for t in cand.split(delim)]
            if is_units_like(tokens):
                hdr_tokens = [h.strip() for h in header_line.split(delim)]
                for h, u in zip(hdr_tokens, tokens):
                    units_map[h] = u
        return df, units_map

    # If header_line_index not provided - fallback auto-extraction like earlier
    # find Channel Information and Channel Data markers
    idx_channel_info = None
    idx_channel_data = None
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if s.startswith("[channel information]"):
            idx_channel_info = i
        if s.startswith("[channel data]"):
            idx_channel_data = i + 1
            break
    header_idx = None
    if idx_channel_info is not None:
        for j in range(idx_channel_info+1, min(idx_channel_info+12, len(lines))):
            ln = lines[j].strip()
            if not ln: continue
            if ln.lower().startswith("offset,") or "engine rpm" in ln.lower() or (ln.count(",") >= 5 or ln.count(";") >= 5 or ln.count("\t") >= 5):
                header_idx = j
                break
    if header_idx is None:
        # try top lines
        for i in range(min(20, len(lines))):
            ln = lines[i].strip()
            if ln.lower().startswith("offset,") or "engine rpm" in ln.lower() or (ln.count(",") >= 5 and any(ch.isalpha() for ch in ln)):
                header_idx = i
                break
    if header_idx is not None and idx_channel_data is not None:
        header_line = lines[header_idx]
        data_lines = [ln for ln in lines[idx_channel_data:] if ln.strip() != ""]
        csv_text = header_line + "\n" + "\n".join(data_lines)
        delim = sep_guess if sep_guess else detect_sep_from_text(header_line)
        try:
            df = pd.read_csv(StringIO(csv_text), sep=delim, engine="c")
        except Exception:
            df = pd.read_csv(StringIO(csv_text), sep=delim, engine="python")
        units_map = {}
        # units detection attempt
        if header_idx + 1 < idx_channel_data:
            cand = lines[header_idx+1].strip()
            tokens = [t.strip() for t in cand.split(delim)]
            if is_units_like(tokens):
                hdr_tokens = [h.strip() for h in header_line.split(delim)]
                for h, u in zip(hdr_tokens, tokens):
                    units_map[h] = u
        return df, units_map

    # final fallback read entire file with guessed delim
    delim = sep_guess if sep_guess else detect_sep_from_text(raw)
    try:
        df = pd.read_csv(StringIO(raw), sep=delim, engine="c")
    except Exception:
        df = pd.read_csv(StringIO(raw), sep=delim, engine="python")
    return df, {}

# -----------------------------
# File upload + persist bytes + file switching
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload VCM Scanner / HP Tuners CSV log (.csv)", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

local_path = st.text_input("Or paste local file path (optional)")
if st.button("Load local file"):
    if os.path.exists(local_path):
        with open(local_path, "rb") as fh:
            st.session_state['uploaded_bytes'] = fh.read()
        st.success("Loaded local file")
    else:
        st.error("Local path not found")

# Compute hash of new uploaded bytes to detect change
new_bytes = None
if uploaded is not None:
    new_bytes = uploaded.getvalue()
elif 'uploaded_bytes' in st.session_state:
    new_bytes = st.session_state['uploaded_bytes']

if new_bytes:
    current_hash = hashlib.md5(new_bytes).hexdigest()
    if 'file_hash' not in st.session_state or st.session_state['file_hash'] != current_hash:
        # File changed, clear old parsed data
        st.session_state['uploaded_bytes'] = new_bytes
        st.session_state['file_hash'] = current_hash
        if 'parsed_df' in st.session_state:
            del st.session_state['parsed_df']
        if 'units_map' in st.session_state:
            del st.session_state['units_map']
        # Reset mappings
        for k in ["rpm_col","map_col","load_col","throttle_col","maf_col","maf_freq_col","stft_col","ltft_col","afr_col","timing_col"]:
            st.session_state[k] = None
    else:
        # Same file, keep data
        pass

if 'uploaded_bytes' not in st.session_state or st.session_state['uploaded_bytes'] is None:
    st.info("Upload a CSV or load a local file to begin.")
    st.stop()

raw_bytes = st.session_state['uploaded_bytes']
file_size_mb = len(raw_bytes) / (1024 * 1024)
if file_size_mb > 50:
    st.warning(f"Large file detected ({file_size_mb:.1f} MB). Parsing may be slow or fail. Consider smaller logs.")

# preview raw header candidates and let user choose if needed
try:
    raw_text_preview = raw_bytes.decode("utf-8", errors="ignore")
except Exception:
    raw_text_preview = raw_bytes.decode("latin1", errors="ignore")

candidates = detect_header_candidates(raw_text_preview, max_lines=40)
st.markdown("### Header candidate selector (use only if parsing seems wrong)")
st.markdown("If the mapping dropdowns below show numeric tokens instead of column names, choose the correct header line here and click Apply Header.")
# show list
options = [f"Line {i}: {ln}" for i, ln in candidates]
selected_header_choice = st.selectbox("Choose header line (auto-detected candidates)", options=["Auto-detect (recommended)"] + options, index=0)
# map to index
if selected_header_choice == "Auto-detect (recommended)":
    header_choice_index = None
else:
    header_choice_index = int(selected_header_choice.split(":")[0].split()[1])

if st.button("Apply Header Selection"):
    # re-parse with explicit header index
    try:
        df, units_map = extract_dataframe_from_raw(raw_bytes, header_choice_index, sep_guess=(None if sep_option=="auto-detect" else ("," if sep_option=="comma (,)" else (";" if sep_option=="semicolon (;)" else "\t"))))
        # Save parsed df and units_map to session state
        if not df.empty:
            st.session_state['parsed_df'] = df.to_csv(index=False).encode('utf-8')
            st.session_state['units_map'] = units_map
            # Reset column mappings to avoid stale references after header change
            for k in ["rpm_col","map_col","load_col","throttle_col","maf_col","maf_freq_col","stft_col","ltft_col","afr_col","timing_col"]:
                st.session_state[k] = None
            st.session_state['df_bytes_parse'] = True
            st.success("Header applied and parsed.")
        else:
            st.error("Parsing resulted in empty DataFrame.")
    except Exception as e:
        st.error(f"Failed to parse with selected header: {e}")
        st.session_state.pop('parsed_df', None)
        st.session_state.pop('units_map', None)

# if user applied header previously, use that parsed_df cached
if 'parsed_df' in st.session_state:
    df = pd.read_csv(BytesIO(st.session_state['parsed_df']))
    units_map = st.session_state.get('units_map', {})
else:
    # initial parse (auto)
    sep_guess = None
    if sep_option == "comma (,)":
        sep_guess = ","
    elif sep_option == "semicolon (;)":
        sep_guess = ";"
    elif sep_option == "tab (\\t)":
        sep_guess = "\t"
    df, units_map = extract_dataframe_from_raw(raw_bytes, header_line_index=None, sep_guess=sep_guess)

st.markdown(f"**Rows loaded:** {len(df)}")
st.dataframe(df.head(50), use_container_width=True)

# -----------------------------
# Column mapping & presets (restored)
# -----------------------------
cols = list(df.columns)

def find_col(keyword):
    if not keyword:
        return None
    kw = keyword.lower()
    for c in cols:
        if kw in str(c).lower():
            return c
    return None

# initialize session state keys for mapping so presets can set them
for k in ["rpm_col","map_col","load_col","throttle_col","maf_col","maf_freq_col","stft_col","ltft_col","afr_col","timing_col"]:
    if k not in st.session_state:
        st.session_state[k] = None

def apply_preset(preset_name):
    try:
        preset_name = preset_name.lower()
        mapping = {}
        if preset_name == "hptuners_gen3":
            mapping = {
                "rpm_col": find_col("engine rpm") or find_col("rpm"),
                "map_col": find_col("manifold absolute pressure") or find_col("map"),
                "load_col": find_col("load"),
                "throttle_col": find_col("throttle"),
                "maf_col": find_col("maf"),
                "maf_freq_col": find_col("freq") or find_col("hz"),
                "stft_col": find_col("short term") or find_col("stft"),
                "ltft_col": find_col("long term") or find_col("ltft"),
                "afr_col": find_col("afr"),
                "timing_col": find_col("ignition") or find_col("ign") or find_col("timing") or find_col("spark") or find_col("advance")
            }
        elif preset_name == "hptuners_gen4":
            mapping = {
                "rpm_col": find_col("rpm"),
                "map_col": find_col("map") or find_col("manifold"),
                "load_col": find_col("load"),
                "throttle_col": find_col("throttle") or find_col("tps"),
                "maf_col": find_col("maf"),
                "maf_freq_col": find_col("maf hz") or find_col("freq"),
                "stft_col": find_col("stft"),
                "ltft_col": find_col("ltft"),
                "afr_col": find_col("afr"),
                "timing_col": find_col("ignition") or find_col("ign") or find_col("timing") or find_col("spark") or find_col("advance")
            }
        elif preset_name == "efilive":
            mapping = {
                "rpm_col": find_col("rpm") or find_col("enginerpm"),
                "map_col": find_col("map") or find_col("manifold"),
                "load_col": find_col("load"),
                "throttle_col": find_col("throttle") or find_col("tp"),
                "maf_col": find_col("maf"),
                "maf_freq_col": find_col("freq"),
                "stft_col": find_col("stft"),
                "ltft_col": find_col("ltft"),
                "afr_col": find_col("afr"),
                "timing_col": find_col("ignition") or find_col("ign") or find_col("timing") or find_col("spark") or find_col("advance")
            }
        elif preset_name == "sct":
            mapping = {
                "rpm_col": find_col("rpm"),
                "map_col": find_col("map"),
                "load_col": find_col("load"),
                "throttle_col": find_col("tps") or find_col("throttle"),
                "maf_col": find_col("maf"),
                "maf_freq_col": find_col("maf freq") or find_col("freq"),
                "stft_col": find_col("stft"),
                "ltft_col": find_col("ltft"),
                "afr_col": find_col("afr"),
                "timing_col": find_col("ignition") or find_col("ign") or find_col("timing") or find_col("spark") or find_col("advance")
            }
        elif preset_name == "megasquirt":
            mapping = {
                "rpm_col": find_col("rpm"),
                "map_col": find_col("map"),
                "load_col": find_col("load"),
                "throttle_col": find_col("tps") or find_col("accel"),
                "maf_col": find_col("maf"),
                "maf_freq_col": find_col("maf hz") or find_col("freq"),
                "stft_col": find_col("stft"),
                "ltft_col": find_col("ltft"),
                "afr_col": find_col("afr"),
                "timing_col": find_col("ignition") or find_col("ign") or find_col("timing") or find_col("spark") or find_col("advance")
            }
        # apply only values that exist in current columns list
        for k, v in mapping.items():
            if v in cols:
                st.session_state[k] = v
            else:
                st.session_state[k] = None
        # Removed st.experimental_rerun() to avoid errors; UI updates naturally
    except Exception as e:
        st.error(f"Preset application failed: {e}")

# Preset buttons
st.markdown("### Presets")
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
with c1:
    if st.button("HP Tuners (Gen 3)"):
        apply_preset("hptuners_gen3")
with c2:
    if st.button("HP Tuners (Gen 4)"):
        apply_preset("hptuners_gen4")
with c3:
    if st.button("EFILive"):
        apply_preset("efilive")
with c4:
    if st.button("SCT"):
        apply_preset("sct")
with c5:
    if st.button("MegaSquirt"):
        apply_preset("megasquirt")

st.subheader("Column mapping (verify or override)")
rpm_col = st.selectbox("RPM column", options=[None] + cols, index=(1 + cols.index(st.session_state["rpm_col"])) if st.session_state["rpm_col"] in cols else 0, key="rpm_col")
map_col = st.selectbox("MAP column", options=[None] + cols, index=(1 + cols.index(st.session_state["map_col"])) if st.session_state["map_col"] in cols else 0, key="map_col")
load_col = st.selectbox("Load column", options=[None] + cols, index=(1 + cols.index(st.session_state["load_col"])) if st.session_state["load_col"] in cols else 0, key="load_col")
throttle_col = st.selectbox("Throttle column", options=[None] + cols, index=(1 + cols.index(st.session_state["throttle_col"])) if st.session_state["throttle_col"] in cols else 0, key="throttle_col")
maf_col = st.selectbox("MAF value column", options=[None] + cols, index=(1 + cols.index(st.session_state["maf_col"])) if st.session_state["maf_col"] in cols else 0, key="maf_col")
maf_freq_col = st.selectbox("MAF frequency column", options=[None] + cols, index=(1 + cols.index(st.session_state["maf_freq_col"])) if st.session_state["maf_freq_col"] in cols else 0, key="maf_freq_col")
stft_col = st.selectbox("STFT column", options=[None] + cols, index=(1 + cols.index(st.session_state["stft_col"])) if st.session_state["stft_col"] in cols else 0, key="stft_col")
ltft_col = st.selectbox("LTFT column", options=[None] + cols, index=(1 + cols.index(st.session_state["ltft_col"])) if st.session_state["ltft_col"] in cols else 0, key="ltft_col")
afr_col = st.selectbox("AFR / Wideband column", options=[None] + cols, index=(1 + cols.index(st.session_state["afr_col"])) if st.session_state["afr_col"] in cols else 0, key="afr_col")
timing_col = st.selectbox("Timing / Ignition column", options=[None] + cols, index=(1 + cols.index(st.session_state["timing_col"])) if st.session_state["timing_col"] in cols else 0, key="timing_col")

col_map = {
    "rpm": rpm_col, "map": map_col, "load": load_col, "throttle": throttle_col,
    "maf": maf_col, "maf_freq": maf_freq_col, "stft": stft_col, "ltft": ltft_col, "afr": afr_col,
    "timing": timing_col
}
st.write("Final mapping:", col_map)

# -----------------------------
# Filtering / steady state
# -----------------------------
def filter_steady_state(df_in):
    mask = pd.Series(True, index=df_in.index)
    try:
        if col_map["rpm"] and col_map["rpm"] in df_in.columns:
            mask &= df_in[col_map["rpm"]].diff().abs().fillna(0) < rpm_thresh
        if col_map["load"] and col_map["load"] in df_in.columns:
            mask &= df_in[col_map["load"]].diff().abs().fillna(0) < load_thresh
        if col_map["throttle"] and col_map["throttle"] in df_in.columns:
            mask &= df_in[col_map["throttle"]].diff().abs().fillna(0) < thr_thresh
    except Exception:
        pass
    # optional gating (not shown fully)
    return df_in[mask]

df_ss = filter_steady_state(df)
st.markdown(f"**Rows after steady-state & optional gates:** {len(df_ss)}")

# -----------------------------
# MAP units handling: convert to kPa if requested/auto-detected
# -----------------------------
map_unit_hint = None
if col_map["map"] and units_map:
    for k in units_map:
        if str(k).strip().lower() == str(col_map["map"]).strip().lower() or "map" in str(k).lower() or "manifold" in str(k).lower():
            map_unit_hint = units_map.get(k)
            break

convert_map_to_kpa = False
if map_unit_option == "psi (convert to kPa)":
    convert_map_to_kpa = True
elif map_unit_option == "Auto-detect from file":
    if map_unit_hint and ("psi" in str(map_unit_hint).lower() or "lb" in str(map_unit_hint).lower()):
        convert_map_to_kpa = True

if col_map["map"] and col_map["map"] in df_ss.columns and convert_map_to_kpa:
    try:
        df_ss["MAP_kPa"] = pd.to_numeric(df_ss[col_map["map"]], errors="coerce") * 6.89475729
        col_map["map"] = "MAP_kPa"
        st.info("Converted MAP to kPa (MAP_kPa column created).")
    except Exception:
        st.warning("Failed to convert MAP to kPa; check the mapped column.")

# -----------------------------
# Binning and VE computation
# -----------------------------
def create_bins(df_in):
    out = df_in.copy()
    if col_map["rpm"] and col_map["rpm"] in out.columns:
        out["RPM_bin"] = (pd.to_numeric(out[col_map["rpm"]], errors="coerce") // rpm_bin) * rpm_bin
    else:
        out["RPM_bin"] = np.nan
    if col_map["map"] and col_map["map"] in out.columns:
        out["MAP_bin"] = (pd.to_numeric(out[col_map["map"]], errors="coerce") // map_bin) * map_bin
    else:
        out["MAP_bin"] = np.nan
    return out

df_binned = create_bins(df_ss)

def compute_ve_recs(df_binned_in, min_samps, target_afr, timing_adj_factor):
    group_cols = []
    if "RPM_bin" in df_binned_in.columns:
        group_cols.append("RPM_bin")
    if "MAP_bin" in df_binned_in.columns:
        group_cols.append("MAP_bin")
    if not group_cols:
        return pd.DataFrame()
    recs = []
    grouped = df_binned_in.groupby(group_cols)
    for name, g in grouped:
        if len(g) < min_samps:
            continue
        stft = g[col_map["stft"]].mean() if col_map["stft"] and col_map["stft"] in g.columns else 0.0
        ltft = g[col_map["ltft"]].mean() if col_map["ltft"] and col_map["ltft"] in g.columns else 0.0
        correction = - (stft + ltft) / 2 if (col_map["ltft"] and col_map["ltft"] in g.columns) else -stft
        row = {}
        if "RPM_bin" in g.columns:
            row["RPM Bin"] = int(g["RPM_bin"].iloc[0]) if not pd.isna(g["RPM_bin"].iloc[0]) else ""
        if "MAP_bin" in g.columns:
            row["MAP Bin"] = g["MAP_bin"].iloc[0]
        row["Samples"] = len(g)
        row["Avg STFT (%)"] = round(stft, 2) if col_map["stft"] else ""
        row["Avg LTFT (%)"] = round(ltft, 2) if col_map["ltft"] else ""
        row["Suggested VE Adjustment (%)"] = round(correction, 2)
        row["Mean AFR"] = round(g[col_map["afr"]].mean(), 3) if col_map["afr"] and col_map["afr"] in g.columns else np.nan
        row["Mean Timing (deg)"] = round(g[col_map["timing"]].mean(), 3) if col_map["timing"] and col_map["timing"] in g.columns else np.nan
        # Timing suggestion based on AFR or fuel trims
        afr_mean = row["Mean AFR"] if not np.isnan(row["Mean AFR"]) else None
        if afr_mean is not None:
            afr_error = afr_mean - target_afr
            timing_adj = -afr_error * timing_adj_factor  # Positive AFR error (rich) -> retard (negative adj)
        else:
            # Use fuel trims as proxy: positive trim (lean) -> advance; negative (rich) -> retard
            trim_avg = (stft + ltft) / 2 if col_map["ltft"] else stft
            timing_adj = -trim_avg * timing_adj_factor  # Adjust factor as needed
        row["Suggested Timing Adj (deg)"] = round(timing_adj, 2)
        recs.append(row)
    return pd.DataFrame(recs)

ve_recs = compute_ve_recs(df_binned, min_samples, target_afr, timing_adj_factor)

# -----------------------------
# VE and Timing display & exporter
# -----------------------------
st.header("VE and Timing Suggestions")
if ve_recs.empty:
    st.info("No VE/timing correction groups met the minimum sample count or RPM/MAP columns are missing.")
else:
    st.dataframe(ve_recs.sort_values(by=["RPM Bin", "MAP Bin"] if "MAP Bin" in ve_recs.columns else ["RPM Bin"]), use_container_width=True)
    st.download_button("Download VE/timing corrections (.csv)", ve_recs.to_csv(index=False), "ve_timing_corrections.csv")

    if "RPM Bin" in ve_recs.columns and "MAP Bin" in ve_recs.columns:
        # VE heatmap
        heat_ve = ve_recs.pivot_table(index="MAP Bin", columns="RPM Bin", values="Suggested VE Adjustment (%)")
        heat_ve_kpa = heat_ve.copy()
        try:
            heat_ve_kpa.index = [float(v) for v in heat_ve_kpa.index]
        except Exception:
            pass
        heat_ve_kpa = heat_ve_kpa.sort_index()
        heat_ve_kpa = heat_ve_kpa.reindex(sorted(heat_ve_kpa.columns), axis=1)

        # Timing heatmap
        heat_timing = ve_recs.pivot_table(index="MAP Bin", columns="RPM Bin", values="Suggested Timing Adj (deg)")
        heat_timing_kpa = heat_timing.copy()
        try:
            heat_timing_kpa.index = [float(v) for v in heat_timing_kpa.index]
        except Exception:
            pass
        heat_timing_kpa = heat_timing_kpa.sort_index()
        heat_timing_kpa = heat_timing_kpa.reindex(sorted(heat_timing_kpa.columns), axis=1)

        # HP-Tuners export builder (for VE; timing separate)
        def _choose_delim(sel):
            return "," if sel.startswith("comma") else ("\t" if "tab" in sel else ";")
        def build_hpt_csv(pivot_df, export_type, variant, delimiter_sel, line_ending_sel, write_bom):
            dfp = pivot_df.copy()
            rpm_headers = [str(int(c)) for c in dfp.columns]
            delim = _choose_delim(delimiter_sel)
            lterm = "\r\n" if line_ending_sel.startswith("CRLF") else "\n"
            out = StringIO()
            writer = csv.writer(out, lineterminator=lterm, delimiter=delim, quoting=csv.QUOTE_MINIMAL)
            if variant.startswith("Variant A"):
                writer.writerow([""] + rpm_headers)
            elif variant.startswith("Variant B"):
                writer.writerow([""] + rpm_headers)
                writer.writerow(["MAP (kPa)"] + ["rpm"] * len(rpm_headers))
            elif variant.startswith("Variant C"):
                writer.writerow(["Manifold Absolute Pressure (kPa)"] + [""] * len(rpm_headers))
                writer.writerow([""] + rpm_headers)
            else:
                writer.writerow(["Manifold Absolute Pressure (kPa)"] + [""] * len(rpm_headers))
                writer.writerow([""] + rpm_headers)
                writer.writerow([""] + [""] * len(rpm_headers))
            for map_val, row in dfp.iterrows():
                try:
                    mv = float(map_val)
                    map_label = str(int(round(mv))) if abs(mv - round(mv)) < 1e-6 else f"{mv:.3f}"
                except Exception:
                    map_label = str(map_val)
                row_vals = []
                for c in dfp.columns:
                    v = dfp.at[map_val, c]
                    if pd.isna(v):
                        row_vals.append("")
                    else:
                        if export_type == "percent":
                            row_vals.append(f"{float(v):.2f}")
                        else:
                            row_vals.append(f"{float(v):.3f}")
                writer.writerow([map_label] + row_vals)
            txt = out.getvalue()
            out.close()
            data = txt.encode("utf-8-sig" if write_bom else "utf-8")
            return data

        exp_type = "percent" if hpt_export_type.startswith("Percent") or "Percent" in hpt_export_type else "absolute"
        csv_bytes = build_hpt_csv(heat_ve_kpa, exp_type, hpt_variant, hpt_delimiter, hpt_line_end, hpt_utf8_bom)
        st.download_button("Download HP‑Tuners VE Import CSV", data=csv_bytes, file_name="hptuners_ve_import.csv", mime="text/csv")

        # SCT and MegaSquirt generic exports
        if "SCT (generic pivot CSV)" in other_export:
            buf = StringIO()
            heat_ve_kpa.to_csv(buf, float_format="%.3f")
            st.download_button("Download SCT-style VE (generic) CSV", data=buf.getvalue().encode("utf-8"), file_name="sct_ve_generic.csv", mime="text/csv")
        if "MegaSquirt (generic pivot CSV)" in other_export:
            buf = StringIO()
            heat_ve_kpa.to_csv(buf, float_format="%.3f")
            st.download_button("Download MegaSquirt-style VE (generic) CSV", data=buf.getvalue().encode("utf-8"), file_name="megasquirt_ve_generic.csv", mime="text/csv")
        if "Timing adjustments (generic CSV)" in other_export:
            buf = StringIO()
            heat_timing_kpa.to_csv(buf, float_format="%.3f")
            st.download_button("Download Timing Adjustments (generic) CSV", data=buf.getvalue().encode("utf-8"), file_name="timing_adjustments_generic.csv", mime="text/csv")

        # VE heatmap
        df_ve_reset = heat_ve_kpa.reset_index()
        map_idx_col = df_ve_reset.columns[0]
        if map_idx_col != "MAP Bin":
            df_ve_reset = df_ve_reset.rename(columns={map_idx_col: "MAP Bin"})
        heat_ve_reset = df_ve_reset.melt(id_vars=["MAP Bin"], var_name="RPM Bin", value_name="VE Adj (%)").dropna()
        if not heat_ve_reset.empty:
            st.subheader("VE Heatmap (MAP × RPM)")
            heat_ve_chart = (
                alt.Chart(heat_ve_reset)
                .mark_rect()
                .encode(
                    x=alt.X("RPM Bin:O", title="RPM Bin"),
                    y=alt.Y("MAP Bin:O", title="MAP (kPa)"),
                    color=alt.Color("VE Adj (%):Q", title="Suggested VE Adj (%)", scale=alt.Scale(scheme="blueorange")),
                    tooltip=["RPM Bin", "MAP Bin", alt.Tooltip("VE Adj (%):Q", format=".2f")]
                ).properties(width=900, height=360)
            )
            st.altair_chart(heat_ve_chart, use_container_width=True)

        # Timing heatmap
        df_timing_reset = heat_timing_kpa.reset_index()
        if map_idx_col != "MAP Bin":
            df_timing_reset = df_timing_reset.rename(columns={map_idx_col: "MAP Bin"})
        heat_timing_reset = df_timing_reset.melt(id_vars=["MAP Bin"], var_name="RPM Bin", value_name="Timing Adj (deg)").dropna()
        if not heat_timing_reset.empty:
            st.subheader("Timing Heatmap (MAP × RPM)")
            heat_timing_chart = (
                alt.Chart(heat_timing_reset)
                .mark_rect()
                .encode(
                    x=alt.X("RPM Bin:O", title="RPM Bin"),
                    y=alt.Y("MAP Bin:O", title="MAP (kPa)"),
                    color=alt.Color("Timing Adj (deg):Q", title="Suggested Timing Adj (deg)", scale=alt.Scale(scheme="redblue")),
                    tooltip=["RPM Bin", "MAP Bin", alt.Tooltip("Timing Adj (deg):Q", format=".2f")]
                ).properties(width=900, height=360)
            )
            st.altair_chart(heat_timing_chart, use_container_width=True)

# -----------------------------
# Simulator
# -----------------------------
st.header("Simulator: Preview AFR / VE / Timing Impact")
if not ve_recs.empty:
    if "MAP Bin" in ve_recs.columns:
        cells = ve_recs.apply(lambda r: f"RPM {r['RPM Bin']} / MAP {round(r['MAP Bin'],3)} (samples {r['Samples']})", axis=1).tolist()
        sel_index = st.selectbox("Pick a VE cell to simulate", options=list(range(len(cells))), format_func=lambda i: cells[i])
        chosen = ve_recs.iloc[sel_index]
        st.markdown(f"**Selected:** RPM {chosen['RPM Bin']} / MAP {chosen['MAP Bin']} — Suggested VE adj: {chosen['Suggested VE Adjustment (%)']}% — Timing adj: {chosen['Suggested Timing Adj (deg)']} deg — Mean AFR: {chosen.get('Mean AFR', 'n/a')}")
        if not np.isnan(chosen.get("Mean AFR", np.nan)):
            mean_afr = chosen["Mean AFR"]
            corr_pct = chosen["Suggested VE Adjustment (%)"]
            timing_adj = chosen["Suggested Timing Adj (deg)"]
            predicted_afr = mean_afr * (1 - corr_pct/100.0) * (1 + timing_adj/100.0)  # Rough approximation
            st.write(f"Average AFR before: {mean_afr:.3f}")
            st.write(f"Predicted AFR after VE & timing adj: {predicted_afr:.3f} (approximation)")
            sim_df = pd.DataFrame({"State": ["Before", "After"], "AFR": [mean_afr, predicted_afr]})
            bar = alt.Chart(sim_df).mark_bar().encode(x="State:N", y="AFR:Q", color="State:N")
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("No AFR data available for this cell to simulate.")
else:
    st.info("No VE cells available to simulate yet.")

# -----------------------------
# MAF analysis & fixed export
# -----------------------------
st.header("MAF Suggestions")
maf_col_ui = maf_col
maf_freq_col_ui = maf_freq_col
maf_df = pd.DataFrame()
if maf_col_ui and maf_freq_col_ui and maf_col_ui in df.columns and maf_freq_col_ui in df.columns:
    df["MAF_bin"] = (pd.to_numeric(df[maf_freq_col_ui], errors="coerce") // maf_bin) * maf_bin
    recs = []
    for mb, g in df.groupby("MAF_bin"):
        if len(g) < min_samples:
            continue
        stft = g[stft_col].mean() if stft_col and stft_col in g.columns else 0.0
        ltft = g[ltft_col].mean() if ltft_col and ltft_col in g.columns else 0.0
        corr = - (stft + ltft) / 2 if ltft_col and ltft_col in g.columns else -stft
        recs.append({"MAF_bin": int(mb), "Samples": len(g), "Avg STFT (%)": round(stft,2), "Avg LTFT (%)": round(ltft,2) if ltft_col else "", "Suggested MAF Adj (%)": round(corr,2)})
    maf_df = pd.DataFrame(recs)

if maf_df.empty:
    st.info("No MAF bins met minimum sample count or columns not set.")
else:
    cols_order = ["MAF_bin", "Samples", "Avg STFT (%)", "Avg LTFT (%)", "Suggested MAF Adj (%)"]
    maf_df = maf_df.reindex(columns=[c for c in cols_order if c in maf_df.columns]).fillna("")
    st.dataframe(maf_df, use_container_width=True)
    st.download_button("Download MAF corrections (.csv)", data=maf_df.to_csv(index=False, float_format="%.3f"), file_name="maf_corrections.csv", mime="text/csv")

# -----------------------------
# Compare logs / historical (enhanced for multi-file comparison)
# -----------------------------
st.header("Compare logs / historical")
compare_file = st.file_uploader("Upload second CSV to compare VE/timing suggestions (optional)", type=["csv"], key="compare")
if compare_file:
    try:
        compare_bytes = compare_file.getvalue()
        df2, _u2 = extract_dataframe_from_raw(compare_bytes, header_line_index=None, sep_guess=None)
        df2_binned = create_bins(filter_steady_state(df2))
        ve2 = compute_ve_recs(df2_binned, min_samples, target_afr, timing_adj_factor)
        st.subheader("Second file VE/timing suggestions (preview)")
        st.dataframe(ve2.head(10))
        if not ve_recs.empty and not ve2.empty and "RPM Bin" in ve_recs.columns and "RPM Bin" in ve2.columns:
            merged = pd.merge(ve_recs, ve2, on=["RPM Bin", "MAP Bin"], how="outer", suffixes=("_A", "_B"))
            def tofloat(x):
                try: return float(x)
                except Exception: return np.nan
            merged["Delta VE (%)"] = merged.apply(lambda r: tofloat(r.get("Suggested VE Adjustment (%)_B")) - tofloat(r.get("Suggested VE Adjustment (%)_A")), axis=1)
            merged["Delta Timing (deg)"] = merged.apply(lambda r: tofloat(r.get("Suggested Timing Adj (deg)_B")) - tofloat(r.get("Suggested Timing Adj (deg)_A")), axis=1)
            st.subheader("Difference between files (B - A)")
            columns_to_show = ["RPM Bin","MAP Bin","Suggested VE Adjustment (%)_A","Suggested VE Adjustment (%)_B","Delta VE (%)","Suggested Timing Adj (deg)_A","Suggested Timing Adj (deg)_B","Delta Timing (deg)"]
            st.dataframe(merged[columns_to_show].head(50))
    except Exception as e:
        st.error(f"Failed to read/compare second file: {e}")

st.markdown("---")
st.markdown("If you still see numeric tokens in column dropdowns, pick the correct header line using the Header candidate selector at the top and press 'Apply Header Selection' — that will re-parse the file using your chosen header row.")