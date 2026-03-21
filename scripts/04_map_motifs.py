"""
Step 4 — Map MoDISco seqlet hits back to genomic coordinates and generate
per-motif, per-sequence logo + gene-map plots.

Usage
-----
    python scripts/04_map_motifs.py --config config/example_config.yaml

    # Override the measurement key at the CLI:
    python scripts/04_map_motifs.py --config config/example_config.yaml \\
        --measurement whole_transcript

Config keys read (under motif_mapping section, or top-level as fallback)
------------------------------------------------------------------------
    experiment_dir    : directory containing modisco_results.h5 and forward.meme
    measurement       : 'element_only' or 'whole_transcript'
    cosi_file_path    : coordinate CSV (same file used in steps 1-3)
    gtf_file          : GTF for gene-map visualisation
    modisco_window    : window (bp) used during MoDISco run  [default 1000]
    model_seq_len     : full Borzoi input length              [default 524288]
    selection_method  : 'modiscolite'                         [default]
    attr_respect_to   : 'element_only' or 'whole_transcript'  [default element_only]
    bigwig_dir        : optional directory of .bw coverage tracks
    motifs_of_interest: optional list of pattern labels to restrict
    motif_regex       : optional regex to filter pattern labels
    motif_limit       : optional integer cap on number of patterns plotted

Outputs (written to experiment_dir/motif_mapping_plots/)
---------------------------------------------------------
    plots/indexing_df.csv
    plots/elements_df.csv
    plots/modiscolite/<pattern_label>/<seq_name>.png  (one per hit)
    plots/modiscolite/<pattern_label>_hits.csv
"""
import os
import sys
import re
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utility_functions import (
    load_and_process_gtf, load_meme, select_longest_basic_transcripts
)
from utils.sequence_functions import (
    create_introns_dataframe, create_elements_dataframe, attribution_native_only
)
from utils.helper_functions import handle_modiscolite

DEFAULTS = {
    "model_seq_len":    524288,
    "modisco_window":   1000,
    "plot":             {"modes": ["combined", "separate"]},
    "bigwig_dir":       None,
    "selection_method": "modiscolite",
    "attr_respect_to":  "element_only",
    "motifs_of_interest": None,
    "motif_regex":      None,
    "motif_limit":      None,
}
MAPPING_SECTION = "motif_mapping"


def parse_args():
    p = argparse.ArgumentParser(description="Map MoDISco hits and generate plots")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p.parse_args()


def load_cfg(args) -> dict:
    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}

    block = dict(raw.get(MAPPING_SECTION, {}))

    def pick(key, default=None):
        if key in block and block[key] is not None:
            return block[key]
        if key in raw and raw[key] is not None:
            return raw[key]
        return default

    cfg = {**DEFAULTS, **block}
    cfg["experiment_dir"]   = pick("experiment_dir")
    cfg["modisco_window"]   = pick("modisco_window", DEFAULTS["modisco_window"])
    cfg["cosi_file_path"]   = pick("cosi_file_path")
    cfg["gtf_file"]         = pick("gtf_file")
    cfg["bigwig_dir"]       = pick("bigwig_dir", DEFAULTS["bigwig_dir"])
    cfg["attr_respect_to"]  = pick("attr_respect_to", DEFAULTS["attr_respect_to"])
    cfg["selection_method"] = pick("selection_method", DEFAULTS["selection_method"])
    cfg["plot"]             = {**DEFAULTS["plot"], **(pick("plot", {}) or {})}

    required = ["experiment_dir", "cosi_file_path", "gtf_file", "attr_respect_to"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Missing required config field(s): {missing}")

    cfg["plot_out_dir"] = os.path.join(cfg["experiment_dir"], "motif_mapping_plots")
    cfg["motif_file"]   = os.path.join(cfg["experiment_dir"], "forward.meme")
    cfg["modisco_h5"]   = os.path.join(cfg["experiment_dir"], "modisco_results.h5")

    # Fallback H5 name used by older grelu versions
    if not os.path.exists(cfg["modisco_h5"]):
        alt = os.path.join(cfg["experiment_dir"], "modisco_report.h5")
        if os.path.exists(alt):
            cfg["modisco_h5"] = alt

    # Attributions and input sequences live two levels above the modisco dir
    attr_root        = os.path.normpath(os.path.join(cfg["experiment_dir"], "..", ".."))
    cfg["attr_pkl"]  = os.path.join(attr_root, "attributions.pkl")
    cfg["seqs_pkl"]  = os.path.join(attr_root, "input_seqs.pkl")

    os.makedirs(cfg["plot_out_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["plot_out_dir"], "plots"), exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_window_start(seq_len: int, win: int) -> int:
    return (seq_len // 2) - (win // 2)


def _prep_indexing_df(cosi_csv, seq_len, modisco_window,
                       attr_respect_to="element_only", gtf_file=None):
    """
    Build the per-sequence indexing DataFrame used for coordinate mapping.

    If attr_respect_to == 'whole_transcript', transcript bounds are read from
    tscript_start/tscript_end columns in the CSV (preferred) or derived from
    the GTF (fallback).
    """
    df = pd.read_csv(cosi_csv)
    chrom_col = "chrom" if "chrom" in df.columns else "chr"

    idx = pd.DataFrame({
        "index":       df.index,
        "unique_ID": (df["unique_ID"].astype(str) if "unique_ID" in df.columns
                      else df["name"].astype(str)),
        "chrom":       df[chrom_col].astype(str),
        "start":       df["start"].astype(int),
        "end":         df["end"].astype(int),
        "strand":      df["strand"].astype(str),
    })

    if attr_respect_to == "whole_transcript":
        idx["element_start"] = idx["start"]
        idx["element_end"]   = idx["end"]

        if "tscript_start" in df.columns and "tscript_end" in df.columns:
            idx["start"] = df["tscript_start"].astype(int)
            idx["end"]   = df["tscript_end"].astype(int)
        else:
            if not gtf_file:
                raise ValueError(
                    "gtf_file must be provided when attr_respect_to == "
                    "'whole_transcript' and the CSV lacks tscript_start/tscript_end."
                )
            idx["gene_name"] = idx["unique_ID"].str.rsplit("_", n=1).str[0]
            gene_list = sorted({n.split("_")[0] for n in idx["unique_ID"].unique()})
            gtf_df = load_and_process_gtf(gtf_file, gene_list)
            gtf_df = select_longest_basic_transcripts(gtf_df)
            tx_df = (
                gtf_df.loc[gtf_df["feature"] == "transcript", ["gene_name", "start", "end"]]
                .sort_values(["gene_name", "start", "end"])
                .drop_duplicates(subset=["gene_name"], keep="first")
                .rename(columns={"start": "t_start", "end": "t_end"})
            )
            idx = idx.merge(tx_df, on="gene_name", how="left")
            have_tx = idx["t_start"].notna() & idx["t_end"].notna()
            idx.loc[have_tx, "start"] = idx.loc[have_tx, "t_start"].astype(int)
            idx.loc[have_tx, "end"]   = idx.loc[have_tx, "t_end"].astype(int)
            missing = (~have_tx).sum()
            if missing:
                print(f"Warning: {missing} rows missing transcript bounds in GTF; "
                      f"using element bounds for those rows.")
            idx = idx.drop(columns=["t_start", "t_end"])

    lengths = (idx["end"] - idx["start"]).astype(int)
    idx["tensor_start"]        = (seq_len // 2) - (lengths // 2)
    idx["tensor_window_start"] = _tensor_window_start(seq_len, modisco_window)
    return idx


def _subset_elements_for_indexing(gtf_file, indexing_df):
    """Build the elements_df (exons/introns/UTRs) for genes in indexing_df."""
    if "gene_name" in indexing_df.columns:
        gene_list = indexing_df["gene_name"].unique().tolist()
    else:
        gene_list = list({n.rsplit("_", 1)[0] for n in indexing_df["unique_ID"].unique()})
    gtf_df      = load_and_process_gtf(gtf_file, gene_list)
    gtf_df      = select_longest_basic_transcripts(gtf_df)
    introns_df  = create_introns_dataframe(gtf_df)
    elements_df = create_elements_dataframe(introns_df, gtf_df)
    return elements_df


def _maybe_load_bigwigs(bw_dir):
    if not bw_dir or not os.path.isdir(bw_dir):
        return {}
    bw = {fn[:-3]: os.path.join(bw_dir, fn)
          for fn in os.listdir(bw_dir) if fn.endswith(".bw")}
    return dict(sorted(bw.items()))


def _select_motifs(cfg, motifs):
    names = list(motifs.keys())
    if cfg.get("motifs_of_interest"):
        allow = [m for m in cfg["motifs_of_interest"] if m in motifs]
        if allow:
            names = allow
    if cfg.get("motif_regex"):
        rx    = re.compile(cfg["motif_regex"])
        names = [m for m in names if rx.search(m)]
    if cfg.get("motif_limit"):
        names = sorted(names)[:int(cfg["motif_limit"])]
    return names or list(motifs.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_cfg(args)

    # Load motifs
    if not os.path.exists(cfg["motif_file"]):
        raise FileNotFoundError(f"Motif file not found: {cfg['motif_file']}")
    motifs             = load_meme(cfg["motif_file"])
    motifs_of_interest = _select_motifs(cfg, motifs)
    print(f"Loaded {len(motifs)} motifs; selected {len(motifs_of_interest)}")

    plot_root = os.path.join(cfg["plot_out_dir"], "plots")

    # Indexing DataFrame (cached)
    idx_df_path = os.path.join(plot_root, "indexing_df.csv")
    if os.path.exists(idx_df_path):
        idx_df = pd.read_csv(idx_df_path)
    else:
        idx_df = _prep_indexing_df(
            cosi_csv=cfg["cosi_file_path"],
            seq_len=int(cfg["model_seq_len"]),
            modisco_window=int(cfg["modisco_window"]),
            attr_respect_to=cfg["attr_respect_to"],
            gtf_file=cfg["gtf_file"] if cfg["attr_respect_to"] == "whole_transcript" else None,
        )
        os.makedirs(plot_root, exist_ok=True)
        idx_df.to_csv(idx_df_path, index=False)

    # Elements DataFrame (cached)
    elements_df_path = os.path.join(plot_root, "elements_df.csv")
    if os.path.exists(elements_df_path):
        elements_df = pd.read_csv(elements_df_path)
    else:
        elements_df = _subset_elements_for_indexing(cfg["gtf_file"], idx_df)
        elements_df.to_csv(elements_df_path, index=False)

    # Attributions
    if not os.path.exists(cfg["attr_pkl"]):
        raise FileNotFoundError(f"Attributions not found: {cfg['attr_pkl']}")
    with open(cfg["attr_pkl"], "rb") as f:
        pt_attributions = pickle.load(f)
    with open(cfg["seqs_pkl"], "rb") as f:
        input_seqs = pickle.load(f)

    native_only = [attribution_native_only(pt_attributions[i], seq)
                   for i, seq in enumerate(input_seqs)]
    final_attributions = np.array(native_only)

    # Optional BigWig tracks
    bw_files = _maybe_load_bigwigs(cfg.get("bigwig_dir"))
    if bw_files:
        print(f"Loaded {len(bw_files)} BigWig tracks")

    # Run selected method
    sel = str(cfg.get("selection_method", "modiscolite")).lower()
    logo_plot_dir = os.path.join(plot_root, sel)
    os.makedirs(logo_plot_dir, exist_ok=True)

    if sel == "modiscolite":
        if not os.path.exists(cfg["modisco_h5"]):
            raise FileNotFoundError(f"MoDISco H5 not found: {cfg['modisco_h5']}")
        handle_modiscolite(
            h5_file=cfg["modisco_h5"],
            motifs_of_interest=motifs_of_interest,
            indexing_df=idx_df,
            pt_attributions=final_attributions,
            logo_plot_dir=logo_plot_dir,
            elements_df=elements_df,
            MODEL_SEQ_LEN=int(cfg["model_seq_len"]),
            MODISCO_WINDOW=int(cfg["modisco_window"]),
            bw_files=bw_files,
            y_scale=cfg.get("y_scale", "log1p"),
            figsize=tuple(cfg.get("figsize", (20, 1.5))),
            attr_respect_to=cfg.get("attr_respect_to", "element_only"),
        )
    else:
        raise ValueError(
            f"Unknown selection_method '{sel}'. Currently only 'modiscolite' is supported."
        )


if __name__ == "__main__":
    main()
