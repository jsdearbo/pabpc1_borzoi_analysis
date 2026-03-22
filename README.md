# PABPC1 Borzoi Attribution Pipeline

Deep-learning attribution and motif discovery pipeline for identifying sequence features associated with PABPC1-dependent mRNA regulation. Uses the [Borzoi](https://github.com/calico/borzoi) sequence-to-function model (via [gReLU](https://github.com/Genentech/gReLU)) to compute per-nucleotide attributions over eCLIP-seq peaks, discovers *de novo* motifs with [TF-MoDISco](https://github.com/jmschrei/tfmodisco-lite), and measures motif enrichment between regulatory subsets.

---

## Overview

The pipeline runs in four sequential steps, each driven by a single YAML config file:

```
01_get_attributions.py   →   02_run_modisco.py   →   03_run_enrichment.py   →   04_map_motifs.py
       ↓                            ↓                         ↓                        ↓
 attributions.pkl           modisco_results.h5          sea.tsv / fimo          logo + gene-map
 input_seqs.pkl             forward.meme               enrichment CSVs             PNG plots
```

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_get_attributions.py` | Loads Borzoi, centers each peak on the input window, computes input×gradient attributions |
| 2 | `02_run_modisco.py` | Masks attributions to the peak ± flank, runs TF-MoDISco, exports motifs to MEME format |
| 3 | `03_run_enrichment.py` | Compares motif enrichment between subsets using SEA (MEME Suite) and FIMO (tangermeme) |
| 4 | `04_map_motifs.py` | Maps MoDISco seqlets back to genomic coordinates and generates logo + gene-map figures |

---

## Requirements

### System dependencies

- **MEME Suite** ≥ 5.5 (provides the `sea` command used in Step 3)
  Install: https://meme-suite.org/meme/doc/install.html
  Verify: `sea --version`

- **conda** or **mamba** (for environment setup)

### Reference genome GTF

A GTF annotation file is required for Steps 3–4 (gene-structure models and enrichment FASTA generation). The manuscript figures use **Ensembl GRCh38 release 110**:

```bash
wget https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz
gunzip Homo_sapiens.GRCh38.110.gtf.gz
```

Then set `gtf_file` in your config to the path of the uncompressed `.gtf` file.

### Python environment

Create the conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate pabpc1-borzoi
```

> **Note on gReLU / Borzoi weights.**
> `grelu` (the gReLU library) is installed from PyPI. On first run, it will automatically download the pretrained Borzoi weights (~2 GB) from the gReLU model registry. An internet connection is required for this step; subsequent runs use the cached weights.

---

## Installation

```bash
git clone https://github.com/<your-username>/pabpc1-borzoi-analysis.git
cd pabpc1-borzoi-analysis
conda env create -f environment.yml
conda activate pabpc1-borzoi
```

No additional installation steps are required — the `utils/` package is imported directly from the repository root by each script.

---

## Input data

### Coordinate file (required for all steps)

A CSV with one row per genomic element (eCLIP-seq peak, RBP site, or intron). Required columns:

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome (e.g. `chr1` or `1`) |
| `start` | Element start position (0-based) |
| `end` | Element end position |
| `strand` | `+` or `-` |
| `unique_ID` | Unique element identifier (or `name`) |

Optional columns used when present:

| Column | Description |
|--------|-------------|
| `expression` | Subset label (e.g. `up` / `down`) — used by steps 2 and 3 to split data |
| `tscript_start` / `tscript_end` | Transcript bounds for `whole_transcript` measurement mode |

### GTF file (required for Step 4)

Any Ensembl or GENCODE GTF. Used to build gene-structure models for the gene-map visualisation. The pipeline selects the longest "basic" transcript per gene.

---

## Usage

### 1. Create a config file

Copy and edit the example:

```bash
cp config/example_config.yaml config/my_run.yaml
# Edit paths and settings in my_run.yaml
```

The key fields to set in your config:

```yaml
experiment_dir:  "/path/to/outputs/my_run/saliency"
coord_file_path: "/path/to/peaks.csv"
gtf_file:        "/path/to/genome.gtf"
species:         "human"          # or "mouse"
task_id:         "ENCFF560YUT"    # ENCODE RNA-seq track ID
```

See [config/example_config.yaml](config/example_config.yaml) for all options with inline documentation.

### 2. Run the pipeline

Each script accepts `--config` and runs end-to-end for that step:

```bash
# Step 1: compute attributions (~minutes to hours depending on dataset size and GPU)
python scripts/01_get_attributions.py --config config/my_run.yaml

# Step 2: motif discovery
python scripts/02_run_modisco.py --config config/my_run.yaml

# Step 3: enrichment analysis
python scripts/03_run_enrichment.py --config config/my_run.yaml

# Step 4: generate plots
python scripts/04_map_motifs.py --config config/my_run.yaml
```

Steps must be run in order: each step reads outputs from the previous one.

---

## Output structure

```
experiment_dir/
├── attributions.pkl              # (N, 4, L) numpy array of attributions
├── input_seqs.pkl                # list of N input sequences (strings)
├── element_names_list.pkl        # list of N element names
├── attribution_mapping.csv       # coord_index → attribution_index mapping
│
├── all_peaks_modisco/
│   ├── masked_50bp_flank/
│   │   ├── modisco_results.h5    # raw TF-MoDISco output
│   │   ├── forward.meme          # de novo motifs (forward strand)
│   │   └── combined.meme         # forward + reverse-complement motifs
│   └── masked_500bp_flank/
│       └── ...
│
├── fasta_files/
│   ├── all_peaks_modisco.fa
│   ├── up_peaks_modisco.fa
│   └── down_peaks_modisco.fa
│
├── enrichment/
│   └── up_peaks_modisco_vs_down_peaks_modisco/
│       ├── sea/
│       │   ├── sea.tsv
│       │   ├── sea_enrichment.csv
│       │   └── sea_enrichment_scatter.png
│       └── fimo/
│           ├── fimo_enrichment.csv
│           └── fimo_enrichment_scatter.png
│
└── motif_mapping_plots/
    └── plots/
        ├── indexing_df.csv
        ├── elements_df.csv
        └── modiscolite/
            └── pos_pattern_0/
                ├── pos_pattern_0_hits.csv
                └── <element_name>.png    # logo + gene-map figures
```

---

## Configuration reference

All pipeline options are documented in [config/example_config.yaml](config/example_config.yaml). Key options per step:

**Step 1**
- `centering_mode`: `element_only` or `whole_transcript` — how to center the input window
- `attr_respect_to`: `element_only` or `whole_transcript` — region over which predictions are aggregated before computing gradients
- `attribution_method`: `inputxgradient` (default) or `saliency`

**Step 2**
- `mask_mode`: `element_only` masks all signal outside peak ± flank (recommended for RBP sites)
- `flank`: list of flank sizes in bp; a separate MoDISco run is produced for each (e.g. `[500, 50]`)
- `groupby_column`: column in the coordinate CSV to split data into subsets

**Step 3**
- `enrichment.source_subset` / `source_flank`: which MoDISco run to use as the motif source
- `enrichment.comparisons`: list of `{primary, control}` pairs

**Step 4**
- `motif_mapping.modisco_window`: must match the `modisco_len` used in Step 2
- `motif_mapping.motifs_of_interest`: optional list to restrict which patterns are plotted

---

## Dependencies

| Package | Role |
|---------|------|
| [gReLU](https://github.com/Genentech/gReLU) | Borzoi model loading, sequence formatting, attribution calculation |
| [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite) | *De novo* motif discovery (called via gReLU) |
| [tangermeme](https://github.com/jmschrei/tangermeme) | FIMO motif scanning |
| [MEME Suite](https://meme-suite.org/) | SEA enrichment analysis |
| [logomaker](https://github.com/jbkinney/logomaker) | Sequence logo plots |
| [pyBigWig](https://github.com/deeptools/pyBigWig) | Optional read-density tracks |

---

## Citation

If you use this pipeline, please cite:

- **Borzoi**: Linder J, et al. *Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation.* Nature Genetics (2025). https://doi.org/10.1038/s41588-024-02053-6
- **gReLU**: Nair S, et al. *gReLU: a Python library to train, interpret, and apply deep learning models to genomics.* (2024). https://github.com/Genentech/gReLU
- **TF-MoDISco / modisco-lite**: Shrikumar A, et al. (2020); Trofimova D & Shrikumar A (2023).
- **MEME Suite**: Bailey TL, et al. *The MEME Suite.* Nucleic Acids Research (2015).

> **This analysis**: [your paper citation here]

---

## License

[MIT](LICENSE) — see LICENSE file.
