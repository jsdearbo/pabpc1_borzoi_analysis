import os
import pickle
import numpy as np
import pandas as pd

exp_dir = "/mnt/data_sda/jdearborn/scratch/pabpc1-borzoi-analysis/outputs/fig3"
with open(os.path.join(exp_dir, "input_seqs.pkl"), "rb") as f:
    input_seqs = pickle.load(f)

print("Total seqs:", len(input_seqs))

lengths = []
types = []
for i, seq in enumerate(input_seqs):
    lengths.append(len(seq) if isinstance(seq, str) else seq.shape[-1])
    types.append(str(type(seq)))

print("Lengths:")
print(pd.Series(lengths).value_counts())
print("Types:")
print(pd.Series(types).value_counts())

mapping_df = pd.read_csv(os.path.join(exp_dir, "attribution_mapping.csv"))
coord_data = pd.read_csv("/mnt/data_sda/jdearborn/scratch/pabpc1-borzoi-analysis/manuscript_runs/subset_peaks_data.csv")

subset_coord_full = coord_data.copy()
subset_mapping = pd.merge(subset_coord_full, mapping_df, left_index=True, right_on="coord_index", how="inner")
subset_indices = subset_mapping["attribution_index"].astype(int).values
subset_input_seqs = [input_seqs[i] for i in subset_indices]

sub_lengths = []
for seq in subset_input_seqs:
    sub_lengths.append(len(seq) if isinstance(seq, str) else seq.shape[-1])
print("Subset Lengths:")
print(pd.Series(sub_lengths).value_counts())
